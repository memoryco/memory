//! Substrate - the neural memory container

use super::engram::MemoryState;
use super::{Association, Config, Engram, EngramId};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Get current unix timestamp in seconds
fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// The neural substrate - holds engrams and their associations
#[derive(Debug)]
pub struct Substrate {
    /// Configuration for decay, propagation, etc.
    pub config: Config,

    /// All engrams in the substrate
    engrams: HashMap<EngramId, Engram>,

    /// Associations between engrams (keyed by source id)
    /// Each source can have multiple targets
    associations: HashMap<EngramId, Vec<Association>>,

    /// Reverse lookup: which engrams point TO this one?
    reverse_associations: HashMap<EngramId, Vec<EngramId>>,

    /// Track recently accessed engrams for Hebbian learning
    recent_accesses: Vec<EngramId>,

    /// How many recent accesses to track for co-activation
    recent_window: usize,

    /// Last time decay was applied (unix timestamp)
    last_decay_at: i64,
}

impl Substrate {
    /// Create a new empty substrate with default config
    pub fn new() -> Self {
        Self::with_config(Config::default())
    }

    /// Create a substrate with custom config
    pub fn with_config(config: Config) -> Self {
        Self {
            config,
            engrams: HashMap::new(),
            associations: HashMap::new(),
            reverse_associations: HashMap::new(),
            recent_accesses: Vec::new(),
            recent_window: 5, // Track last 5 accesses for co-activation
            last_decay_at: now_unix(),
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Insert an engram directly (used when loading from storage)
    /// This bypasses the normal create flow - use for hydration only
    pub fn insert_engram(&mut self, engram: Engram) {
        self.engrams.insert(engram.id, engram);
    }

    /// Insert an association directly (used when loading from storage)
    /// This bypasses the normal associate flow - use for hydration only
    pub fn insert_association(&mut self, assoc: Association) {
        // Add to forward lookup
        self.associations
            .entry(assoc.from)
            .or_default()
            .push(assoc.clone());

        // Add to reverse lookup
        self.reverse_associations
            .entry(assoc.to)
            .or_default()
            .push(assoc.from);
    }

    /// Add a new engram to the substrate
    pub fn add(&mut self, engram: Engram) -> EngramId {
        let id = engram.id;
        self.engrams.insert(id, engram);
        id
    }

    /// Create and add a new engram, returning its ID
    pub fn create(&mut self, content: impl Into<String>) -> EngramId {
        let engram = Engram::new(content);
        self.add(engram)
    }

    /// Create and add a new engram with an explicit creation timestamp
    pub fn create_with_timestamp(
        &mut self,
        content: impl Into<String>,
        created_at: i64,
    ) -> EngramId {
        let engram = Engram::new_with_timestamp(content, created_at);
        self.add(engram)
    }

    /// Create an engram with tags
    pub fn create_with_tags(&mut self, content: impl Into<String>, tags: Vec<String>) -> EngramId {
        let engram = Engram::with_tags(content, tags);
        self.add(engram)
    }

    /// Get an engram by ID (immutable) - works for ANY state
    pub fn get(&self, id: &EngramId) -> Option<&Engram> {
        self.engrams.get(id)
    }

    /// Get an engram by ID (mutable) - only if writable
    #[allow(dead_code)]
    pub fn get_mut(&mut self, id: &EngramId) -> Option<&mut Engram> {
        self.engrams.get_mut(id).filter(|e| e.state.is_writable())
    }

    /// Get an engram by ID (mutable) - bypass write check for internal ops
    fn get_mut_unchecked(&mut self, id: &EngramId) -> Option<&mut Engram> {
        self.engrams.get_mut(id)
    }

    /// Remove an engram from the substrate entirely
    /// Also removes all associations from/to this engram
    /// Returns the removed engram if it existed
    pub fn remove(&mut self, id: &EngramId) -> Option<Engram> {
        // Remove the engram
        let removed = self.engrams.remove(id);

        if removed.is_some() {
            // Remove from recent accesses
            self.recent_accesses.retain(|&acc_id| acc_id != *id);

            // Remove associations FROM this engram
            self.associations.remove(id);

            // Remove associations TO this engram (in other engrams' association lists)
            for assocs in self.associations.values_mut() {
                assocs.retain(|a| a.to != *id);
            }

            // Update reverse associations
            self.reverse_associations.remove(id);
            for sources in self.reverse_associations.values_mut() {
                sources.retain(|&source_id| source_id != *id);
            }
        }

        removed
    }

    /// Stimulate an engram and propagate to neighbors
    /// This is the main "access" operation
    /// Returns list of affected engrams and any resurrections that occurred
    pub fn stimulate(&mut self, id: EngramId, amount: f64) -> StimulationResult {
        let mut result = StimulationResult::default();
        result.affected.push(id);

        // Stimulate the target engram
        if let Some(engram) = self.get_mut_unchecked(&id) {
            if let Some(old_state) = engram.stimulate(amount) {
                result.resurrections.push((id, old_state));
            }
        } else {
            return result;
        }

        // Only do Hebbian learning and propagation for searchable engrams
        if self
            .get(&id)
            .map(|e| e.state.can_propagate())
            .unwrap_or(false)
        {
            // Hebbian learning: strengthen associations with recently accessed engrams
            let modified = self.apply_hebbian_learning(id);
            result.modified_associations = modified;

            // Track this access
            self.recent_accesses.push(id);
            if self.recent_accesses.len() > self.recent_window {
                self.recent_accesses.remove(0);
            }

            // Propagate to neighbors
            let propagated = self.propagate(id, amount);
            result.affected.extend(propagated);
        }

        result
    }

    // ==================
    // RECALL METHODS
    // ==================

    /// Recall a memory - stimulates it AND returns a clone
    /// Use this when actually referencing a memory in conversation.
    /// This is the "I'm using this memory" operation.
    ///
    /// Unlike search (passive), recall:
    /// - Strengthens the memory (stimulation)
    /// - Can resurrect archived memories
    /// - Triggers Hebbian learning with other recent recalls
    /// - Propagates to associated memories
    pub fn recall(&mut self, id: EngramId) -> RecallResult {
        let strength = self.config.recall_strength;
        self.recall_with_strength(id, strength)
    }

    /// Recall with custom stimulation strength
    pub fn recall_with_strength(&mut self, id: EngramId, strength: f64) -> RecallResult {
        let stimulation = self.stimulate(id, strength);
        let engram = self.get(&id).cloned();

        RecallResult {
            engram,
            resurrected: !stimulation.resurrections.is_empty(),
            previous_state: stimulation.resurrections.first().map(|(_, s)| *s),
            affected_ids: stimulation.affected,
            modified_associations: stimulation.modified_associations,
        }
    }

    /// Recall multiple memories at once
    /// Good for when referencing several related memories together
    pub fn recall_many(&mut self, ids: &[EngramId]) -> Vec<RecallResult> {
        ids.iter().map(|id| self.recall(*id)).collect()
    }

    /// Propagate stimulation to connected engrams
    fn propagate(&mut self, source_id: EngramId, source_energy: f64) -> Vec<EngramId> {
        let mut affected = Vec::new();

        // Get associations from this source
        let assocs = match self.associations.get(&source_id) {
            Some(a) => a.clone(),
            None => return affected,
        };

        for assoc in assocs {
            let propagated_energy =
                assoc.propagation_energy(source_energy, self.config.propagation_damping);

            // Only propagate if there's meaningful energy
            if propagated_energy > 0.01
                && let Some(target) = self.get_mut_unchecked(&assoc.to)
            {
                target.stimulate(propagated_energy);
                affected.push(assoc.to);
            }
        }

        affected
    }

    /// Apply Hebbian learning: strengthen connections between co-accessed engrams
    /// Returns list of (from, to) pairs that were modified
    fn apply_hebbian_learning(&mut self, current_id: EngramId) -> Vec<(EngramId, EngramId)> {
        let learning_rate = self.config.hebbian_learning_rate;
        let mut modified = Vec::new();

        // Clone to avoid borrow conflict
        let recent: Vec<EngramId> = self.recent_accesses.clone();

        for recent_id in recent {
            if recent_id != current_id {
                // Only form associations with searchable engrams
                let recent_searchable = self
                    .get(&recent_id)
                    .map(|e| e.is_searchable())
                    .unwrap_or(false);

                if recent_searchable {
                    // Strengthen or create association in both directions
                    if self.strengthen_or_create_association(recent_id, current_id, learning_rate) {
                        modified.push((recent_id, current_id));
                    }
                    if self.strengthen_or_create_association(current_id, recent_id, learning_rate) {
                        modified.push((current_id, recent_id));
                    }
                }
            }
        }

        modified
    }

    /// Strengthen an existing association or create a new one
    /// Returns true if an association was created or strengthened
    fn strengthen_or_create_association(
        &mut self,
        from: EngramId,
        to: EngramId,
        amount: f64,
    ) -> bool {
        let assocs = self.associations.entry(from).or_default();

        if let Some(assoc) = assocs.iter_mut().find(|a| a.to == to) {
            assoc.strengthen(amount);
            true
        } else {
            // Create new association with initial weight based on learning rate
            let mut new_assoc = Association::new(from, to);
            new_assoc.weight = amount.min(0.5); // Don't start too strong
            assocs.push(new_assoc);

            // Update reverse lookup
            self.reverse_associations.entry(to).or_default().push(from);
            true
        }
    }

    /// Manually create an association between two engrams
    pub fn associate(&mut self, from: EngramId, to: EngramId, weight: f64, ordinal: Option<u32>) {
        let assoc = Association::with_ordinal(from, to, weight, ordinal);

        self.associations.entry(from).or_default().push(assoc);

        self.reverse_associations.entry(to).or_default().push(from);
    }

    /// Get the last decay timestamp (for persistence)
    pub fn last_decay_at(&self) -> i64 {
        self.last_decay_at
    }

    /// Set the last decay timestamp (for hydration from storage)
    pub fn set_last_decay_at(&mut self, timestamp: i64) {
        self.last_decay_at = timestamp;
    }

    /// Apply time-based decay to all engrams
    /// Calculates decay based on elapsed time since last decay
    /// Returns true if decay was applied, false if not enough time elapsed
    pub fn apply_time_decay(&mut self) -> bool {
        let now = now_unix();
        let elapsed_hours = (now - self.last_decay_at) as f64 / 3600.0;

        // Only decay if enough time has passed
        if elapsed_hours < self.config.decay_interval_hours {
            return false;
        }

        // Convert elapsed time to days and calculate total decay
        let elapsed_days = elapsed_hours / 24.0;
        let total_decay = self.config.decay_rate_per_day * elapsed_days;

        // Apply decay to all engrams
        for engram in self.engrams.values_mut() {
            engram.decay(total_decay);
        }

        // Also decay associations (configurable rate)
        let assoc_decay = total_decay * self.config.association_decay_rate;
        for assocs in self.associations.values_mut() {
            for assoc in assocs.iter_mut() {
                assoc.weaken(assoc_decay);
            }
        }

        self.last_decay_at = now;
        true
    }

    /// Legacy tick_decay - applies one "tick" worth of decay
    /// Prefer apply_time_decay() for automatic time-based decay
    #[deprecated(note = "Use apply_time_decay() for time-based decay")]
    pub fn tick_decay(&mut self) {
        // For backwards compat, treat one tick as 1 hour
        let decay_rate = self.config.decay_rate_per_day / 24.0;

        for engram in self.engrams.values_mut() {
            engram.decay(decay_rate);
        }

        // Also decay associations (configurable rate)
        let assoc_decay = decay_rate * self.config.association_decay_rate;
        for assocs in self.associations.values_mut() {
            for assoc in assocs.iter_mut() {
                assoc.weaken(assoc_decay);
            }
        }
    }

    /// Prune weak associations below the minimum weight threshold
    /// Returns the number of associations pruned
    pub fn prune_weak_associations(&mut self) -> usize {
        let min_weight = self.config.min_association_weight;
        let mut pruned = 0;

        for assocs in self.associations.values_mut() {
            let before = assocs.len();
            assocs.retain(|a| a.weight >= min_weight);
            pruned += before - assocs.len();
        }

        // Also clean up reverse associations for pruned ones
        // This is a bit expensive but ensures consistency
        self.rebuild_reverse_associations();

        pruned
    }

    /// Rebuild the reverse association index from scratch
    fn rebuild_reverse_associations(&mut self) {
        self.reverse_associations.clear();

        for (from_id, assocs) in &self.associations {
            for assoc in assocs {
                self.reverse_associations
                    .entry(assoc.to)
                    .or_default()
                    .push(*from_id);
            }
        }
    }

    /// Get all engrams that are currently searchable (Active or Dormant)
    pub fn searchable_engrams(&self) -> impl Iterator<Item = &Engram> {
        self.engrams.values().filter(|e| e.is_searchable())
    }

    /// Get all engrams in deep storage
    #[allow(dead_code)]
    pub fn deep_engrams(&self) -> impl Iterator<Item = &Engram> {
        self.engrams.values().filter(|e| e.is_deep())
    }

    /// Get all archived engrams
    pub fn archived_engrams(&self) -> impl Iterator<Item = &Engram> {
        self.engrams.values().filter(|e| e.is_archived())
    }

    /// Get all engrams (for iteration/inspection)
    pub fn all_engrams(&self) -> impl Iterator<Item = &Engram> {
        self.engrams.values()
    }

    // ==================
    // DISCOVERY METHODS
    // ==================

    /// List all unique tags across all engrams
    pub fn list_tags(&self) -> Vec<String> {
        let mut tags: Vec<String> = self
            .engrams
            .values()
            .flat_map(|e| e.tags.iter().cloned())
            .collect();
        tags.sort();
        tags.dedup();
        tags
    }

    /// List tags with their counts
    pub fn list_tags_with_counts(&self) -> Vec<(String, usize)> {
        let mut tag_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        for engram in self.engrams.values() {
            for tag in &engram.tags {
                *tag_counts.entry(tag.clone()).or_insert(0) += 1;
            }
        }

        let mut result: Vec<(String, usize)> = tag_counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by count descending
        result
    }

    // ==================
    // SEARCH METHODS
    // ==================

    /// Search engrams by content (case-insensitive substring match)
    /// Only searches Active and Dormant by default
    pub fn search(&self, query: &str) -> Vec<&Engram> {
        self.search_with_options(query, SearchOptions::default())
    }

    /// Search engrams with options
    pub fn search_with_options(&self, query: &str, options: SearchOptions) -> Vec<&Engram> {
        let q = query.to_lowercase();

        let mut results: Vec<&Engram> = self
            .engrams
            .values()
            .filter(|e| {
                // State filter
                let state_ok = match options.include_states {
                    StateFilter::SearchableOnly => e.is_searchable(),
                    StateFilter::IncludeDeep => e.is_searchable() || e.is_deep(),
                    StateFilter::IncludeAll => true,
                };

                if !state_ok {
                    return false;
                }

                // Content match
                e.content.to_lowercase().contains(&q)
            })
            .collect();

        // Sort by energy (strongest first) if requested
        if options.sort_by_energy {
            results.sort_by(|a, b| {
                b.energy
                    .partial_cmp(&a.energy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Apply limit
        if let Some(limit) = options.limit {
            results.truncate(limit);
        }

        results
    }

    /// Search engrams by tag
    pub fn search_by_tag(&self, tag: &str) -> Vec<&Engram> {
        self.search_by_tag_with_options(tag, SearchOptions::default())
    }

    /// Search engrams by tag with options
    pub fn search_by_tag_with_options(&self, tag: &str, options: SearchOptions) -> Vec<&Engram> {
        let tag_lower = tag.to_lowercase();

        let mut results: Vec<&Engram> = self
            .engrams
            .values()
            .filter(|e| {
                // State filter
                let state_ok = match options.include_states {
                    StateFilter::SearchableOnly => e.is_searchable(),
                    StateFilter::IncludeDeep => e.is_searchable() || e.is_deep(),
                    StateFilter::IncludeAll => true,
                };

                if !state_ok {
                    return false;
                }

                // Tag match (case insensitive)
                e.tags.iter().any(|t| t.to_lowercase() == tag_lower)
            })
            .collect();

        if options.sort_by_energy {
            results.sort_by(|a, b| {
                b.energy
                    .partial_cmp(&a.energy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        if let Some(limit) = options.limit {
            results.truncate(limit);
        }

        results
    }

    /// Search engrams by multiple tags
    pub fn search_by_tags(&self, tags: &[&str], mode: TagMatchMode) -> Vec<&Engram> {
        self.search_by_tags_with_options(tags, mode, SearchOptions::default())
    }

    /// Search engrams by multiple tags with options
    pub fn search_by_tags_with_options(
        &self,
        tags: &[&str],
        mode: TagMatchMode,
        options: SearchOptions,
    ) -> Vec<&Engram> {
        let tags_lower: Vec<String> = tags.iter().map(|t| t.to_lowercase()).collect();

        let mut results: Vec<&Engram> = self
            .engrams
            .values()
            .filter(|e| {
                // State filter
                let state_ok = match options.include_states {
                    StateFilter::SearchableOnly => e.is_searchable(),
                    StateFilter::IncludeDeep => e.is_searchable() || e.is_deep(),
                    StateFilter::IncludeAll => true,
                };

                if !state_ok {
                    return false;
                }

                // Tag match based on mode
                let engram_tags: Vec<String> = e.tags.iter().map(|t| t.to_lowercase()).collect();

                match mode {
                    TagMatchMode::All => tags_lower.iter().all(|t| engram_tags.contains(t)),
                    TagMatchMode::Any => tags_lower.iter().any(|t| engram_tags.contains(t)),
                }
            })
            .collect();

        if options.sort_by_energy {
            results.sort_by(|a, b| {
                b.energy
                    .partial_cmp(&a.energy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        if let Some(limit) = options.limit {
            results.truncate(limit);
        }

        results
    }

    /// Find engrams associated with a given engram
    /// Follows outbound associations and returns connected engrams
    pub fn find_associated(&self, id: &EngramId) -> Vec<(&Engram, f64)> {
        let mut results: Vec<(&Engram, f64)> = Vec::new();

        if let Some(assocs) = self.associations.get(id) {
            for assoc in assocs {
                if let Some(target) = self.engrams.get(&assoc.to) {
                    results.push((target, assoc.weight));
                }
            }
        }

        // Sort by weight descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Get associations from a specific engram
    pub fn associations_from(&self, id: &EngramId) -> Option<&Vec<Association>> {
        self.associations.get(id)
    }

    /// Get IDs of engrams that point to the given engram
    pub fn associations_to(&self, id: &EngramId) -> Option<&Vec<EngramId>> {
        self.reverse_associations.get(id)
    }

    /// Get all associations in the substrate
    pub fn all_associations(&self) -> Vec<&Association> {
        self.associations.values().flat_map(|v| v.iter()).collect()
    }

    /// Count of engrams in the substrate (all states)
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.engrams.len()
    }

    /// Is the substrate empty?
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.engrams.is_empty()
    }

    /// Get substrate statistics
    pub fn stats(&self) -> SubstrateStats {
        let total = self.engrams.len();

        let mut active = 0;
        let mut dormant = 0;
        let mut deep = 0;
        let mut archived = 0;
        let mut total_energy = 0.0;

        for e in self.engrams.values() {
            total_energy += e.energy;
            match e.state {
                MemoryState::Active => active += 1,
                MemoryState::Dormant => dormant += 1,
                MemoryState::Deep => deep += 1,
                MemoryState::Archived => archived += 1,
            }
        }

        let avg_energy = if total > 0 {
            total_energy / total as f64
        } else {
            0.0
        };

        let association_count: usize = self.associations.values().map(|v| v.len()).sum();

        SubstrateStats {
            total_engrams: total,
            active_engrams: active,
            dormant_engrams: dormant,
            deep_engrams: deep,
            archived_engrams: archived,
            total_associations: association_count,
            average_energy: avg_energy,
        }
    }
}

impl Default for Substrate {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a stimulation operation
#[derive(Debug, Default)]
pub struct StimulationResult {
    /// All engrams that were affected (directly or through propagation)
    pub affected: Vec<EngramId>,
    /// Engrams that were resurrected from Deep or Archived state
    pub resurrections: Vec<(EngramId, MemoryState)>,
    /// Associations that were created or strengthened by Hebbian learning
    /// Stored as (from_id, to_id) pairs
    pub modified_associations: Vec<(EngramId, EngramId)>,
}

/// Result of a recall operation
#[derive(Debug, Clone)]
pub struct RecallResult {
    /// The recalled engram (None if not found)
    pub engram: Option<Engram>,
    /// Whether this recall resurrected the memory
    pub resurrected: bool,
    /// The state it was resurrected from (if applicable)
    pub previous_state: Option<MemoryState>,
    /// IDs of all engrams affected (the recalled one + propagation targets)
    pub affected_ids: Vec<EngramId>,
    /// Associations that were created or strengthened by Hebbian learning
    /// Stored as (from_id, to_id) pairs
    pub modified_associations: Vec<(EngramId, EngramId)>,
}

impl RecallResult {
    /// Check if the recall found a memory
    pub fn found(&self) -> bool {
        self.engram.is_some()
    }

    /// Get the content if found
    pub fn content(&self) -> Option<&str> {
        self.engram.as_ref().map(|e| e.content.as_str())
    }

    /// How many engrams were affected
    pub fn affected_count(&self) -> usize {
        self.affected_ids.len()
    }
}

/// Statistics about the substrate state
#[derive(Debug, Clone)]
pub struct SubstrateStats {
    pub total_engrams: usize,
    pub active_engrams: usize,
    pub dormant_engrams: usize,
    pub deep_engrams: usize,
    pub archived_engrams: usize,
    pub total_associations: usize,
    pub average_energy: f64,
}

/// Options for search queries
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Which memory states to include in search
    pub include_states: StateFilter,
    /// Sort results by energy (strongest first)
    pub sort_by_energy: bool,
    /// Maximum number of results to return
    pub limit: Option<usize>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            include_states: StateFilter::SearchableOnly,
            sort_by_energy: true,
            limit: None,
        }
    }
}

impl SearchOptions {
    /// Include all states including archived
    pub fn include_all(mut self) -> Self {
        self.include_states = StateFilter::IncludeAll;
        self
    }

    /// Include deep storage but not archived
    pub fn include_deep(mut self) -> Self {
        self.include_states = StateFilter::IncludeDeep;
        self
    }

    /// Limit results
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Don't sort by energy
    pub fn unsorted(mut self) -> Self {
        self.sort_by_energy = false;
        self
    }
}

/// Filter for which memory states to include in search
#[derive(Debug, Clone, Copy, Default)]
pub enum StateFilter {
    /// Only Active and Dormant (default)
    #[default]
    SearchableOnly,
    /// Active, Dormant, and Deep
    IncludeDeep,
    /// All states including Archived
    IncludeAll,
}

/// How to match multiple tags
#[derive(Debug, Clone, Copy)]
pub enum TagMatchMode {
    /// Must match ALL specified tags
    All,
    /// Must match ANY of the specified tags
    Any,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_substrate() -> Substrate {
        let mut substrate = Substrate::new();

        substrate.create_with_tags(
            "Rust FFI patterns discussion",
            vec!["work".into(), "rust".into()],
        );
        substrate.create_with_tags(
            "Swift concurrency debugging",
            vec!["work".into(), "swift".into()],
        );
        substrate.create_with_tags("Wedding venue planning", vec!["personal".into()]);
        substrate.create_with_tags("Beer tasting notes", vec!["personal".into()]);
        substrate.create_with_tags(
            "TwilioDataCore architecture",
            vec!["work".into(), "rust".into()],
        );

        substrate
    }

    #[test]
    fn create_and_retrieve_engram() {
        let mut substrate = Substrate::new();
        let id = substrate.create("test memory");

        let engram = substrate.get(&id).unwrap();
        assert_eq!(engram.content, "test memory");
        assert_eq!(engram.energy, 1.0);
    }

    #[test]
    fn stimulate_updates_energy_and_access() {
        let mut substrate = Substrate::new();
        let id = substrate.create("test");

        // Drain some energy first
        substrate.get_mut(&id).unwrap().energy = 0.5;

        substrate.stimulate(id, 0.3);

        let engram = substrate.get(&id).unwrap();
        assert!((engram.energy - 0.8).abs() < 0.001);
        assert_eq!(engram.access_count, 1);
    }

    #[test]
    fn manual_association() {
        let mut substrate = Substrate::new();
        let a = substrate.create("memory A");
        let b = substrate.create("memory B");

        substrate.associate(a, b, 0.8, None);

        let assocs = substrate.associations_from(&a).unwrap();
        assert_eq!(assocs.len(), 1);
        assert_eq!(assocs[0].to, b);
        assert_eq!(assocs[0].weight, 0.8);
    }

    #[test]
    fn propagation_affects_neighbors() {
        let mut substrate = Substrate::new();
        let a = substrate.create("memory A");
        let b = substrate.create("memory B");

        // Set B to lower energy so we can see the increase
        substrate.get_mut(&b).unwrap().energy = 0.5;

        // Create strong association A -> B
        substrate.associate(a, b, 0.8, None);

        // Stimulate A
        let result = substrate.stimulate(a, 1.0);

        // B should be affected
        assert!(result.affected.contains(&b));

        // B's energy should have increased
        let b_energy = substrate.get(&b).unwrap().energy;
        assert!(b_energy > 0.5);
    }

    #[test]
    fn hebbian_learning_creates_associations() {
        let mut substrate = Substrate::new();
        let a = substrate.create("memory A");
        let b = substrate.create("memory B");

        // Access both in sequence
        substrate.stimulate(a, 1.0);
        substrate.stimulate(b, 1.0);

        // Should have created associations between them
        let assocs_from_a = substrate.associations_from(&a);
        let assocs_from_b = substrate.associations_from(&b);

        // A should now point to B (created when B was accessed after A)
        assert!(assocs_from_a.is_some());
        assert!(assocs_from_b.is_some());
    }

    #[test]
    fn decay_reduces_energy() {
        let mut substrate = Substrate::new();
        let id = substrate.create("test");

        // Backdate last decay so apply_time_decay actually fires
        substrate.set_last_decay_at(substrate.last_decay_at - 86400);
        substrate.apply_time_decay();

        let engram = substrate.get(&id).unwrap();
        assert!(engram.energy < 1.0);
    }

    #[test]
    fn engrams_sink_to_deep_and_archive() {
        let mut substrate = Substrate::new();
        let id = substrate.create("sinking memory");

        // Manually decay - the new time-based decay uses smaller increments
        // So we just directly manipulate energy to test state transitions
        {
            let e = substrate.get_mut_unchecked(&id).unwrap();
            e.energy = 0.25;
            e.state = MemoryState::from_energy(e.energy);
        }
        assert_eq!(substrate.get(&id).unwrap().state, MemoryState::Dormant);

        {
            let e = substrate.get_mut_unchecked(&id).unwrap();
            e.energy = 0.05;
            e.state = MemoryState::from_energy(e.energy);
        }
        assert_eq!(substrate.get(&id).unwrap().state, MemoryState::Deep);

        {
            let e = substrate.get_mut_unchecked(&id).unwrap();
            e.energy = 0.01;
            e.state = MemoryState::from_energy(e.energy);
        }
        assert_eq!(substrate.get(&id).unwrap().state, MemoryState::Archived);
        assert!(substrate.get(&id).unwrap().energy > 0.0); // But never deleted!
    }

    #[test]
    fn resurrection_from_archive() {
        let mut substrate = Substrate::new();
        let id = substrate.create("archived memory");

        // Force to archived state
        {
            let e = substrate.get_mut_unchecked(&id).unwrap();
            e.energy = 0.01;
            e.state = MemoryState::Archived;
        }

        // Stimulate with strong signal
        let result = substrate.stimulate(id, 0.5);

        // Should have been resurrected
        assert_eq!(result.resurrections.len(), 1);
        assert_eq!(result.resurrections[0].1, MemoryState::Archived);

        // Now active again
        let engram = substrate.get(&id).unwrap();
        assert_eq!(engram.state, MemoryState::Active);
    }

    #[test]
    fn stats_track_all_states() {
        let mut substrate = Substrate::new();
        let _a = substrate.create("A");
        let b = substrate.create("B");
        let c = substrate.create("C");
        let d = substrate.create("D");

        // Set different energy levels to get different states
        substrate.get_mut(&b).unwrap().energy = 0.2;
        substrate.get_mut(&b).unwrap().state = MemoryState::Dormant;

        {
            let e = substrate.get_mut_unchecked(&c).unwrap();
            e.energy = 0.05;
            e.state = MemoryState::Deep;
        }

        {
            let e = substrate.get_mut_unchecked(&d).unwrap();
            e.energy = 0.01;
            e.state = MemoryState::Archived;
        }

        let stats = substrate.stats();

        assert_eq!(stats.total_engrams, 4);
        assert_eq!(stats.active_engrams, 1); // a
        assert_eq!(stats.dormant_engrams, 1); // b
        assert_eq!(stats.deep_engrams, 1); // c
        assert_eq!(stats.archived_engrams, 1); // d
    }

    // =================
    // DISCOVERY TESTS
    // =================

    #[test]
    fn list_tags_returns_unique_sorted() {
        let substrate = build_test_substrate();
        let tags = substrate.list_tags();

        assert_eq!(tags.len(), 4); // work, rust, swift, personal
        assert!(tags.contains(&"work".to_string()));
        assert!(tags.contains(&"rust".to_string()));
        assert!(tags.contains(&"personal".to_string()));
    }

    #[test]
    fn list_tags_with_counts() {
        let substrate = build_test_substrate();
        let tag_counts = substrate.list_tags_with_counts();

        // work should have highest count (3)
        assert_eq!(tag_counts[0].0, "work");
        assert_eq!(tag_counts[0].1, 3);

        // rust and personal should have 2 each
        let rust_count = tag_counts.iter().find(|(t, _)| t == "rust").unwrap().1;
        let personal_count = tag_counts.iter().find(|(t, _)| t == "personal").unwrap().1;
        assert_eq!(rust_count, 2);
        assert_eq!(personal_count, 2);
    }

    // =================
    // SEARCH TESTS
    // =================

    #[test]
    fn search_by_content() {
        let substrate = build_test_substrate();
        let results = substrate.search("rust");

        // Should find "Rust FFI patterns" and "TwilioDataCore architecture" (has rust tag but also matches content)
        assert_eq!(results.len(), 1); // Only "Rust FFI patterns" has "rust" in content
        assert!(results[0].content.to_lowercase().contains("rust"));
    }

    #[test]
    fn search_is_case_insensitive() {
        let substrate = build_test_substrate();

        let results1 = substrate.search("rust");
        let results2 = substrate.search("RUST");
        let results3 = substrate.search("Rust");

        assert_eq!(results1.len(), results2.len());
        assert_eq!(results2.len(), results3.len());
    }

    #[test]
    fn search_by_tag() {
        let substrate = build_test_substrate();
        let results = substrate.search_by_tag("work");

        assert_eq!(results.len(), 3); // 3 work-tagged items
    }

    #[test]
    fn search_by_tag_case_insensitive() {
        let substrate = build_test_substrate();

        let results1 = substrate.search_by_tag("work");
        let results2 = substrate.search_by_tag("WORK");

        assert_eq!(results1.len(), results2.len());
    }

    #[test]
    fn search_by_tags_all_mode() {
        let substrate = build_test_substrate();

        // Must have BOTH work AND rust
        let results = substrate.search_by_tags(&["work", "rust"], TagMatchMode::All);

        assert_eq!(results.len(), 2); // "Rust FFI" and "TwilioDataCore"
    }

    #[test]
    fn search_by_tags_any_mode() {
        let substrate = build_test_substrate();

        // Must have work OR personal
        let results = substrate.search_by_tags(&["work", "personal"], TagMatchMode::Any);

        assert_eq!(results.len(), 5); // All of them
    }

    #[test]
    fn search_respects_state_filter() {
        let mut substrate = build_test_substrate();

        // Archive one of the engrams
        let archived_id = substrate.search("Wedding")[0].id;
        {
            let e = substrate.get_mut_unchecked(&archived_id).unwrap();
            e.energy = 0.01;
            e.state = MemoryState::Archived;
        }

        // Default search should not find archived
        let default_results = substrate.search_by_tag("personal");
        assert_eq!(default_results.len(), 1); // Only Beer

        // With include_all, should find both
        let all_results = substrate
            .search_by_tag_with_options("personal", SearchOptions::default().include_all());
        assert_eq!(all_results.len(), 2); // Beer and Wedding
    }

    #[test]
    fn search_with_limit() {
        let substrate = build_test_substrate();

        let results =
            substrate.search_by_tag_with_options("work", SearchOptions::default().with_limit(2));

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn find_associated() {
        let mut substrate = Substrate::new();
        let a = substrate.create("Memory A");
        let b = substrate.create("Memory B");
        let c = substrate.create("Memory C");

        substrate.associate(a, b, 0.8, None);
        substrate.associate(a, c, 0.5, None);

        let associated = substrate.find_associated(&a);

        assert_eq!(associated.len(), 2);
        // Should be sorted by weight descending
        assert_eq!(associated[0].1, 0.8); // B first
        assert_eq!(associated[1].1, 0.5); // C second
    }

    // =================
    // RECALL TESTS
    // =================

    #[test]
    fn recall_returns_engram() {
        let mut substrate = Substrate::new();
        let id = substrate.create("Test memory");

        let result = substrate.recall(id);

        assert!(result.found());
        assert_eq!(result.content(), Some("Test memory"));
    }

    #[test]
    fn recall_stimulates_memory() {
        let mut substrate = Substrate::new();
        let id = substrate.create("Test memory");

        // Lower energy first
        substrate.get_mut(&id).unwrap().energy = 0.5;

        let result = substrate.recall(id);

        // Energy should have increased
        assert!(result.engram.unwrap().energy > 0.5);
    }

    #[test]
    fn recall_triggers_hebbian_learning() {
        let mut substrate = Substrate::new();
        let a = substrate.create("Memory A");
        let b = substrate.create("Memory B");

        // Recall both in sequence
        substrate.recall(a);
        substrate.recall(b);

        // Should have formed associations
        let assocs = substrate.associations_from(&a);
        assert!(assocs.is_some());
        assert!(!assocs.unwrap().is_empty());
    }

    #[test]
    fn recall_resurrects_archived() {
        let mut substrate = Substrate::new();
        let id = substrate.create("Archived memory");

        // Archive it
        {
            let e = substrate.get_mut_unchecked(&id).unwrap();
            e.energy = 0.01;
            e.state = MemoryState::Archived;
        }

        // Recall should resurrect it
        let result = substrate.recall_with_strength(id, 0.5);

        assert!(result.resurrected);
        assert_eq!(result.previous_state, Some(MemoryState::Archived));
        assert!(result.engram.unwrap().state != MemoryState::Archived);
    }

    #[test]
    fn recall_many() {
        let mut substrate = Substrate::new();
        let a = substrate.create("Memory A");
        let b = substrate.create("Memory B");
        let c = substrate.create("Memory C");

        let results = substrate.recall_many(&[a, b, c]);

        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.found()));
    }

    #[test]
    fn recall_not_found() {
        let mut substrate = Substrate::new();
        let fake_id = uuid::Uuid::new_v4();

        let result = substrate.recall(fake_id);

        assert!(!result.found());
        assert!(result.content().is_none());
    }
}
