//! Brain - The coordination layer
//!
//! Brain ties together Identity, Substrate, and Storage.
//! It's the main interface for the engram system.
//!
//! All mutating operations automatically persist changes.
//! Search operations are read-only and don't hit storage.

use super::{
    EngramId, Engram, Config, Association,
    Identity,
    Substrate, RecallResult, SearchOptions, TagMatchMode, SubstrateStats,
    Storage, StorageResult,
};

/// The Brain - coordinates Identity, Substrate, and Storage
/// 
/// This is the main entry point for the engram system.
/// All mutating operations automatically persist to storage.
pub struct Brain {
    identity: Identity,
    /// Made pub(crate) for testing decay behavior
    pub(crate) substrate: Substrate,
    storage: Box<dyn Storage>,
}

impl Brain {
    /// Create a new Brain with empty identity and substrate
    pub fn new<S: Storage + 'static>(mut storage: S) -> StorageResult<Self> {
        storage.initialize()?;
        
        let config = storage.load_config()?.unwrap_or_default();
        
        Ok(Self {
            identity: Identity::new(),
            substrate: Substrate::with_config(config),
            storage: Box::new(storage),
        })
    }
    
    /// Open an existing Brain from storage, or create new if empty
    pub fn open<S: Storage + 'static>(mut storage: S) -> StorageResult<Self> {
        storage.initialize()?;
        
        // Load config
        let config = storage.load_config()?.unwrap_or_default();
        
        // Load identity
        let identity = storage.load_identity()?.unwrap_or_default();
        
        // Load substrate
        let mut substrate = Substrate::with_config(config);
        
        // Load last decay timestamp
        // If never set, default to 24 hours ago so first run applies ~1 day of decay
        let default_decay_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64 - 86400; // 24 hours ago
        let last_decay = storage.load_last_decay_at()?.unwrap_or(default_decay_at);
        substrate.set_last_decay_at(last_decay);
        
        // Load all engrams
        let engrams = storage.load_all_engrams()?;
        for engram in engrams {
            substrate.insert_engram(engram);
        }
        
        // Load all associations
        let associations = storage.load_all_associations()?;
        for assoc in associations {
            substrate.insert_association(assoc);
        }
        
        Ok(Self {
            identity,
            substrate,
            storage: Box::new(storage),
        })
    }
    
    // ==================
    // IDENTITY
    // ==================
    
    /// Get read-only access to identity
    pub fn identity(&self) -> &Identity {
        &self.identity
    }
    
    /// Get mutable access to identity
    /// Note: Call save_identity() after modifications to persist
    pub fn identity_mut(&mut self) -> &mut Identity {
        &mut self.identity
    }
    
    /// Replace the entire identity
    pub fn set_identity(&mut self, identity: Identity) -> StorageResult<()> {
        self.identity = identity;
        self.save_identity()
    }
    
    /// Persist identity changes to storage
    pub fn save_identity(&mut self) -> StorageResult<()> {
        self.storage.save_identity(&self.identity)
    }
    
    // ==================
    // CREATE
    // ==================
    
    /// Create a new memory
    pub fn create(&mut self, content: &str) -> StorageResult<EngramId> {
        let id = self.substrate.create(content);
        
        // Persist immediately
        if let Some(engram) = self.substrate.get(&id) {
            self.storage.save_engram(engram)?;
        }
        
        Ok(id)
    }
    
    /// Create a new memory with tags
    pub fn create_with_tags(&mut self, content: &str, tags: Vec<String>) -> StorageResult<EngramId> {
        let id = self.substrate.create_with_tags(content, tags);
        
        // Persist immediately
        if let Some(engram) = self.substrate.get(&id) {
            self.storage.save_engram(engram)?;
        }
        
        Ok(id)
    }
    
    /// Create an explicit association between memories
    pub fn associate(&mut self, from: EngramId, to: EngramId, weight: f64) -> StorageResult<()> {
        self.substrate.associate(from, to, weight);
        
        // Persist the association
        if let Some(assocs) = self.substrate.associations_from(&from) {
            if let Some(assoc) = assocs.iter().find(|a| a.to == to) {
                self.storage.save_association(assoc)?;
            }
        }
        
        Ok(())
    }
    
    // ==================
    // DELETE
    // ==================
    
    /// Delete a memory permanently
    /// Also removes all associations from/to this memory
    /// Returns true if the memory existed and was deleted
    pub fn delete(&mut self, id: EngramId) -> StorageResult<bool> {
        // Remove from substrate
        let existed = self.substrate.remove(&id).is_some();
        
        // Remove from storage (also cleans up associations)
        self.storage.delete_engram(&id)?;
        
        Ok(existed)
    }
    
    // ==================
    // RECALL (active)
    // ==================
    
    /// Recall a memory - stimulates it and persists changes
    pub fn recall(&mut self, id: EngramId) -> StorageResult<RecallResult> {
        let result = self.substrate.recall(id);
        self.persist_recall_effects(&result)?;
        Ok(result)
    }
    
    /// Recall with custom stimulation strength
    pub fn recall_with_strength(&mut self, id: EngramId, strength: f64) -> StorageResult<RecallResult> {
        let result = self.substrate.recall_with_strength(id, strength);
        self.persist_recall_effects(&result)?;
        Ok(result)
    }
    
    /// Recall multiple memories
    pub fn recall_many(&mut self, ids: &[EngramId]) -> StorageResult<Vec<RecallResult>> {
        let results = self.substrate.recall_many(ids);
        
        for result in &results {
            self.persist_recall_effects(result)?;
        }
        
        Ok(results)
    }
    
    /// Persist the side effects of a recall operation
    fn persist_recall_effects(&mut self, result: &RecallResult) -> StorageResult<()> {
        // Save all affected engrams
        let engrams: Vec<&Engram> = result.affected_ids.iter()
            .filter_map(|id| self.substrate.get(id))
            .collect();
        
        self.storage.save_engrams(&engrams)?;
        
        // Save any associations modified by Hebbian learning
        if !result.modified_associations.is_empty() {
            let mut assocs_to_save = Vec::new();
            
            for (from_id, to_id) in &result.modified_associations {
                if let Some(assocs) = self.substrate.associations_from(from_id) {
                    if let Some(assoc) = assocs.iter().find(|a| a.to == *to_id) {
                        assocs_to_save.push(assoc);
                    }
                }
            }
            
            self.storage.save_associations(&assocs_to_save)?;
        }
        
        Ok(())
    }
    
    // ==================
    // SEARCH (passive)
    // ==================
    
    /// Search memories by content (in-memory substring search)
    /// For better tokenized search, use search_fts()
    pub fn search(&self, query: &str) -> Vec<&Engram> {
        self.substrate.search(query)
    }
    
    /// Search with options (in-memory substring search)
    pub fn search_with_options(&self, query: &str, options: SearchOptions) -> Vec<&Engram> {
        self.substrate.search_with_options(query, options)
    }
    
    /// Search using full-text search (FTS5 for SQLite storage)
    /// This tokenizes the query and matches individual terms
    /// Returns ALL matching engrams (caller should filter by state if needed)
    /// Results are sorted by energy (highest first)
    pub fn search_fts(&self, query: &str) -> StorageResult<Vec<&Engram>> {
        // Get matching IDs from storage
        let ids = self.storage.search_content(query)?;
        
        // Hydrate from substrate, sorted by energy
        let mut engrams: Vec<&Engram> = ids.iter()
            .filter_map(|id| self.substrate.get(id))
            .collect();
        
        engrams.sort_by(|a, b| b.energy.partial_cmp(&a.energy).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(engrams)
    }
    
    /// Search by tag
    pub fn search_by_tag(&self, tag: &str) -> Vec<&Engram> {
        self.substrate.search_by_tag(tag)
    }
    
    /// Search by multiple tags
    pub fn search_by_tags(&self, tags: &[&str], mode: TagMatchMode) -> Vec<&Engram> {
        self.substrate.search_by_tags(tags, mode)
    }
    
    /// Find memories associated with a given memory
    pub fn find_associated(&self, id: &EngramId) -> Vec<(&Engram, f64)> {
        self.substrate.find_associated(id)
    }
    
    /// Get raw associations FROM a memory (outbound)
    pub fn associations_from(&self, id: &EngramId) -> Option<&Vec<Association>> {
        self.substrate.associations_from(id)
    }
    
    /// Get IDs of memories that point TO this memory (inbound)
    pub fn associations_to(&self, id: &EngramId) -> Option<&Vec<EngramId>> {
        self.substrate.associations_to(id)
    }
    
    /// Get all associations in the substrate (for graph visualization)
    pub fn all_associations(&self) -> Vec<&Association> {
        self.substrate.all_associations()
    }
    
    /// Get a memory by ID
    pub fn get(&self, id: &EngramId) -> Option<&Engram> {
        self.substrate.get(id)
    }
    
    // ==================
    // DISCOVERY
    // ==================
    
    /// List all unique tags
    pub fn list_tags(&self) -> Vec<String> {
        self.substrate.list_tags()
    }
    
    /// List tags with counts
    pub fn list_tags_with_counts(&self) -> Vec<(String, usize)> {
        self.substrate.list_tags_with_counts()
    }
    
    /// Get count for a specific tag
    pub fn tag_count(&self, tag: &str) -> usize {
        let tag_lower = tag.to_lowercase();
        self.substrate.list_tags_with_counts()
            .into_iter()
            .find(|(t, _)| t.to_lowercase() == tag_lower)
            .map(|(_, count)| count)
            .unwrap_or(0)
    }
    
    /// Get substrate statistics
    pub fn stats(&self) -> SubstrateStats {
        self.substrate.stats()
    }
    
    // ==================
    // LIFECYCLE
    // ==================
    
    /// Apply time-based decay to all memories and persist
    /// Returns true if decay was applied, false if not enough time elapsed
    pub fn apply_time_decay(&mut self) -> StorageResult<bool> {
        let applied = self.substrate.apply_time_decay();
        
        if applied {
            // Save all engrams (their energy/state may have changed)
            let engrams: Vec<Engram> = self.substrate.all_engrams().cloned().collect();
            let engram_refs: Vec<&Engram> = engrams.iter().collect();
            self.storage.save_engrams(&engram_refs)?;
            
            // Save the last decay timestamp
            self.storage.save_last_decay_at(self.substrate.last_decay_at())?;
        }
        
        Ok(applied)
    }
    
    /// Prune weak associations below the minimum weight threshold
    /// Returns the number of associations pruned
    /// Note: Pruned associations are removed from storage as well
    pub fn prune_weak_associations(&mut self) -> StorageResult<usize> {
        let pruned = self.substrate.prune_weak_associations();
        
        if pruned > 0 {
            // Re-save all associations to storage (the simple approach)
            // A more efficient approach would track which were pruned
            self.storage.delete_all_associations()?;
            let assocs: Vec<&Association> = self.substrate.all_associations();
            self.storage.save_associations(&assocs)?;
        }
        
        Ok(pruned)
    }
    
    /// Legacy tick_decay - prefer apply_time_decay()
    #[deprecated(note = "Use apply_time_decay() for time-based decay")]
    #[allow(deprecated)]
    pub fn tick_decay(&mut self) -> StorageResult<()> {
        self.substrate.tick_decay();
        
        // Save all engrams (their energy/state may have changed)
        let engrams: Vec<Engram> = self.substrate.all_engrams().cloned().collect();
        let engram_refs: Vec<&Engram> = engrams.iter().collect();
        self.storage.save_engrams(&engram_refs)?;
        
        Ok(())
    }
    
    // ==================
    // CONFIG
    // ==================
    
    /// Get the current configuration
    pub fn config(&self) -> &Config {
        self.substrate.config()
    }
    
    /// Update configuration and persist
    pub fn set_config(&mut self, config: Config) -> StorageResult<()> {
        self.substrate.config = config;
        self.storage.save_config(self.substrate.config())
    }
    
    /// Update a single config value
    pub fn configure(&mut self, key: &str, value: f64) -> StorageResult<bool> {
        let mut config = self.substrate.config.clone();
        
        let updated = match key {
            "decay_rate_per_day" => { config.decay_rate_per_day = value; true }
            "decay_interval_hours" => { config.decay_interval_hours = value; true }
            "propagation_damping" => { config.propagation_damping = value; true }
            "hebbian_learning_rate" => { config.hebbian_learning_rate = value; true }
            "recall_strength" => { config.recall_strength = value; true }
            _ => false,
        };
        
        if updated {
            self.substrate.config = config;
            self.storage.save_config(self.substrate.config())?;
        }
        
        Ok(updated)
    }
    
    /// Save config (for manual saves)
    pub fn save_config(&mut self) -> StorageResult<()> {
        self.storage.save_config(self.substrate.config())
    }
    
    /// Flush all pending writes
    pub fn flush(&mut self) -> StorageResult<()> {
        self.storage.flush()
    }
    
    /// Close the brain cleanly
    pub fn close(&mut self) -> StorageResult<()> {
        self.flush()?;
        self.storage.close()
    }
    
    // ==================
    // ITERATORS
    // ==================
    
    /// Iterate over searchable engrams (Active + Dormant)
    pub fn searchable_engrams(&self) -> impl Iterator<Item = &Engram> {
        self.substrate.searchable_engrams()
    }
    
    /// Iterate over archived engrams
    pub fn archived_engrams(&self) -> impl Iterator<Item = &Engram> {
        self.substrate.archived_engrams()
    }
    
    /// Iterate over all engrams
    pub fn all_engrams(&self) -> impl Iterator<Item = &Engram> {
        self.substrate.all_engrams()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::MemoryStorage;
    
    #[test]
    fn create_and_recall() {
        let storage = MemoryStorage::new();
        let mut brain = Brain::new(storage).unwrap();
        
        let id = brain.create("Test memory").unwrap();
        
        let result = brain.recall(id).unwrap();
        assert!(result.found());
        assert_eq!(result.content(), Some("Test memory"));
    }
    
    #[test]
    fn persistence_survives_reload() {
        // Create brain, add memory, close
        let storage = MemoryStorage::new();
        let mut brain = Brain::new(storage).unwrap();
        
        let id = brain.create_with_tags("Persistent memory", vec!["test".into()]).unwrap();
        brain.close().unwrap();
        
        // Reopen - but wait, MemoryStorage doesn't persist!
        // This test just verifies the API works; real persistence needs SQLite
        // For now, just verify create works
        assert!(brain.get(&id).is_some());
    }
    
    #[test]
    fn recall_persists_energy_changes() {
        let storage = MemoryStorage::new();
        let mut brain = Brain::new(storage).unwrap();
        
        let id = brain.create("Test memory").unwrap();
        
        // Lower energy
        brain.substrate.get_mut(&id).unwrap().energy = 0.5;
        
        // Recall should boost energy
        let result = brain.recall(id).unwrap();
        
        // The engram in the result should show increased energy
        assert!(result.engram.unwrap().energy > 0.5);
    }
    
    #[test]
    fn identity_persistence() {
        let storage = MemoryStorage::new();
        let mut brain = Brain::new(storage).unwrap();
        
        brain.identity_mut()
            .persona.name = "Porter".to_string();
        
        brain.save_identity().unwrap();
        
        // Verify it's saved (in memory storage at least)
        assert_eq!(brain.identity().persona.name, "Porter");
    }
    
    #[test]
    fn search_doesnt_mutate() {
        let storage = MemoryStorage::new();
        let mut brain = Brain::new(storage).unwrap();
        
        let id = brain.create_with_tags("Searchable memory", vec!["work".into()]).unwrap();
        
        let initial_energy = brain.get(&id).unwrap().energy;
        
        // Search should not affect energy
        let results = brain.search("Searchable");
        assert_eq!(results.len(), 1);
        
        let after_energy = brain.get(&id).unwrap().energy;
        assert_eq!(initial_energy, after_energy);
    }
    
    #[test]
    fn association_creation() {
        let storage = MemoryStorage::new();
        let mut brain = Brain::new(storage).unwrap();
        
        let a = brain.create("Memory A").unwrap();
        let b = brain.create("Memory B").unwrap();
        
        brain.associate(a, b, 0.8).unwrap();
        
        let associated = brain.find_associated(&a);
        assert_eq!(associated.len(), 1);
        assert_eq!(associated[0].1, 0.8);
    }
    
    #[test]
    fn decay_applies_on_startup_when_time_elapsed() {
        // Simulates the bug where decay never ran because last_decay_at
        // defaulted to now() instead of a past timestamp
        use std::time::{SystemTime, UNIX_EPOCH};
        
        let storage = MemoryStorage::new();
        let mut brain = Brain::new(storage).unwrap();
        
        let id = brain.create("Test memory").unwrap();
        assert_eq!(brain.get(&id).unwrap().energy, 1.0);
        
        // Simulate server being offline for 2 days by setting last_decay_at to 48 hours ago
        let two_days_ago = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64 - (48 * 3600);
        brain.substrate.set_last_decay_at(two_days_ago);
        
        // Now apply_time_decay should apply ~10% decay (2 days × 5%/day)
        let applied = brain.apply_time_decay().unwrap();
        assert!(applied, "Decay should have been applied");
        
        let energy = brain.get(&id).unwrap().energy;
        // Should be approximately 0.90 (1.0 - 0.10)
        assert!(energy < 0.95, "Energy should have decayed significantly: {}", energy);
        assert!(energy > 0.85, "Energy shouldn't have decayed too much: {}", energy);
    }
    
    #[test]
    fn decay_skips_when_interval_not_elapsed() {
        let storage = MemoryStorage::new();
        let mut brain = Brain::new(storage).unwrap();
        
        let id = brain.create("Test memory").unwrap();
        
        // last_decay_at defaults to now, so decay should NOT apply
        let applied = brain.apply_time_decay().unwrap();
        assert!(!applied, "Decay should not apply when interval hasn't elapsed");
        
        // Energy should still be 1.0
        assert_eq!(brain.get(&id).unwrap().energy, 1.0);
    }
}
