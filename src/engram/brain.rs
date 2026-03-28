//! Brain - The coordination layer
//!
//! Brain ties together Identity, Substrate, and Storage.
//! It's the main interface for the engram system.
//!
//! All mutating operations automatically persist changes.
//! Search operations are read-only and don't hit storage.

use std::path::PathBuf;

use std::collections::HashSet;

use super::persistence::{PersistenceWork, PersistenceWorker};
use super::{
    Association, Config, Engram, EngramId, Identity, RecallResult, SearchOptions, SimilarityResult,
    Storage, StorageResult, Substrate, SubstrateStats, TagMatchMode,
};

/// The Brain - coordinates Identity, Substrate, and Storage
///
/// This is the main entry point for the engram system.
/// All mutating operations automatically persist to storage.
pub struct Brain {
    identity: Identity,
    /// Made pub(crate) for testing decay behavior
    pub(crate) substrate: Substrate,
    /// Interior-mutable storage so read-only Brain methods (&self) can perform
    /// storage queries without requiring an exclusive borrow. This enables
    /// Arc<RwLock<Brain>> callers to hold a read lock while still accessing
    /// SQLite (e.g. during vector search or embedding lookups).
    storage: std::sync::Mutex<Box<dyn Storage>>,
    /// Background worker for async recall persistence
    /// None for in-memory storage (no persistence needed)
    persistence_worker: Option<PersistenceWorker>,
}

impl Brain {
    /// Create a new Brain with empty identity and substrate.
    /// `config` is loaded externally (from config.toml) and passed in.
    /// Note: This constructor doesn't create a persistence worker since
    /// we don't have a db path. Use `open_path` for full persistence support.
    pub fn new<S: Storage + 'static>(mut storage: S, mut config: Config) -> StorageResult<Self> {
        storage.initialize()?;

        // Load runtime state (embedding_model_active) from SQLite metadata.
        // This is NOT stored in config.toml — it tracks what model was actually used.
        if let Ok(Some(active)) = storage.get_metadata("embedding_model_active") {
            config.embedding_model_active = Some(active);
        }

        let mut brain = Self {
            identity: Identity::new(),
            substrate: Substrate::with_config(config),
            storage: std::sync::Mutex::new(Box::new(storage)),
            persistence_worker: None,
        };

        brain.check_model_mismatch()?;
        brain.ensure_fts_index()?;

        Ok(brain)
    }

    /// Open an existing Brain from storage, or create new if empty.
    /// `config` is loaded externally (from config.toml) and passed in.
    /// Note: This constructor doesn't create a persistence worker since
    /// we don't have a db path. Use `open_path` for full persistence support.
    pub fn open<S: Storage + 'static>(mut storage: S, mut config: Config) -> StorageResult<Self> {
        storage.initialize()?;

        // Load runtime state (embedding_model_active) from SQLite metadata.
        if let Ok(Some(active)) = storage.get_metadata("embedding_model_active") {
            config.embedding_model_active = Some(active);
        }

        // Load identity
        let identity = storage.load_identity()?.unwrap_or_default();

        // Load substrate
        let mut substrate = Substrate::with_config(config);

        // Load last decay timestamp
        // If never set, default to 24 hours ago so first run applies ~1 day of decay
        let default_decay_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64
            - 86400; // 24 hours ago
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

        let mut brain = Self {
            identity,
            substrate,
            storage: std::sync::Mutex::new(Box::new(storage)),
            persistence_worker: None,
        };

        brain.check_model_mismatch()?;
        brain.ensure_fts_index()?;

        Ok(brain)
    }

    /// Open a Brain from a database path with full async persistence support.
    /// `config` is loaded externally (from config.toml) and passed in.
    /// This is the preferred way to open a Brain for production use.
    pub fn open_path(db_path: impl Into<PathBuf>, mut config: Config) -> StorageResult<Self> {
        use super::storage::EngramStorage;

        let path = db_path.into();
        let mut storage = EngramStorage::open(&path)?;
        storage.initialize()?;

        // Load runtime state (embedding_model_active) from SQLite metadata.
        if let Ok(Some(active)) = storage.get_metadata("embedding_model_active") {
            config.embedding_model_active = Some(active);
        }

        // Load identity
        let identity = storage.load_identity()?.unwrap_or_default();

        // Load substrate
        let mut substrate = Substrate::with_config(config);

        // Load last decay timestamp
        let default_decay_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64
            - 86400;
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

        // Create persistence worker with its own connection
        let persistence_worker = PersistenceWorker::new(&path);
        eprintln!("[brain] Persistence worker started for {:?}", path);

        let mut brain = Self {
            identity,
            substrate,
            storage: std::sync::Mutex::new(Box::new(storage)),
            persistence_worker: Some(persistence_worker),
        };

        brain.check_model_mismatch()?;
        brain.ensure_fts_index()?;

        Ok(brain)
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
        self.storage.lock().unwrap().save_identity(&self.identity)
    }

    // ==================
    // CREATE
    // ==================

    /// Create a new memory
    pub fn create(&mut self, content: &str) -> StorageResult<EngramId> {
        let id = self.substrate.create(content);

        // Persist immediately
        if let Some(engram) = self.substrate.get(&id) {
            self.storage.lock().unwrap().save_engram(engram)?;
        }

        Ok(id)
    }

    /// Create a new memory with an explicit creation timestamp
    pub fn create_with_timestamp(
        &mut self,
        content: &str,
        created_at: i64,
    ) -> StorageResult<EngramId> {
        let id = self.substrate.create_with_timestamp(content, created_at);

        // Persist immediately
        if let Some(engram) = self.substrate.get(&id) {
            self.storage.lock().unwrap().save_engram(engram)?;
        }

        Ok(id)
    }

    /// Create a new memory with tags
    pub fn create_with_tags(
        &mut self,
        content: &str,
        tags: Vec<String>,
    ) -> StorageResult<EngramId> {
        let id = self.substrate.create_with_tags(content, tags);

        // Persist immediately
        if let Some(engram) = self.substrate.get(&id) {
            self.storage.lock().unwrap().save_engram(engram)?;
        }

        Ok(id)
    }

    /// Create an explicit association between memories
    pub fn associate(&mut self, from: EngramId, to: EngramId, weight: f64) -> StorageResult<()> {
        self.associate_with_ordinal(from, to, weight, None)
    }

    /// Create an explicit association with optional ordinal (for ordered chains)
    pub fn associate_with_ordinal(
        &mut self,
        from: EngramId,
        to: EngramId,
        weight: f64,
        ordinal: Option<u32>,
    ) -> StorageResult<()> {
        self.substrate.associate(from, to, weight, ordinal);

        // Persist the association
        if let Some(assocs) = self.substrate.associations_from(&from)
            && let Some(assoc) = assocs.iter().find(|a| a.to == to)
        {
            self.storage.lock().unwrap().save_association(assoc)?;
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
        self.storage.lock().unwrap().delete_engram(&id)?;

        Ok(existed)
    }

    // ==================
    // RECALL (active)
    // ==================

    /// Recall a memory - stimulates it and persists changes asynchronously
    pub fn recall(&mut self, id: EngramId) -> StorageResult<RecallResult> {
        let result = self.substrate.recall(id);
        self.async_persist_recall_effects(&result);
        Ok(result)
    }

    /// Recall with custom stimulation strength
    pub fn recall_with_strength(
        &mut self,
        id: EngramId,
        strength: f64,
    ) -> StorageResult<RecallResult> {
        let result = self.substrate.recall_with_strength(id, strength);
        self.async_persist_recall_effects(&result);
        Ok(result)
    }

    /// Recall multiple memories
    pub fn recall_many(&mut self, ids: &[EngramId]) -> StorageResult<Vec<RecallResult>> {
        let results = self.substrate.recall_many(ids);

        // Build combined work for all recalls
        let mut work = PersistenceWork::new();
        for result in &results {
            self.build_persistence_work_into(result, &mut work);
        }

        // Send to worker (fire and forget)
        if let Some(ref worker) = self.persistence_worker {
            worker.send(work);
        }

        Ok(results)
    }

    /// Build persistence work from a recall result (helper)
    fn build_persistence_work_into(&self, result: &RecallResult, work: &mut PersistenceWork) {
        // Gather energy updates for affected engrams
        for id in &result.affected_ids {
            if let Some(engram) = self.substrate.get(id) {
                work.energy_updates.push((*id, engram.energy, engram.state));
            }
        }

        // Gather associations modified by Hebbian learning
        for (from_id, to_id) in &result.modified_associations {
            if let Some(assocs) = self.substrate.associations_from(from_id)
                && let Some(assoc) = assocs.iter().find(|a| a.to == *to_id)
            {
                work.associations.push(assoc.clone());
            }
        }
    }

    /// Send recall effects to background worker for async persistence
    fn async_persist_recall_effects(&self, result: &RecallResult) {
        // Skip if no persistence worker (in-memory storage)
        let worker = match &self.persistence_worker {
            Some(w) => w,
            None => return,
        };

        let mut work = PersistenceWork::new();
        self.build_persistence_work_into(result, &mut work);
        worker.send(work);
    }

    // ==================
    // SEARCH (passive)
    // ==================

    /// Search memories by content (in-memory substring search)
    pub fn search(&self, query: &str) -> Vec<&Engram> {
        self.substrate.search(query)
    }

    /// Search with options (in-memory substring search)
    pub fn search_with_options(&self, query: &str, options: SearchOptions) -> Vec<&Engram> {
        self.substrate.search_with_options(query, options)
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

    /// Get a memory by ID (substrate only - no storage fallback)
    pub fn get(&self, id: &EngramId) -> Option<&Engram> {
        self.substrate.get(id)
    }

    /// Get a memory by ID, falling back to storage if not in substrate.
    /// This handles cross-process writes where another process created a memory
    /// that this process's substrate doesn't know about yet.
    pub fn get_or_load(&mut self, id: &EngramId) -> Option<&Engram> {
        // Check substrate first (fast path)
        if self.substrate.get(id).is_some() {
            return self.substrate.get(id);
        }

        // Fall back to storage (handles cross-process writes)
        if let Ok(Some(engram)) = self.storage.lock().unwrap().load_engram(id) {
            self.substrate.insert_engram(engram);
            return self.substrate.get(id);
        }

        None
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
        self.substrate
            .list_tags_with_counts()
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

    /// Sync substrate with storage to pick up cross-process writes.
    /// Does a lightweight count check first — only reloads if counts diverge.
    /// Returns true if new engrams were loaded.
    pub fn sync_from_storage(&mut self) -> StorageResult<bool> {
        let db_count = self.storage.lock().unwrap().count_engrams()?;
        let substrate_count = self.substrate.len();

        if db_count <= substrate_count {
            return Ok(false);
        }

        // New engrams exist in DB that substrate doesn't have — load them
        let engrams = self.storage.lock().unwrap().load_all_engrams()?;
        let mut loaded = 0;
        for engram in engrams {
            if self.substrate.get(&engram.id).is_none() {
                self.substrate.insert_engram(engram);
                loaded += 1;
            }
        }

        if loaded > 0 {
            // Also sync associations (new engrams may have associations)
            let associations = self.storage.lock().unwrap().load_all_associations()?;
            for assoc in associations {
                let exists = self
                    .substrate
                    .associations_from(&assoc.from)
                    .map(|a| a.iter().any(|existing| existing.to == assoc.to))
                    .unwrap_or(false);
                if !exists {
                    self.substrate.insert_association(assoc);
                }
            }

            eprintln!("[brain] Synced {} new engrams from storage", loaded);
        }

        Ok(loaded > 0)
    }

    /// Apply time-based decay to all memories and persist
    /// Returns true if decay was applied, false if not enough time elapsed
    pub fn apply_time_decay(&mut self) -> StorageResult<bool> {
        let applied = self.substrate.apply_time_decay();

        if applied {
            // Save only energy/state for all engrams (skips FTS rebuild)
            let updates: Vec<_> = self
                .substrate
                .all_engrams()
                .map(|e| (&e.id, e.energy, e.state))
                .collect();
            self.storage.lock().unwrap().save_engram_energies(&updates)?;

            // Save the last decay timestamp
            self.storage
                .lock()
                .unwrap()
                .save_last_decay_at(self.substrate.last_decay_at())?;
        }

        Ok(applied)
    }

    /// Remove associations that reference non-existent engrams.
    ///
    /// Orphan associations can accumulate when the async persistence worker
    /// re-writes associations after `delete_engram` has already cleaned them up.
    /// This is called during server startup maintenance.
    pub fn prune_orphan_associations(&mut self) -> StorageResult<usize> {
        self.storage.lock().unwrap().prune_orphan_associations()
    }

    /// Prune weak associations below the minimum weight threshold
    /// Returns the number of associations pruned
    /// Note: Pruned associations are removed from storage as well
    pub fn prune_weak_associations(&mut self) -> StorageResult<usize> {
        let pruned = self.substrate.prune_weak_associations();

        if pruned > 0 {
            // Re-save all associations to storage (the simple approach)
            // A more efficient approach would track which were pruned
            self.storage.lock().unwrap().delete_all_associations()?;
            let assocs: Vec<&Association> = self.substrate.all_associations();
            self.storage.lock().unwrap().save_associations(&assocs)?;
        }

        Ok(pruned)
    }

    /// Legacy tick_decay - prefer apply_time_decay()
    #[deprecated(note = "Use apply_time_decay() for time-based decay")]
    #[allow(deprecated)]
    pub fn tick_decay(&mut self) -> StorageResult<()> {
        self.substrate.tick_decay();

        // Save only energy/state for all engrams (skips FTS rebuild)
        let updates: Vec<_> = self
            .substrate
            .all_engrams()
            .map(|e| (&e.id, e.energy, e.state))
            .collect();
        self.storage.lock().unwrap().save_engram_energies(&updates)?;

        Ok(())
    }

    // ==================
    // CONFIG
    // ==================

    /// Get the current configuration
    pub fn config(&self) -> &Config {
        self.substrate.config()
    }

    /// Update the in-memory configuration.
    /// Callers (e.g. config_set tool) are responsible for persisting to config.toml.
    pub fn set_config(&mut self, config: Config) -> StorageResult<()> {
        self.substrate.config = config;
        Ok(())
    }

    /// Update a single config value
    pub fn configure(&mut self, key: &str, value: f64) -> StorageResult<bool> {
        let mut config = self.substrate.config.clone();

        let updated = match key {
            "decay_rate_per_day" => {
                config.decay_rate_per_day = value;
                true
            }
            "decay_interval_hours" => {
                config.decay_interval_hours = value;
                true
            }
            "propagation_damping" => {
                config.propagation_damping = value;
                true
            }
            "hebbian_learning_rate" => {
                config.hebbian_learning_rate = value;
                true
            }
            "recall_strength" => {
                config.recall_strength = value;
                true
            }
            "search_follow_associations" => {
                config.search_follow_associations = value != 0.0;
                true
            }
            "search_association_depth" => {
                config.search_association_depth = value.clamp(0.0, 5.0) as u8;
                true
            }
            "rerank_mode" => {
                // rerank_mode is a string — handled directly in config_set, not here
                false
            }
            "rerank_candidates" => {
                config.rerank_candidates = value.clamp(5.0, 200.0) as usize;
                true
            }
            "hybrid_search_enabled" => {
                config.hybrid_search_enabled = value != 0.0;
                true
            }
            "query_expansion_enabled" => {
                config.query_expansion_enabled = value != 0.0;
                true
            }
            "llm_rerank_candidates" => {
                config.llm_rerank_candidates = value.clamp(1.0, 100.0) as usize;
                true
            }
            "search_min_score" => {
                config.search_min_score = value.clamp(0.0, 1.0);
                true
            }
            "composite_limit_min" => {
                config.composite_limit_min = value.clamp(1.0, 100.0) as usize;
                true
            }
            "composite_limit_max" => {
                config.composite_limit_max = value.clamp(1.0, 200.0) as usize;
                true
            }
            "association_cap_min" => {
                config.association_cap_min = value.clamp(1.0, 50.0) as usize;
                true
            }
            "association_cap_max" => {
                config.association_cap_max = value.clamp(1.0, 50.0) as usize;
                true
            }
            "debug" => {
                config.debug = value != 0.0;
                true
            }
            _ => false,
        };

        if updated {
            self.substrate.config = config;
        }

        Ok(updated)
    }

    /// No-op: config is now persisted to config.toml by the config_set tool.
    /// Kept for API compatibility.
    pub fn save_config(&mut self) -> StorageResult<()> {
        Ok(())
    }


    /// Get a metadata value by key
    pub fn get_metadata(&mut self, key: &str) -> StorageResult<Option<String>> {
        self.storage.lock().unwrap().get_metadata(key)
    }

    /// Set a metadata key/value pair
    pub fn set_metadata(&mut self, key: &str, value: &str) -> StorageResult<()> {
        self.storage.lock().unwrap().set_metadata(key, value)
    }

    /// Delete sessions whose `last_seen_at` is strictly less than `expire_before`.
    /// Returns the number of sessions deleted.
    pub fn delete_expired_sessions(&mut self, expire_before: i64) -> StorageResult<usize> {
        self.storage
            .lock()
            .unwrap()
            .delete_expired_sessions(expire_before)
    }

    /// Flush all pending writes
    pub fn flush(&mut self) -> StorageResult<()> {
        self.storage.lock().unwrap().flush()
    }

    /// Close the brain cleanly
    pub fn close(&mut self) -> StorageResult<()> {
        self.flush()?;
        self.storage.lock().unwrap().close()
    }

    // ==================
    // VECTOR SEARCH
    // ==================

    /// Find engrams semantically similar to the given embedding
    /// Returns (id, score, content) tuples sorted by descending similarity
    ///
    /// Takes `&self` so callers holding an RwLock read guard can call this
    /// without an exclusive borrow. Storage is locked internally.
    pub fn find_similar_by_embedding(
        &self,
        query_embedding: &[f32],
        limit: usize,
        min_score: f32,
    ) -> StorageResult<Vec<SimilarityResult>> {
        self.storage
            .lock()
            .unwrap()
            .find_similar_by_embedding(query_embedding, limit, min_score)
    }

    /// Discover memories reachable via associations from the given seed IDs.
    ///
    /// This is a read-only operation: no stimulation, no Hebbian learning, no energy changes.
    /// For each discovered memory not already in `seen_ids`, looks up its embedding,
    /// computes cosine similarity against query_embedding, and returns a scored result.
    ///
    /// Returns a vec of (id, blended_score, similarity, association_weight) for discovered memories.
    /// The caller is responsible for merging these into the final result set.
    /// Discover memories reachable via associations.
    ///
    /// Takes `&self` so callers holding an RwLock read guard can call this.
    /// Storage is locked internally per embedding lookup.
    /// After `sync_from_storage()`, all active engrams are in the substrate cache,
    /// so energy lookups use `get()` (substrate only) rather than `get_or_load`.
    pub fn discover_associated_memories(
        &self,
        query_embedding: &[f32],
        seed_ids: &[EngramId],
        seen_ids: &mut HashSet<EngramId>,
        depth: u8,
    ) -> StorageResult<Vec<AssociationDiscovery>> {
        use crate::embedding::cosine_similarity;

        const MIN_ASSOCIATION_WEIGHT: f64 = 0.15;
        const ASSOCIATION_BONUS_FACTOR: f64 = 0.1;

        if depth == 0 {
            return Ok(Vec::new());
        }

        let mut all_discoveries: Vec<AssociationDiscovery> = Vec::new();
        let mut frontier: Vec<EngramId> = seed_ids.to_vec();

        for _hop in 0..depth {
            let mut next_frontier: Vec<EngramId> = Vec::new();

            // Collect all candidate (target_id, association_weight, ordinal) tuples from frontier
            let mut candidates: Vec<(EngramId, f64, Option<u32>)> = Vec::new();
            for seed_id in &frontier {
                if let Some(assocs) = self.substrate.associations_from(seed_id) {
                    for assoc in assocs {
                        if assoc.weight >= MIN_ASSOCIATION_WEIGHT && !seen_ids.contains(&assoc.to) {
                            candidates.push((assoc.to, assoc.weight, assoc.ordinal));
                            seen_ids.insert(assoc.to);
                        }
                    }
                }
            }

            if candidates.is_empty() {
                break;
            }

            // Sort by ordinal ascending (None/null sorts last) so procedure chains
            // are traversed step-by-step
            candidates.sort_by(|a, b| match (a.2, b.2) {
                (Some(oa), Some(ob)) => oa.cmp(&ob),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            });

            // For each candidate, look up embedding and score against query
            for (candidate_id, assoc_weight, _ordinal) in candidates {
                let embedding = match self.storage.lock().unwrap().get_embedding(&candidate_id)? {
                    Some(emb) => emb,
                    None => continue,
                };

                let similarity = cosine_similarity(query_embedding, &embedding);

                // Get energy from substrate cache (sync_from_storage ensures all
                // engrams are loaded; use get() to avoid nested storage lock).
                let energy = match self.get(&candidate_id) {
                    Some(engram) => engram.energy,
                    None => continue,
                };

                // Blended score: multiplicative (same as search) + association bonus
                let blended = similarity * (0.5 + energy as f32 * 0.5)
                    + (assoc_weight as f32 * ASSOCIATION_BONUS_FACTOR as f32);

                all_discoveries.push(AssociationDiscovery {
                    id: candidate_id,
                    similarity,
                    energy,
                    blended_score: blended,
                    association_weight: assoc_weight,
                });

                next_frontier.push(candidate_id);
            }

            frontier = next_frontier;
        }

        Ok(all_discoveries)
    }

    /// Count engrams that have embeddings
    pub fn count_with_embeddings(&self) -> StorageResult<usize> {
        self.storage.lock().unwrap().count_with_embeddings()
    }

    /// Count engrams that need embeddings (for backfill progress)
    pub fn count_without_embeddings(&self) -> StorageResult<usize> {
        self.storage.lock().unwrap().count_without_embeddings()
    }

    /// Get IDs of engrams that need embeddings (for backfill)
    pub fn get_ids_without_embeddings(&self, limit: usize) -> StorageResult<Vec<EngramId>> {
        self.storage.lock().unwrap().get_ids_without_embeddings(limit)
    }

    /// Get IDs of engrams that have no enrichments (for incremental enrichment backfill)
    pub fn get_ids_without_enrichments(&self) -> StorageResult<Vec<EngramId>> {
        self.storage.lock().unwrap().get_ids_without_enrichments()
    }

    /// Get embedding for an engram.
    ///
    /// Takes `&self` so callers holding an RwLock read guard can call this
    /// during diversity shaping without an exclusive borrow.
    pub fn get_embedding(&self, id: &EngramId) -> StorageResult<Option<Vec<f32>>> {
        self.storage.lock().unwrap().get_embedding(id)
    }

    /// Set embedding for an engram (used during backfill).
    ///
    /// Takes `&self` so enrichment background threads can write embeddings
    /// while search holds an RwLock read guard. Storage is locked internally.
    pub fn set_embedding(&self, id: &EngramId, embedding: &[f32]) -> StorageResult<()> {
        self.storage.lock().unwrap().set_embedding(id, embedding)
    }

    /// Store enrichment embeddings for an engram (multi-vector support).
    /// Replaces any existing enrichments for this engram.
    ///
    /// Takes `&self` so enrichment threads can write while search holds a read lock.
    pub fn set_enrichment_embeddings(
        &self,
        id: &EngramId,
        embeddings: &[Vec<f32>],
        source: &str,
    ) -> StorageResult<()> {
        self.storage
            .lock()
            .unwrap()
            .set_enrichment_embeddings(id, embeddings, source)
    }

    /// Delete all enrichment embeddings for an engram.
    pub fn delete_enrichments(&mut self, id: &EngramId) -> StorageResult<()> {
        self.storage.lock().unwrap().delete_enrichments(id)
    }

    /// Count total enrichment vectors across all engrams.
    pub fn count_enrichments(&self) -> StorageResult<usize> {
        self.storage.lock().unwrap().count_enrichments()
    }

    // ── Session Context ──────────────────────────────────────────────────────

    /// Load a session by ID. Returns None if the session does not exist.
    ///
    /// Takes `&self` (uses internal Mutex) so callers holding an RwLock read
    /// guard can load session state without upgrading to a write lock.
    pub fn load_session(
        &self,
        session_id: &str,
    ) -> StorageResult<Option<crate::engram::session::SessionContext>> {
        self.storage.lock().unwrap().load_session(session_id)
    }

    /// Save (upsert) a session.
    ///
    /// Takes `&self` (uses internal Mutex) following the same pattern as
    /// `set_embedding` and `set_enrichment_embeddings`.
    pub fn save_session(
        &self,
        session: &crate::engram::session::SessionContext,
    ) -> StorageResult<()> {
        self.storage.lock().unwrap().save_session(session)
    }

    /// Find similar memories to a given memory ID
    /// Convenience wrapper around find_similar_by_embedding
    pub fn find_similar_to(
        &mut self,
        id: &EngramId,
        limit: usize,
        min_score: f32,
    ) -> StorageResult<Vec<SimilarityResult>> {
        let embedding = match self.storage.lock().unwrap().get_embedding(id)? {
            Some(emb) => emb,
            None => return Ok(vec![]),
        };

        let mut results =
            self.storage
                .lock()
                .unwrap()
                .find_similar_by_embedding(&embedding, limit + 1, min_score)?;

        // Filter out self
        results.retain(|r| r.id != *id);
        results.truncate(limit);

        Ok(results)
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

    /// Check for embedding model mismatch and clear stale vectors if needed.
    ///
    /// Triggers when `embedding_model` != `embedding_model_active` in config.
    /// Steps: clear all embeddings/enrichments/sessions → update `embedding_model_active`.
    ///
    /// Re-generation is intentionally NOT done here. Run `memoryco generate` to rebuild.
    pub fn check_model_mismatch(&mut self) -> StorageResult<()> {
        let config = self.substrate.config();
        let desired = config.embedding_model.clone();
        let active = config.embedding_model_active.clone();

        // If active matches desired (the common case), nothing to do.
        if active.as_deref() == Some(desired.as_str()) {
            return Ok(());
        }

        let old = active.as_deref().unwrap_or("(none)");
        eprintln!(
            "[brain] Embedding model changed ({} → {}). Cleared stale vectors. \
             Run 'memoryco generate' to rebuild.",
            old, desired
        );

        // Clear all existing embeddings, enrichments, and sessions.
        // Session centroids have wrong dimensions after a model switch —
        // comparing them against new-dimension query embeddings produces garbage.
        let cleared = self.storage.lock().unwrap().clear_all_embeddings()?;
        let cleared_enrichments = self.storage.lock().unwrap().clear_all_enrichments()?;
        let cleared_sessions = self.storage.lock().unwrap().clear_all_sessions()?;
        eprintln!(
            "[brain] Cleared {} embeddings, {} enrichments, {} sessions",
            cleared, cleared_enrichments, cleared_sessions
        );

        // Update in-memory config and persist the active model to SQLite metadata.
        // embedding_model_active is runtime state, not stored in config.toml.
        let mut new_config = self.substrate.config().clone();
        new_config.embedding_model_active = Some(desired.clone());
        self.substrate.config = new_config;
        self.storage
            .lock()
            .unwrap()
            .set_metadata("embedding_model_active", &desired)?;

        Ok(())
    }


    // ==================
    // FTS5 KEYWORD SEARCH
    // ==================

    /// Ensure the FTS5 index is populated from existing engrams.
    /// Called during Brain initialization. If the FTS5 table is empty but
    /// engrams exist, backfills the index from the engrams table.
    fn ensure_fts_index(&mut self) -> StorageResult<()> {
        let backfilled = self.storage.lock().unwrap().ensure_fts_populated()?;
        if backfilled > 0 {
            eprintln!("[brain] Backfilled {} engrams into FTS5 index", backfilled);
        }
        Ok(())
    }

    /// Search engrams by keyword using FTS5/BM25.
    ///
    /// Takes `&self` so callers holding an RwLock read guard can call this
    /// during hybrid search without an exclusive borrow.
    pub fn keyword_search(
        &self,
        query: &str,
        limit: usize,
    ) -> StorageResult<Vec<SimilarityResult>> {
        self.storage.lock().unwrap().keyword_search(query, limit)
    }

    // ==================
    // ACCESS LOG
    // ==================

    /// Log a search→recall cycle for membed training data extraction.
    pub fn log_access(
        &mut self,
        query_text: &str,
        result_ids: &[EngramId],
        recalled_ids: &[EngramId],
    ) -> StorageResult<()> {
        self.storage
            .lock()
            .unwrap()
            .log_access(query_text, result_ids, recalled_ids)
    }
}

/// A memory discovered via association-following during search
#[derive(Debug, Clone)]
pub struct AssociationDiscovery {
    pub id: EngramId,
    pub similarity: f32,
    pub energy: f64,
    pub blended_score: f32,
    pub association_weight: f64,
}

#[cfg(test)]
mod tests {
    use super::super::MemoryStorage;
    use super::*;

    #[test]
    fn create_and_recall() {
        let storage = MemoryStorage::new();
        let mut brain = Brain::new(storage, Config::default()).unwrap();

        let id = brain.create("Test memory").unwrap();

        let result = brain.recall(id).unwrap();
        assert!(result.found());
        assert_eq!(result.content(), Some("Test memory"));
    }

    #[test]
    fn persistence_survives_reload() {
        // Create brain, add memory, close
        let storage = MemoryStorage::new();
        let mut brain = Brain::new(storage, Config::default()).unwrap();

        let id = brain
            .create_with_tags("Persistent memory", vec!["test".into()])
            .unwrap();
        brain.close().unwrap();

        // Reopen - but wait, MemoryStorage doesn't persist!
        // This test just verifies the API works; real persistence needs SQLite
        // For now, just verify create works
        assert!(brain.get(&id).is_some());
    }

    #[test]
    fn recall_persists_energy_changes() {
        let storage = MemoryStorage::new();
        let mut brain = Brain::new(storage, Config::default()).unwrap();

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
        let mut brain = Brain::new(storage, Config::default()).unwrap();

        brain.identity_mut().persona.name = "Porter".to_string();

        brain.save_identity().unwrap();

        // Verify it's saved (in memory storage at least)
        assert_eq!(brain.identity().persona.name, "Porter");
    }

    #[test]
    fn search_doesnt_mutate() {
        let storage = MemoryStorage::new();
        let mut brain = Brain::new(storage, Config::default()).unwrap();

        let id = brain
            .create_with_tags("Searchable memory", vec!["work".into()])
            .unwrap();

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
        let mut brain = Brain::new(storage, Config::default()).unwrap();

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
        let mut brain = Brain::new(storage, Config::default()).unwrap();

        let id = brain.create("Test memory").unwrap();
        assert_eq!(brain.get(&id).unwrap().energy, 1.0);

        // Simulate server being offline for 2 days by setting last_decay_at to 48 hours ago
        let two_days_ago = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
            - (48 * 3600);
        brain.substrate.set_last_decay_at(two_days_ago);

        // Now apply_time_decay should apply ~10% decay (2 days × 5%/day)
        let applied = brain.apply_time_decay().unwrap();
        assert!(applied, "Decay should have been applied");

        let energy = brain.get(&id).unwrap().energy;
        // Should be approximately 0.90 (1.0 - 0.10)
        assert!(
            energy < 0.95,
            "Energy should have decayed significantly: {}",
            energy
        );
        assert!(
            energy > 0.85,
            "Energy shouldn't have decayed too much: {}",
            energy
        );
    }

    #[test]
    fn decay_skips_when_interval_not_elapsed() {
        let storage = MemoryStorage::new();
        let mut brain = Brain::new(storage, Config::default()).unwrap();

        let id = brain.create("Test memory").unwrap();

        // last_decay_at defaults to now, so decay should NOT apply
        let applied = brain.apply_time_decay().unwrap();
        assert!(
            !applied,
            "Decay should not apply when interval hasn't elapsed"
        );

        // Energy should still be 1.0
        assert_eq!(brain.get(&id).unwrap().energy, 1.0);
    }

    #[test]
    fn cross_process_get_or_load() {
        // Simulate two processes sharing the same SQLite file.
        // Process A creates a memory, Process B (opened before the write)
        // should be able to see it via get_or_load().
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("brain.db");

        // "Process A" — create and write a memory
        let mut brain_a = Brain::open_path(&db_path, Config::default()).unwrap();
        let id = brain_a.create("artichokes are delicious").unwrap();
        brain_a.flush().unwrap();

        // "Process B" — opened AFTER A's write, but simulating a stale substrate
        // by opening a fresh Brain (its substrate will load the memory).
        // To truly test staleness, we open B, then have A create another memory.
        let mut brain_b = Brain::open_path(&db_path, Config::default()).unwrap();

        // B can see the first memory (loaded at startup)
        assert!(
            brain_b.get(&id).is_some(),
            "B should see memory loaded at startup"
        );

        // Now A creates a NEW memory that B doesn't know about
        let id2 = brain_a.create("the artichoke memory was a test").unwrap();
        brain_a.flush().unwrap();

        // B's substrate doesn't have id2
        assert!(
            brain_b.get(&id2).is_none(),
            "B's substrate should NOT have the new memory"
        );

        // But get_or_load falls back to storage and finds it
        assert!(
            brain_b.get_or_load(&id2).is_some(),
            "get_or_load should find cross-process write"
        );
        assert_eq!(
            brain_b.get(&id2).unwrap().content,
            "the artichoke memory was a test",
        );
    }

    #[test]
    fn cross_process_sync_from_storage() {
        // sync_from_storage should bulk-load all new cross-process memories
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("brain.db");

        let mut brain_a = Brain::open_path(&db_path, Config::default()).unwrap();
        let mut brain_b = Brain::open_path(&db_path, Config::default()).unwrap();

        // A creates several memories after B is already open
        let id1 = brain_a.create("memory one").unwrap();
        let id2 = brain_a.create("memory two").unwrap();
        let id3 = brain_a.create("memory three").unwrap();
        brain_a.flush().unwrap();

        // B doesn't see any of them
        assert!(brain_b.get(&id1).is_none());
        assert!(brain_b.get(&id2).is_none());
        assert!(brain_b.get(&id3).is_none());

        // After sync, B sees all of them
        let synced = brain_b.sync_from_storage().unwrap();
        assert!(synced, "sync should have loaded new memories");
        assert!(brain_b.get(&id1).is_some());
        assert!(brain_b.get(&id2).is_some());
        assert!(brain_b.get(&id3).is_some());

        // Calling sync again with no changes should be a no-op
        let synced_again = brain_b.sync_from_storage().unwrap();
        assert!(!synced_again, "sync should be no-op when nothing changed");
    }

    #[test]
    fn count_engrams_storage() {
        use super::super::storage::EngramStorage;

        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("brain.db");

        let mut storage = EngramStorage::open(&db_path).unwrap();
        storage.initialize().unwrap();

        assert_eq!(storage.count_engrams().unwrap(), 0);

        let e1 = Engram::new("first");
        let e2 = Engram::new("second");
        storage.save_engram(&e1).unwrap();
        storage.save_engram(&e2).unwrap();

        assert_eq!(storage.count_engrams().unwrap(), 2);
    }

    // ==================
    // ASSOCIATION-FOLLOWING SEARCH TESTS
    // ==================

    /// Helper: create a Brain backed by in-memory SQLite (supports embeddings)
    fn brain_with_sqlite() -> Brain {
        use super::super::storage::EngramStorage;
        let storage = EngramStorage::in_memory().unwrap();
        Brain::new(storage, Config::default()).unwrap()
    }

    #[test]
    fn association_following_disabled_returns_no_discoveries() {
        let mut brain = brain_with_sqlite();

        // Create two engrams with embeddings
        let id_a = brain.create("Rust programming language").unwrap();
        brain.set_embedding(&id_a, &[1.0, 0.0, 0.0]).unwrap();

        let id_b = brain.create("Rust borrow checker rules").unwrap();
        brain.set_embedding(&id_b, &[0.1, 0.9, 0.0]).unwrap();

        // Create a strong association A -> B
        brain.associate(id_a, id_b, 0.8).unwrap();

        // With config disabled (default), discover_associated_memories still works
        // when called directly, but the search tool won't call it.
        // Test that depth=0 returns nothing:
        let mut seen = std::collections::HashSet::new();
        seen.insert(id_a);
        let discoveries = brain
            .discover_associated_memories(
                &[1.0, 0.0, 0.0],
                &[id_a],
                &mut seen,
                0, // depth=0 means no following
            )
            .unwrap();

        assert!(
            discoveries.is_empty(),
            "depth=0 should return no discoveries"
        );
    }

    #[test]
    fn association_following_discovers_related_memory() {
        let mut brain = brain_with_sqlite();

        // Memory A: very similar to query embedding
        let id_a = brain.create("Rust programming language").unwrap();
        brain.set_embedding(&id_a, &[1.0, 0.0, 0.0]).unwrap();

        // Memory B: NOT similar to query (orthogonal), but associated to A
        let id_b = brain.create("Cargo package manager details").unwrap();
        brain.set_embedding(&id_b, &[0.1, 0.9, 0.0]).unwrap();

        // Strong association A -> B
        brain.associate(id_a, id_b, 0.8).unwrap();

        // Query is close to A but not B
        let query = [1.0, 0.0, 0.0];

        let mut seen = std::collections::HashSet::new();
        seen.insert(id_a);

        let discoveries = brain
            .discover_associated_memories(&query, &[id_a], &mut seen, 1)
            .unwrap();

        // Should discover B via association from A
        assert_eq!(discoveries.len(), 1);
        assert_eq!(discoveries[0].id, id_b);
        assert!(discoveries[0].association_weight > 0.0);
        // Blended score should include association bonus on top of base score.
        // Actual formula: similarity * (0.5 + energy * 0.5) + assoc_weight * 0.1
        let sim = crate::embedding::cosine_similarity(&query, &[0.1, 0.9, 0.0]);
        let energy = 1.0_f32; // new engram has full energy
        let base_without_bonus = sim * (0.5 + energy * 0.5);
        assert!(
            discoveries[0].blended_score > base_without_bonus,
            "blended score {} should be > base {} due to association bonus",
            discoveries[0].blended_score,
            base_without_bonus
        );
    }

    #[test]
    fn association_following_handles_cycles() {
        let mut brain = brain_with_sqlite();

        let id_a = brain.create("Memory A").unwrap();
        brain.set_embedding(&id_a, &[1.0, 0.0, 0.0]).unwrap();

        let id_b = brain.create("Memory B").unwrap();
        brain.set_embedding(&id_b, &[0.5, 0.5, 0.0]).unwrap();

        // Create cycle: A -> B -> A
        brain.associate(id_a, id_b, 0.5).unwrap();
        brain.associate(id_b, id_a, 0.5).unwrap();

        let mut seen = std::collections::HashSet::new();
        seen.insert(id_a);

        // Should not infinite loop — A is already in seen_ids
        let discoveries = brain
            .discover_associated_memories(
                &[1.0, 0.0, 0.0],
                &[id_a],
                &mut seen,
                3, // Multiple hops, but cycle should be caught
            )
            .unwrap();

        // Should discover B (once) but not re-discover A
        assert_eq!(discoveries.len(), 1);
        assert_eq!(discoveries[0].id, id_b);
    }

    #[test]
    fn association_following_respects_weight_threshold() {
        let mut brain = brain_with_sqlite();

        let id_a = brain.create("Memory A").unwrap();
        brain.set_embedding(&id_a, &[1.0, 0.0, 0.0]).unwrap();

        let id_b = brain.create("Memory B - weak link").unwrap();
        brain.set_embedding(&id_b, &[0.5, 0.5, 0.0]).unwrap();

        let id_c = brain.create("Memory C - strong link").unwrap();
        brain.set_embedding(&id_c, &[0.3, 0.7, 0.0]).unwrap();

        // Weak association A -> B (below 0.15 threshold)
        brain.associate(id_a, id_b, 0.10).unwrap();
        // Strong association A -> C (above threshold)
        brain.associate(id_a, id_c, 0.50).unwrap();

        let mut seen = std::collections::HashSet::new();
        seen.insert(id_a);

        let discoveries = brain
            .discover_associated_memories(&[1.0, 0.0, 0.0], &[id_a], &mut seen, 1)
            .unwrap();

        // Should only discover C (strong link), not B (weak link)
        assert_eq!(discoveries.len(), 1);
        assert_eq!(discoveries[0].id, id_c);
    }

    #[test]
    fn association_following_no_associations_returns_empty() {
        let mut brain = brain_with_sqlite();

        let id_a = brain.create("Isolated memory").unwrap();
        brain.set_embedding(&id_a, &[1.0, 0.0, 0.0]).unwrap();

        // No associations at all
        let mut seen = std::collections::HashSet::new();
        seen.insert(id_a);

        let discoveries = brain
            .discover_associated_memories(&[1.0, 0.0, 0.0], &[id_a], &mut seen, 1)
            .unwrap();

        assert!(
            discoveries.is_empty(),
            "no associations means no discoveries"
        );
    }

    #[test]
    fn association_following_multi_hop() {
        let mut brain = brain_with_sqlite();

        // Chain: A -> B -> C
        let id_a = brain.create("Memory A").unwrap();
        brain.set_embedding(&id_a, &[1.0, 0.0, 0.0]).unwrap();

        let id_b = brain.create("Memory B").unwrap();
        brain.set_embedding(&id_b, &[0.5, 0.5, 0.0]).unwrap();

        let id_c = brain.create("Memory C").unwrap();
        brain.set_embedding(&id_c, &[0.0, 1.0, 0.0]).unwrap();

        brain.associate(id_a, id_b, 0.6).unwrap();
        brain.associate(id_b, id_c, 0.6).unwrap();

        // depth=1: should only find B (direct from A)
        let mut seen1 = std::collections::HashSet::new();
        seen1.insert(id_a);
        let discoveries_1 = brain
            .discover_associated_memories(&[1.0, 0.0, 0.0], &[id_a], &mut seen1, 1)
            .unwrap();
        assert_eq!(discoveries_1.len(), 1);
        assert_eq!(discoveries_1[0].id, id_b);

        // depth=2: should find both B and C
        let mut seen2 = std::collections::HashSet::new();
        seen2.insert(id_a);
        let discoveries_2 = brain
            .discover_associated_memories(&[1.0, 0.0, 0.0], &[id_a], &mut seen2, 2)
            .unwrap();
        assert_eq!(discoveries_2.len(), 2);
        let ids: std::collections::HashSet<_> = discoveries_2.iter().map(|d| d.id).collect();
        assert!(ids.contains(&id_b));
        assert!(ids.contains(&id_c));
    }

    #[test]
    fn association_following_is_passive() {
        let mut brain = brain_with_sqlite();

        let id_a = brain.create("Memory A").unwrap();
        brain.set_embedding(&id_a, &[1.0, 0.0, 0.0]).unwrap();

        let id_b = brain.create("Memory B").unwrap();
        brain.set_embedding(&id_b, &[0.5, 0.5, 0.0]).unwrap();

        brain.associate(id_a, id_b, 0.5).unwrap();

        // Record energy before
        let energy_a_before = brain.get(&id_a).unwrap().energy;
        let energy_b_before = brain.get(&id_b).unwrap().energy;
        let access_a_before = brain.get(&id_a).unwrap().access_count;
        let access_b_before = brain.get(&id_b).unwrap().access_count;

        // Run association discovery
        let mut seen = std::collections::HashSet::new();
        seen.insert(id_a);
        let _discoveries = brain
            .discover_associated_memories(&[1.0, 0.0, 0.0], &[id_a], &mut seen, 1)
            .unwrap();

        // Energy and access count should be UNCHANGED (passive operation)
        assert_eq!(brain.get(&id_a).unwrap().energy, energy_a_before);
        assert_eq!(brain.get(&id_b).unwrap().energy, energy_b_before);
        assert_eq!(brain.get(&id_a).unwrap().access_count, access_a_before);
        assert_eq!(brain.get(&id_b).unwrap().access_count, access_b_before);
    }

    #[test]
    fn config_defaults_for_search_association() {
        let config = Config::default();
        assert!(config.search_follow_associations);
        assert_eq!(config.search_association_depth, 1);
    }

    #[test]
    fn config_search_association_roundtrip() {
        let storage = MemoryStorage::new();
        let mut brain = Brain::new(storage, Config::default()).unwrap();

        // Default: enabled
        assert!(brain.config().search_follow_associations);
        assert_eq!(brain.config().search_association_depth, 1);

        // Enable via configure()
        assert!(brain.configure("search_follow_associations", 1.0).unwrap());
        assert!(brain.config().search_follow_associations);

        // Disable via configure()
        assert!(brain.configure("search_follow_associations", 0.0).unwrap());
        assert!(!brain.config().search_follow_associations);

        // Set depth
        assert!(brain.configure("search_association_depth", 3.0).unwrap());
        assert_eq!(brain.config().search_association_depth, 3);

        // Depth is capped at 5
        assert!(brain.configure("search_association_depth", 10.0).unwrap());
        assert_eq!(brain.config().search_association_depth, 5);
    }

    // ==================
    // ORDINAL TESTS
    // ==================

    #[test]
    fn associate_without_ordinal_backward_compat() {
        let storage = MemoryStorage::new();
        let mut brain = Brain::new(storage, Config::default()).unwrap();

        let a = brain.create("Memory A").unwrap();
        let b = brain.create("Memory B").unwrap();

        // Old-style associate() should still work perfectly
        brain.associate(a, b, 0.8).unwrap();

        let assocs = brain.associations_from(&a).unwrap();
        assert_eq!(assocs.len(), 1);
        assert_eq!(assocs[0].ordinal, None);
        assert!((assocs[0].weight - 0.8).abs() < 0.001);
    }

    #[test]
    fn associate_with_ordinal() {
        let storage = MemoryStorage::new();
        let mut brain = Brain::new(storage, Config::default()).unwrap();

        let anchor = brain.create("Deploy procedure").unwrap();
        let step1 = brain.create("Pull latest code").unwrap();
        let step2 = brain.create("Run tests").unwrap();
        let step3 = brain.create("Deploy to staging").unwrap();

        brain
            .associate_with_ordinal(anchor, step1, 0.9, Some(1))
            .unwrap();
        brain
            .associate_with_ordinal(anchor, step2, 0.9, Some(2))
            .unwrap();
        brain
            .associate_with_ordinal(anchor, step3, 0.9, Some(3))
            .unwrap();

        let assocs = brain.associations_from(&anchor).unwrap();
        assert_eq!(assocs.len(), 3);

        // Verify ordinals
        let mut ordinals: Vec<u32> = assocs.iter().filter_map(|a| a.ordinal).collect();
        ordinals.sort();
        assert_eq!(ordinals, vec![1, 2, 3]);
    }

    #[test]
    fn ordinal_persists_through_diesel() {
        // Test ordinal roundtrip through actual SQLite storage
        use super::super::storage::EngramStorage;

        let storage = EngramStorage::in_memory().unwrap();
        let mut brain = Brain::open(storage, Config::default()).unwrap();

        let a = brain.create("Anchor").unwrap();
        let b = brain.create("Step 1").unwrap();

        brain.associate_with_ordinal(a, b, 0.8, Some(5)).unwrap();

        let assocs = brain.associations_from(&a).unwrap();
        assert_eq!(assocs.len(), 1);
        assert_eq!(assocs[0].ordinal, Some(5));
    }

    #[test]
    fn mixed_ordinal_and_hebbian_associations() {
        let storage = MemoryStorage::new();
        let mut brain = Brain::new(storage, Config::default()).unwrap();

        let anchor = brain.create("Procedure anchor").unwrap();
        let step1 = brain.create("Step 1").unwrap();
        let related = brain.create("Related note").unwrap();

        // Ordered chain step
        brain
            .associate_with_ordinal(anchor, step1, 0.9, Some(1))
            .unwrap();
        // Unordered association (like Hebbian would create)
        brain.associate(anchor, related, 0.4).unwrap();

        let assocs = brain.associations_from(&anchor).unwrap();
        assert_eq!(assocs.len(), 2);

        let ordered: Vec<_> = assocs.iter().filter(|a| a.ordinal.is_some()).collect();
        let unordered: Vec<_> = assocs.iter().filter(|a| a.ordinal.is_none()).collect();
        assert_eq!(ordered.len(), 1);
        assert_eq!(unordered.len(), 1);
        assert_eq!(ordered[0].ordinal, Some(1));
    }

    #[test]
    fn ordinal_survives_prune_weak_associations() {
        let storage = MemoryStorage::new();
        let mut brain = Brain::new(storage, Config::default()).unwrap();

        let anchor = brain.create("Procedure").unwrap();
        let step1 = brain.create("Strong step").unwrap();
        let weak = brain.create("Weak association").unwrap();

        // Strong ordered step
        brain
            .associate_with_ordinal(anchor, step1, 0.9, Some(1))
            .unwrap();
        // Weak unordered association (below default min_association_weight)
        brain.associate(anchor, weak, 0.01).unwrap();

        let pruned = brain.prune_weak_associations().unwrap();
        assert_eq!(pruned, 1); // Only the weak one should be pruned

        let assocs = brain.associations_from(&anchor).unwrap();
        assert_eq!(assocs.len(), 1);
        assert_eq!(assocs[0].ordinal, Some(1));
        assert_eq!(assocs[0].to, step1);
    }

    // ==================
    // ORDERED TRAVERSAL TESTS (Step 6)
    // ==================

    #[test]
    fn discover_ordered_traversal_respects_ordinal_order() {
        // Chain: anchor → B(ord:1) → C(ord:2) → D(ord:3)
        // All associations are from anchor, so discoveries should come back in ordinal order
        let mut brain = brain_with_sqlite();

        let anchor = brain.create("Deploy procedure").unwrap();
        brain.set_embedding(&anchor, &[1.0, 0.0, 0.0]).unwrap();

        let step_b = brain.create("Pull latest code").unwrap();
        brain.set_embedding(&step_b, &[0.3, 0.7, 0.0]).unwrap();

        let step_c = brain.create("Run tests").unwrap();
        brain.set_embedding(&step_c, &[0.2, 0.5, 0.3]).unwrap();

        let step_d = brain.create("Deploy to staging").unwrap();
        brain.set_embedding(&step_d, &[0.1, 0.3, 0.6]).unwrap();

        // Create ordered associations from anchor — intentionally out of order
        brain
            .associate_with_ordinal(anchor, step_d, 0.9, Some(3))
            .unwrap();
        brain
            .associate_with_ordinal(anchor, step_b, 0.9, Some(1))
            .unwrap();
        brain
            .associate_with_ordinal(anchor, step_c, 0.9, Some(2))
            .unwrap();

        let mut seen = std::collections::HashSet::new();
        seen.insert(anchor);

        let discoveries = brain
            .discover_associated_memories(&[1.0, 0.0, 0.0], &[anchor], &mut seen, 1)
            .unwrap();

        // Should discover all 3 steps, in ordinal order
        assert_eq!(discoveries.len(), 3);
        assert_eq!(
            discoveries[0].id, step_b,
            "First discovery should be step B (ord:1)"
        );
        assert_eq!(
            discoveries[1].id, step_c,
            "Second discovery should be step C (ord:2)"
        );
        assert_eq!(
            discoveries[2].id, step_d,
            "Third discovery should be step D (ord:3)"
        );
    }

    #[test]
    fn discover_mixed_ordinal_and_null_ordinal_first() {
        // Some associations have ordinals, some don't
        // Ordinal ones should be processed first (ascending), then null
        let mut brain = brain_with_sqlite();

        let anchor = brain.create("Mixed procedure").unwrap();
        brain.set_embedding(&anchor, &[1.0, 0.0, 0.0]).unwrap();

        let step1 = brain.create("Ordered step 1").unwrap();
        brain.set_embedding(&step1, &[0.3, 0.7, 0.0]).unwrap();

        let step2 = brain.create("Ordered step 2").unwrap();
        brain.set_embedding(&step2, &[0.2, 0.5, 0.3]).unwrap();

        let unordered_a = brain.create("Related note A").unwrap();
        brain.set_embedding(&unordered_a, &[0.1, 0.3, 0.6]).unwrap();

        let unordered_b = brain.create("Related note B").unwrap();
        brain.set_embedding(&unordered_b, &[0.0, 0.1, 0.9]).unwrap();

        // Mix of ordered and unordered — create in scrambled order
        brain.associate(anchor, unordered_a, 0.5).unwrap(); // no ordinal
        brain
            .associate_with_ordinal(anchor, step2, 0.9, Some(2))
            .unwrap();
        brain.associate(anchor, unordered_b, 0.5).unwrap(); // no ordinal
        brain
            .associate_with_ordinal(anchor, step1, 0.9, Some(1))
            .unwrap();

        let mut seen = std::collections::HashSet::new();
        seen.insert(anchor);

        let discoveries = brain
            .discover_associated_memories(&[1.0, 0.0, 0.0], &[anchor], &mut seen, 1)
            .unwrap();

        assert_eq!(discoveries.len(), 4);

        // First two should be the ordered steps (1, 2)
        assert_eq!(discoveries[0].id, step1, "First should be ordered step 1");
        assert_eq!(discoveries[1].id, step2, "Second should be ordered step 2");

        // Last two are unordered — their relative order doesn't matter,
        // just verify they're both there
        let unordered_ids: std::collections::HashSet<_> =
            discoveries[2..].iter().map(|d| d.id).collect();
        assert!(
            unordered_ids.contains(&unordered_a),
            "Should contain unordered A"
        );
        assert!(
            unordered_ids.contains(&unordered_b),
            "Should contain unordered B"
        );
    }

    #[test]
    fn discover_no_ordinals_unchanged_behavior() {
        // When no associations have ordinals, behavior should be exactly as before
        let mut brain = brain_with_sqlite();

        let anchor = brain.create("Root memory").unwrap();
        brain.set_embedding(&anchor, &[1.0, 0.0, 0.0]).unwrap();

        let target_a = brain.create("Related A").unwrap();
        brain.set_embedding(&target_a, &[0.5, 0.5, 0.0]).unwrap();

        let target_b = brain.create("Related B").unwrap();
        brain.set_embedding(&target_b, &[0.3, 0.7, 0.0]).unwrap();

        // No ordinals
        brain.associate(anchor, target_a, 0.7).unwrap();
        brain.associate(anchor, target_b, 0.6).unwrap();

        let mut seen = std::collections::HashSet::new();
        seen.insert(anchor);

        let discoveries = brain
            .discover_associated_memories(&[1.0, 0.0, 0.0], &[anchor], &mut seen, 1)
            .unwrap();

        // Both should be discovered
        assert_eq!(discoveries.len(), 2);
        let ids: std::collections::HashSet<_> = discoveries.iter().map(|d| d.id).collect();
        assert!(ids.contains(&target_a));
        assert!(ids.contains(&target_b));
    }

    #[test]
    fn discover_ordinal_traversal_multi_hop_chain() {
        // Test multi-hop with ordinals: anchor → B(ord:1) → C(ord:1)
        // At depth=2, should discover B first, then C
        let mut brain = brain_with_sqlite();

        let anchor = brain.create("Root").unwrap();
        brain.set_embedding(&anchor, &[1.0, 0.0, 0.0]).unwrap();

        let step_b = brain.create("Step B").unwrap();
        brain.set_embedding(&step_b, &[0.5, 0.5, 0.0]).unwrap();

        let step_c = brain.create("Step C").unwrap();
        brain.set_embedding(&step_c, &[0.0, 1.0, 0.0]).unwrap();

        brain
            .associate_with_ordinal(anchor, step_b, 0.9, Some(1))
            .unwrap();
        brain
            .associate_with_ordinal(step_b, step_c, 0.9, Some(1))
            .unwrap();

        let mut seen = std::collections::HashSet::new();
        seen.insert(anchor);

        let discoveries = brain
            .discover_associated_memories(&[1.0, 0.0, 0.0], &[anchor], &mut seen, 2)
            .unwrap();

        assert_eq!(discoveries.len(), 2);
        assert_eq!(discoveries[0].id, step_b, "First hop should discover B");
        assert_eq!(discoveries[1].id, step_c, "Second hop should discover C");
    }

    // ==================
    // EMBEDDING MODEL MIGRATION TESTS
    // ==================

    #[test]
    fn config_defaults_embedding_model_for_fresh_install() {
        let config = Config::default();
        assert_eq!(
            config.embedding_model,
            crate::embedding::default_embedding_model()
        );
        assert!(config.embedding_model_active.is_none());
    }

    #[test]
    fn config_embedding_model_backwards_compat_deserialization() {
        // Simulate loading a config from an old database that doesn't have
        // embedding_model fields. Serde defaults should fill them in.
        let old_json = r#"{
            "decay_rate_per_day": 0.05,
            "decay_interval_hours": 1.0,
            "propagation_damping": 0.5,
            "hebbian_learning_rate": 0.1,
            "recall_strength": 0.2,
            "blocked_tags": [],
            "max_tag_cardinality": 20,
            "association_decay_rate": 1.0,
            "min_association_weight": 0.05,
            "search_follow_associations": true,
            "search_association_depth": 1
        }"#;

        let config: Config = serde_json::from_str(old_json).unwrap();
        assert_eq!(
            config.embedding_model,
            crate::embedding::default_embedding_model()
        );
        assert!(
            config.embedding_model_active.is_none(),
            "Old configs should have embedding_model_active=None"
        );
    }

    #[test]
    fn mismatch_check_triggers_when_active_is_none() {
        // Fresh database: embedding_model_active is None, embedding_model is set.
        // Brain::new calls check_model_mismatch(), which sets embedding_model_active.
        let brain = brain_with_sqlite();

        assert!(
            brain.config().embedding_model_active.is_some(),
            "After Brain::new, embedding_model_active should be set"
        );
        assert_eq!(
            brain.config().embedding_model_active.as_deref(),
            Some(brain.config().embedding_model.as_str()),
            "Active should match desired after mismatch check"
        );
    }

    #[test]
    fn mismatch_check_clears_stale_embeddings_on_model_change() {
        // Create brain with one model, then change config and verify mismatch is detected.
        let mut brain = brain_with_sqlite();

        // Create some engrams with embeddings
        let id_a = brain.create("Rust programming").unwrap();
        brain.set_embedding(&id_a, &[1.0, 0.0, 0.0]).unwrap();

        let id_b = brain.create("Python scripting").unwrap();
        brain.set_embedding(&id_b, &[0.0, 1.0, 0.0]).unwrap();

        assert_eq!(brain.count_with_embeddings().unwrap(), 2);

        // Simulate a config change to a new model
        let mut config = brain.config().clone();
        config.embedding_model = "BGESmallENV15".to_string();
        // Keep active as the old model — this mismatch triggers the check
        brain.set_config(config).unwrap();

        assert_ne!(
            brain.config().embedding_model,
            brain
                .config()
                .embedding_model_active
                .as_deref()
                .unwrap_or(""),
        );

        // Run the mismatch check directly (as Brain::new/open would).
        // The check clears stale vectors and updates embedding_model_active.
        // Re-generation is NOT done here — that's `memoryco generate`.
        let _ = brain.check_model_mismatch();

        // After the check, active should match desired and old embeddings are gone
        assert_eq!(
            brain.config().embedding_model_active.as_deref(),
            Some("BGESmallENV15"),
            "After mismatch check, active model should match desired"
        );
        assert_eq!(
            brain.count_with_embeddings().unwrap(),
            0,
            "Stale embeddings should be cleared on model change"
        );
    }

    #[test]
    fn mismatch_check_skips_when_models_match() {
        let mut brain = brain_with_sqlite();

        // After init, active should already match desired
        let config = brain.config().clone();
        assert_eq!(
            config.embedding_model_active.as_deref(),
            Some(config.embedding_model.as_str())
        );

        // Create an engram with embedding
        let id = brain.create("Test memory").unwrap();
        brain.set_embedding(&id, &[1.0, 0.0, 0.0]).unwrap();

        // Mismatch check should be a no-op when models match
        brain.check_model_mismatch().unwrap();

        // Embedding should still exist (not cleared)
        assert_eq!(brain.count_with_embeddings().unwrap(), 1);
    }

    #[test]
    fn clear_all_embeddings_works() {
        let mut brain = brain_with_sqlite();

        let id_a = brain.create("Memory A").unwrap();
        brain.set_embedding(&id_a, &[1.0, 0.0, 0.0]).unwrap();

        let id_b = brain.create("Memory B").unwrap();
        brain.set_embedding(&id_b, &[0.0, 1.0, 0.0]).unwrap();

        let id_c = brain.create("Memory C (no embedding)").unwrap();

        assert_eq!(brain.count_with_embeddings().unwrap(), 2);
        assert_eq!(brain.count_without_embeddings().unwrap(), 1);

        // Clear all embeddings
        let cleared = brain.storage.lock().unwrap().clear_all_embeddings().unwrap();
        assert_eq!(cleared, 2, "Should have cleared 2 embeddings");

        // All should now be without embeddings
        assert_eq!(brain.count_with_embeddings().unwrap(), 0);
        assert_eq!(brain.count_without_embeddings().unwrap(), 3);

        // Engrams themselves should still exist (no data loss)
        assert!(brain.get(&id_a).is_some());
        assert!(brain.get(&id_b).is_some());
        assert!(brain.get(&id_c).is_some());
    }

    #[test]
    fn clear_all_embeddings_returns_zero_when_none_exist() {
        let mut brain = brain_with_sqlite();

        brain.create("No embedding memory").unwrap();

        let cleared = brain.storage.lock().unwrap().clear_all_embeddings().unwrap();
        assert_eq!(cleared, 0);
    }

    // ==================
    // RERANK CONFIG TESTS
    // ==================

    #[test]
    fn config_defaults_rerank_mode() {
        let config = Config::default();
        assert_eq!(config.rerank_mode, "cross-encoder");
        assert_eq!(config.rerank_candidates, 30);
    }

    #[test]
    fn config_rerank_roundtrip() {
        let storage = MemoryStorage::new();
        let mut brain = Brain::new(storage, Config::default()).unwrap();

        // Default: cross-encoder mode with 30 candidates
        assert_eq!(brain.config().rerank_mode, "cross-encoder");
        assert_eq!(brain.config().rerank_candidates, 30);

        // rerank_mode is a string — configure() returns false for it (handled via set_config)
        assert!(!brain.configure("rerank_mode", 0.0).unwrap());

        // Set candidates
        assert!(brain.configure("rerank_candidates", 50.0).unwrap());
        assert_eq!(brain.config().rerank_candidates, 50);

        // Candidates are clamped to [5, 200]
        assert!(brain.configure("rerank_candidates", 1.0).unwrap());
        assert_eq!(brain.config().rerank_candidates, 5);

        assert!(brain.configure("rerank_candidates", 999.0).unwrap());
        assert_eq!(brain.config().rerank_candidates, 200);
    }

    #[test]
    fn config_rerank_backwards_compat_deserialization() {
        // Old config JSON without rerank fields should deserialize with defaults
        let old_json = r#"{
            "decay_rate_per_day": 0.05,
            "decay_interval_hours": 1.0,
            "propagation_damping": 0.5,
            "hebbian_learning_rate": 0.1,
            "recall_strength": 0.2,
            "blocked_tags": [],
            "max_tag_cardinality": 20,
            "association_decay_rate": 1.0,
            "min_association_weight": 0.05,
            "search_follow_associations": true,
            "search_association_depth": 1
        }"#;

        let config: Config = serde_json::from_str(old_json).unwrap();
        assert_eq!(config.rerank_mode, "cross-encoder", "rerank should default to cross-encoder");
        assert_eq!(
            config.rerank_candidates, 30,
            "rerank_candidates should default to 30"
        );
    }

    // ==================
    // HYBRID SEARCH / FTS5 TESTS
    // ==================

    #[test]
    fn config_defaults_hybrid_search_enabled() {
        let config = Config::default();
        assert!(config.hybrid_search_enabled);
    }

    #[test]
    fn config_hybrid_search_roundtrip() {
        let storage = MemoryStorage::new();
        let mut brain = Brain::new(storage, Config::default()).unwrap();

        // Default: enabled
        assert!(brain.config().hybrid_search_enabled);

        // Disable via configure()
        assert!(brain.configure("hybrid_search_enabled", 0.0).unwrap());
        assert!(!brain.config().hybrid_search_enabled);

        // Re-enable
        assert!(brain.configure("hybrid_search_enabled", 1.0).unwrap());
        assert!(brain.config().hybrid_search_enabled);
    }

    #[test]
    fn config_hybrid_search_backwards_compat_deserialization() {
        // Old config JSON without hybrid_search_enabled should deserialize with default true
        let old_json = r#"{
            "decay_rate_per_day": 0.05,
            "decay_interval_hours": 1.0,
            "propagation_damping": 0.5,
            "hebbian_learning_rate": 0.1,
            "recall_strength": 0.2,
            "blocked_tags": [],
            "max_tag_cardinality": 20,
            "association_decay_rate": 1.0,
            "min_association_weight": 0.05,
            "search_follow_associations": true,
            "search_association_depth": 1
        }"#;

        let config: Config = serde_json::from_str(old_json).unwrap();
        assert!(
            config.hybrid_search_enabled,
            "hybrid_search should default to enabled"
        );
    }

    #[test]
    fn fts_backfill_populates_existing_engrams() {
        // This test verifies the backfill flow through diesel.rs tests
        // (where we have access to raw SQL). Here we just verify that
        // keyword_search works end-to-end through Brain after normal create.
        let mut brain = brain_with_sqlite();

        brain.create("Memory about quantum computing").unwrap();
        brain.create("Memory about neural networks").unwrap();

        // Brain.new already called ensure_fts_index, so FTS should be populated
        let results = brain.keyword_search("quantum", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("quantum"));
    }

    #[test]
    fn keyword_search_via_brain() {
        let mut brain = brain_with_sqlite();

        brain.create("Rust programming language is fast").unwrap();
        brain
            .create("Python is a dynamic scripting language")
            .unwrap();
        brain.create("Cooking recipes for pasta").unwrap();

        let results = brain.keyword_search("language", 10).unwrap();
        assert_eq!(results.len(), 2, "Both language memories should match");
    }

    #[test]
    fn get_embedding_returns_none_for_missing_engram() {
        let mut brain = brain_with_sqlite();

        // Create a valid engram so the DB is initialized
        brain.create("Test memory").unwrap();

        // Query embedding for a non-existent engram ID
        let missing_id = uuid::Uuid::new_v4();
        let result = brain.get_embedding(&missing_id);
        assert!(
            result.is_ok(),
            "get_embedding should return Ok for missing engram, got: {:?}",
            result
        );
        assert_eq!(
            result.unwrap(),
            None,
            "get_embedding should return None for missing engram"
        );
    }

    #[test]
    fn prune_orphan_associations_removes_dangling() {
        let mut brain = brain_with_sqlite();

        let a = brain.create("Memory A").unwrap();
        let b = brain.create("Memory B").unwrap();
        let c = brain.create("Memory C").unwrap();

        // Create valid associations
        brain.associate(a, b, 0.8).unwrap();
        brain.associate(b, c, 0.6).unwrap();

        // Delete memory B — this cleans up associations in storage
        brain.delete(b).unwrap();

        // Simulate the persistence worker race: re-insert orphan associations
        // directly into storage after the delete has cleaned them up.
        let orphan_assoc = Association::with_weight(a, b, 0.8);
        brain
            .storage
            .lock()
            .unwrap()
            .save_associations(&[&orphan_assoc])
            .unwrap();

        // Verify the orphan exists in storage
        let all_assocs = brain.storage.lock().unwrap().load_all_associations().unwrap();
        assert!(
            all_assocs.iter().any(|assoc| assoc.to == b),
            "Orphan association should exist before pruning"
        );

        // Prune orphans
        let pruned = brain.prune_orphan_associations().unwrap();
        assert!(
            pruned >= 1,
            "Should have pruned at least 1 orphan, pruned: {}",
            pruned
        );

        // Verify no orphans remain
        let remaining = brain.storage.lock().unwrap().load_all_associations().unwrap();
        for assoc in &remaining {
            assert!(
                brain.get(&assoc.from).is_some() || assoc.from == a || assoc.from == c,
                "Association from {} should reference an existing engram",
                assoc.from
            );
            assert!(
                assoc.to != b,
                "No associations should point to deleted engram B"
            );
        }
    }

    #[test]
    fn discover_associations_skips_dangling_references() {
        let mut brain = brain_with_sqlite();

        let a = brain.create("Memory about Rust").unwrap();
        let b = brain.create("Memory about safety").unwrap();
        let c = brain.create("Memory about types").unwrap();

        // Set embeddings so discover_associated_memories can compute similarity
        let emb_a: Vec<f32> = vec![0.9, 0.1, 0.0, 0.0];
        let emb_b: Vec<f32> = vec![0.8, 0.2, 0.0, 0.0];
        let emb_c: Vec<f32> = vec![0.7, 0.3, 0.0, 0.0];
        brain.set_embedding(&a, &emb_a).unwrap();
        brain.set_embedding(&b, &emb_b).unwrap();
        brain.set_embedding(&c, &emb_c).unwrap();

        brain.associate(a, b, 0.8).unwrap();
        brain.associate(a, c, 0.6).unwrap();

        // Delete B, then re-insert orphan association (simulating persistence race)
        brain.delete(b).unwrap();
        let orphan_assoc = Association::with_weight(a, b, 0.8);
        brain
            .storage
            .lock()
            .unwrap()
            .save_associations(&[&orphan_assoc])
            .unwrap();

        // Reload associations into substrate so it sees the orphan
        let _ = brain.sync_from_storage();

        // discover_associated_memories should NOT error when encountering
        // the dangling association to B (whose embedding row is gone)
        let query_emb: Vec<f32> = vec![0.85, 0.15, 0.0, 0.0];
        let mut seen = std::collections::HashSet::new();
        seen.insert(a);

        let result = brain.discover_associated_memories(&query_emb, &[a], &mut seen, 1);
        assert!(
            result.is_ok(),
            "discover_associated_memories should not error on dangling associations: {:?}",
            result
        );

        // Only C should be discovered (B is deleted)
        let discoveries = result.unwrap();
        assert!(
            !discoveries.iter().any(|d| d.id == b),
            "Deleted engram B should not appear in discoveries"
        );
    }
}
