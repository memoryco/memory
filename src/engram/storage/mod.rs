//! Storage trait - abstract persistence layer
//!
//! This module defines the contract for storing and retrieving engram data.
//! Implementations can be SQLite, in-memory, Postgres, Redis, whatever.
//! The key is: the rest of the system doesn't care.

mod diesel;
#[cfg(test)]
pub(crate) mod memory;
pub mod models;
pub mod rrf;
pub mod schema;
mod vector;

pub use diesel::EngramStorage;
pub use vector::{SimilarityResult, VectorSearch};

// Re-export unified storage types from the foundation
pub use crate::storage::{StorageError, StorageResult};

use super::{Association, Config, Engram, EngramId, MemoryState};
use crate::identity::Identity;

/// The storage contract
///
/// Implementations of this trait provide persistence for the engram system.
/// All operations are synchronous. If you need async, wrap at a higher level.
///
/// Design notes:
/// - Identity is saved/loaded as a whole (it's small and always needed)
/// - Engrams and Associations are fine-grained for flexibility
/// - Delete is available for housekeeping (duplicates, corrections)
/// - Save operations are upserts (insert or update)
pub trait Storage: Send {
    // ==================
    // IDENTITY
    // ==================

    /// Save the entire identity (upsert)
    fn save_identity(&mut self, identity: &Identity) -> StorageResult<()>;

    /// Load the identity (returns None if not yet created)
    fn load_identity(&mut self) -> StorageResult<Option<Identity>>;

    // ==================
    // ENGRAMS
    // ==================

    /// Save a single engram (upsert by id)
    fn save_engram(&mut self, engram: &Engram) -> StorageResult<()>;

    /// Save multiple engrams in a batch (for efficiency)
    fn save_engrams(&mut self, engrams: &[&Engram]) -> StorageResult<()> {
        // Default implementation: save one by one
        // Implementations can override for batch optimization
        for engram in engrams {
            self.save_engram(engram)?;
        }
        Ok(())
    }

    /// Load a single engram by ID
    fn load_engram(&mut self, id: &EngramId) -> StorageResult<Option<Engram>>;

    /// Load all engrams
    fn load_all_engrams(&mut self) -> StorageResult<Vec<Engram>>;

    /// Load engrams by state (for partial loading)
    fn load_engrams_by_state(&mut self, state: MemoryState) -> StorageResult<Vec<Engram>>;

    /// Load engrams by tag
    fn load_engrams_by_tag(&mut self, tag: &str) -> StorageResult<Vec<Engram>>;

    /// Delete an engram by ID (also removes associated associations)
    /// Returns true if the engram existed and was deleted
    fn delete_engram(&mut self, id: &EngramId) -> StorageResult<bool>;

    /// Count total engrams in storage
    /// Used for lightweight cross-process sync detection
    fn count_engrams(&mut self) -> StorageResult<usize>;

    /// Update only energy and state for engrams (efficient bulk update)
    /// Used by decay and recall effects where content hasn't changed
    /// Default implementation is no-op; implementations should override
    fn save_engram_energies(
        &mut self,
        updates: &[(&EngramId, f64, MemoryState)],
    ) -> StorageResult<()> {
        // Default: no optimized implementation, caller should use save_engrams
        let _ = updates;
        Ok(())
    }

    // ==================
    // ASSOCIATIONS
    // ==================

    /// Save a single association (upsert by from+to)
    fn save_association(&mut self, assoc: &Association) -> StorageResult<()>;

    /// Save multiple associations in a batch
    fn save_associations(&mut self, assocs: &[&Association]) -> StorageResult<()> {
        // Default implementation: save one by one
        for assoc in assocs {
            self.save_association(assoc)?;
        }
        Ok(())
    }

    /// Load all associations from a given engram
    fn load_associations_from(&mut self, from: &EngramId) -> StorageResult<Vec<Association>>;

    /// Load all associations
    fn load_all_associations(&mut self) -> StorageResult<Vec<Association>>;

    /// Delete all associations (for bulk operations like pruning)
    fn delete_all_associations(&mut self) -> StorageResult<()>;

    // ==================
    // CONFIG
    // ==================

    /// Save substrate configuration
    fn save_config(&mut self, config: &Config) -> StorageResult<()>;

    /// Load substrate configuration
    fn load_config(&mut self) -> StorageResult<Option<Config>>;

    /// Save the last decay timestamp
    fn save_last_decay_at(&mut self, timestamp: i64) -> StorageResult<()>;

    /// Load the last decay timestamp
    fn load_last_decay_at(&mut self) -> StorageResult<Option<i64>>;

    /// Get a metadata value by key
    fn get_metadata(&mut self, key: &str) -> StorageResult<Option<String>>;

    /// Set a metadata key/value pair
    fn set_metadata(&mut self, key: &str, value: &str) -> StorageResult<()>;

    // ==================
    // LIFECYCLE
    // ==================

    /// Initialize storage (create tables, files, etc.)
    /// Called once on first use
    fn initialize(&mut self) -> StorageResult<()>;

    /// Flush any pending writes
    /// Some implementations may buffer; this ensures durability
    fn flush(&mut self) -> StorageResult<()> {
        // Default: no-op (for implementations that don't buffer)
        Ok(())
    }

    /// Close the storage cleanly
    fn close(&mut self) -> StorageResult<()> {
        self.flush()
    }

    // ==================
    // VECTOR SEARCH
    // ==================

    /// Find engrams similar to the given embedding
    /// Returns (id, score, content) tuples sorted by descending similarity
    fn find_similar_by_embedding(
        &mut self,
        query_embedding: &[f32],
        limit: usize,
        min_score: f32,
    ) -> StorageResult<Vec<SimilarityResult>> {
        // Default: no vector search support
        let _ = (query_embedding, limit, min_score);
        Ok(Vec::new())
    }

    /// Count engrams that have embeddings
    fn count_with_embeddings(&mut self) -> StorageResult<usize> {
        Ok(0)
    }

    /// Count engrams that need embeddings
    fn count_without_embeddings(&mut self) -> StorageResult<usize> {
        Ok(0)
    }

    /// Get IDs of engrams that need embeddings (for backfill)
    fn get_ids_without_embeddings(&mut self, limit: usize) -> StorageResult<Vec<EngramId>> {
        let _ = limit;
        Ok(Vec::new())
    }

    /// Update embedding for a single engram
    fn set_embedding(&mut self, id: &EngramId, embedding: &[f32]) -> StorageResult<()> {
        let _ = (id, embedding);
        Ok(())
    }

    /// Get embedding for a single engram
    fn get_embedding(&mut self, id: &EngramId) -> StorageResult<Option<Vec<f32>>> {
        let _ = id;
        Ok(None)
    }

    /// Clear all embeddings (set to NULL) for migration purposes.
    /// Returns the number of affected rows.
    fn clear_all_embeddings(&mut self) -> StorageResult<usize> {
        Ok(0)
    }

    // ==================
    // KEYWORD SEARCH (FTS5)
    // ==================

    /// Search engrams by keyword using FTS5/BM25.
    /// Returns results sorted by BM25 relevance score (higher = more relevant).
    fn keyword_search(
        &mut self,
        query: &str,
        limit: usize,
    ) -> StorageResult<Vec<SimilarityResult>> {
        let _ = (query, limit);
        Ok(Vec::new())
    }

    /// Ensure the FTS5 index is populated from existing engrams.
    /// Returns the number of engrams backfilled.
    fn ensure_fts_populated(&mut self) -> StorageResult<usize> {
        Ok(0)
    }

    // ==================
    // ACCESS LOG
    // ==================

    /// Log a search→recall cycle for training data extraction.
    /// Records the query text, which engrams were returned, and which were recalled.
    fn log_access(
        &mut self,
        query_text: &str,
        result_ids: &[EngramId],
        recalled_ids: &[EngramId],
    ) -> StorageResult<()> {
        let _ = (query_text, result_ids, recalled_ids);
        Ok(())
    }
}
