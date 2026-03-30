//! Storage trait - abstract persistence layer
//!
//! This module defines the contract for storing and retrieving memory data.
//! Implementations can be SQLite, in-memory, Postgres, Redis, whatever.
//! The key is: the rest of the system doesn't care.

mod diesel;
#[cfg(test)]
pub(crate) mod memory;
pub mod models;
pub mod rrf;
pub mod schema;
mod vector;

pub use diesel::MemoryStorage;
pub use vector::{SimilarityResult, VectorSearch};

// Re-export unified storage types from the foundation
pub use crate::storage::{StorageError, StorageResult};

use super::{Association, Config, Memory, MemoryId, MemoryState};
use crate::memory_core::session::SessionContext;
use crate::identity::Identity;

/// The storage contract
///
/// Implementations of this trait provide persistence for the memory system.
/// All operations are synchronous. If you need async, wrap at a higher level.
///
/// Design notes:
/// - Identity is saved/loaded as a whole (it's small and always needed)
/// - Memories and Associations are fine-grained for flexibility
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
    // MEMORIES
    // ==================

    /// Save a single memory (upsert by id)
    fn save_memory(&mut self, mem: &Memory) -> StorageResult<()>;

    /// Save multiple memories in a batch (for efficiency)
    fn save_memories(&mut self, memories: &[&Memory]) -> StorageResult<()> {
        // Default implementation: save one by one
        // Implementations can override for batch optimization
        for mem in memories {
            self.save_memory(mem)?;
        }
        Ok(())
    }

    /// Load a single memory by ID
    fn load_memory(&mut self, id: &MemoryId) -> StorageResult<Option<Memory>>;

    /// Load all memories
    fn load_all_memories(&mut self) -> StorageResult<Vec<Memory>>;

    /// Load memories by state (for partial loading)
    fn load_memories_by_state(&mut self, state: MemoryState) -> StorageResult<Vec<Memory>>;

    /// Load memories by tag
    fn load_memories_by_tag(&mut self, tag: &str) -> StorageResult<Vec<Memory>>;

    /// Delete a memory by ID (also removes associated associations)
    /// Returns true if the memory existed and was deleted
    fn delete_memory(&mut self, id: &MemoryId) -> StorageResult<bool>;

    /// Count total memories in storage
    /// Used for lightweight cross-process sync detection
    fn count_memories(&mut self) -> StorageResult<usize>;

    /// Update only energy and state for memories (efficient bulk update)
    /// Used by decay and recall effects where content hasn't changed
    /// Default implementation is no-op; implementations should override
    fn save_memory_energies(
        &mut self,
        updates: &[(&MemoryId, f64, MemoryState)],
    ) -> StorageResult<()> {
        // Default: no optimized implementation, caller should use save_memories
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

    /// Load all associations from a given memory
    fn load_associations_from(&mut self, from: &MemoryId) -> StorageResult<Vec<Association>>;

    /// Load all associations
    fn load_all_associations(&mut self) -> StorageResult<Vec<Association>>;

    /// Delete all associations (for bulk operations like pruning)
    fn delete_all_associations(&mut self) -> StorageResult<()>;

    /// Delete associations that reference non-existent memories.
    /// Returns the number of orphaned associations removed.
    fn prune_orphan_associations(&mut self) -> StorageResult<usize>;

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

    /// Find memories similar to the given embedding
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

    /// Count memories that have embeddings
    fn count_with_embeddings(&mut self) -> StorageResult<usize> {
        Ok(0)
    }

    /// Count memories that need embeddings
    fn count_without_embeddings(&mut self) -> StorageResult<usize> {
        Ok(0)
    }

    /// Get IDs of memories that need embeddings (for backfill)
    fn get_ids_without_embeddings(&mut self, limit: usize) -> StorageResult<Vec<MemoryId>> {
        let _ = limit;
        Ok(Vec::new())
    }

    /// Get IDs of memories that have no enrichments (for incremental enrichment backfill)
    fn get_ids_without_enrichments(&mut self) -> StorageResult<Vec<MemoryId>> {
        Ok(Vec::new())
    }

    /// Update embedding for a single memory
    fn set_embedding(&mut self, id: &MemoryId, embedding: &[f32]) -> StorageResult<()> {
        let _ = (id, embedding);
        Ok(())
    }

    /// Get embedding for a single memory
    fn get_embedding(&mut self, id: &MemoryId) -> StorageResult<Option<Vec<f32>>> {
        let _ = id;
        Ok(None)
    }

    /// Clear all embeddings (set to NULL) for migration purposes.
    /// Returns the number of affected rows.
    fn clear_all_embeddings(&mut self) -> StorageResult<usize> {
        Ok(0)
    }

    /// Store enrichment embeddings for a memory (multi-vector support).
    /// Replaces any existing enrichments for this memory.
    fn set_enrichment_embeddings(
        &mut self,
        id: &MemoryId,
        embeddings: &[Vec<f32>],
        source: &str,
    ) -> StorageResult<()> {
        let _ = (id, embeddings, source);
        Ok(())
    }

    /// Delete all enrichment embeddings for a memory.
    fn delete_enrichments(&mut self, id: &MemoryId) -> StorageResult<()> {
        let _ = id;
        Ok(())
    }

    /// Count total enrichment vectors across all memories.
    fn count_enrichments(&mut self) -> StorageResult<usize> {
        Ok(0)
    }

    /// Clear all enrichment embeddings for migration purposes.
    /// Returns the number of affected rows.
    fn clear_all_enrichments(&mut self) -> StorageResult<usize> {
        Ok(0)
    }

    // ==================
    // KEYWORD SEARCH (FTS5)
    // ==================

    /// Search memories by keyword using FTS5/BM25.
    /// Returns results sorted by BM25 relevance score (higher = more relevant).
    fn keyword_search(
        &mut self,
        query: &str,
        limit: usize,
    ) -> StorageResult<Vec<SimilarityResult>> {
        let _ = (query, limit);
        Ok(Vec::new())
    }

    /// Ensure the FTS5 index is populated from existing memorys.
    /// Returns the number of memories backfilled.
    fn ensure_fts_populated(&mut self) -> StorageResult<usize> {
        Ok(0)
    }

    // ==================
    // ACCESS LOG
    // ==================

    /// Log a search→recall cycle for training data extraction.
    /// Records the query text, which memories were returned, and which were recalled.
    fn log_access(
        &mut self,
        query_text: &str,
        result_ids: &[MemoryId],
        recalled_ids: &[MemoryId],
    ) -> StorageResult<()> {
        let _ = (query_text, result_ids, recalled_ids);
        Ok(())
    }

    // ==================
    // SESSIONS
    // ==================

    /// Load a session by ID. Returns None if the session does not exist.
    fn load_session(&mut self, session_id: &str) -> StorageResult<Option<SessionContext>> {
        let _ = session_id;
        Ok(None)
    }

    /// Save (upsert) a session.
    fn save_session(&mut self, session: &SessionContext) -> StorageResult<()> {
        let _ = session;
        Ok(())
    }

    /// Delete sessions whose `last_seen_at` is strictly less than `expire_before`.
    /// Returns the number of sessions deleted.
    fn delete_expired_sessions(&mut self, expire_before: i64) -> StorageResult<usize> {
        let _ = expire_before;
        Ok(0)
    }

    /// Delete all sessions. Used during embedding model migration since session
    /// centroids have wrong dimensions after a model switch.
    /// Returns the number of sessions deleted.
    fn clear_all_sessions(&mut self) -> StorageResult<usize> {
        Ok(0)
    }
}
