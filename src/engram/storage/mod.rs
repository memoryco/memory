//! Storage trait - abstract persistence layer
//!
//! This module defines the contract for storing and retrieving engram data.
//! Implementations can be SQLite, in-memory, Postgres, Redis, whatever.
//! The key is: the rest of the system doesn't care.

mod memory;
mod sqlite;
mod vector;

pub use memory::MemoryStorage;
pub use sqlite::SqliteStorage;
pub use vector::{VectorSearch, SimilarityResult};

use super::{EngramId, Engram, Association, Config, MemoryState};
use super::identity::Identity;
use std::fmt;

/// Result type for storage operations
pub type StorageResult<T> = Result<T, StorageError>;

/// Errors that can occur during storage operations
#[derive(Debug)]
pub enum StorageError {
    /// Item not found
    NotFound(String),
    /// IO error (file, network, etc.)
    Io(String),
    /// Serialization/deserialization error
    Serialization(String),
    /// Database/query error
    Database(String),
    /// Configuration error
    Config(String),
    /// Generic error with context
    Other(String),
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageError::NotFound(msg) => write!(f, "Not found: {}", msg),
            StorageError::Io(msg) => write!(f, "IO error: {}", msg),
            StorageError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            StorageError::Database(msg) => write!(f, "Database error: {}", msg),
            StorageError::Config(msg) => write!(f, "Config error: {}", msg),
            StorageError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for StorageError {}

// Convenience conversions
impl From<std::io::Error> for StorageError {
    fn from(err: std::io::Error) -> Self {
        StorageError::Io(err.to_string())
    }
}

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
    fn load_identity(&self) -> StorageResult<Option<Identity>>;
    
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
    fn load_engram(&self, id: &EngramId) -> StorageResult<Option<Engram>>;
    
    /// Load all engrams
    fn load_all_engrams(&self) -> StorageResult<Vec<Engram>>;
    
    /// Load engrams by state (for partial loading)
    fn load_engrams_by_state(&self, state: MemoryState) -> StorageResult<Vec<Engram>>;
    
    /// Load engrams by tag
    fn load_engrams_by_tag(&self, tag: &str) -> StorageResult<Vec<Engram>>;
    
    /// Search engrams by content using full-text search
    /// Returns IDs of matching engrams
    /// Default implementation does simple substring matching
    fn search_content(&self, query: &str) -> StorageResult<Vec<EngramId>> {
        // Default: no search support, return empty
        let _ = query;
        Ok(Vec::new())
    }
    
    /// Delete an engram by ID (also removes associated associations)
    /// Returns true if the engram existed and was deleted
    fn delete_engram(&mut self, id: &EngramId) -> StorageResult<bool>;
    
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
    fn load_associations_from(&self, from: &EngramId) -> StorageResult<Vec<Association>>;
    
    /// Load all associations
    fn load_all_associations(&self) -> StorageResult<Vec<Association>>;
    
    /// Delete all associations (for bulk operations like pruning)
    fn delete_all_associations(&mut self) -> StorageResult<()>;
    
    // ==================
    // CONFIG
    // ==================
    
    /// Save substrate configuration
    fn save_config(&mut self, config: &Config) -> StorageResult<()>;
    
    /// Load substrate configuration
    fn load_config(&self) -> StorageResult<Option<Config>>;
    
    /// Save the last decay timestamp
    fn save_last_decay_at(&mut self, timestamp: i64) -> StorageResult<()>;
    
    /// Load the last decay timestamp
    fn load_last_decay_at(&self) -> StorageResult<Option<i64>>;
    
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
        &self,
        query_embedding: &[f32],
        limit: usize,
        min_score: f32,
    ) -> StorageResult<Vec<SimilarityResult>> {
        // Default: no vector search support
        let _ = (query_embedding, limit, min_score);
        Ok(Vec::new())
    }
    
    /// Count engrams that have embeddings
    fn count_with_embeddings(&self) -> StorageResult<usize> {
        Ok(0)
    }
    
    /// Count engrams that need embeddings
    fn count_without_embeddings(&self) -> StorageResult<usize> {
        Ok(0)
    }
    
    /// Get IDs of engrams that need embeddings (for backfill)
    fn get_ids_without_embeddings(&self, limit: usize) -> StorageResult<Vec<EngramId>> {
        let _ = limit;
        Ok(Vec::new())
    }
    
    /// Update embedding for a single engram
    fn set_embedding(&mut self, id: &EngramId, embedding: &[f32]) -> StorageResult<()> {
        let _ = (id, embedding);
        Ok(())
    }
    
    /// Get embedding for a single engram
    fn get_embedding(&self, id: &EngramId) -> StorageResult<Option<Vec<f32>>> {
        let _ = id;
        Ok(None)
    }
}
