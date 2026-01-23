//! Storage trait - abstract persistence layer
//!
//! This trait defines the contract for storing and retrieving engram data.
//! Implementations can be SQLite, JSON files, Redis, whatever.
//! The key is: the rest of the system doesn't care.

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
}

/// In-memory storage implementation (for testing)
#[derive(Debug, Default)]
pub struct MemoryStorage {
    identity: Option<Identity>,
    engrams: std::collections::HashMap<EngramId, Engram>,
    associations: Vec<Association>,
    config: Option<Config>,
    last_decay_at: Option<i64>,
}

impl MemoryStorage {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Storage for MemoryStorage {
    fn save_identity(&mut self, identity: &Identity) -> StorageResult<()> {
        self.identity = Some(identity.clone());
        Ok(())
    }
    
    fn load_identity(&self) -> StorageResult<Option<Identity>> {
        Ok(self.identity.clone())
    }
    
    fn save_engram(&mut self, engram: &Engram) -> StorageResult<()> {
        self.engrams.insert(engram.id, engram.clone());
        Ok(())
    }
    
    fn load_engram(&self, id: &EngramId) -> StorageResult<Option<Engram>> {
        Ok(self.engrams.get(id).cloned())
    }
    
    fn load_all_engrams(&self) -> StorageResult<Vec<Engram>> {
        Ok(self.engrams.values().cloned().collect())
    }
    
    fn load_engrams_by_state(&self, state: MemoryState) -> StorageResult<Vec<Engram>> {
        Ok(self.engrams.values()
            .filter(|e| e.state == state)
            .cloned()
            .collect())
    }
    
    fn load_engrams_by_tag(&self, tag: &str) -> StorageResult<Vec<Engram>> {
        let tag_lower = tag.to_lowercase();
        Ok(self.engrams.values()
            .filter(|e| e.tags.iter().any(|t| t.to_lowercase() == tag_lower))
            .cloned()
            .collect())
    }
    
    fn search_content(&self, query: &str) -> StorageResult<Vec<EngramId>> {
        // Simple in-memory search: tokenize query and match any token
        let tokens: Vec<String> = query.split_whitespace()
            .map(|t| t.to_lowercase())
            .collect();
        
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        
        Ok(self.engrams.values()
            .filter(|e| {
                let content_lower = e.content.to_lowercase();
                let tags_lower: Vec<String> = e.tags.iter().map(|t| t.to_lowercase()).collect();
                tokens.iter().any(|t| content_lower.contains(t) || tags_lower.iter().any(|tag| tag.contains(t)))
            })
            .map(|e| e.id)
            .collect())
    }
    
    fn delete_engram(&mut self, id: &EngramId) -> StorageResult<bool> {
        // Remove the engram
        let existed = self.engrams.remove(id).is_some();
        
        // Remove associations from/to this engram
        self.associations.retain(|a| a.from != *id && a.to != *id);
        
        Ok(existed)
    }
    
    fn save_association(&mut self, assoc: &Association) -> StorageResult<()> {
        // Upsert: remove existing if present, then add
        self.associations.retain(|a| !(a.from == assoc.from && a.to == assoc.to));
        self.associations.push(assoc.clone());
        Ok(())
    }
    
    fn load_associations_from(&self, from: &EngramId) -> StorageResult<Vec<Association>> {
        Ok(self.associations.iter()
            .filter(|a| &a.from == from)
            .cloned()
            .collect())
    }
    
    fn load_all_associations(&self) -> StorageResult<Vec<Association>> {
        Ok(self.associations.clone())
    }
    
    fn delete_all_associations(&mut self) -> StorageResult<()> {
        self.associations.clear();
        Ok(())
    }
    
    fn save_config(&mut self, config: &Config) -> StorageResult<()> {
        self.config = Some(config.clone());
        Ok(())
    }
    
    fn load_config(&self) -> StorageResult<Option<Config>> {
        Ok(self.config.clone())
    }
    
    fn save_last_decay_at(&mut self, timestamp: i64) -> StorageResult<()> {
        self.last_decay_at = Some(timestamp);
        Ok(())
    }
    
    fn load_last_decay_at(&self) -> StorageResult<Option<i64>> {
        Ok(self.last_decay_at)
    }
    
    fn initialize(&mut self) -> StorageResult<()> {
        // Nothing to do for in-memory
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::{Engram, MemoryState};
    
    #[test]
    fn memory_storage_engrams() {
        let mut storage = MemoryStorage::new();
        storage.initialize().unwrap();
        
        let engram = Engram::new("Test memory");
        let id = engram.id;
        
        storage.save_engram(&engram).unwrap();
        
        let loaded = storage.load_engram(&id).unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().content, "Test memory");
    }
    
    #[test]
    fn memory_storage_associations() {
        let mut storage = MemoryStorage::new();
        storage.initialize().unwrap();
        
        let e1 = Engram::new("Memory 1");
        let e2 = Engram::new("Memory 2");
        
        let assoc = Association::new(e1.id, e2.id);
        
        storage.save_association(&assoc).unwrap();
        
        let loaded = storage.load_associations_from(&e1.id).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].to, e2.id);
    }
    
    #[test]
    fn memory_storage_upsert() {
        let mut storage = MemoryStorage::new();
        storage.initialize().unwrap();
        
        let mut engram = Engram::new("Original");
        let id = engram.id;
        
        storage.save_engram(&engram).unwrap();
        
        engram.content = "Updated".to_string();
        storage.save_engram(&engram).unwrap();
        
        let loaded = storage.load_engram(&id).unwrap().unwrap();
        assert_eq!(loaded.content, "Updated");
        
        // Should still be just one engram
        assert_eq!(storage.load_all_engrams().unwrap().len(), 1);
    }
    
    #[test]
    fn memory_storage_load_by_state() {
        let mut storage = MemoryStorage::new();
        storage.initialize().unwrap();
        
        let mut active = Engram::new("Active memory");
        active.state = MemoryState::Active;
        
        let mut archived = Engram::new("Archived memory");
        archived.state = MemoryState::Archived;
        archived.energy = 0.01;
        
        storage.save_engram(&active).unwrap();
        storage.save_engram(&archived).unwrap();
        
        let active_only = storage.load_engrams_by_state(MemoryState::Active).unwrap();
        assert_eq!(active_only.len(), 1);
        assert_eq!(active_only[0].content, "Active memory");
    }
    
    #[test]
    fn memory_storage_load_by_tag() {
        let mut storage = MemoryStorage::new();
        storage.initialize().unwrap();
        
        let mut work = Engram::new("Work memory");
        work.tags = vec!["work".to_string(), "rust".to_string()];
        
        let mut personal = Engram::new("Personal memory");
        personal.tags = vec!["personal".to_string()];
        
        storage.save_engram(&work).unwrap();
        storage.save_engram(&personal).unwrap();
        
        let work_only = storage.load_engrams_by_tag("work").unwrap();
        assert_eq!(work_only.len(), 1);
        assert_eq!(work_only[0].content, "Work memory");
    }
}
