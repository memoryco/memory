//! In-memory storage implementation
//!
//! Useful for testing and ephemeral usage.

use super::{Storage, StorageResult};
use crate::engram::{EngramId, Engram, Association, Config, MemoryState, Identity};
use std::collections::HashMap;

/// In-memory storage implementation (for testing)
#[derive(Debug, Default)]
pub struct MemoryStorage {
    identity: Option<Identity>,
    engrams: HashMap<EngramId, Engram>,
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
    
    fn load_identity(&mut self) -> StorageResult<Option<Identity>> {
        Ok(self.identity.clone())
    }
    
    fn save_engram(&mut self, engram: &Engram) -> StorageResult<()> {
        self.engrams.insert(engram.id, engram.clone());
        Ok(())
    }
    
    fn load_engram(&mut self, id: &EngramId) -> StorageResult<Option<Engram>> {
        Ok(self.engrams.get(id).cloned())
    }
    
    fn load_all_engrams(&mut self) -> StorageResult<Vec<Engram>> {
        Ok(self.engrams.values().cloned().collect())
    }
    
    fn load_engrams_by_state(&mut self, state: MemoryState) -> StorageResult<Vec<Engram>> {
        Ok(self.engrams.values()
            .filter(|e| e.state == state)
            .cloned()
            .collect())
    }
    
    fn load_engrams_by_tag(&mut self, tag: &str) -> StorageResult<Vec<Engram>> {
        let tag_lower = tag.to_lowercase();
        Ok(self.engrams.values()
            .filter(|e| e.tags.iter().any(|t| t.to_lowercase() == tag_lower))
            .cloned()
            .collect())
    }
    
    fn delete_engram(&mut self, id: &EngramId) -> StorageResult<bool> {
        // Remove the engram
        let existed = self.engrams.remove(id).is_some();
        
        // Remove associations from/to this engram
        self.associations.retain(|a| a.from != *id && a.to != *id);
        
        Ok(existed)
    }
    
    fn count_engrams(&mut self) -> StorageResult<usize> {
        Ok(self.engrams.len())
    }
    
    fn save_association(&mut self, assoc: &Association) -> StorageResult<()> {
        // Upsert: remove existing if present, then add
        self.associations.retain(|a| !(a.from == assoc.from && a.to == assoc.to));
        self.associations.push(assoc.clone());
        Ok(())
    }
    
    fn load_associations_from(&mut self, from: &EngramId) -> StorageResult<Vec<Association>> {
        Ok(self.associations.iter()
            .filter(|a| &a.from == from)
            .cloned()
            .collect())
    }
    
    fn load_all_associations(&mut self) -> StorageResult<Vec<Association>> {
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
    
    fn load_config(&mut self) -> StorageResult<Option<Config>> {
        Ok(self.config.clone())
    }
    
    fn save_last_decay_at(&mut self, timestamp: i64) -> StorageResult<()> {
        self.last_decay_at = Some(timestamp);
        Ok(())
    }
    
    fn load_last_decay_at(&mut self) -> StorageResult<Option<i64>> {
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
