//! In-memory storage implementation
//!
//! Useful for testing and ephemeral usage.

use super::{Storage, StorageResult};
use crate::memory_core::{Association, Config, Memory, MemoryId, Identity, MemoryState};
use std::collections::HashMap;

/// In-memory storage implementation (for testing)
#[derive(Debug, Default)]
pub struct MemoryStorage {
    identity: Option<Identity>,
    memories: HashMap<MemoryId, Memory>,
    associations: Vec<Association>,
    config: Option<Config>,
    last_decay_at: Option<i64>,
    metadata: HashMap<String, String>,
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

    fn save_memory(&mut self, mem: &Memory) -> StorageResult<()> {
        self.memories.insert(mem.id, mem.clone());
        Ok(())
    }

    fn load_memory(&mut self, id: &MemoryId) -> StorageResult<Option<Memory>> {
        Ok(self.memories.get(id).cloned())
    }

    fn load_all_memories(&mut self) -> StorageResult<Vec<Memory>> {
        Ok(self.memories.values().cloned().collect())
    }

    fn load_memories_by_state(&mut self, state: MemoryState) -> StorageResult<Vec<Memory>> {
        Ok(self
            .memories
            .values()
            .filter(|e| e.state == state)
            .cloned()
            .collect())
    }

    fn load_memories_by_tag(&mut self, tag: &str) -> StorageResult<Vec<Memory>> {
        let tag_lower = tag.to_lowercase();
        Ok(self
            .memories
            .values()
            .filter(|e| e.tags.iter().any(|t| t.to_lowercase() == tag_lower))
            .cloned()
            .collect())
    }

    fn delete_memory(&mut self, id: &MemoryId) -> StorageResult<bool> {
        // Remove the memory
        let existed = self.memories.remove(id).is_some();

        // Remove associations from/to this memory
        self.associations.retain(|a| a.from != *id && a.to != *id);

        Ok(existed)
    }

    fn count_memories(&mut self) -> StorageResult<usize> {
        Ok(self.memories.len())
    }

    fn save_association(&mut self, assoc: &Association) -> StorageResult<()> {
        // Upsert: remove existing if present, then add
        self.associations
            .retain(|a| !(a.from == assoc.from && a.to == assoc.to));
        self.associations.push(assoc.clone());
        Ok(())
    }

    fn load_associations_from(&mut self, from: &MemoryId) -> StorageResult<Vec<Association>> {
        Ok(self
            .associations
            .iter()
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

    fn prune_orphan_associations(&mut self) -> StorageResult<usize> {
        let before = self.associations.len();
        self.associations.retain(|a| {
            self.memories.contains_key(&a.from)
                && self.memories.contains_key(&a.to)
        });
        Ok(before - self.associations.len())
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

    fn get_metadata(&mut self, key: &str) -> StorageResult<Option<String>> {
        Ok(self.metadata.get(key).cloned())
    }

    fn set_metadata(&mut self, key: &str, value: &str) -> StorageResult<()> {
        self.metadata.insert(key.to_string(), value.to_string());
        Ok(())
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
    fn memory_storage_memories() {
        let mut storage = MemoryStorage::new();
        storage.initialize().unwrap();

        let mem = Memory::new("Test memory");
        let id = mem.id;

        storage.save_memory(&mem).unwrap();

        let loaded = storage.load_memory(&id).unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().content, "Test memory");
    }

    #[test]
    fn memory_storage_associations() {
        let mut storage = MemoryStorage::new();
        storage.initialize().unwrap();

        let e1 = Memory::new("Memory 1");
        let e2 = Memory::new("Memory 2");

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

        let mut mem = Memory::new("Original");
        let id = mem.id;

        storage.save_memory(&mem).unwrap();

        mem.content = "Updated".to_string();
        storage.save_memory(&mem).unwrap();

        let loaded = storage.load_memory(&id).unwrap().unwrap();
        assert_eq!(loaded.content, "Updated");

        // Should still be just one memory
        assert_eq!(storage.load_all_memories().unwrap().len(), 1);
    }

    #[test]
    fn memory_storage_load_by_state() {
        let mut storage = MemoryStorage::new();
        storage.initialize().unwrap();

        let mut active = Memory::new("Active memory");
        active.state = MemoryState::Active;

        let mut archived = Memory::new("Archived memory");
        archived.state = MemoryState::Archived;
        archived.energy = 0.01;

        storage.save_memory(&active).unwrap();
        storage.save_memory(&archived).unwrap();

        let active_only = storage.load_memories_by_state(MemoryState::Active).unwrap();
        assert_eq!(active_only.len(), 1);
        assert_eq!(active_only[0].content, "Active memory");
    }

    #[test]
    fn memory_storage_load_by_tag() {
        let mut storage = MemoryStorage::new();
        storage.initialize().unwrap();

        let mut work = Memory::new("Work memory");
        work.tags = vec!["work".to_string(), "rust".to_string()];

        let mut personal = Memory::new("Personal memory");
        personal.tags = vec!["personal".to_string()];

        storage.save_memory(&work).unwrap();
        storage.save_memory(&personal).unwrap();

        let work_only = storage.load_memories_by_tag("work").unwrap();
        assert_eq!(work_only.len(), 1);
        assert_eq!(work_only[0].content, "Work memory");
    }
}
