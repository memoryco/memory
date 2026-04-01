//! Async persistence for recall operations
//!
//! This module handles fire-and-forget persistence of recall side effects
//! (energy updates and Hebbian learning) without blocking the main thread.
//!
//! The in-memory Substrate state is always authoritative. The background
//! worker just ensures durability to disk. If the process crashes before
//! the worker finishes, we lose some association strengthening - acceptable
//! since Hebbian learning is cumulative over many recalls.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// How long to wait for the worker to drain its queue on shutdown
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

use super::storage::{MemoryStorage, Storage};
use super::{Association, MemoryId, MemoryState};

/// Work item for the persistence worker
/// Contains all owned data needed to persist recall effects
#[derive(Debug)]
pub struct PersistenceWork {
    /// Energy updates: (id, new_energy, new_state)
    pub energy_updates: Vec<(MemoryId, f64, MemoryState)>,
    /// Associations that were created or strengthened
    pub associations: Vec<Association>,
}

impl PersistenceWork {
    /// Create a new empty work item
    pub fn new() -> Self {
        Self {
            energy_updates: Vec::new(),
            associations: Vec::new(),
        }
    }

    /// Check if there's actually work to do
    pub fn is_empty(&self) -> bool {
        self.energy_updates.is_empty() && self.associations.is_empty()
    }
}

impl Default for PersistenceWork {
    fn default() -> Self {
        Self::new()
    }
}

/// Background worker for async persistence
///
/// Spawns a dedicated thread with its own database connection.
/// Work is sent via channel and processed sequentially.
/// Graceful shutdown on Drop (waits for queue to drain).
pub struct PersistenceWorker {
    sender: Option<Sender<PersistenceWork>>,
    handle: Option<JoinHandle<()>>,
}

impl PersistenceWorker {
    /// Create a new persistence worker with its own DB connection
    pub fn new(db_path: &Path) -> Self {
        let (sender, receiver) = mpsc::channel();
        let path = db_path.to_path_buf();

        let handle = thread::spawn(move || {
            Self::worker_loop(path, receiver);
        });

        Self {
            sender: Some(sender),
            handle: Some(handle),
        }
    }

    /// The worker thread's main loop
    fn worker_loop(db_path: PathBuf, receiver: Receiver<PersistenceWork>) {
        // Open our own storage connection
        let mut storage = match MemoryStorage::open(&db_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[persistence] Failed to open storage: {:?}", e);
                return;
            }
        };

        eprintln!("[persistence] Worker started");

        // Process work until channel closes
        while let Ok(work) = receiver.recv() {
            if work.is_empty() {
                continue;
            }

            eprintln!(
                "[persistence] Processing: {} energy updates, {} associations",
                work.energy_updates.len(),
                work.associations.len()
            );

            // Persist energy updates
            if !work.energy_updates.is_empty() {
                let updates: Vec<_> = work
                    .energy_updates
                    .iter()
                    .map(|(id, energy, state)| (id, *energy, *state))
                    .collect();

                if let Err(e) = storage.save_memory_energies(&updates) {
                    eprintln!("[persistence] Failed to save energies: {:?}", e);
                }
            }

            // Persist associations
            if !work.associations.is_empty() {
                let assoc_refs: Vec<_> = work.associations.iter().collect();
                if let Err(e) = storage.save_associations(&assoc_refs) {
                    eprintln!("[persistence] Failed to save associations: {:?}", e);
                }
            }
        }

        eprintln!("[persistence] Worker shutting down");
    }

    /// Send work to the background thread (fire and forget)
    /// Returns immediately - does not wait for persistence
    pub fn send(&self, work: PersistenceWork) {
        if work.is_empty() {
            return;
        }

        // Ignore send errors - if the worker is dead, we just lose this batch
        // The in-memory state is still correct
        if let Some(ref sender) = self.sender {
            if let Err(e) = sender.send(work) {
                eprintln!("[persistence] Failed to send work: {:?}", e);
            }
        }
    }
}

impl Drop for PersistenceWorker {
    fn drop(&mut self) {
        // Drop sender first to close the channel. The worker's recv() will
        // return Err once all queued items are consumed, causing it to exit.
        self.sender.take();

        if let Some(handle) = self.handle.take() {
            // Join with a timeout so we don't hang forever if the worker is stuck.
            // We use a watchdog thread since JoinHandle has no native timeout.
            let (done_tx, done_rx) = mpsc::channel();
            let join_thread = thread::spawn(move || {
                let result = handle.join();
                let _ = done_tx.send(result);
            });

            match done_rx.recv_timeout(SHUTDOWN_TIMEOUT) {
                Ok(Ok(())) => eprintln!("[persistence] Worker shut down cleanly"),
                Ok(Err(_)) => eprintln!("[persistence] Worker thread panicked during shutdown"),
                Err(_) => eprintln!(
                    "[persistence] Worker did not shut down within {:?}, detaching",
                    SHUTDOWN_TIMEOUT
                ),
            }
            drop(join_thread);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_core::storage::Storage;
    use crate::memory_core::Association;
    use tempfile::tempdir;
    use uuid::Uuid;

    fn setup_worker() -> (PersistenceWorker, PathBuf, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let mut storage = MemoryStorage::open(&db_path).unwrap();
        storage.initialize().unwrap();
        drop(storage);

        let worker = PersistenceWorker::new(&db_path);
        (worker, db_path, dir)
    }

    #[test]
    fn persistence_work_empty_check() {
        let work = PersistenceWork::new();
        assert!(work.is_empty());

        let mut work = PersistenceWork::new();
        work.energy_updates
            .push((Uuid::new_v4(), 0.5, MemoryState::Active));
        assert!(!work.is_empty());
    }

    #[test]
    fn persistence_work_default() {
        let work = PersistenceWork::default();
        assert!(work.is_empty());
    }

    #[test]
    fn worker_starts_and_stops_cleanly() {
        let (worker, _db_path, _dir) = setup_worker();

        // Send some empty work (should be no-op)
        worker.send(PersistenceWork::new());

        // Drop worker — should shut down cleanly via the new flush path
        drop(worker);
    }

    /// Helper: initialize DB and insert test memories, then create worker.
    /// This avoids SQLite locking races between the worker thread opening
    /// its connection and the test inserting seed data.
    fn setup_worker_with_memories(ids: &[Uuid]) -> (PersistenceWorker, PathBuf, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let mut storage = MemoryStorage::open(&db_path).unwrap();
        storage.initialize().unwrap();
        for &id in ids {
            let mut m = crate::memory_core::Memory::new(format!("memory {}", id));
            m.id = id;
            storage.save_memory(&m).unwrap();
        }
        drop(storage);

        let worker = PersistenceWorker::new(&db_path);
        (worker, db_path, dir)
    }

    #[test]
    fn worker_flushes_energy_updates_on_drop() {
        let id = Uuid::new_v4();
        let (worker, db_path, _dir) = setup_worker_with_memories(&[id]);

        // Queue an energy update
        let mut work = PersistenceWork::new();
        work.energy_updates
            .push((id, 0.75, MemoryState::Active));
        worker.send(work);

        // Drop should flush the queue before returning
        drop(worker);

        // Verify the energy was actually persisted
        let mut storage = MemoryStorage::open(&db_path).unwrap();
        let loaded = storage.load_memory(&id).unwrap().expect("memory should exist");
        assert!(
            (loaded.energy - 0.75).abs() < 1e-6,
            "energy should be 0.75, got {}",
            loaded.energy
        );
    }

    #[test]
    fn worker_flushes_associations_on_drop() {
        let id_a = Uuid::new_v4();
        let id_b = Uuid::new_v4();
        let (worker, db_path, _dir) = setup_worker_with_memories(&[id_a, id_b]);

        // Queue an association
        let mut work = PersistenceWork::new();
        work.associations
            .push(Association::with_weight(id_a, id_b, 0.8));
        worker.send(work);

        // Drop should flush
        drop(worker);

        // Verify association was persisted
        let mut storage = MemoryStorage::open(&db_path).unwrap();
        let assocs = storage.load_associations_from(&id_a).unwrap();
        assert_eq!(assocs.len(), 1);
        assert_eq!(assocs[0].to, id_b);
        assert!((assocs[0].weight - 0.8).abs() < 1e-6);
    }

    #[test]
    fn worker_flushes_multiple_queued_items_on_drop() {
        let ids: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();
        let (worker, db_path, _dir) = setup_worker_with_memories(&ids);

        // Queue multiple work items
        for &id in &ids {
            let mut work = PersistenceWork::new();
            work.energy_updates
                .push((id, 0.42, MemoryState::Active));
            worker.send(work);
        }

        // Drop should flush all of them
        drop(worker);

        // Verify all were persisted
        let mut storage = MemoryStorage::open(&db_path).unwrap();
        for &id in &ids {
            let loaded = storage.load_memory(&id).unwrap().expect("memory should exist");
            assert!(
                (loaded.energy - 0.42).abs() < 1e-6,
                "memory {} energy should be 0.42, got {}",
                id,
                loaded.energy
            );
        }
    }

    #[test]
    fn send_after_drop_is_harmless() {
        let (worker, _db_path, _dir) = setup_worker();

        // Drop the worker
        drop(worker);

        // Can't send after drop (ownership moved), but verify the pattern
        // where sender is None doesn't panic
        // This is implicitly tested — the worker's Drop takes the sender,
        // so any cloned sender would get a SendError, which is handled.
    }

    #[test]
    fn worker_handles_empty_work_items() {
        let (worker, _db_path, _dir) = setup_worker();

        // Send a bunch of empty work — should be filtered out
        for _ in 0..10 {
            worker.send(PersistenceWork::new());
        }

        drop(worker);
    }

    #[test]
    fn worker_handles_mixed_work() {
        let id_a = Uuid::new_v4();
        let id_b = Uuid::new_v4();
        let (worker, db_path, _dir) = setup_worker_with_memories(&[id_a, id_b]);

        // Single work item with both energy updates and associations
        let mut work = PersistenceWork::new();
        work.energy_updates
            .push((id_a, 0.9, MemoryState::Active));
        work.energy_updates
            .push((id_b, 0.3, MemoryState::Dormant));
        work.associations
            .push(Association::with_weight(id_a, id_b, 0.6));
        worker.send(work);

        drop(worker);

        let mut storage = MemoryStorage::open(&db_path).unwrap();
        let a = storage.load_memory(&id_a).unwrap().unwrap();
        let b = storage.load_memory(&id_b).unwrap().unwrap();
        assert!((a.energy - 0.9).abs() < 1e-6);
        assert!((b.energy - 0.3).abs() < 1e-6);

        let assocs = storage.load_associations_from(&id_a).unwrap();
        assert_eq!(assocs.len(), 1);
    }
}
