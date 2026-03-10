//! Storage trait for identity items
//!
//! This module defines the contract for storing and retrieving identity items.
//! The storage is flat - all identity types share one table with a type discriminator.

mod diesel;
pub mod models;
pub mod schema;

pub use diesel::DieselIdentityStorage;
pub use models::{IdentityItemRow, IdentityItemType, NewIdentityItem};

// Re-export unified storage types from the foundation
pub use crate::storage::{StorageError, StorageResult};

/// The storage contract for identity items
///
/// Implementations of this trait provide persistence for identity data.
/// All operations are synchronous. The storage layer is dumb - it just
/// stores/retrieves rows. The semantic meaning of fields is handled
/// by the higher-level IdentityStore.
pub trait IdentityStorage: Send {
    /// Initialize storage (create tables, etc.)
    fn initialize(&mut self) -> StorageResult<()>;

    /// Add a new identity item, returns the generated ID
    fn add_item(
        &mut self,
        item_type: IdentityItemType,
        content: &str,
        secondary: Option<&str>,
        tertiary: Option<&str>,
        category: Option<&str>,
    ) -> StorageResult<String>;

    /// Remove an identity item by ID
    fn remove_item(&mut self, id: &str) -> StorageResult<bool>;

    /// Get a single item by ID
    fn get_item(&mut self, id: &str) -> StorageResult<Option<IdentityItemRow>>;

    /// List all items, optionally filtered by type
    fn list_items(
        &mut self,
        item_type: Option<IdentityItemType>,
    ) -> StorageResult<Vec<IdentityItemRow>>;

    /// Flush any pending writes
    fn flush(&mut self) -> StorageResult<()> {
        Ok(())
    }

    /// Close the storage cleanly
    fn close(&mut self) -> StorageResult<()> {
        self.flush()
    }
}
