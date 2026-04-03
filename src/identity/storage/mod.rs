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

    /// Delete all items matching a raw type string.
    /// Used for cleaning up removed types (e.g., "instruction") that are
    /// no longer in the IdentityItemType enum.
    fn delete_items_by_type_str(&mut self, type_str: &str) -> StorageResult<usize>;

    /// List items by raw type string (bypasses enum parsing).
    /// Used during migration to read rows with legacy type values.
    fn list_items_by_type_str(&mut self, type_str: &str) -> StorageResult<Vec<IdentityItemRow>>;

    /// Update all items matching old_type to new_type (raw strings).
    /// Returns the number of rows updated. Used during migration.
    fn update_item_type(&mut self, old_type: &str, new_type: &str) -> StorageResult<usize>;

    /// Update all items matching old_type to new_type AND set their category (raw strings).
    /// Returns the number of rows updated. Used during migration when type conversion
    /// also requires setting metadata (e.g., antipattern → rule with category="negative").
    fn update_item_type_with_category(
        &mut self,
        old_type: &str,
        new_type: &str,
        category: Option<&str>,
    ) -> StorageResult<usize>;

    /// Begin a database transaction
    fn begin_transaction(&mut self) -> StorageResult<()> {
        Ok(())
    }

    /// Commit the current transaction
    fn commit_transaction(&mut self) -> StorageResult<()> {
        Ok(())
    }

    /// Rollback the current transaction
    fn rollback_transaction(&mut self) -> StorageResult<()> {
        Ok(())
    }

    /// Flush any pending writes
    fn flush(&mut self) -> StorageResult<()> {
        Ok(())
    }

    /// Close the storage cleanly
    fn close(&mut self) -> StorageResult<()> {
        self.flush()
    }
}
