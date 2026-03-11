//! Diesel-backed storage implementation for Identity items
//!
//! This module provides database storage using Diesel ORM.

use super::schema::identity_items;
use super::{
    IdentityItemRow, IdentityItemType, IdentityStorage, NewIdentityItem, StorageError,
    StorageResult,
};

use diesel::connection::SimpleConnection;
use diesel::prelude::*;
use std::path::Path;

#[cfg(feature = "sqlite")]
use diesel::sqlite::SqliteConnection as DbConnection;

#[cfg(feature = "postgres")]
use diesel::pg::PgConnection as DbConnection;

/// Database-backed storage implementation for Identity items
pub struct DieselIdentityStorage {
    conn: DbConnection,
}

impl DieselIdentityStorage {
    /// Open storage at the given path (SQLite)
    #[cfg(feature = "sqlite")]
    pub fn open<P: AsRef<Path>>(path: P) -> StorageResult<Self> {
        let path_str = path.as_ref().to_string_lossy();
        let mut conn = DbConnection::establish(&path_str)
            .map_err(|e| StorageError::Database(e.to_string()))?;

        conn.batch_execute(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA foreign_keys = ON;
            PRAGMA busy_timeout = 5000;
        "#,
        )
        .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(Self { conn })
    }

    /// Open storage at the given connection URL (Postgres)
    #[cfg(feature = "postgres")]
    pub fn open(connection_url: &str) -> StorageResult<Self> {
        let conn = DbConnection::establish(connection_url)
            .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(Self { conn })
    }

    /// Create an in-memory database (for testing) - SQLite only
    #[cfg(test)]
    pub fn in_memory() -> StorageResult<Self> {
        let mut conn = DbConnection::establish(":memory:")
            .map_err(|e| StorageError::Database(e.to_string()))?;

        conn.batch_execute("PRAGMA foreign_keys = ON;")
            .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(Self { conn })
    }

    /// Create the database schema (SQLite)
    #[cfg(feature = "sqlite")]
    fn create_schema(&mut self) -> StorageResult<()> {
        self.conn
            .batch_execute(
                r#"
            CREATE TABLE IF NOT EXISTS identity_items (
                id TEXT PRIMARY KEY,
                item_type TEXT NOT NULL,
                content TEXT NOT NULL,
                secondary TEXT,
                tertiary TEXT,
                category TEXT,
                created_at INTEGER NOT NULL
            );
            
            -- Index for type lookups (most common query pattern)
            CREATE INDEX IF NOT EXISTS idx_identity_items_type ON identity_items(item_type);
        "#,
            )
            .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(())
    }

    /// Generate a short ID (4 hex chars from UUID)
    fn generate_short_id() -> String {
        let uuid = uuid::Uuid::new_v4();
        let hex = uuid.simple().to_string();
        hex.chars().take(8).collect() // 8 chars gives us plenty of uniqueness for identity items
    }
}

impl IdentityStorage for DieselIdentityStorage {
    #[cfg(feature = "sqlite")]
    fn initialize(&mut self) -> StorageResult<()> {
        self.create_schema()
    }

    #[cfg(feature = "postgres")]
    fn initialize(&mut self) -> StorageResult<()> {
        // Postgres schema is managed via migrations
        Ok(())
    }

    fn add_item(
        &mut self,
        item_type: IdentityItemType,
        content: &str,
        secondary: Option<&str>,
        tertiary: Option<&str>,
        category: Option<&str>,
    ) -> StorageResult<String> {
        let id = Self::generate_short_id();
        let type_str = item_type.to_string();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        diesel::insert_into(identity_items::table)
            .values(NewIdentityItem {
                id: &id,
                item_type: &type_str,
                content,
                secondary,
                tertiary,
                category,
                created_at: now,
            })
            .execute(&mut self.conn)?;

        Ok(id)
    }

    fn remove_item(&mut self, id: &str) -> StorageResult<bool> {
        let deleted = diesel::delete(identity_items::table.filter(identity_items::id.eq(id)))
            .execute(&mut self.conn)?;

        Ok(deleted > 0)
    }

    fn get_item(&mut self, id: &str) -> StorageResult<Option<IdentityItemRow>> {
        let result = identity_items::table
            .filter(identity_items::id.eq(id))
            .select(IdentityItemRow::as_select())
            .first(&mut self.conn)
            .optional()?;

        Ok(result)
    }

    fn list_items(
        &mut self,
        item_type: Option<IdentityItemType>,
    ) -> StorageResult<Vec<IdentityItemRow>> {
        let rows = match item_type {
            Some(t) => {
                let type_str = t.to_string();
                identity_items::table
                    .filter(identity_items::item_type.eq(type_str))
                    .order(identity_items::created_at.asc())
                    .select(IdentityItemRow::as_select())
                    .load(&mut self.conn)?
            }
            None => identity_items::table
                .order(identity_items::created_at.asc())
                .select(IdentityItemRow::as_select())
                .load(&mut self.conn)?,
        };

        Ok(rows)
    }

    #[cfg(feature = "sqlite")]
    fn flush(&mut self) -> StorageResult<()> {
        self.conn.batch_execute("PRAGMA wal_checkpoint(PASSIVE);")?;
        Ok(())
    }

    #[cfg(feature = "postgres")]
    fn flush(&mut self) -> StorageResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_get_item() {
        let mut storage = DieselIdentityStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let id = storage
            .add_item(
                IdentityItemType::Antipattern,
                "using tokio block_on in async context",
                Some("use .await instead"),
                Some("blocks the runtime"),
                None,
            )
            .unwrap();

        let item = storage.get_item(&id).unwrap();
        assert!(item.is_some());
        let item = item.unwrap();
        assert_eq!(item.content, "using tokio block_on in async context");
        assert_eq!(item.secondary.as_deref(), Some("use .await instead"));
        assert_eq!(item.tertiary.as_deref(), Some("blocks the runtime"));
    }

    #[test]
    fn list_items_by_type() {
        let mut storage = DieselIdentityStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        storage
            .add_item(IdentityItemType::Expertise, "Rust", None, None, None)
            .unwrap();
        storage
            .add_item(IdentityItemType::Expertise, "Python", None, None, None)
            .unwrap();
        storage
            .add_item(IdentityItemType::Trait, "curious", None, None, None)
            .unwrap();

        let expertise = storage
            .list_items(Some(IdentityItemType::Expertise))
            .unwrap();
        assert_eq!(expertise.len(), 2);

        let traits = storage.list_items(Some(IdentityItemType::Trait)).unwrap();
        assert_eq!(traits.len(), 1);

        let all = storage.list_items(None).unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn remove_item() {
        let mut storage = DieselIdentityStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let id = storage
            .add_item(IdentityItemType::Expertise, "Rust", None, None, None)
            .unwrap();

        let removed = storage.remove_item(&id).unwrap();
        assert!(removed);

        let item = storage.get_item(&id).unwrap();
        assert!(item.is_none());

        // Second remove returns false
        let removed_again = storage.remove_item(&id).unwrap();
        assert!(!removed_again);
    }

    #[test]
    fn persona_fields() {
        let mut storage = DieselIdentityStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        storage
            .add_item(IdentityItemType::PersonaName, "Porter", None, None, None)
            .unwrap();
        storage
            .add_item(
                IdentityItemType::PersonaDescription,
                "A software engineer who...",
                None,
                None,
                None,
            )
            .unwrap();

        let name = storage
            .list_items(Some(IdentityItemType::PersonaName))
            .unwrap();
        assert_eq!(name.len(), 1);
        assert_eq!(name[0].content, "Porter");

        let desc = storage
            .list_items(Some(IdentityItemType::PersonaDescription))
            .unwrap();
        assert_eq!(desc.len(), 1);
    }
}
