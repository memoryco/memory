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

    /// Get mutable reference to the underlying connection (for testing)
    #[cfg(test)]
    pub fn conn_mut(&mut self) -> &mut DbConnection {
        &mut self.conn
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

    fn delete_items_by_type_str(&mut self, type_str: &str) -> StorageResult<usize> {
        let deleted =
            diesel::delete(identity_items::table.filter(identity_items::item_type.eq(type_str)))
                .execute(&mut self.conn)?;
        Ok(deleted)
    }

    fn list_items_by_type_str(&mut self, type_str: &str) -> StorageResult<Vec<IdentityItemRow>> {
        let rows = identity_items::table
            .filter(identity_items::item_type.eq(type_str))
            .order(identity_items::created_at.asc())
            .select(IdentityItemRow::as_select())
            .load(&mut self.conn)?;
        Ok(rows)
    }

    fn update_item_type(&mut self, old_type: &str, new_type: &str) -> StorageResult<usize> {
        let updated = diesel::update(
            identity_items::table.filter(identity_items::item_type.eq(old_type)),
        )
        .set(identity_items::item_type.eq(new_type))
        .execute(&mut self.conn)?;
        Ok(updated)
    }

    fn update_item_type_with_category(
        &mut self,
        old_type: &str,
        new_type: &str,
        category: Option<&str>,
    ) -> StorageResult<usize> {
        let updated = diesel::update(
            identity_items::table.filter(identity_items::item_type.eq(old_type)),
        )
        .set((
            identity_items::item_type.eq(new_type),
            identity_items::category.eq(category),
        ))
        .execute(&mut self.conn)?;
        Ok(updated)
    }

    fn begin_transaction(&mut self) -> StorageResult<()> {
        self.conn
            .batch_execute("BEGIN IMMEDIATE")
            .map_err(|e| StorageError::Database(format!("Failed to begin transaction: {}", e)))?;
        Ok(())
    }

    fn commit_transaction(&mut self) -> StorageResult<()> {
        self.conn
            .batch_execute("COMMIT")
            .map_err(|e| StorageError::Database(format!("Failed to commit transaction: {}", e)))?;
        Ok(())
    }

    fn rollback_transaction(&mut self) -> StorageResult<()> {
        self.conn
            .batch_execute("ROLLBACK")
            .map_err(|e| StorageError::Database(format!("Failed to rollback transaction: {}", e)))?;
        Ok(())
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
                IdentityItemType::Rule,
                "Don't use tokio block_on in async context",
                Some("use .await instead"),
                Some("blocks the runtime"),
                None,
            )
            .unwrap();

        let item = storage.get_item(&id).unwrap();
        assert!(item.is_some());
        let item = item.unwrap();
        assert_eq!(item.content, "Don't use tokio block_on in async context");
        assert_eq!(item.secondary.as_deref(), Some("use .await instead"));
        assert_eq!(item.tertiary.as_deref(), Some("blocks the runtime"));
    }

    #[test]
    fn list_items_by_type() {
        let mut storage = DieselIdentityStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        storage
            .add_item(IdentityItemType::Rule, "Always run tests", None, None, None)
            .unwrap();
        storage
            .add_item(IdentityItemType::Rule, "Don't skip linting", None, None, None)
            .unwrap();
        storage
            .add_item(IdentityItemType::Value, "Quality matters", None, None, None)
            .unwrap();

        let rules = storage
            .list_items(Some(IdentityItemType::Rule))
            .unwrap();
        assert_eq!(rules.len(), 2);

        let values = storage.list_items(Some(IdentityItemType::Value)).unwrap();
        assert_eq!(values.len(), 1);

        let all = storage.list_items(None).unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn remove_item() {
        let mut storage = DieselIdentityStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let id = storage
            .add_item(IdentityItemType::Rule, "Always run tests", None, None, None)
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

    #[test]
    fn list_and_update_by_raw_type_str() {
        let mut storage = DieselIdentityStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        // Insert some rules
        storage
            .add_item(IdentityItemType::Rule, "Always test", None, None, None)
            .unwrap();

        // Verify list_items_by_type_str works
        let rules = storage.list_items_by_type_str("rule").unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].content, "Always test");

        // Empty for non-existent type
        let empty = storage.list_items_by_type_str("nonexistent").unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn update_item_type_raw() {
        let mut storage = DieselIdentityStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        // Manually insert with a raw type string (simulating legacy data)
        let id = "test1234".to_string();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        diesel::insert_into(identity_items::table)
            .values(NewIdentityItem {
                id: &id,
                item_type: "directive",
                content: "Use full cargo path",
                secondary: None,
                tertiary: None,
                category: None,
                created_at: now,
            })
            .execute(storage.conn_mut())
            .unwrap();

        // Update directive -> rule
        let updated = storage.update_item_type("directive", "rule").unwrap();
        assert_eq!(updated, 1);

        // Verify it's now a rule
        let rules = storage.list_items_by_type_str("rule").unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].content, "Use full cargo path");

        // Old type should be empty
        let directives = storage.list_items_by_type_str("directive").unwrap();
        assert!(directives.is_empty());
    }
}
