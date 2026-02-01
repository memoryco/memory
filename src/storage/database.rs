//! Database wrapper providing common SQLite setup and utilities.

use rusqlite::Connection;
use std::path::Path;

use super::error::{StorageError, StorageResult};

/// A wrapper around SQLite connections with sensible defaults.
///
/// Provides:
/// - Automatic WAL mode for better concurrency
/// - Foreign key enforcement
/// - Common pragma settings for reliability
/// - Schema initialization helpers
///
/// # Example
///
/// ```ignore
/// let db = Database::open("mydata.db")?;
/// db.initialize_schema(r#"
///     CREATE TABLE IF NOT EXISTS items (
///         id TEXT PRIMARY KEY,
///         data BLOB NOT NULL
///     );
/// "#)?;
///
/// // Use the connection for queries
/// db.conn().execute("INSERT INTO items VALUES (?1, ?2)", params![id, data])?;
/// ```
pub struct Database {
    conn: Connection,
}

impl Database {
    /// Open or create a database file with sensible defaults.
    ///
    /// Applies standard pragmas:
    /// - `journal_mode = WAL` - Write-ahead logging for better concurrency
    /// - `foreign_keys = ON` - Enforce foreign key constraints
    /// - `synchronous = NORMAL` - Balance between safety and speed
    pub fn open(path: impl AsRef<Path>) -> StorageResult<Self> {
        let conn = Connection::open(path)?;
        Self::apply_pragmas(&conn)?;
        Ok(Self { conn })
    }

    /// Create an in-memory database (useful for testing).
    pub fn in_memory() -> StorageResult<Self> {
        let conn = Connection::open_in_memory()?;
        Self::apply_pragmas(&conn)?;
        Ok(Self { conn })
    }

    /// Open with a custom connection (for special cases like read-only).
    pub fn from_connection(conn: Connection) -> StorageResult<Self> {
        Self::apply_pragmas(&conn)?;
        Ok(Self { conn })
    }

    /// Initialize the database schema.
    ///
    /// Runs the provided SQL which should use `CREATE TABLE IF NOT EXISTS`
    /// and similar idempotent statements. Safe to call multiple times.
    pub fn initialize_schema(&self, sql: &str) -> StorageResult<()> {
        self.conn
            .execute_batch(sql)
            .map_err(|e| StorageError::Schema(e.to_string()))
    }

    /// Get a reference to the underlying connection.
    #[inline]
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    /// Get a mutable reference to the underlying connection.
    #[inline]
    pub fn conn_mut(&mut self) -> &mut Connection {
        &mut self.conn
    }

    /// Execute a function within a transaction.
    ///
    /// The transaction is committed if the function returns `Ok`,
    /// or rolled back if it returns `Err`.
    pub fn transaction<T, F>(&mut self, f: F) -> StorageResult<T>
    where
        F: FnOnce(&Connection) -> StorageResult<T>,
    {
        let tx = self.conn.transaction()?;
        let result = f(&tx)?;
        tx.commit()?;
        Ok(result)
    }

    /// Apply standard pragmas for performance and reliability.
    fn apply_pragmas(conn: &Connection) -> StorageResult<()> {
        conn.execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;
            PRAGMA synchronous = NORMAL;
            "#,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_database() {
        let db = Database::in_memory().unwrap();
        db.initialize_schema(
            r#"
            CREATE TABLE IF NOT EXISTS test (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
        "#,
        )
        .unwrap();

        db.conn()
            .execute("INSERT INTO test (name) VALUES (?1)", ["hello"])
            .unwrap();

        let name: String = db
            .conn()
            .query_row("SELECT name FROM test WHERE id = 1", [], |row| row.get(0))
            .unwrap();

        assert_eq!(name, "hello");
    }

    #[test]
    fn test_transaction_commit() {
        let mut db = Database::in_memory().unwrap();
        db.initialize_schema("CREATE TABLE IF NOT EXISTS t (x INTEGER);")
            .unwrap();

        db.transaction(|conn| {
            conn.execute("INSERT INTO t VALUES (1)", [])?;
            conn.execute("INSERT INTO t VALUES (2)", [])?;
            Ok(())
        })
        .unwrap();

        let count: i32 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM t", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_transaction_rollback() {
        let mut db = Database::in_memory().unwrap();
        db.initialize_schema("CREATE TABLE IF NOT EXISTS t (x INTEGER);")
            .unwrap();

        let result: StorageResult<()> = db.transaction(|conn| {
            conn.execute("INSERT INTO t VALUES (1)", [])?;
            Err(StorageError::Database("simulated failure".into()))
        });

        assert!(result.is_err());

        let count: i32 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM t", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0); // rolled back
    }
}
