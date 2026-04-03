//! Diesel-backed storage implementation for Memories
//!
//! This module provides database storage using Diesel ORM.
//! The actual database backend (SQLite, Postgres, MySQL) is selected
//! via feature flags at compile time.

use super::models::*;
use super::schema::{associations, memories, identity, metadata};
use super::{SimilarityResult, Storage, StorageError, StorageResult};
use crate::memory_core::session::SessionContext;
use crate::memory_core::{Association, Config, Memory, MemoryId, Identity, MemoryState};

use diesel::connection::SimpleConnection;
use diesel::prelude::*;
use std::path::Path;

// Conditional imports based on feature flags
#[cfg(feature = "sqlite")]
use diesel::sqlite::SqliteConnection as DbConnection;

#[cfg(feature = "sqlite")]
use super::VectorSearch;

#[cfg(feature = "postgres")]
use diesel::pg::PgConnection as DbConnection;

/// Register sqlite-vec as an auto-extension (idempotent, process-global).
///
/// Called from both `open()` and `in_memory()` so the extension is available
/// whether the caller is the real binary (which also registers in `main()`)
/// or a test harness.
#[cfg(feature = "sqlite")]
fn ensure_sqlite_vec_registered() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        unsafe {
            libsqlite3_sys::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
    });
}

/// Database-backed storage implementation for Memories
///
/// The underlying database is selected at compile time via feature flags:
/// - `sqlite` (default) - Uses SQLite, suitable for local/embedded use
/// - `postgres` - Uses PostgreSQL, suitable for server/SaaS deployments
pub struct MemoryStorage {
    conn: DbConnection,
    db_path: Option<std::path::PathBuf>,
}

impl MemoryStorage {
    /// Open storage at the given connection string/path
    ///
    /// For SQLite: Pass a file path (e.g., "/path/to/brain.db")
    /// For Postgres: Pass a connection URL (e.g., "postgres://user:pass@host/db")
    #[cfg(feature = "sqlite")]
    pub fn open<P: AsRef<Path>>(path: P) -> StorageResult<Self> {
        ensure_sqlite_vec_registered();
        let path_str = path.as_ref().to_string_lossy();
        let mut conn = DbConnection::establish(&path_str)
            .map_err(|e| StorageError::Database(e.to_string()))?;

        let db_path = Some(std::path::PathBuf::from(path.as_ref()));

        // Apply SQLite pragmas for performance
        conn.batch_execute(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA foreign_keys = ON;
            PRAGMA busy_timeout = 5000;
        "#,
        )
        .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(Self { conn, db_path })
    }

    /// Open storage at the given connection URL
    #[cfg(feature = "postgres")]
    pub fn open(connection_url: &str) -> StorageResult<Self> {
        let conn = DbConnection::establish(connection_url)
            .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(Self { conn, db_path: None })
    }

    /// Create an in-memory database (for testing) - SQLite only
    #[cfg(test)]
    pub fn in_memory() -> StorageResult<Self> {
        ensure_sqlite_vec_registered();
        let mut conn = DbConnection::establish(":memory:")
            .map_err(|e| StorageError::Database(e.to_string()))?;

        conn.batch_execute("PRAGMA foreign_keys = ON;")
            .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(Self { conn, db_path: None })
    }

    /// Get mutable reference to the underlying connection
    #[cfg(test)]
    pub fn connection(&mut self) -> &mut DbConnection {
        &mut self.conn
    }

    /// Create the database schema (SQLite)
    #[cfg(feature = "sqlite")]
    fn create_schema(&mut self) -> StorageResult<()> {
        self.conn
            .batch_execute(
                r#"
            -- Identity table (single row, JSON blob)
            CREATE TABLE IF NOT EXISTS identity (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                data TEXT NOT NULL
            );
            
            -- Config table (single row, JSON blob)
            CREATE TABLE IF NOT EXISTS config (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                data TEXT NOT NULL
            );
            
            -- Metadata table (key-value for misc settings)
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            
            -- Memories table
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                energy REAL NOT NULL,
                state TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at INTEGER NOT NULL,
                last_accessed INTEGER NOT NULL,
                access_count INTEGER NOT NULL,
                tags TEXT NOT NULL,
                embedding BLOB
            );
            
            -- Indices for common queries
            CREATE INDEX IF NOT EXISTS idx_memories_state ON memories(state);
            CREATE INDEX IF NOT EXISTS idx_memories_energy ON memories(energy);
            
            -- Associations table
            CREATE TABLE IF NOT EXISTS associations (
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                weight REAL NOT NULL,
                created_at INTEGER NOT NULL,
                last_activated INTEGER NOT NULL,
                co_activation_count INTEGER NOT NULL,
                PRIMARY KEY (from_id, to_id)
            );
            
            -- Index for association lookups
            CREATE INDEX IF NOT EXISTS idx_associations_from ON associations(from_id);

            -- Access log: records search→recall cycles for training data extraction
            CREATE TABLE IF NOT EXISTS access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                query_text TEXT NOT NULL,
                result_ids TEXT NOT NULL,
                recalled_ids TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_access_log_timestamp ON access_log(timestamp);

            -- Memory enrichment embeddings (multi-vector support)
            CREATE TABLE IF NOT EXISTS memory_enrichments (
                memory_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                source TEXT NOT NULL DEFAULT 'llm',
                created_at INTEGER NOT NULL,
                PRIMARY KEY (memory_id, seq),
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_enrichments_memory ON memory_enrichments(memory_id);
        "#,
            )
            .map_err(|e| StorageError::Database(e.to_string()))?;

        // FTS5 virtual table for BM25 keyword search
        self.conn
            .batch_execute(
                r#"
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                content,
                memory_id UNINDEXED,
                tokenize='porter unicode61'
            );
        "#,
            )
            .map_err(|e| StorageError::Database(e.to_string()))?;

        // vec0 virtual tables for SIMD-accelerated cosine similarity (sqlite-vec)
        self.conn
            .batch_execute(
                r#"
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
                memory_id TEXT,
                embedding float[2048] distance_metric=cosine
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS enrichment_vec USING vec0(
                memory_id TEXT,
                embedding float[2048] distance_metric=cosine
            );
        "#,
            )
            .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(())
    }

    /// Migrate existing databases: rename `engrams` table to `memories`.
    /// Also renames `engram_fts` to `memory_fts` and the FTS `engram_id` column.
    #[cfg(feature = "sqlite")]
    fn migrate_engrams_to_memories(&mut self) -> StorageResult<()> {
        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }

        let result: Result<CountResult, _> = diesel::sql_query(
            "SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='table' AND name='engrams'",
        )
        .get_result(&mut self.conn);

        let has_old_table = match result {
            Ok(r) => r.cnt > 0,
            Err(_) => false,
        };

        if has_old_table {
            // Back up the database before any destructive migration operations.
            if let Some(ref path) = self.db_path {
                let backup_path = path.with_extension("db.pre-migration");
                match std::fs::copy(path, &backup_path) {
                    Ok(_) => eprintln!("[migration] Backup created at {}", backup_path.display()),
                    Err(e) => eprintln!("[migration] WARNING: Could not back up database: {}. Proceeding anyway.", e),
                }
            } else {
                eprintln!("[migration] WARNING: About to migrate engrams → memories. Back up your database NOW if you haven't.");
            }

            // Wrap all destructive operations in a transaction so a crash
            // mid-migration can't leave the database in a half-migrated state.
            self.conn
                .batch_execute("BEGIN IMMEDIATE")
                .map_err(|e| StorageError::Database(
                    format!("Failed to start migration transaction: {}", e)
                ))?;

            // Run the migration body; on error, roll back.
            let migration_result = self.run_engram_migration_body();
            match migration_result {
                Ok(()) => {
                    self.conn
                        .batch_execute("COMMIT")
                        .map_err(|e| StorageError::Database(
                            format!("Failed to commit migration: {}", e)
                        ))?;
                }
                Err(e) => {
                    let _ = self.conn.batch_execute("ROLLBACK");
                    return Err(e);
                }
            }
        }

        Ok(())
    }

    /// Inner migration body, called within a transaction by `migrate_engrams_to_memories`.
    #[cfg(feature = "sqlite")]
    fn run_engram_migration_body(&mut self) -> StorageResult<()> {
        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }

        // Check if 'memories' table already exists
        let has_new_table: Result<CountResult, _> = diesel::sql_query(
                "SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='table' AND name='memories'",
            )
            .get_result(&mut self.conn);
            let memories_exists = matches!(has_new_table, Ok(r) if r.cnt > 0);

            if memories_exists {
                // Both tables exist. Count rows in each to decide which has real data.
                let engram_count: i32 = diesel::sql_query(
                    "SELECT COUNT(*) as cnt FROM engrams",
                )
                .get_result::<CountResult>(&mut self.conn)
                .map(|r| r.cnt)
                .map_err(|e| StorageError::Database(
                    format!("Migration aborted: cannot count engrams rows: {}", e)
                ))?;

                let memory_count: i32 = diesel::sql_query(
                    "SELECT COUNT(*) as cnt FROM memories",
                )
                .get_result::<CountResult>(&mut self.conn)
                .map(|r| r.cnt)
                .map_err(|e| StorageError::Database(
                    format!("Migration aborted: cannot count memories rows: {}", e)
                ))?;

                if memory_count > 0 && engram_count == 0 {
                    // Migration already completed — memories has the data,
                    // engrams is a ghost (e.g. from Syncthing sync). Drop the ghost.
                    self.conn
                        .batch_execute("DROP TABLE engrams")
                        .map_err(|e| StorageError::Database(e.to_string()))?;
                    eprintln!(
                        "[migration] Dropped stale 'engrams' table ({} rows in 'memories' already)",
                        memory_count
                    );
                    // Clean up any old enrichments/FTS/indexes that came along for the ride
                    self.conn
                        .batch_execute(
                            r#"
                            DROP TABLE IF EXISTS engram_enrichments;
                            DROP TABLE IF EXISTS engram_fts;
                            DROP INDEX IF EXISTS idx_engrams_state;
                            DROP INDEX IF EXISTS idx_engrams_energy;
                            DROP INDEX IF EXISTS idx_enrichments_engram;
                        "#,
                        )
                        .map_err(|e| StorageError::Database(e.to_string()))?;
                    return Ok(());
                } else if memory_count > 0 && engram_count > 0 {
                    // Both have data — this shouldn't happen. Refuse to migrate
                    // rather than risk data loss.
                    return Err(StorageError::Database(format!(
                        "MIGRATION CONFLICT: both 'engrams' ({} rows) and 'memories' ({} rows) contain data. \
                         Manual intervention required — inspect the database and drop the stale table.",
                        engram_count, memory_count
                    )));
                } else {
                    // engrams has data (or both empty) — drop memories and rename.
                    self.conn
                        .batch_execute("DROP TABLE memories")
                        .map_err(|e| StorageError::Database(e.to_string()))?;
                    eprintln!(
                        "[migration] Dropped empty 'memories' table (engrams has {} rows)",
                        engram_count
                    );
                }
            }

            self.conn
                .batch_execute("ALTER TABLE engrams RENAME TO memories")
                .map_err(|e| StorageError::Database(e.to_string()))?;
            eprintln!("[migration] Renamed table engrams → memories");

            // Rebuild FTS table with new name and column name
            let has_old_fts: Result<CountResult, _> = diesel::sql_query(
                "SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='table' AND name='engram_fts'",
            )
            .get_result(&mut self.conn);

            if matches!(has_old_fts, Ok(r) if r.cnt > 0) {
                self.conn
                    .batch_execute(
                        r#"
                        DROP TABLE IF EXISTS engram_fts;
                        DROP TABLE IF EXISTS memory_fts;
                        CREATE VIRTUAL TABLE memory_fts USING fts5(
                            content,
                            memory_id UNINDEXED,
                            tokenize='porter unicode61'
                        );
                        INSERT INTO memory_fts(content, memory_id) SELECT content, id FROM memories;
                    "#,
                    )
                    .map_err(|e| StorageError::Database(e.to_string()))?;
                eprintln!("[migration] Rebuilt FTS table as memory_fts");
            }

            // Handle engram_enrichments → memory_enrichments migration
            let has_old_enrichments: Result<CountResult, _> = diesel::sql_query(
                "SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='table' AND name='engram_enrichments'",
            )
            .get_result(&mut self.conn);

            if matches!(has_old_enrichments, Ok(r) if r.cnt > 0) {
                // Check if memory_enrichments also exists
                let has_new_enrichments: Result<CountResult, _> = diesel::sql_query(
                    "SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='table' AND name='memory_enrichments'",
                )
                .get_result(&mut self.conn);

                if matches!(has_new_enrichments, Ok(r) if r.cnt > 0) {
                    let old_count: i32 = diesel::sql_query(
                        "SELECT COUNT(*) as cnt FROM engram_enrichments",
                    )
                    .get_result::<CountResult>(&mut self.conn)
                    .map(|r| r.cnt)
                    .map_err(|e| StorageError::Database(
                        format!("Migration aborted: cannot count engram_enrichments rows: {}", e)
                    ))?;

                    let new_count: i32 = diesel::sql_query(
                        "SELECT COUNT(*) as cnt FROM memory_enrichments",
                    )
                    .get_result::<CountResult>(&mut self.conn)
                    .map(|r| r.cnt)
                    .map_err(|e| StorageError::Database(
                        format!("Migration aborted: cannot count memory_enrichments rows: {}", e)
                    ))?;

                    if new_count > 0 && old_count == 0 {
                        // Already migrated, ghost table. Drop the ghost.
                        self.conn
                            .batch_execute("DROP TABLE engram_enrichments")
                            .map_err(|e| StorageError::Database(e.to_string()))?;
                        eprintln!("[migration] Dropped stale 'engram_enrichments' table");
                    } else if new_count > 0 && old_count > 0 {
                        return Err(StorageError::Database(format!(
                            "MIGRATION CONFLICT: both 'engram_enrichments' ({} rows) and 'memory_enrichments' ({} rows) contain data.",
                            old_count, new_count
                        )));
                    } else {
                        self.conn
                            .batch_execute("DROP TABLE memory_enrichments")
                            .map_err(|e| StorageError::Database(e.to_string()))?;
                        self.conn
                            .batch_execute("ALTER TABLE engram_enrichments RENAME TO memory_enrichments")
                            .map_err(|e| StorageError::Database(e.to_string()))?;
                        eprintln!("[migration] Renamed table engram_enrichments → memory_enrichments");
                    }
                } else {
                    // Only old table exists — straightforward rename
                    self.conn
                        .batch_execute("ALTER TABLE engram_enrichments RENAME TO memory_enrichments")
                        .map_err(|e| StorageError::Database(e.to_string()))?;
                    eprintln!("[migration] Renamed table engram_enrichments → memory_enrichments");
                }
            }

            // Rename engram_id column in memory_enrichments if it still has the old name
            let has_old_column: Result<CountResult, _> = diesel::sql_query(
                "SELECT COUNT(*) as cnt FROM pragma_table_info('memory_enrichments') WHERE name = 'engram_id'",
            )
            .get_result(&mut self.conn);

            if matches!(has_old_column, Ok(r) if r.cnt > 0) {
                self.conn
                    .batch_execute("ALTER TABLE memory_enrichments RENAME COLUMN engram_id TO memory_id")
                    .map_err(|e| StorageError::Database(e.to_string()))?;
                eprintln!("[migration] Renamed memory_enrichments.engram_id → memory_id");
            }

            // SQLite ALTER TABLE RENAME doesn't rename indexes — they keep their
            // old names but still work. We drop the old-named indexes here so
            // create_schema() can recreate them with the new names. No data is
            // lost — indexes are derived lookup structures, not stored data.
            self.conn
                .batch_execute(
                    r#"
                    DROP INDEX IF EXISTS idx_engrams_state;
                    DROP INDEX IF EXISTS idx_engrams_energy;
                    DROP INDEX IF EXISTS idx_enrichments_engram;
                "#,
                )
                .map_err(|e| StorageError::Database(e.to_string()))?;
            eprintln!("[migration] Rebuilding indexes with new names (no data affected)");

        Ok(())
    }

    /// Migrate existing database to add embedding column if missing (SQLite)
    #[cfg(feature = "sqlite")]
    fn migrate_embeddings(&mut self) -> StorageResult<()> {
        // Check if embedding column exists using raw SQL
        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }

        let result: Result<CountResult, _> = diesel::sql_query(
            "SELECT COUNT(*) as cnt FROM pragma_table_info('memories') WHERE name = 'embedding'",
        )
        .get_result(&mut self.conn);

        let has_embedding = match result {
            Ok(r) => r.cnt > 0,
            Err(_) => false,
        };

        if !has_embedding {
            self.conn
                .batch_execute("ALTER TABLE memories ADD COLUMN embedding BLOB")
                .map_err(|e| StorageError::Database(e.to_string()))?;
        }

        Ok(())
    }

    /// Migrate existing database to add memory_enrichments table if missing (SQLite)
    #[cfg(feature = "sqlite")]
    fn migrate_enrichments_table(&mut self) -> StorageResult<()> {
        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }

        let result: Result<CountResult, _> = diesel::sql_query(
            "SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='table' AND name='memory_enrichments'",
        )
        .get_result(&mut self.conn);

        let has_table = match result {
            Ok(r) => r.cnt > 0,
            Err(_) => false,
        };

        if !has_table {
            self.conn
                .batch_execute(
                    r#"
                CREATE TABLE IF NOT EXISTS memory_enrichments (
                    memory_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    source TEXT NOT NULL DEFAULT 'llm',
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY (memory_id, seq),
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_enrichments_memory ON memory_enrichments(memory_id);
            "#,
                )
                .map_err(|e| StorageError::Database(e.to_string()))?;
        }

        Ok(())
    }

    /// Migrate existing database to add ordinal column to associations if missing (SQLite)
    #[cfg(feature = "sqlite")]
    fn migrate_association_ordinal(&mut self) -> StorageResult<()> {
        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }

        let result: Result<CountResult, _> = diesel::sql_query(
            "SELECT COUNT(*) as cnt FROM pragma_table_info('associations') WHERE name = 'ordinal'",
        )
        .get_result(&mut self.conn);

        let has_ordinal = match result {
            Ok(r) => r.cnt > 0,
            Err(_) => false,
        };

        if !has_ordinal {
            self.conn
                .batch_execute("ALTER TABLE associations ADD COLUMN ordinal INTEGER")
                .map_err(|e| StorageError::Database(e.to_string()))?;
        }

        Ok(())
    }

    /// Migrate existing database to add the sessions table if missing (SQLite)
    #[cfg(feature = "sqlite")]
    fn migrate_sessions_table(&mut self) -> StorageResult<()> {
        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }

        let result: Result<CountResult, _> = diesel::sql_query(
            "SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='table' AND name='sessions'",
        )
        .get_result(&mut self.conn);

        let has_table = match result {
            Ok(r) => r.cnt > 0,
            Err(_) => false,
        };

        if !has_table {
            self.conn
                .batch_execute(
                    r#"
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id         TEXT PRIMARY KEY,
                    queries            TEXT NOT NULL DEFAULT '[]',
                    centroid           BLOB,
                    query_count        INTEGER NOT NULL DEFAULT 0,
                    created_at         INTEGER NOT NULL,
                    last_seen_at       INTEGER NOT NULL,
                    search_result_ids  TEXT NOT NULL DEFAULT '[]',
                    recalled_ids       TEXT NOT NULL DEFAULT '[]',
                    created_ids        TEXT NOT NULL DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_last_seen ON sessions(last_seen_at);
            "#,
                )
                .map_err(|e| StorageError::Database(e.to_string()))?;
        } else {
            // Table exists — add new columns if missing (upgrade from old schema)
            let col_check: Result<CountResult, _> = diesel::sql_query(
                "SELECT COUNT(*) as cnt FROM pragma_table_info('sessions') WHERE name = 'search_result_ids'",
            )
            .get_result(&mut self.conn);

            let has_new_columns = match col_check {
                Ok(r) => r.cnt > 0,
                Err(_) => false,
            };

            if !has_new_columns {
                self.conn
                    .batch_execute(
                        r#"
                    ALTER TABLE sessions ADD COLUMN search_result_ids TEXT NOT NULL DEFAULT '[]';
                    ALTER TABLE sessions ADD COLUMN recalled_ids TEXT NOT NULL DEFAULT '[]';
                    ALTER TABLE sessions ADD COLUMN created_ids TEXT NOT NULL DEFAULT '[]';
                "#,
                    )
                    .map_err(|e| StorageError::Database(e.to_string()))?;
            }
        }

        Ok(())
    }

    /// Backfill vec0 virtual tables from existing embeddings.
    ///
    /// Runs once: if memory_vec is empty but memories has embeddings, we
    /// bulk-copy everything into the vec0 tables so that sqlite-vec can
    /// serve similarity queries immediately.
    #[cfg(feature = "sqlite")]
    fn migrate_vec0_backfill(&mut self) -> StorageResult<()> {
        use diesel::connection::SimpleConnection;

        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }

        // Compare counts: if vec0 has fewer rows than the source tables,
        // rebuild from scratch. This repairs partial failures and missed syncs.
        let vec_count: CountResult =
            diesel::sql_query("SELECT COUNT(*) as cnt FROM memory_vec")
                .get_result(&mut self.conn)
                .map_err(|e| StorageError::Database(e.to_string()))?;
        let embed_count: CountResult = diesel::sql_query(
            "SELECT COUNT(*) as cnt FROM memories WHERE embedding IS NOT NULL",
        )
        .get_result(&mut self.conn)
        .map_err(|e| StorageError::Database(e.to_string()))?;

        if embed_count.cnt > 0 && vec_count.cnt < embed_count.cnt {
            // Clear and rebuild to avoid duplicates
            let _ = self.conn.batch_execute("DELETE FROM memory_vec");
            match self.conn.batch_execute(
                "INSERT INTO memory_vec(memory_id, embedding) \
                 SELECT id, embedding FROM memories WHERE embedding IS NOT NULL",
            ) {
                Ok(()) => eprintln!(
                    "[migration] Backfilled {} memory embeddings into memory_vec",
                    embed_count.cnt
                ),
                Err(e) => eprintln!(
                    "[migration] Could not backfill memory_vec (dimension mismatch?): {}",
                    e
                ),
            }
        }

        let evec_count: CountResult =
            diesel::sql_query("SELECT COUNT(*) as cnt FROM enrichment_vec")
                .get_result(&mut self.conn)
                .map_err(|e| StorageError::Database(e.to_string()))?;
        let enrich_count: CountResult =
            diesel::sql_query("SELECT COUNT(*) as cnt FROM memory_enrichments")
                .get_result(&mut self.conn)
                .map_err(|e| StorageError::Database(e.to_string()))?;

        if enrich_count.cnt > 0 && evec_count.cnt < enrich_count.cnt {
            let _ = self.conn.batch_execute("DELETE FROM enrichment_vec");
            match self.conn.batch_execute(
                "INSERT INTO enrichment_vec(memory_id, embedding) \
                 SELECT memory_id, embedding FROM memory_enrichments",
            ) {
                Ok(()) => eprintln!(
                    "[migration] Backfilled {} enrichment embeddings into enrichment_vec",
                    enrich_count.cnt
                ),
                Err(e) => eprintln!(
                    "[migration] Could not backfill enrichment_vec (dimension mismatch?): {}",
                    e
                ),
            }
        }

        Ok(())
    }
}

impl Storage for MemoryStorage {
    #[cfg(feature = "sqlite")]
    fn initialize(&mut self) -> StorageResult<()> {
        self.migrate_engrams_to_memories()?;
        self.create_schema()?;
        self.migrate_embeddings()?;
        self.migrate_association_ordinal()?;
        self.migrate_enrichments_table()?;
        self.migrate_sessions_table()?;
        self.migrate_vec0_backfill()?;
        Ok(())
    }

    #[cfg(feature = "postgres")]
    fn initialize(&mut self) -> StorageResult<()> {
        // Postgres schema is managed via migrations (diesel_cli)
        // TODO: Add postgres schema setup or use migrations
        Ok(())
    }

    // ==================
    // IDENTITY
    // ==================

    fn save_identity(&mut self, ident: &Identity) -> StorageResult<()> {
        let json = serde_json::to_string(ident)?;

        diesel::replace_into(identity::table)
            .values(NewIdentity { id: 1, data: &json })
            .execute(&mut self.conn)?;

        Ok(())
    }

    fn load_identity(&mut self) -> StorageResult<Option<Identity>> {
        let result: Option<IdentityRow> = identity::table
            .filter(identity::id.eq(1))
            .select(IdentityRow::as_select())
            .first(&mut self.conn)
            .optional()?;

        match result {
            Some(row) => {
                let ident: Identity = serde_json::from_str(&row.data)?;
                Ok(Some(ident))
            }
            None => Ok(None),
        }
    }

    // ==================
    // MEMORIES
    // ==================

    fn save_memory(&mut self, mem: &Memory) -> StorageResult<()> {
        let id_str = mem.id.to_string();
        let state_str = state_to_str(mem.state);
        let tags_json = serde_json::to_string(&mem.tags)?;

        diesel::replace_into(memories::table)
            .values(NewMemory {
                id: &id_str,
                content: &mem.content,
                energy: mem.energy as f32,
                state: state_str,
                confidence: mem.confidence as f32,
                created_at: mem.created_at,
                last_accessed: mem.last_accessed,
                access_count: mem.access_count as i64,
                tags: tags_json,
                embedding: mem.embedding.as_ref().map(|e| embedding_to_bytes(e)),
            })
            .execute(&mut self.conn)?;

        // Sync FTS5: delete existing entry (if any) then insert fresh
        diesel::sql_query("DELETE FROM memory_fts WHERE memory_id = ?1")
            .bind::<diesel::sql_types::Text, _>(&id_str)
            .execute(&mut self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;
        diesel::sql_query("INSERT INTO memory_fts(content, memory_id) VALUES (?1, ?2)")
            .bind::<diesel::sql_types::Text, _>(&mem.content)
            .bind::<diesel::sql_types::Text, _>(&id_str)
            .execute(&mut self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;

        // Always remove any existing vec0 row to prevent stale entries.
        diesel::sql_query("DELETE FROM memory_vec WHERE memory_id = ?")
            .bind::<diesel::sql_types::Text, _>(&id_str)
            .execute(&mut self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;
        // Only insert when dimensions match the vec0 table.
        if let Some(ref embedding) = mem.embedding {
            if embedding.len() == super::VEC0_DIMENSIONS {
                let bytes = embedding_to_bytes(embedding);
                diesel::sql_query(
                    "INSERT INTO memory_vec(memory_id, embedding) VALUES (?, ?)",
                )
                .bind::<diesel::sql_types::Text, _>(&id_str)
                .bind::<diesel::sql_types::Binary, _>(&bytes[..])
                .execute(&mut self.conn)
                .map_err(|e| StorageError::Database(e.to_string()))?;
            }
        }

        Ok(())
    }

    fn save_memories(&mut self, memory_list: &[&Memory]) -> StorageResult<()> {
        self.conn
            .transaction::<_, diesel::result::Error, _>(|conn| {
                for mem in memory_list {
                    let id_str = mem.id.to_string();
                    let state_str = state_to_str(mem.state);
                    let tags_json = serde_json::to_string(&mem.tags)
                        .map_err(|e| diesel::result::Error::QueryBuilderError(Box::new(e)))?;

                    diesel::replace_into(memories::table)
                        .values(NewMemory {
                            id: &id_str,
                            content: &mem.content,
                            energy: mem.energy as f32,
                            state: state_str,
                            confidence: mem.confidence as f32,
                            created_at: mem.created_at,
                            last_accessed: mem.last_accessed,
                            access_count: mem.access_count as i64,
                            tags: tags_json,
                            embedding: mem.embedding.as_ref().map(|e| embedding_to_bytes(e)),
                        })
                        .execute(conn)?;

                    // Sync FTS5: delete existing entry (if any) then insert fresh
                    diesel::sql_query("DELETE FROM memory_fts WHERE memory_id = ?1")
                        .bind::<diesel::sql_types::Text, _>(&id_str)
                        .execute(conn)?;
                    diesel::sql_query("INSERT INTO memory_fts(content, memory_id) VALUES (?1, ?2)")
                        .bind::<diesel::sql_types::Text, _>(&mem.content)
                        .bind::<diesel::sql_types::Text, _>(&id_str)
                        .execute(conn)?;

                    // Always remove stale vec0 row, conditionally insert.
                    diesel::sql_query("DELETE FROM memory_vec WHERE memory_id = ?")
                        .bind::<diesel::sql_types::Text, _>(&id_str)
                        .execute(conn)?;
                    if let Some(ref embedding) = mem.embedding {
                        if embedding.len() == super::VEC0_DIMENSIONS {
                            let bytes = embedding_to_bytes(embedding);
                            diesel::sql_query(
                                "INSERT INTO memory_vec(memory_id, embedding) VALUES (?, ?)",
                            )
                            .bind::<diesel::sql_types::Text, _>(&id_str)
                            .bind::<diesel::sql_types::Binary, _>(&bytes[..])
                            .execute(conn)?;
                        }
                    }
                }
                Ok(())
            })?;

        Ok(())
    }

    fn load_memory(&mut self, id: &MemoryId) -> StorageResult<Option<Memory>> {
        let id_str = id.to_string();

        let result: Option<MemoryRow> = memories::table
            .filter(memories::id.eq(&id_str))
            .select(MemoryRow::as_select())
            .first(&mut self.conn)
            .optional()?;

        match result {
            Some(row) => Ok(Some(row.into_memory()?)),
            None => Ok(None),
        }
    }

    fn load_all_memories(&mut self) -> StorageResult<Vec<Memory>> {
        let rows: Vec<MemoryRow> = memories::table
            .select(MemoryRow::as_select())
            .load(&mut self.conn)?;

        rows.into_iter().map(|row| row.into_memory()).collect()
    }

    fn load_memories_by_state(&mut self, state: MemoryState) -> StorageResult<Vec<Memory>> {
        let state_str = state_to_str(state);

        let rows: Vec<MemoryRow> = memories::table
            .filter(memories::state.eq(state_str))
            .select(MemoryRow::as_select())
            .load(&mut self.conn)?;

        rows.into_iter().map(|row| row.into_memory()).collect()
    }

    fn load_memories_by_tag(&mut self, tag: &str) -> StorageResult<Vec<Memory>> {
        // Tags are stored as JSON array, search with LIKE using parameterized query
        let pattern = format!("%\"{}%", tag.to_lowercase());

        let rows: Vec<MemoryRow> = diesel::sql_query(
            "SELECT id, content, energy, state, confidence, created_at, last_accessed, access_count, tags, embedding FROM memories WHERE LOWER(tags) LIKE ?1"
        )
        .bind::<diesel::sql_types::Text, _>(&pattern)
        .load(&mut self.conn)
        .map_err(|e| StorageError::Database(e.to_string()))?;

        rows.into_iter().map(|row| row.into_memory()).collect()
    }

    fn delete_memory(&mut self, id: &MemoryId) -> StorageResult<bool> {
        let id_str = id.to_string();

        // Delete the memory
        let deleted = diesel::delete(memories::table.filter(memories::id.eq(&id_str)))
            .execute(&mut self.conn)?;

        // Delete from FTS5 index
        diesel::sql_query("DELETE FROM memory_fts WHERE memory_id = ?1")
            .bind::<diesel::sql_types::Text, _>(&id_str)
            .execute(&mut self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;

        // Delete from vec0 tables
        diesel::sql_query("DELETE FROM memory_vec WHERE memory_id = ?")
            .bind::<diesel::sql_types::Text, _>(&id_str)
            .execute(&mut self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;
        diesel::sql_query("DELETE FROM enrichment_vec WHERE memory_id = ?")
            .bind::<diesel::sql_types::Text, _>(&id_str)
            .execute(&mut self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;

        // Delete associations from/to this memory
        diesel::delete(
            associations::table.filter(
                associations::from_id
                    .eq(&id_str)
                    .or(associations::to_id.eq(&id_str)),
            ),
        )
        .execute(&mut self.conn)?;

        Ok(deleted > 0)
    }

    fn count_memories(&mut self) -> StorageResult<usize> {
        let count: i64 = memories::table.count().get_result(&mut self.conn)?;
        Ok(count as usize)
    }

    fn save_memory_energies(
        &mut self,
        updates: &[(&MemoryId, f64, MemoryState)],
    ) -> StorageResult<()> {
        if updates.is_empty() {
            return Ok(());
        }

        self.conn
            .transaction::<_, diesel::result::Error, _>(|conn| {
                for (id, energy, state) in updates {
                    let id_str = id.to_string();
                    let state_str = state_to_str(*state);

                    diesel::update(memories::table.filter(memories::id.eq(&id_str)))
                        .set((
                            memories::energy.eq(*energy as f32),
                            memories::state.eq(state_str),
                        ))
                        .execute(conn)?;
                }
                Ok(())
            })?;

        Ok(())
    }

    // ==================
    // ASSOCIATIONS
    // ==================

    fn save_association(&mut self, assoc: &Association) -> StorageResult<()> {
        let from_str = assoc.from.to_string();
        let to_str = assoc.to.to_string();

        diesel::replace_into(associations::table)
            .values(NewAssociation {
                from_id: &from_str,
                to_id: &to_str,
                weight: assoc.weight as f32,
                created_at: assoc.created_at,
                last_activated: assoc.last_activated,
                co_activation_count: assoc.co_activation_count as i64,
                ordinal: assoc.ordinal.map(|v| v as i32),
            })
            .execute(&mut self.conn)?;

        Ok(())
    }

    fn save_associations(&mut self, assocs: &[&Association]) -> StorageResult<()> {
        self.conn
            .transaction::<_, diesel::result::Error, _>(|conn| {
                for assoc in assocs {
                    let from_str = assoc.from.to_string();
                    let to_str = assoc.to.to_string();

                    diesel::replace_into(associations::table)
                        .values(NewAssociation {
                            from_id: &from_str,
                            to_id: &to_str,
                            weight: assoc.weight as f32,
                            created_at: assoc.created_at,
                            last_activated: assoc.last_activated,
                            co_activation_count: assoc.co_activation_count as i64,
                            ordinal: assoc.ordinal.map(|v| v as i32),
                        })
                        .execute(conn)?;
                }
                Ok(())
            })?;

        Ok(())
    }

    fn load_associations_from(&mut self, from: &MemoryId) -> StorageResult<Vec<Association>> {
        let from_str = from.to_string();

        let rows: Vec<AssociationRow> = associations::table
            .filter(associations::from_id.eq(&from_str))
            .select(AssociationRow::as_select())
            .load(&mut self.conn)?;

        rows.into_iter().map(|row| row.into_association()).collect()
    }

    fn load_all_associations(&mut self) -> StorageResult<Vec<Association>> {
        let rows: Vec<AssociationRow> = associations::table
            .select(AssociationRow::as_select())
            .load(&mut self.conn)?;

        rows.into_iter().map(|row| row.into_association()).collect()
    }

    fn delete_all_associations(&mut self) -> StorageResult<()> {
        diesel::delete(associations::table).execute(&mut self.conn)?;
        Ok(())
    }

    fn prune_orphan_associations(&mut self) -> StorageResult<usize> {
        let deleted = diesel::sql_query(
            "DELETE FROM associations \
             WHERE from_id NOT IN (SELECT id FROM memories) \
             OR to_id NOT IN (SELECT id FROM memories)",
        )
        .execute(&mut self.conn)
        .map_err(|e| StorageError::Database(e.to_string()))?;
        Ok(deleted)
    }

    // ==================
    // CONFIG
    // ==================

    fn save_config(&mut self, _cfg: &Config) -> StorageResult<()> {
        // Config is now stored in config.toml, not SQLite.
        // This is a no-op kept for trait compatibility.
        Ok(())
    }

    fn load_config(&mut self) -> StorageResult<Option<Config>> {
        // Config is now loaded from config.toml at startup.
        // This is a no-op kept for trait compatibility.
        Ok(None)
    }

    fn save_last_decay_at(&mut self, timestamp: i64) -> StorageResult<()> {
        diesel::replace_into(metadata::table)
            .values(NewMetadata {
                key: "last_decay_at",
                value: &timestamp.to_string(),
            })
            .execute(&mut self.conn)?;

        Ok(())
    }

    fn load_last_decay_at(&mut self) -> StorageResult<Option<i64>> {
        let result: Option<MetadataRow> = metadata::table
            .filter(metadata::key.eq("last_decay_at"))
            .select(MetadataRow::as_select())
            .first(&mut self.conn)
            .optional()?;

        match result {
            Some(row) => {
                let timestamp: i64 = row.value.parse().map_err(|e: std::num::ParseIntError| {
                    StorageError::Serialization(e.to_string())
                })?;
                Ok(Some(timestamp))
            }
            None => Ok(None),
        }
    }

    fn get_metadata(&mut self, key: &str) -> StorageResult<Option<String>> {
        let result: Option<MetadataRow> = metadata::table
            .filter(metadata::key.eq(key))
            .select(MetadataRow::as_select())
            .first(&mut self.conn)
            .optional()?;

        Ok(result.map(|row| row.value))
    }

    fn set_metadata(&mut self, key: &str, value: &str) -> StorageResult<()> {
        diesel::replace_into(metadata::table)
            .values(NewMetadata { key, value })
            .execute(&mut self.conn)?;

        Ok(())
    }

    // ==================
    // LIFECYCLE
    // ==================

    #[cfg(feature = "sqlite")]
    fn flush(&mut self) -> StorageResult<()> {
        self.conn.batch_execute("PRAGMA wal_checkpoint(PASSIVE);")?;
        Ok(())
    }

    #[cfg(feature = "postgres")]
    fn flush(&mut self) -> StorageResult<()> {
        // Postgres doesn't need explicit flush
        Ok(())
    }

    // ==================
    // VECTOR SEARCH
    // ==================

    #[cfg(feature = "sqlite")]
    fn find_similar_by_embedding(
        &mut self,
        query_embedding: &[f32],
        limit: usize,
        min_score: f32,
    ) -> StorageResult<Vec<SimilarityResult>> {
        let mut vs = VectorSearch::new(&mut self.conn);
        vs.find_similar(query_embedding, limit, min_score)
    }

    #[cfg(feature = "sqlite")]
    fn count_with_embeddings(&mut self) -> StorageResult<usize> {
        let mut vs = VectorSearch::new(&mut self.conn);
        vs.count_with_embeddings()
    }

    #[cfg(feature = "sqlite")]
    fn count_without_embeddings(&mut self) -> StorageResult<usize> {
        let mut vs = VectorSearch::new(&mut self.conn);
        vs.count_without_embeddings()
    }

    #[cfg(feature = "sqlite")]
    fn get_ids_without_embeddings(&mut self, limit: usize) -> StorageResult<Vec<MemoryId>> {
        let mut vs = VectorSearch::new(&mut self.conn);
        vs.get_ids_without_embeddings(limit)
    }

    #[cfg(feature = "sqlite")]
    fn get_ids_without_enrichments(&mut self) -> StorageResult<Vec<MemoryId>> {
        let mut vs = VectorSearch::new(&mut self.conn);
        vs.get_ids_without_enrichments()
    }

    #[cfg(feature = "sqlite")]
    fn set_embedding(&mut self, id: &MemoryId, embedding: &[f32]) -> StorageResult<()> {
        let mut vs = VectorSearch::new(&mut self.conn);
        vs.set_embedding(id, embedding)
    }

    #[cfg(feature = "sqlite")]
    fn get_embedding(&mut self, id: &MemoryId) -> StorageResult<Option<Vec<f32>>> {
        let mut vs = VectorSearch::new(&mut self.conn);
        vs.get_embedding(id)
    }

    // Postgres vector search - TODO: implement with pgvector
    #[cfg(feature = "postgres")]
    fn find_similar_by_embedding(
        &mut self,
        _query_embedding: &[f32],
        _limit: usize,
        _min_score: f32,
    ) -> StorageResult<Vec<SimilarityResult>> {
        // TODO: Implement with pgvector extension
        Ok(Vec::new())
    }

    #[cfg(feature = "postgres")]
    fn count_with_embeddings(&mut self) -> StorageResult<usize> {
        Ok(0) // TODO
    }

    #[cfg(feature = "postgres")]
    fn count_without_embeddings(&mut self) -> StorageResult<usize> {
        Ok(0) // TODO
    }

    #[cfg(feature = "postgres")]
    fn get_ids_without_embeddings(&mut self, _limit: usize) -> StorageResult<Vec<MemoryId>> {
        Ok(Vec::new()) // TODO
    }

    #[cfg(feature = "postgres")]
    fn get_ids_without_enrichments(&mut self) -> StorageResult<Vec<MemoryId>> {
        Ok(Vec::new()) // TODO
    }

    #[cfg(feature = "postgres")]
    fn set_embedding(&mut self, _id: &MemoryId, _embedding: &[f32]) -> StorageResult<()> {
        Ok(()) // TODO
    }

    #[cfg(feature = "postgres")]
    fn get_embedding(&mut self, _id: &MemoryId) -> StorageResult<Option<Vec<f32>>> {
        Ok(None) // TODO
    }

    #[cfg(feature = "sqlite")]
    fn clear_all_embeddings(&mut self) -> StorageResult<usize> {
        let mut vs = VectorSearch::new(&mut self.conn);
        vs.clear_all_embeddings()
    }

    #[cfg(feature = "postgres")]
    fn clear_all_embeddings(&mut self) -> StorageResult<usize> {
        Ok(0) // TODO
    }

    #[cfg(feature = "sqlite")]
    fn set_enrichment_embeddings(
        &mut self,
        id: &MemoryId,
        embeddings: &[Vec<f32>],
        source: &str,
    ) -> StorageResult<()> {
        let mut vs = VectorSearch::new(&mut self.conn);
        vs.set_enrichment_embeddings(id, embeddings, source)
    }

    #[cfg(feature = "sqlite")]
    fn delete_enrichments(&mut self, id: &MemoryId) -> StorageResult<()> {
        let mut vs = VectorSearch::new(&mut self.conn);
        vs.delete_enrichments(id)
    }

    #[cfg(feature = "sqlite")]
    fn count_enrichments(&mut self) -> StorageResult<usize> {
        let mut vs = VectorSearch::new(&mut self.conn);
        vs.count_enrichments()
    }

    #[cfg(feature = "sqlite")]
    fn clear_all_enrichments(&mut self) -> StorageResult<usize> {
        let mut vs = VectorSearch::new(&mut self.conn);
        vs.clear_all_enrichments()
    }

    #[cfg(feature = "postgres")]
    fn clear_all_enrichments(&mut self) -> StorageResult<usize> {
        Ok(0) // TODO
    }

    // ==================
    // KEYWORD SEARCH (FTS5)
    // ==================

    #[cfg(feature = "sqlite")]
    fn keyword_search(
        &mut self,
        query: &str,
        limit: usize,
    ) -> StorageResult<Vec<SimilarityResult>> {
        let sanitized = sanitize_fts_query(query);
        if sanitized.is_empty() {
            return Ok(Vec::new());
        }

        #[derive(QueryableByName, Debug)]
        struct FtsRow {
            #[diesel(sql_type = diesel::sql_types::Text)]
            id: String,
            #[diesel(sql_type = diesel::sql_types::Text)]
            content: String,
            #[diesel(sql_type = diesel::sql_types::Float)]
            score: f32,
        }

        let rows: Vec<FtsRow> = diesel::sql_query(
            "SELECT e.id, e.content, -bm25(memory_fts) as score \
             FROM memory_fts f \
             JOIN memories e ON e.id = f.memory_id \
             WHERE memory_fts MATCH ?1 \
             ORDER BY bm25(memory_fts) \
             LIMIT ?2",
        )
        .bind::<diesel::sql_types::Text, _>(&sanitized)
        .bind::<diesel::sql_types::BigInt, _>(limit as i64)
        .load(&mut self.conn)
        .map_err(|e| StorageError::Database(e.to_string()))?;

        let mut results = Vec::with_capacity(rows.len());
        for row in rows {
            let id = uuid::Uuid::parse_str(&row.id)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
            results.push(SimilarityResult {
                id,
                score: row.score,
                content: row.content,
            });
        }

        Ok(results)
    }

    #[cfg(feature = "postgres")]
    fn keyword_search(
        &mut self,
        _query: &str,
        _limit: usize,
    ) -> StorageResult<Vec<SimilarityResult>> {
        Ok(Vec::new()) // TODO
    }

    #[cfg(feature = "sqlite")]
    fn ensure_fts_populated(&mut self) -> StorageResult<usize> {
        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }

        let fts_count: CountResult = diesel::sql_query("SELECT COUNT(*) as cnt FROM memory_fts")
            .get_result(&mut self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;

        let memory_count: CountResult = diesel::sql_query("SELECT COUNT(*) as cnt FROM memories")
            .get_result(&mut self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;

        if fts_count.cnt > 0 || memory_count.cnt == 0 {
            return Ok(0);
        }

        // Backfill FTS5 from existing memorys
        let affected = diesel::sql_query(
            "INSERT INTO memory_fts(content, memory_id) SELECT content, id FROM memories",
        )
        .execute(&mut self.conn)
        .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(affected)
    }

    #[cfg(feature = "postgres")]
    fn ensure_fts_populated(&mut self) -> StorageResult<usize> {
        Ok(0) // TODO
    }

    // ==================
    // ACCESS LOG
    // ==================

    fn log_access(
        &mut self,
        query_text: &str,
        result_ids: &[MemoryId],
        recalled_ids: &[MemoryId],
    ) -> StorageResult<()> {
        use super::models::NewAccessLogEntry;
        use super::schema::access_log;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let result_ids_json = serde_json::to_string(
            &result_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>(),
        )
        .unwrap_or_else(|_| "[]".to_string());

        let recalled_ids_json = serde_json::to_string(
            &recalled_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>(),
        )
        .unwrap_or_else(|_| "[]".to_string());

        let entry = NewAccessLogEntry {
            timestamp: now,
            query_text,
            result_ids: &result_ids_json,
            recalled_ids: &recalled_ids_json,
        };

        ::diesel::insert_into(access_log::table)
            .values(&entry)
            .execute(&mut self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(())
    }

    // ==================
    // SESSIONS
    // ==================

    #[cfg(feature = "sqlite")]
    fn load_session(&mut self, session_id: &str) -> StorageResult<Option<SessionContext>> {
        #[derive(QueryableByName)]
        struct SessionRow {
            #[diesel(sql_type = diesel::sql_types::Text)]
            session_id: String,
            #[diesel(sql_type = diesel::sql_types::Text)]
            queries: String,
            #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Blob>)]
            centroid: Option<Vec<u8>>,
            #[diesel(sql_type = diesel::sql_types::BigInt)]
            query_count: i64,
            #[diesel(sql_type = diesel::sql_types::BigInt)]
            created_at: i64,
            #[diesel(sql_type = diesel::sql_types::BigInt)]
            last_seen_at: i64,
            #[diesel(sql_type = diesel::sql_types::Text)]
            search_result_ids: String,
            #[diesel(sql_type = diesel::sql_types::Text)]
            recalled_ids: String,
            #[diesel(sql_type = diesel::sql_types::Text)]
            created_ids: String,
        }

        let result: Option<SessionRow> = diesel::sql_query(
            "SELECT session_id, queries, centroid, query_count, created_at, last_seen_at, \
             search_result_ids, recalled_ids, created_ids \
             FROM sessions WHERE session_id = ?1",
        )
        .bind::<diesel::sql_types::Text, _>(session_id)
        .get_result::<SessionRow>(&mut self.conn)
        .optional()
        .map_err(|e| StorageError::Database(e.to_string()))?;

        match result {
            None => Ok(None),
            Some(row) => {
                let queries: Vec<String> =
                    serde_json::from_str(&row.queries).unwrap_or_default();
                let centroid = row.centroid.as_deref().and_then(bytes_to_embedding);
                let search_result_ids: Vec<String> =
                    serde_json::from_str(&row.search_result_ids).unwrap_or_default();
                let recalled_ids: Vec<String> =
                    serde_json::from_str(&row.recalled_ids).unwrap_or_default();
                let created_ids: Vec<String> =
                    serde_json::from_str(&row.created_ids).unwrap_or_default();
                Ok(Some(SessionContext {
                    session_id: row.session_id,
                    queries,
                    centroid,
                    query_count: row.query_count as usize,
                    created_at: row.created_at,
                    last_seen_at: row.last_seen_at,
                    search_result_ids,
                    recalled_ids,
                    created_ids,
                }))
            }
        }
    }

    #[cfg(feature = "sqlite")]
    fn save_session(&mut self, session: &SessionContext) -> StorageResult<()> {
        let queries_json = serde_json::to_string(&session.queries)
            .unwrap_or_else(|_| "[]".to_string());
        let centroid_bytes: Option<Vec<u8>> =
            session.centroid.as_deref().map(embedding_to_bytes);
        let search_result_ids_json = serde_json::to_string(&session.search_result_ids)
            .unwrap_or_else(|_| "[]".to_string());
        let recalled_ids_json = serde_json::to_string(&session.recalled_ids)
            .unwrap_or_else(|_| "[]".to_string());
        let created_ids_json = serde_json::to_string(&session.created_ids)
            .unwrap_or_else(|_| "[]".to_string());

        diesel::sql_query(
            "INSERT OR REPLACE INTO sessions \
             (session_id, queries, centroid, query_count, created_at, last_seen_at, \
              search_result_ids, recalled_ids, created_ids) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        )
        .bind::<diesel::sql_types::Text, _>(&session.session_id)
        .bind::<diesel::sql_types::Text, _>(&queries_json)
        .bind::<diesel::sql_types::Nullable<diesel::sql_types::Blob>, _>(centroid_bytes)
        .bind::<diesel::sql_types::BigInt, _>(session.query_count as i64)
        .bind::<diesel::sql_types::BigInt, _>(session.created_at)
        .bind::<diesel::sql_types::BigInt, _>(session.last_seen_at)
        .bind::<diesel::sql_types::Text, _>(&search_result_ids_json)
        .bind::<diesel::sql_types::Text, _>(&recalled_ids_json)
        .bind::<diesel::sql_types::Text, _>(&created_ids_json)
        .execute(&mut self.conn)
        .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(())
    }

    #[cfg(feature = "sqlite")]
    fn delete_expired_sessions(&mut self, expire_before: i64) -> StorageResult<usize> {
        let count = diesel::sql_query("DELETE FROM sessions WHERE last_seen_at < ?1")
            .bind::<diesel::sql_types::BigInt, _>(expire_before)
            .execute(&mut self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;
        Ok(count)
    }

    fn clear_all_sessions(&mut self) -> StorageResult<usize> {
        let count = diesel::sql_query("DELETE FROM sessions")
            .execute(&mut self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;
        Ok(count)
    }
}

/// Sanitize a user query for FTS5 MATCH syntax.
///
/// Splits the query into words, wraps each in double quotes to treat as literal
/// terms, and joins with spaces. This prevents FTS5 operator injection (AND, OR,
/// NOT, NEAR) and handles special characters safely.
fn sanitize_fts_query(query: &str) -> String {
    query
        .split_whitespace()
        .filter(|w| !w.is_empty())
        .map(|w| format!("\"{}\"", w.replace('"', "")))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{Preference, Value};

    #[test]
    fn save_and_load_memory() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let mem = Memory::with_tags("Test memory", vec!["work".into(), "rust".into()]);
        let id = mem.id;

        storage.save_memory(&mem).unwrap();

        let loaded = storage.load_memory(&id).unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.content, "Test memory");
        assert_eq!(loaded.tags, vec!["work", "rust"]);
    }

    #[test]
    fn save_and_load_identity() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let identity = Identity::new()
            .with_persona("Porter", "A test assistant")
            .with_value(Value::new("Test value"))
            .with_preference(Preference::new("Rust").over("JavaScript"));

        storage.save_identity(&identity).unwrap();

        let loaded = storage.load_identity().unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.persona.name, "Porter");
    }

    #[test]
    fn save_and_load_association() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let e1 = Memory::new("Memory 1");
        let e2 = Memory::new("Memory 2");
        let assoc = Association::with_weight(e1.id, e2.id, 0.8);

        storage.save_association(&assoc).unwrap();

        let loaded = storage.load_associations_from(&e1.id).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].to, e2.id);
        assert!((loaded[0].weight - 0.8).abs() < 0.001);
    }

    #[test]
    fn batch_save_memories() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let e1 = Memory::new("Memory 1");
        let e2 = Memory::new("Memory 2");
        let e3 = Memory::new("Memory 3");

        storage.save_memories(&[&e1, &e2, &e3]).unwrap();

        let all = storage.load_all_memories().unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn load_by_state() {
        let mut storage = MemoryStorage::in_memory().unwrap();
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

        let archived_only = storage
            .load_memories_by_state(MemoryState::Archived)
            .unwrap();
        assert_eq!(archived_only.len(), 1);
        assert_eq!(archived_only[0].content, "Archived memory");
    }

    #[test]
    fn load_by_tag() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let work = Memory::with_tags("Work memory", vec!["work".into()]);
        let personal = Memory::with_tags("Personal memory", vec!["personal".into()]);

        storage.save_memory(&work).unwrap();
        storage.save_memory(&personal).unwrap();

        let work_only = storage.load_memories_by_tag("work").unwrap();
        assert_eq!(work_only.len(), 1);
        assert_eq!(work_only[0].content, "Work memory");
    }

    #[test]
    fn config_persistence() {
        // Config is now stored in config.toml, not SQLite.
        // save_config/load_config are no-ops kept for trait compatibility.
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let config = Config {
            decay_rate_per_day: 0.1,
            ..Default::default()
        };

        // save_config is a no-op
        storage.save_config(&config).unwrap();

        // load_config always returns None (config lives in config.toml)
        let loaded = storage.load_config().unwrap();
        assert!(loaded.is_none(), "load_config should return None (config is in config.toml)");
    }

    #[test]
    fn brain_with_diesel() {
        use crate::memory_core::Brain;

        // Create a brain with Diesel backend
        let storage = MemoryStorage::in_memory().unwrap();
        let mut brain = Brain::open(storage, crate::memory_core::Config::default()).unwrap();

        // Create some memories
        let rust_talk = brain
            .create_with_tags(
                "Discussed Rust FFI patterns",
                vec!["work".into(), "rust".into()],
            )
            .unwrap();

        let ffi_fact = brain
            .create_with_tags(
                "TwilioDataCore uses opaque handles",
                vec!["work".into(), "technical".into()],
            )
            .unwrap();

        // Create association
        brain.associate(rust_talk, ffi_fact, 0.8).unwrap();

        // Recall a memory
        let result = brain.recall(rust_talk).unwrap();
        assert!(result.found());
        assert_eq!(result.content(), Some("Discussed Rust FFI patterns"));

        // Find associated
        let associated = brain.find_associated(&rust_talk);
        assert_eq!(associated.len(), 1);

        // Stats
        let stats = brain.stats();
        assert_eq!(stats.total_memories, 2);
    }

    // ==================
    // ORDINAL TESTS
    // ==================

    #[test]
    fn migration_adds_ordinal_column() {
        // Fresh DB should have ordinal column after initialize
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        // Verify by saving an association with ordinal
        let e1 = Memory::new("Step 1");
        let e2 = Memory::new("Step 2");
        let assoc = Association::with_ordinal(e1.id, e2.id, 0.8, Some(1));

        storage.save_association(&assoc).unwrap();
        let loaded = storage.load_associations_from(&e1.id).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].ordinal, Some(1));
    }

    #[test]
    fn ordinal_roundtrip_with_value() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let e1 = Memory::new("Anchor");
        let e2 = Memory::new("Step 1");
        let e3 = Memory::new("Step 2");
        let e4 = Memory::new("Step 3");

        let a1 = Association::with_ordinal(e1.id, e2.id, 0.9, Some(1));
        let a2 = Association::with_ordinal(e1.id, e3.id, 0.9, Some(2));
        let a3 = Association::with_ordinal(e1.id, e4.id, 0.9, Some(3));

        storage.save_association(&a1).unwrap();
        storage.save_association(&a2).unwrap();
        storage.save_association(&a3).unwrap();

        let loaded = storage.load_associations_from(&e1.id).unwrap();
        assert_eq!(loaded.len(), 3);

        // Verify each ordinal value survived roundtrip
        let ordinals: Vec<Option<u32>> = loaded.iter().map(|a| a.ordinal).collect();
        assert!(ordinals.contains(&Some(1)));
        assert!(ordinals.contains(&Some(2)));
        assert!(ordinals.contains(&Some(3)));
    }

    #[test]
    fn ordinal_roundtrip_null() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let e1 = Memory::new("Memory A");
        let e2 = Memory::new("Memory B");

        // Legacy association without ordinal
        let assoc = Association::with_weight(e1.id, e2.id, 0.7);
        assert_eq!(assoc.ordinal, None);

        storage.save_association(&assoc).unwrap();
        let loaded = storage.load_associations_from(&e1.id).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].ordinal, None);
    }

    #[test]
    fn mixed_ordinal_and_null_associations() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let anchor = Memory::new("Procedure anchor");
        let step1 = Memory::new("Step 1");
        let step2 = Memory::new("Step 2");
        let related = Memory::new("Related note");

        // Ordered chain
        let a1 = Association::with_ordinal(anchor.id, step1.id, 0.9, Some(1));
        let a2 = Association::with_ordinal(anchor.id, step2.id, 0.9, Some(2));
        // Unordered (Hebbian-style)
        let a3 = Association::with_weight(anchor.id, related.id, 0.4);

        storage.save_association(&a1).unwrap();
        storage.save_association(&a2).unwrap();
        storage.save_association(&a3).unwrap();

        let loaded = storage.load_associations_from(&anchor.id).unwrap();
        assert_eq!(loaded.len(), 3);

        // Check that we have both ordinal and null associations
        let with_ordinal: Vec<_> = loaded.iter().filter(|a| a.ordinal.is_some()).collect();
        let without_ordinal: Vec<_> = loaded.iter().filter(|a| a.ordinal.is_none()).collect();
        assert_eq!(with_ordinal.len(), 2);
        assert_eq!(without_ordinal.len(), 1);
        assert_eq!(without_ordinal[0].to, related.id);
    }

    #[test]
    fn ordinal_chain_five_steps() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let anchor = Memory::new("Deploy procedure");
        let steps: Vec<Memory> = (1..=5)
            .map(|i| Memory::new(format!("Step {}", i)))
            .collect();

        for (i, step) in steps.iter().enumerate() {
            let assoc = Association::with_ordinal(anchor.id, step.id, 0.9, Some((i + 1) as u32));
            storage.save_association(&assoc).unwrap();
        }

        let loaded = storage.load_associations_from(&anchor.id).unwrap();
        assert_eq!(loaded.len(), 5);

        // Verify all ordinals 1-5 are present
        let mut ordinals: Vec<u32> = loaded.iter().filter_map(|a| a.ordinal).collect();
        ordinals.sort();
        assert_eq!(ordinals, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn ordinal_survives_batch_save() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let e1 = Memory::new("A");
        let e2 = Memory::new("B");
        let e3 = Memory::new("C");

        let a1 = Association::with_ordinal(e1.id, e2.id, 0.8, Some(1));
        let a2 = Association::with_ordinal(e1.id, e3.id, 0.8, Some(2));

        storage.save_associations(&[&a1, &a2]).unwrap();

        let loaded = storage.load_associations_from(&e1.id).unwrap();
        assert_eq!(loaded.len(), 2);

        let ordinals: Vec<Option<u32>> = loaded.iter().map(|a| a.ordinal).collect();
        assert!(ordinals.contains(&Some(1)));
        assert!(ordinals.contains(&Some(2)));
    }

    #[test]
    fn load_all_associations_includes_ordinal() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let e1 = Memory::new("X");
        let e2 = Memory::new("Y");

        let assoc = Association::with_ordinal(e1.id, e2.id, 0.7, Some(42));
        storage.save_association(&assoc).unwrap();

        let all = storage.load_all_associations().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].ordinal, Some(42));
    }

    // ==================
    // FTS5 TESTS
    // ==================

    #[test]
    fn fts_table_created_on_init() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        // Verify the FTS5 table exists by querying it
        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }

        let result: CountResult = diesel::sql_query("SELECT COUNT(*) as cnt FROM memory_fts")
            .get_result(&mut storage.conn)
            .expect("memory_fts table should exist after init");
        assert_eq!(result.cnt, 0);
    }

    #[test]
    fn fts_populated_on_create() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let mem = Memory::new("Rust programming language features");
        storage.save_memory(&mem).unwrap();

        // Check FTS5 has the entry
        #[derive(QueryableByName)]
        struct FtsCheck {
            #[diesel(sql_type = diesel::sql_types::Text)]
            memory_id: String,
        }

        let rows: Vec<FtsCheck> =
            diesel::sql_query("SELECT memory_id FROM memory_fts WHERE memory_fts MATCH '\"Rust\"'")
                .load(&mut storage.conn)
                .unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].memory_id, mem.id.to_string());
    }

    #[test]
    fn fts_cleaned_on_delete() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let mem = Memory::new("Temporary memory for testing");
        let id = mem.id;
        storage.save_memory(&mem).unwrap();

        // Verify it's in FTS
        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }

        let before: CountResult = diesel::sql_query("SELECT COUNT(*) as cnt FROM memory_fts")
            .get_result(&mut storage.conn)
            .unwrap();
        assert_eq!(before.cnt, 1);

        // Delete the memory
        storage.delete_memory(&id).unwrap();

        // FTS should be cleaned
        let after: CountResult = diesel::sql_query("SELECT COUNT(*) as cnt FROM memory_fts")
            .get_result(&mut storage.conn)
            .unwrap();
        assert_eq!(after.cnt, 0);
    }

    #[test]
    fn keyword_search_basic() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        storage
            .save_memory(&Memory::new("Rust programming language"))
            .unwrap();
        storage
            .save_memory(&Memory::new("Python programming language"))
            .unwrap();
        storage
            .save_memory(&Memory::new("Cooking delicious pasta"))
            .unwrap();

        let results = storage.keyword_search("programming", 10).unwrap();
        assert_eq!(results.len(), 2, "Should find both programming memories");

        // Verify content matches
        let contents: Vec<&str> = results.iter().map(|r| r.content.as_str()).collect();
        assert!(contents.iter().any(|c| c.contains("Rust")));
        assert!(contents.iter().any(|c| c.contains("Python")));
    }

    #[test]
    fn keyword_search_stemming() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        storage
            .save_memory(&Memory::new("Researching machine learning techniques"))
            .unwrap();

        // "research" should match "Researching" via porter stemmer
        let results = storage.keyword_search("research", 10).unwrap();
        assert_eq!(
            results.len(),
            1,
            "Porter stemmer should match 'research' to 'Researching'"
        );
    }

    #[test]
    fn keyword_search_no_results() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        storage
            .save_memory(&Memory::new("Rust programming"))
            .unwrap();

        let results = storage.keyword_search("quantum physics", 10).unwrap();
        assert!(results.is_empty(), "No memories about quantum physics");
    }

    #[test]
    fn keyword_search_empty_query() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        storage.save_memory(&Memory::new("Some content")).unwrap();

        let results = storage.keyword_search("", 10).unwrap();
        assert!(results.is_empty(), "Empty query should return no results");
    }

    #[test]
    fn fts_updated_on_content_change() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let mut mem = Memory::new("Original content about dogs");
        storage.save_memory(&mem).unwrap();

        // Should find "dogs"
        let results = storage.keyword_search("dogs", 10).unwrap();
        assert_eq!(results.len(), 1);

        // Update content
        mem.content = "Updated content about cats".to_string();
        storage.save_memory(&mem).unwrap();

        // Should no longer find "dogs"
        let results = storage.keyword_search("dogs", 10).unwrap();
        assert!(results.is_empty(), "Old content should not be findable");

        // Should find "cats"
        let results = storage.keyword_search("cats", 10).unwrap();
        assert_eq!(results.len(), 1, "New content should be findable");
    }

    #[test]
    fn ensure_fts_populated_backfills() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        // Create memories directly in the memories table without FTS sync
        // by using raw SQL to bypass the save_memory FTS logic
        let e1 = Memory::new("Memory about Rust");
        let e2 = Memory::new("Memory about Python");

        // Save normally (which populates FTS)
        storage.save_memory(&e1).unwrap();
        storage.save_memory(&e2).unwrap();

        // Clear FTS manually to simulate pre-FTS data
        diesel::sql_query("DELETE FROM memory_fts")
            .execute(&mut storage.conn)
            .unwrap();

        // Verify FTS is empty
        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }

        let count: CountResult = diesel::sql_query("SELECT COUNT(*) as cnt FROM memory_fts")
            .get_result(&mut storage.conn)
            .unwrap();
        assert_eq!(count.cnt, 0, "FTS should be empty after manual clear");

        // Run backfill
        let backfilled = storage.ensure_fts_populated().unwrap();
        assert_eq!(backfilled, 2, "Should backfill 2 memories");

        // Verify search works after backfill
        let results = storage.keyword_search("Rust", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Rust"));
    }

    #[test]
    fn ensure_fts_populated_noop_when_already_populated() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        storage
            .save_memory(&Memory::new("Already indexed"))
            .unwrap();

        // FTS already has data, should be a no-op
        let backfilled = storage.ensure_fts_populated().unwrap();
        assert_eq!(
            backfilled, 0,
            "Should not backfill when FTS already has data"
        );
    }

    #[test]
    fn sanitize_fts_query_handles_special_chars() {
        assert_eq!(sanitize_fts_query("hello world"), "\"hello\" \"world\"");
        assert_eq!(sanitize_fts_query("AND OR NOT"), "\"AND\" \"OR\" \"NOT\"");
        assert_eq!(sanitize_fts_query("test\"injection"), "\"testinjection\"");
        assert_eq!(sanitize_fts_query(""), "");
        assert_eq!(sanitize_fts_query("   "), "");
    }

    #[test]
    fn fts_batch_save_populates() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let e1 = Memory::new("Batch item alpha");
        let e2 = Memory::new("Batch item beta");
        let e3 = Memory::new("Batch item gamma");

        storage.save_memories(&[&e1, &e2, &e3]).unwrap();

        let results = storage.keyword_search("batch", 10).unwrap();
        assert_eq!(results.len(), 3, "All batch-saved memories should be in FTS");
    }

    #[test]
    fn access_log_records_search_recall_cycle() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let e1 = Memory::new("Brandon prefers ale");
        let e2 = Memory::new("Brandon lives in Oregon City");
        let e3 = Memory::new("Brandon is an engineer");

        // Log an access cycle: searched for "drink preference",
        // got all 3 results, recalled only e1
        storage
            .log_access(
                "what does Brandon like to drink",
                &[e1.id, e2.id, e3.id],
                &[e1.id],
            )
            .unwrap();

        // Verify the entry was written
        #[derive(QueryableByName)]
        struct LogRow {
            #[diesel(sql_type = diesel::sql_types::Text)]
            query_text: String,
            #[diesel(sql_type = diesel::sql_types::Text)]
            result_ids: String,
            #[diesel(sql_type = diesel::sql_types::Text)]
            recalled_ids: String,
        }

        let rows: Vec<LogRow> =
            diesel::sql_query("SELECT query_text, result_ids, recalled_ids FROM access_log")
                .load(&mut storage.conn)
                .unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].query_text, "what does Brandon like to drink");

        // Verify result_ids contains all 3
        let result_ids: Vec<String> = serde_json::from_str(&rows[0].result_ids).unwrap();
        assert_eq!(result_ids.len(), 3);

        // Verify recalled_ids contains only e1
        let recalled_ids: Vec<String> = serde_json::from_str(&rows[0].recalled_ids).unwrap();
        assert_eq!(recalled_ids.len(), 1);
        assert_eq!(recalled_ids[0], e1.id.to_string());
    }

    #[test]
    fn access_log_multiple_entries() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let e1 = Memory::new("Memory one");
        let e2 = Memory::new("Memory two");

        storage
            .log_access("first query", &[e1.id], &[e1.id])
            .unwrap();
        storage
            .log_access("second query", &[e1.id, e2.id], &[e2.id])
            .unwrap();

        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }

        let count: CountResult = diesel::sql_query("SELECT COUNT(*) as cnt FROM access_log")
            .get_result(&mut storage.conn)
            .unwrap();

        assert_eq!(count.cnt, 2);
    }

    // ── Migration safety tests ─────────────────────────────────────────────

    /// Helper: create the old `engrams` table schema in a fresh in-memory db.
    fn create_old_engrams_table(storage: &mut MemoryStorage) {
        storage
            .conn
            .batch_execute(
                r#"
                CREATE TABLE engrams (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    energy REAL NOT NULL,
                    state TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at INTEGER NOT NULL,
                    last_accessed INTEGER NOT NULL,
                    access_count INTEGER NOT NULL,
                    tags TEXT NOT NULL,
                    embedding BLOB
                );
            "#,
            )
            .unwrap();
    }

    /// Helper: create the new `memories` table with some data.
    fn create_memories_table_with_data(storage: &mut MemoryStorage, count: usize) {
        storage
            .conn
            .batch_execute(
                r#"
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    energy REAL NOT NULL,
                    state TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at INTEGER NOT NULL,
                    last_accessed INTEGER NOT NULL,
                    access_count INTEGER NOT NULL,
                    tags TEXT NOT NULL,
                    embedding BLOB
                );
            "#,
            )
            .unwrap();
        for i in 0..count {
            let id = uuid::Uuid::new_v4().to_string();
            diesel::sql_query(format!(
                "INSERT INTO memories (id, content, energy, state, confidence, created_at, last_accessed, access_count, tags) \
                 VALUES ('{}', 'memory {}', 1.0, 'Active', 0.8, 1000000, 1000000, 1, '[]')",
                id, i
            ))
            .execute(&mut storage.conn)
            .unwrap();
        }
    }

    /// Helper: insert rows into the engrams table.
    fn insert_engram_rows(storage: &mut MemoryStorage, count: usize) {
        for i in 0..count {
            let id = uuid::Uuid::new_v4().to_string();
            diesel::sql_query(format!(
                "INSERT INTO engrams (id, content, energy, state, confidence, created_at, last_accessed, access_count, tags) \
                 VALUES ('{}', 'engram {}', 1.0, 'Active', 0.8, 1000000, 1000000, 1, '[]')",
                id, i
            ))
            .execute(&mut storage.conn)
            .unwrap();
        }
    }

    /// Helper: check if a table exists.
    fn table_exists(storage: &mut MemoryStorage, name: &str) -> bool {
        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }
        let result: CountResult = diesel::sql_query(format!(
            "SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='table' AND name='{}'",
            name
        ))
        .get_result(&mut storage.conn)
        .unwrap();
        result.cnt > 0
    }

    /// Helper: count rows in a table.
    fn row_count(storage: &mut MemoryStorage, table: &str) -> i32 {
        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }
        diesel::sql_query(format!("SELECT COUNT(*) as cnt FROM {}", table))
            .get_result::<CountResult>(&mut storage.conn)
            .map(|r| r.cnt)
            .unwrap_or(0)
    }

    #[test]
    fn migration_ghost_engrams_preserves_memories() {
        // Scenario: migration already ran (data in memories), then a ghost
        // engrams table reappeared (e.g. Syncthing sync). Must NOT drop memories.
        let mut storage = MemoryStorage::in_memory().unwrap();
        create_memories_table_with_data(&mut storage, 50);
        create_old_engrams_table(&mut storage);  // empty ghost

        assert_eq!(row_count(&mut storage, "memories"), 50);
        assert_eq!(row_count(&mut storage, "engrams"), 0);

        storage.initialize().unwrap();

        // memories must still have all 50 rows
        assert!(table_exists(&mut storage, "memories"));
        assert_eq!(row_count(&mut storage, "memories"), 50);
        // engrams ghost must be gone
        assert!(!table_exists(&mut storage, "engrams"));
    }

    #[test]
    fn migration_normal_rename_works() {
        // Scenario: first run after rename. engrams has data, memories doesn't exist.
        let mut storage = MemoryStorage::in_memory().unwrap();
        create_old_engrams_table(&mut storage);
        insert_engram_rows(&mut storage, 25);

        assert_eq!(row_count(&mut storage, "engrams"), 25);

        storage.initialize().unwrap();

        assert!(table_exists(&mut storage, "memories"));
        assert!(!table_exists(&mut storage, "engrams"));
        assert_eq!(row_count(&mut storage, "memories"), 25);
    }

    #[test]
    fn migration_partial_drops_empty_memories_and_renames() {
        // Scenario: partial previous migration left an empty memories table,
        // and engrams still has the data.
        let mut storage = MemoryStorage::in_memory().unwrap();
        create_old_engrams_table(&mut storage);
        insert_engram_rows(&mut storage, 30);
        create_memories_table_with_data(&mut storage, 0); // empty memories

        assert_eq!(row_count(&mut storage, "engrams"), 30);
        assert_eq!(row_count(&mut storage, "memories"), 0);

        storage.initialize().unwrap();

        assert!(table_exists(&mut storage, "memories"));
        assert!(!table_exists(&mut storage, "engrams"));
        assert_eq!(row_count(&mut storage, "memories"), 30);
    }

    #[test]
    fn migration_both_have_data_returns_error() {
        // Scenario: somehow both tables have data. Must refuse to migrate.
        let mut storage = MemoryStorage::in_memory().unwrap();
        create_old_engrams_table(&mut storage);
        insert_engram_rows(&mut storage, 10);
        create_memories_table_with_data(&mut storage, 20);

        let result = storage.initialize();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("MIGRATION CONFLICT"),
            "Expected MIGRATION CONFLICT error, got: {}",
            err_msg
        );
    }

    #[test]
    fn migration_no_engrams_table_is_noop() {
        // Scenario: clean install, no engrams table ever existed.
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        assert!(table_exists(&mut storage, "memories"));
        assert!(!table_exists(&mut storage, "engrams"));
    }

    #[test]
    fn migration_fts_table_rebuilt() {
        // Scenario: old engram_fts exists with data, migration should rebuild as memory_fts.
        let mut storage = MemoryStorage::in_memory().unwrap();
        create_old_engrams_table(&mut storage);
        insert_engram_rows(&mut storage, 5);

        // Create old FTS table with data
        storage.conn.batch_execute(r#"
            CREATE VIRTUAL TABLE engram_fts USING fts5(
                content,
                engram_id UNINDEXED,
                tokenize='porter unicode61'
            );
            INSERT INTO engram_fts(content, engram_id) SELECT content, id FROM engrams;
        "#).unwrap();

        assert!(table_exists(&mut storage, "engram_fts"));
        assert_eq!(row_count(&mut storage, "engram_fts"), 5);

        storage.initialize().unwrap();

        // engram_fts should be gone, memory_fts should exist with data
        assert!(!table_exists(&mut storage, "engram_fts"));
        assert!(table_exists(&mut storage, "memory_fts"));
        assert_eq!(row_count(&mut storage, "memory_fts"), 5);
    }

    #[test]
    fn migration_enrichments_table_renamed() {
        // Scenario: old engram_enrichments exists with data, should become memory_enrichments.
        let mut storage = MemoryStorage::in_memory().unwrap();
        create_old_engrams_table(&mut storage);
        insert_engram_rows(&mut storage, 3);

        // Create old enrichments table with data
        storage.conn.batch_execute(r#"
            CREATE TABLE engram_enrichments (
                engram_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                source TEXT NOT NULL DEFAULT 'llm',
                created_at INTEGER NOT NULL,
                PRIMARY KEY (engram_id, seq)
            );
        "#).unwrap();

        // Get an engram id to use as FK
        #[derive(QueryableByName)]
        struct IdRow {
            #[diesel(sql_type = diesel::sql_types::Text)]
            id: String,
        }
        let ids: Vec<IdRow> = diesel::sql_query("SELECT id FROM engrams LIMIT 1")
            .load(&mut storage.conn).unwrap();
        let eid = &ids[0].id;

        diesel::sql_query(
            "INSERT INTO engram_enrichments (engram_id, seq, embedding, source, created_at) VALUES (?1, 1, X'00', 'llm', 1000000)"
        )
        .bind::<diesel::sql_types::Text, _>(eid)
        .execute(&mut storage.conn).unwrap();

        assert!(table_exists(&mut storage, "engram_enrichments"));
        assert_eq!(row_count(&mut storage, "engram_enrichments"), 1);

        storage.initialize().unwrap();

        assert!(!table_exists(&mut storage, "engram_enrichments"));
        assert!(table_exists(&mut storage, "memory_enrichments"));
        assert_eq!(row_count(&mut storage, "memory_enrichments"), 1);
    }

    #[test]
    fn migration_column_rename_engram_id_to_memory_id() {
        // Scenario: memory_enrichments exists with old engram_id column name.
        // Need an engrams table so the main migration triggers, which also
        // checks for the column rename.
        let mut storage = MemoryStorage::in_memory().unwrap();
        create_old_engrams_table(&mut storage);
        insert_engram_rows(&mut storage, 2);

        // Create memory_enrichments with old column name
        storage.conn.batch_execute(r#"
            CREATE TABLE memory_enrichments (
                engram_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                source TEXT NOT NULL DEFAULT 'llm',
                created_at INTEGER NOT NULL,
                PRIMARY KEY (engram_id, seq)
            );
            INSERT INTO memory_enrichments (engram_id, seq, embedding, source, created_at)
                VALUES ('test-id', 1, X'00', 'llm', 1000000);
        "#).unwrap();

        // Verify old column name exists
        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }
        let old_col: CountResult = diesel::sql_query(
            "SELECT COUNT(*) as cnt FROM pragma_table_info('memory_enrichments') WHERE name = 'engram_id'"
        ).get_result(&mut storage.conn).unwrap();
        assert_eq!(old_col.cnt, 1, "engram_id column should exist before migration");

        storage.initialize().unwrap();

        // Verify column was renamed
        let new_col: CountResult = diesel::sql_query(
            "SELECT COUNT(*) as cnt FROM pragma_table_info('memory_enrichments') WHERE name = 'memory_id'"
        ).get_result(&mut storage.conn).unwrap();
        assert_eq!(new_col.cnt, 1, "memory_id column should exist after migration");

        let old_col: CountResult = diesel::sql_query(
            "SELECT COUNT(*) as cnt FROM pragma_table_info('memory_enrichments') WHERE name = 'engram_id'"
        ).get_result(&mut storage.conn).unwrap();
        assert_eq!(old_col.cnt, 0, "engram_id column should be gone after migration");

        // Data should be preserved
        assert_eq!(row_count(&mut storage, "memory_enrichments"), 1);
    }

    #[test]
    fn migration_idempotent_double_initialize() {
        // Scenario: initialize() called twice. Second run should succeed
        // without errors and data should remain intact.
        let mut storage = MemoryStorage::in_memory().unwrap();

        // First initialize — creates schema from scratch
        storage.initialize().unwrap();

        // Insert some data between runs
        let m = Memory::new("Idempotency test memory");
        storage.save_memory(&m).unwrap();
        assert_eq!(row_count(&mut storage, "memories"), 1);

        // Second initialize — should be a no-op, not error
        storage.initialize().unwrap();

        // Data should still be intact
        assert_eq!(row_count(&mut storage, "memories"), 1);
        let loaded = storage.load_memory(&m.id).unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().content, "Idempotency test memory");
    }

    /// Helper: check if an index exists.
    fn index_exists(storage: &mut MemoryStorage, name: &str) -> bool {
        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }
        let result: CountResult = diesel::sql_query(format!(
            "SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='index' AND name='{}'",
            name
        ))
        .get_result(&mut storage.conn)
        .unwrap();
        result.cnt > 0
    }

    #[test]
    fn migration_index_cleanup() {
        // Scenario: old engram-named indexes exist, migration should drop them
        // and create_schema should recreate them with new names.
        let mut storage = MemoryStorage::in_memory().unwrap();
        create_old_engrams_table(&mut storage);
        insert_engram_rows(&mut storage, 3);

        // Create old indexes
        storage.conn.batch_execute(r#"
            CREATE INDEX idx_engrams_state ON engrams(state);
            CREATE INDEX idx_engrams_energy ON engrams(energy);
        "#).unwrap();

        assert!(index_exists(&mut storage, "idx_engrams_state"));
        assert!(index_exists(&mut storage, "idx_engrams_energy"));

        storage.initialize().unwrap();

        // Old indexes should be gone
        assert!(!index_exists(&mut storage, "idx_engrams_state"));
        assert!(!index_exists(&mut storage, "idx_engrams_energy"));
        // New indexes should exist (created by create_schema)
        assert!(index_exists(&mut storage, "idx_memories_state"));
        assert!(index_exists(&mut storage, "idx_memories_energy"));
    }

    // ==================
    // VEC0 SYNC TESTS
    // ==================

    /// Helper: create a 2048-dim embedding (matches vec0 table schema).
    fn make_embedding(seed: f32) -> Vec<f32> {
        (0..2048).map(|i| ((i as f32) * seed).sin()).collect()
    }

    /// Helper: count rows in memory_vec for a given memory_id.
    fn memory_vec_count(storage: &mut MemoryStorage, memory_id: &str) -> i32 {
        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }
        diesel::sql_query("SELECT COUNT(*) as cnt FROM memory_vec WHERE memory_id = ?")
            .bind::<diesel::sql_types::Text, _>(memory_id)
            .get_result::<CountResult>(&mut storage.conn)
            .map(|r| r.cnt)
            .unwrap_or(0)
    }

    /// Helper: count rows in enrichment_vec for a given memory_id.
    fn enrichment_vec_count(storage: &mut MemoryStorage, memory_id: &str) -> i32 {
        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }
        diesel::sql_query("SELECT COUNT(*) as cnt FROM enrichment_vec WHERE memory_id = ?")
            .bind::<diesel::sql_types::Text, _>(memory_id)
            .get_result::<CountResult>(&mut storage.conn)
            .map(|r| r.cnt)
            .unwrap_or(0)
    }

    #[test]
    fn vec0_save_memory_with_embedding_populates_memory_vec() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let mut mem = Memory::new("Rust is great");
        mem.embedding = Some(make_embedding(1.0));
        let id_str = mem.id.to_string();

        storage.save_memory(&mem).unwrap();

        assert_eq!(memory_vec_count(&mut storage, &id_str), 1);
    }

    #[test]
    fn vec0_save_memory_without_embedding_has_no_vec_row() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let mem = Memory::new("No embedding here");
        let id_str = mem.id.to_string();

        storage.save_memory(&mem).unwrap();

        assert_eq!(memory_vec_count(&mut storage, &id_str), 0);
    }

    #[test]
    fn vec0_save_memory_replaces_existing_vec_row() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let mut mem = Memory::new("First embedding");
        mem.embedding = Some(make_embedding(1.0));
        let id_str = mem.id.to_string();

        storage.save_memory(&mem).unwrap();
        assert_eq!(memory_vec_count(&mut storage, &id_str), 1);

        // Update embedding and re-save
        mem.embedding = Some(make_embedding(2.0));
        storage.save_memory(&mem).unwrap();
        assert_eq!(
            memory_vec_count(&mut storage, &id_str),
            1,
            "Should still be 1 row after update"
        );
    }

    #[test]
    fn vec0_save_memory_removes_vec_row_when_embedding_cleared() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let mut mem = Memory::new("Had embedding");
        mem.embedding = Some(make_embedding(1.0));
        let id_str = mem.id.to_string();

        storage.save_memory(&mem).unwrap();
        assert_eq!(memory_vec_count(&mut storage, &id_str), 1);

        // Clear embedding and re-save
        mem.embedding = None;
        storage.save_memory(&mem).unwrap();
        assert_eq!(
            memory_vec_count(&mut storage, &id_str),
            0,
            "Vec row should be removed when embedding is cleared"
        );
    }

    #[test]
    fn vec0_delete_memory_removes_from_both_vec_tables() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let mut mem = Memory::new("Will be deleted");
        mem.embedding = Some(make_embedding(1.0));
        let id_str = mem.id.to_string();

        storage.save_memory(&mem).unwrap();
        assert_eq!(memory_vec_count(&mut storage, &id_str), 1);

        // Add enrichment vectors
        {
            let mut vs = VectorSearch::new(storage.connection());
            vs.set_enrichment_embeddings(&mem.id, &[make_embedding(3.0)], "llm")
                .unwrap();
        }
        assert_eq!(enrichment_vec_count(&mut storage, &id_str), 1);

        // Delete the memory
        storage.delete_memory(&mem.id).unwrap();

        assert_eq!(memory_vec_count(&mut storage, &id_str), 0);
        assert_eq!(enrichment_vec_count(&mut storage, &id_str), 0);
    }

    #[test]
    fn vec0_set_embedding_populates_memory_vec() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        // Create memory without embedding
        let mem = Memory::new("Get embedding later");
        let id_str = mem.id.to_string();
        storage.save_memory(&mem).unwrap();
        assert_eq!(memory_vec_count(&mut storage, &id_str), 0);

        // Set embedding via VectorSearch
        {
            let mut vs = VectorSearch::new(storage.connection());
            vs.set_embedding(&mem.id, &make_embedding(1.0)).unwrap();
        }
        assert_eq!(memory_vec_count(&mut storage, &id_str), 1);
    }

    #[test]
    fn vec0_clear_all_embeddings_empties_memory_vec() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let mut e1 = Memory::new("First");
        e1.embedding = Some(make_embedding(1.0));
        let mut e2 = Memory::new("Second");
        e2.embedding = Some(make_embedding(2.0));
        storage.save_memory(&e1).unwrap();
        storage.save_memory(&e2).unwrap();
        assert_eq!(row_count(&mut storage, "memory_vec"), 2);

        {
            let mut vs = VectorSearch::new(storage.connection());
            vs.clear_all_embeddings().unwrap();
        }
        assert_eq!(row_count(&mut storage, "memory_vec"), 0);
    }

    #[test]
    fn vec0_set_enrichment_embeddings_populates_enrichment_vec() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let mem = Memory::new("Enrichable");
        let id_str = mem.id.to_string();
        storage.save_memory(&mem).unwrap();

        {
            let mut vs = VectorSearch::new(storage.connection());
            vs.set_enrichment_embeddings(
                &mem.id,
                &[make_embedding(1.0), make_embedding(2.0)],
                "llm",
            )
            .unwrap();
        }
        assert_eq!(enrichment_vec_count(&mut storage, &id_str), 2);
    }

    #[test]
    fn vec0_re_enrichment_replaces_enrichment_vec_rows() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let mem = Memory::new("Re-enrich me");
        let id_str = mem.id.to_string();
        storage.save_memory(&mem).unwrap();

        // First enrichment: 3 vectors
        {
            let mut vs = VectorSearch::new(storage.connection());
            vs.set_enrichment_embeddings(
                &mem.id,
                &[make_embedding(1.0), make_embedding(2.0), make_embedding(3.0)],
                "llm",
            )
            .unwrap();
        }
        assert_eq!(enrichment_vec_count(&mut storage, &id_str), 3);

        // Re-enrich: 1 vector — old ones should be gone
        {
            let mut vs = VectorSearch::new(storage.connection());
            vs.set_enrichment_embeddings(&mem.id, &[make_embedding(4.0)], "llm")
                .unwrap();
        }
        assert_eq!(
            enrichment_vec_count(&mut storage, &id_str),
            1,
            "Old enrichment_vec rows should be replaced"
        );
    }

    #[test]
    fn vec0_delete_enrichments_clears_enrichment_vec() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let mem = Memory::new("Delete enrichments");
        let id_str = mem.id.to_string();
        storage.save_memory(&mem).unwrap();

        {
            let mut vs = VectorSearch::new(storage.connection());
            vs.set_enrichment_embeddings(&mem.id, &[make_embedding(1.0)], "llm")
                .unwrap();
        }
        assert_eq!(enrichment_vec_count(&mut storage, &id_str), 1);

        {
            let mut vs = VectorSearch::new(storage.connection());
            vs.delete_enrichments(&mem.id).unwrap();
        }
        assert_eq!(enrichment_vec_count(&mut storage, &id_str), 0);
    }

    #[test]
    fn vec0_clear_all_enrichments_empties_enrichment_vec() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let e1 = Memory::new("First");
        let e2 = Memory::new("Second");
        storage.save_memory(&e1).unwrap();
        storage.save_memory(&e2).unwrap();

        {
            let mut vs = VectorSearch::new(storage.connection());
            vs.set_enrichment_embeddings(&e1.id, &[make_embedding(1.0)], "llm")
                .unwrap();
            vs.set_enrichment_embeddings(&e2.id, &[make_embedding(2.0)], "llm")
                .unwrap();
        }
        assert_eq!(row_count(&mut storage, "enrichment_vec"), 2);

        {
            let mut vs = VectorSearch::new(storage.connection());
            vs.clear_all_enrichments().unwrap();
        }
        assert_eq!(row_count(&mut storage, "enrichment_vec"), 0);
    }

    #[test]
    fn vec0_migration_backfills_from_existing_embeddings() {
        // Simulate a database that has embeddings but empty vec0 tables
        // (like upgrading from a pre-sqlite-vec version).
        let mut storage = MemoryStorage::in_memory().unwrap();

        // Manually create the old schema without vec0 tables
        storage.conn.batch_execute(r#"
            CREATE TABLE IF NOT EXISTS identity (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS config (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                energy REAL NOT NULL,
                state TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at INTEGER NOT NULL,
                last_accessed INTEGER NOT NULL,
                access_count INTEGER NOT NULL,
                tags TEXT NOT NULL,
                embedding BLOB
            );
            CREATE TABLE IF NOT EXISTS associations (
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                weight REAL NOT NULL,
                created_at INTEGER NOT NULL,
                last_activated INTEGER NOT NULL,
                co_activation_count INTEGER NOT NULL,
                PRIMARY KEY (from_id, to_id)
            );
            CREATE TABLE IF NOT EXISTS memory_enrichments (
                memory_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                source TEXT NOT NULL DEFAULT 'llm',
                created_at INTEGER NOT NULL,
                PRIMARY KEY (memory_id, seq),
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                content,
                memory_id UNINDEXED,
                tokenize='porter unicode61'
            );
        "#).unwrap();

        // Insert a memory with a 2048-dim embedding directly
        let id1 = uuid::Uuid::new_v4().to_string();
        let emb1 = make_embedding(1.0);
        let bytes1 = embedding_to_bytes(&emb1);
        diesel::sql_query(
            "INSERT INTO memories (id, content, energy, state, confidence, created_at, last_accessed, access_count, tags, embedding) \
             VALUES (?, 'test memory', 1.0, 'active', 0.8, 1000000, 1000000, 1, '[]', ?)"
        )
        .bind::<diesel::sql_types::Text, _>(&id1)
        .bind::<diesel::sql_types::Binary, _>(&bytes1[..])
        .execute(&mut storage.conn)
        .unwrap();

        // Insert an enrichment embedding
        let enr_bytes = embedding_to_bytes(&make_embedding(2.0));
        diesel::sql_query(
            "INSERT INTO memory_enrichments (memory_id, seq, embedding, source, created_at) \
             VALUES (?, 0, ?, 'llm', 1000000)"
        )
        .bind::<diesel::sql_types::Text, _>(&id1)
        .bind::<diesel::sql_types::Binary, _>(&enr_bytes[..])
        .execute(&mut storage.conn)
        .unwrap();

        // Now call initialize() — it should create vec0 tables and backfill
        storage.initialize().unwrap();

        assert!(table_exists(&mut storage, "memory_vec"));
        assert!(table_exists(&mut storage, "enrichment_vec"));
        assert_eq!(memory_vec_count(&mut storage, &id1), 1);
        assert_eq!(enrichment_vec_count(&mut storage, &id1), 1);
    }

    #[test]
    fn vec0_tables_created_on_fresh_init() {
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        assert!(table_exists(&mut storage, "memory_vec"));
        assert!(table_exists(&mut storage, "enrichment_vec"));
    }

    #[test]
    fn vec0_small_dimension_embeddings_are_best_effort() {
        // Verify that saving a memory with a non-2048-dim embedding
        // succeeds (the primary save works, vec0 insert is silently skipped).
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let mut mem = Memory::new("Small embedding");
        mem.embedding = Some(vec![1.0, 0.0, 0.0]); // 3-dim, won't fit vec0
        let id_str = mem.id.to_string();

        // Should not error — vec0 insert is best-effort
        storage.save_memory(&mem).unwrap();

        // Primary storage should still have the embedding (load via VectorSearch)
        let mut vs = VectorSearch::new(storage.connection());
        let emb = vs.get_embedding(&mem.id).unwrap();
        assert!(emb.is_some(), "Primary embedding column should be populated");
        assert_eq!(emb.unwrap().len(), 3);

        // But vec0 should NOT have it (dimension mismatch)
        drop(vs);
        assert_eq!(
            memory_vec_count(&mut storage, &id_str),
            0,
            "vec0 should reject mismatched dimensions silently"
        );
    }

    #[test]
    fn vec0_non_2048_replacement_clears_vec_row() {
        // A memory starts with a 2048-dim embedding (synced to vec0), then
        // gets replaced with a 3-dim embedding (model migration or test).
        // The old vec0 row must be removed — otherwise searches return stale
        // results that don't match the primary embedding blob.
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let mut mem = Memory::new("Will change dimension");
        mem.embedding = Some(make_embedding(1.0)); // 2048-dim
        let id_str = mem.id.to_string();

        storage.save_memory(&mem).unwrap();
        assert_eq!(memory_vec_count(&mut storage, &id_str), 1, "2048-dim should populate vec0");

        // Replace with a 3-dim embedding (simulates model migration or test fixture)
        let small_embedding = vec![0.5, 0.3, 0.1];
        mem.embedding = Some(small_embedding.clone());
        storage.save_memory(&mem).unwrap();

        // vec0 must NOT have the old row
        assert_eq!(
            memory_vec_count(&mut storage, &id_str),
            0,
            "vec0 row must be removed when embedding dimension changes away from 2048"
        );

        // Primary blob must hold the new 3-dim embedding
        let mut vs = VectorSearch::new(storage.connection());
        let loaded = vs.get_embedding(&mem.id).unwrap().expect("Primary embedding should exist");
        assert_eq!(loaded.len(), 3);
        assert!((loaded[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn vec0_non_2048_replacement_via_set_embedding_clears_vec_row() {
        // Same test but using the VectorSearch::set_embedding path directly.
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let mem = Memory::new("Via set_embedding");
        let id_str = mem.id.to_string();
        storage.save_memory(&mem).unwrap();

        // Set a 2048-dim embedding
        {
            let mut vs = VectorSearch::new(storage.connection());
            vs.set_embedding(&mem.id, &make_embedding(2.0)).unwrap();
        }
        assert_eq!(memory_vec_count(&mut storage, &id_str), 1);

        // Replace with a 3-dim embedding
        let small = vec![1.0, 0.0, 0.0];
        {
            let mut vs = VectorSearch::new(storage.connection());
            vs.set_embedding(&mem.id, &small).unwrap();
        }
        assert_eq!(
            memory_vec_count(&mut storage, &id_str),
            0,
            "set_embedding with non-2048 dims must remove old vec0 row"
        );

        // Primary blob must hold the new value
        let mut vs = VectorSearch::new(storage.connection());
        let loaded = vs.get_embedding(&mem.id).unwrap().expect("should exist");
        assert_eq!(loaded.len(), 3);
    }

    #[test]
    fn vec0_high_enrichment_fan_out_equivalence() {
        // A single memory with many enrichment vectors should have exact
        // row-count parity between memory_enrichments and enrichment_vec,
        // and search must find the memory via enrichment similarity.
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let mem = Memory::new("Memory with many enrichment vectors");
        let id_str = mem.id.to_string();
        storage.save_memory(&mem).unwrap();

        // Create 20 distinct enrichment vectors (high fan-out)
        let enrichments: Vec<Vec<f32>> = (0..20)
            .map(|i| make_embedding(0.1 * (i as f32) + 0.5))
            .collect();

        {
            let mut vs = VectorSearch::new(storage.connection());
            vs.set_enrichment_embeddings(&mem.id, &enrichments, "llm")
                .unwrap();
        }

        // Row-count parity: enrichment_vec should have exactly 20 rows
        assert_eq!(
            enrichment_vec_count(&mut storage, &id_str),
            20,
            "enrichment_vec must match enrichment count for high fan-out"
        );

        // Verify the primary enrichments table also has 20
        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }
        let primary_count: i32 =
            diesel::sql_query("SELECT COUNT(*) as cnt FROM memory_enrichments WHERE memory_id = ?")
                .bind::<diesel::sql_types::Text, _>(&id_str)
                .get_result::<CountResult>(&mut storage.conn)
                .map(|r| r.cnt)
                .unwrap_or(0);
        assert_eq!(primary_count, 20, "memory_enrichments must also have 20 rows");

        // Search using one of the enrichment vectors — should find the memory
        {
            let mut vs = VectorSearch::new(storage.connection());
            let results = vs.find_similar(&enrichments[0], 5, 0.0).unwrap();
            assert!(
                results.iter().any(|r| r.id == mem.id),
                "Search via enrichment vector must find the parent memory"
            );
        }
    }

    #[test]
    fn vec0_enrichment_non_2048_replacement_after_2048() {
        // Enrichments start as 2048-dim (synced to enrichment_vec), then
        // get replaced with 3-dim vectors. The old enrichment_vec rows must
        // be cleaned up — no stale entries.
        let mut storage = MemoryStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let mem = Memory::new("Enrichment dimension change");
        let id_str = mem.id.to_string();
        storage.save_memory(&mem).unwrap();

        // Write 5 enrichment vectors at 2048 dimensions
        let big_enrichments: Vec<Vec<f32>> = (0..5)
            .map(|i| make_embedding(1.0 + i as f32))
            .collect();
        {
            let mut vs = VectorSearch::new(storage.connection());
            vs.set_enrichment_embeddings(&mem.id, &big_enrichments, "llm")
                .unwrap();
        }
        assert_eq!(enrichment_vec_count(&mut storage, &id_str), 5);

        // Replace with 3-dim enrichment vectors (model migration)
        let small_enrichments: Vec<Vec<f32>> = (0..3)
            .map(|i| vec![i as f32, 0.0, 1.0])
            .collect();
        {
            let mut vs = VectorSearch::new(storage.connection());
            vs.set_enrichment_embeddings(&mem.id, &small_enrichments, "llm")
                .unwrap();
        }

        // All old vec0 rows must be gone, and no new ones inserted (wrong dims)
        assert_eq!(
            enrichment_vec_count(&mut storage, &id_str),
            0,
            "enrichment_vec must be empty after replacing 2048-dim with 3-dim"
        );

        // Primary table should have the 3 new enrichments
        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }
        let primary_count: i32 =
            diesel::sql_query("SELECT COUNT(*) as cnt FROM memory_enrichments WHERE memory_id = ?")
                .bind::<diesel::sql_types::Text, _>(&id_str)
                .get_result::<CountResult>(&mut storage.conn)
                .map(|r| r.cnt)
                .unwrap_or(0);
        assert_eq!(primary_count, 3, "primary enrichments should have the 3 new rows");
    }
}
