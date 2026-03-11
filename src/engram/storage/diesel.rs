//! Diesel-backed storage implementation for Engrams
//!
//! This module provides database storage using Diesel ORM.
//! The actual database backend (SQLite, Postgres, MySQL) is selected
//! via feature flags at compile time.

use super::models::*;
use super::schema::{associations, config, engrams, identity, metadata};
use super::{SimilarityResult, Storage, StorageError, StorageResult};
use crate::engram::{Association, Config, Engram, EngramId, Identity, MemoryState};

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

/// Database-backed storage implementation for Engrams
///
/// The underlying database is selected at compile time via feature flags:
/// - `sqlite` (default) - Uses SQLite, suitable for local/embedded use
/// - `postgres` - Uses PostgreSQL, suitable for server/SaaS deployments
pub struct EngramStorage {
    conn: DbConnection,
}

impl EngramStorage {
    /// Open storage at the given connection string/path
    ///
    /// For SQLite: Pass a file path (e.g., "/path/to/brain.db")
    /// For Postgres: Pass a connection URL (e.g., "postgres://user:pass@host/db")
    #[cfg(feature = "sqlite")]
    pub fn open<P: AsRef<Path>>(path: P) -> StorageResult<Self> {
        let path_str = path.as_ref().to_string_lossy();
        let mut conn = DbConnection::establish(&path_str)
            .map_err(|e| StorageError::Database(e.to_string()))?;

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

        Ok(Self { conn })
    }

    /// Open storage at the given connection URL
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
            
            -- Engrams table
            CREATE TABLE IF NOT EXISTS engrams (
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
            CREATE INDEX IF NOT EXISTS idx_engrams_state ON engrams(state);
            CREATE INDEX IF NOT EXISTS idx_engrams_energy ON engrams(energy);
            
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

            -- Engram enrichment embeddings (multi-vector support)
            CREATE TABLE IF NOT EXISTS engram_enrichments (
                engram_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                source TEXT NOT NULL DEFAULT 'llm',
                created_at INTEGER NOT NULL,
                PRIMARY KEY (engram_id, seq),
                FOREIGN KEY (engram_id) REFERENCES engrams(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_enrichments_engram ON engram_enrichments(engram_id);
        "#,
            )
            .map_err(|e| StorageError::Database(e.to_string()))?;

        // FTS5 virtual table for BM25 keyword search
        self.conn
            .batch_execute(
                r#"
            CREATE VIRTUAL TABLE IF NOT EXISTS engram_fts USING fts5(
                content,
                engram_id UNINDEXED,
                tokenize='porter unicode61'
            );
        "#,
            )
            .map_err(|e| StorageError::Database(e.to_string()))?;

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
            "SELECT COUNT(*) as cnt FROM pragma_table_info('engrams') WHERE name = 'embedding'",
        )
        .get_result(&mut self.conn);

        let has_embedding = match result {
            Ok(r) => r.cnt > 0,
            Err(_) => false,
        };

        if !has_embedding {
            self.conn
                .batch_execute("ALTER TABLE engrams ADD COLUMN embedding BLOB")
                .map_err(|e| StorageError::Database(e.to_string()))?;
        }

        Ok(())
    }

    /// Migrate existing database to add engram_enrichments table if missing (SQLite)
    #[cfg(feature = "sqlite")]
    fn migrate_enrichments_table(&mut self) -> StorageResult<()> {
        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }

        let result: Result<CountResult, _> = diesel::sql_query(
            "SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='table' AND name='engram_enrichments'",
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
                CREATE TABLE IF NOT EXISTS engram_enrichments (
                    engram_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    source TEXT NOT NULL DEFAULT 'llm',
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY (engram_id, seq),
                    FOREIGN KEY (engram_id) REFERENCES engrams(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_enrichments_engram ON engram_enrichments(engram_id);
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
}

impl Storage for EngramStorage {
    #[cfg(feature = "sqlite")]
    fn initialize(&mut self) -> StorageResult<()> {
        self.create_schema()?;
        self.migrate_embeddings()?;
        self.migrate_association_ordinal()?;
        self.migrate_enrichments_table()?;
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
    // ENGRAMS
    // ==================

    fn save_engram(&mut self, engram: &Engram) -> StorageResult<()> {
        let id_str = engram.id.to_string();
        let state_str = state_to_str(engram.state);
        let tags_json = serde_json::to_string(&engram.tags)?;

        diesel::replace_into(engrams::table)
            .values(NewEngram {
                id: &id_str,
                content: &engram.content,
                energy: engram.energy as f32,
                state: state_str,
                confidence: engram.confidence as f32,
                created_at: engram.created_at,
                last_accessed: engram.last_accessed,
                access_count: engram.access_count as i64,
                tags: tags_json,
                embedding: engram.embedding.as_ref().map(|e| embedding_to_bytes(e)),
            })
            .execute(&mut self.conn)?;

        // Sync FTS5: delete existing entry (if any) then insert fresh
        diesel::sql_query("DELETE FROM engram_fts WHERE engram_id = ?1")
            .bind::<diesel::sql_types::Text, _>(&id_str)
            .execute(&mut self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;
        diesel::sql_query("INSERT INTO engram_fts(content, engram_id) VALUES (?1, ?2)")
            .bind::<diesel::sql_types::Text, _>(&engram.content)
            .bind::<diesel::sql_types::Text, _>(&id_str)
            .execute(&mut self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(())
    }

    fn save_engrams(&mut self, engram_list: &[&Engram]) -> StorageResult<()> {
        self.conn
            .transaction::<_, diesel::result::Error, _>(|conn| {
                for engram in engram_list {
                    let id_str = engram.id.to_string();
                    let state_str = state_to_str(engram.state);
                    let tags_json = serde_json::to_string(&engram.tags)
                        .map_err(|e| diesel::result::Error::QueryBuilderError(Box::new(e)))?;

                    diesel::replace_into(engrams::table)
                        .values(NewEngram {
                            id: &id_str,
                            content: &engram.content,
                            energy: engram.energy as f32,
                            state: state_str,
                            confidence: engram.confidence as f32,
                            created_at: engram.created_at,
                            last_accessed: engram.last_accessed,
                            access_count: engram.access_count as i64,
                            tags: tags_json,
                            embedding: engram.embedding.as_ref().map(|e| embedding_to_bytes(e)),
                        })
                        .execute(conn)?;

                    // Sync FTS5: delete existing entry (if any) then insert fresh
                    diesel::sql_query("DELETE FROM engram_fts WHERE engram_id = ?1")
                        .bind::<diesel::sql_types::Text, _>(&id_str)
                        .execute(conn)?;
                    diesel::sql_query("INSERT INTO engram_fts(content, engram_id) VALUES (?1, ?2)")
                        .bind::<diesel::sql_types::Text, _>(&engram.content)
                        .bind::<diesel::sql_types::Text, _>(&id_str)
                        .execute(conn)?;
                }
                Ok(())
            })?;

        Ok(())
    }

    fn load_engram(&mut self, id: &EngramId) -> StorageResult<Option<Engram>> {
        let id_str = id.to_string();

        let result: Option<EngramRow> = engrams::table
            .filter(engrams::id.eq(&id_str))
            .select(EngramRow::as_select())
            .first(&mut self.conn)
            .optional()?;

        match result {
            Some(row) => Ok(Some(row.into_engram()?)),
            None => Ok(None),
        }
    }

    fn load_all_engrams(&mut self) -> StorageResult<Vec<Engram>> {
        let rows: Vec<EngramRow> = engrams::table
            .select(EngramRow::as_select())
            .load(&mut self.conn)?;

        rows.into_iter().map(|row| row.into_engram()).collect()
    }

    fn load_engrams_by_state(&mut self, state: MemoryState) -> StorageResult<Vec<Engram>> {
        let state_str = state_to_str(state);

        let rows: Vec<EngramRow> = engrams::table
            .filter(engrams::state.eq(state_str))
            .select(EngramRow::as_select())
            .load(&mut self.conn)?;

        rows.into_iter().map(|row| row.into_engram()).collect()
    }

    fn load_engrams_by_tag(&mut self, tag: &str) -> StorageResult<Vec<Engram>> {
        // Tags are stored as JSON array, search with LIKE
        let pattern = format!("%\"{}%", tag.to_lowercase());

        let rows: Vec<EngramRow> = engrams::table
            .filter(diesel::dsl::sql::<diesel::sql_types::Bool>(&format!(
                "LOWER(tags) LIKE '{}'",
                pattern.replace("'", "''")
            )))
            .select(EngramRow::as_select())
            .load(&mut self.conn)?;

        rows.into_iter().map(|row| row.into_engram()).collect()
    }

    fn delete_engram(&mut self, id: &EngramId) -> StorageResult<bool> {
        let id_str = id.to_string();

        // Delete the engram
        let deleted = diesel::delete(engrams::table.filter(engrams::id.eq(&id_str)))
            .execute(&mut self.conn)?;

        // Delete from FTS5 index
        diesel::sql_query("DELETE FROM engram_fts WHERE engram_id = ?1")
            .bind::<diesel::sql_types::Text, _>(&id_str)
            .execute(&mut self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;

        // Delete associations from/to this engram
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

    fn count_engrams(&mut self) -> StorageResult<usize> {
        let count: i64 = engrams::table.count().get_result(&mut self.conn)?;
        Ok(count as usize)
    }

    fn save_engram_energies(
        &mut self,
        updates: &[(&EngramId, f64, MemoryState)],
    ) -> StorageResult<()> {
        if updates.is_empty() {
            return Ok(());
        }

        self.conn
            .transaction::<_, diesel::result::Error, _>(|conn| {
                for (id, energy, state) in updates {
                    let id_str = id.to_string();
                    let state_str = state_to_str(*state);

                    diesel::update(engrams::table.filter(engrams::id.eq(&id_str)))
                        .set((
                            engrams::energy.eq(*energy as f32),
                            engrams::state.eq(state_str),
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

    fn load_associations_from(&mut self, from: &EngramId) -> StorageResult<Vec<Association>> {
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

    // ==================
    // CONFIG
    // ==================

    fn save_config(&mut self, cfg: &Config) -> StorageResult<()> {
        let json = serde_json::to_string(cfg)?;

        diesel::replace_into(config::table)
            .values(NewConfig { id: 1, data: &json })
            .execute(&mut self.conn)?;

        Ok(())
    }

    fn load_config(&mut self) -> StorageResult<Option<Config>> {
        let result: Option<ConfigRow> = config::table
            .filter(config::id.eq(1))
            .select(ConfigRow::as_select())
            .first(&mut self.conn)
            .optional()?;

        match result {
            Some(row) => {
                let cfg: Config = serde_json::from_str(&row.data)?;
                Ok(Some(cfg))
            }
            None => Ok(None),
        }
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
    fn get_ids_without_embeddings(&mut self, limit: usize) -> StorageResult<Vec<EngramId>> {
        let mut vs = VectorSearch::new(&mut self.conn);
        vs.get_ids_without_embeddings(limit)
    }

    #[cfg(feature = "sqlite")]
    fn set_embedding(&mut self, id: &EngramId, embedding: &[f32]) -> StorageResult<()> {
        let mut vs = VectorSearch::new(&mut self.conn);
        vs.set_embedding(id, embedding)
    }

    #[cfg(feature = "sqlite")]
    fn get_embedding(&mut self, id: &EngramId) -> StorageResult<Option<Vec<f32>>> {
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
    fn get_ids_without_embeddings(&mut self, _limit: usize) -> StorageResult<Vec<EngramId>> {
        Ok(Vec::new()) // TODO
    }

    #[cfg(feature = "postgres")]
    fn set_embedding(&mut self, _id: &EngramId, _embedding: &[f32]) -> StorageResult<()> {
        Ok(()) // TODO
    }

    #[cfg(feature = "postgres")]
    fn get_embedding(&mut self, _id: &EngramId) -> StorageResult<Option<Vec<f32>>> {
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
        id: &EngramId,
        embeddings: &[Vec<f32>],
        source: &str,
    ) -> StorageResult<()> {
        let mut vs = VectorSearch::new(&mut self.conn);
        vs.set_enrichment_embeddings(id, embeddings, source)
    }

    #[cfg(feature = "sqlite")]
    fn delete_enrichments(&mut self, id: &EngramId) -> StorageResult<()> {
        let mut vs = VectorSearch::new(&mut self.conn);
        vs.delete_enrichments(id)
    }

    #[cfg(feature = "sqlite")]
    fn count_enrichments(&mut self) -> StorageResult<usize> {
        let mut vs = VectorSearch::new(&mut self.conn);
        vs.count_enrichments()
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
            "SELECT e.id, e.content, -bm25(engram_fts) as score \
             FROM engram_fts f \
             JOIN engrams e ON e.id = f.engram_id \
             WHERE engram_fts MATCH ?1 \
             ORDER BY bm25(engram_fts) \
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

        let fts_count: CountResult = diesel::sql_query("SELECT COUNT(*) as cnt FROM engram_fts")
            .get_result(&mut self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;

        let engram_count: CountResult = diesel::sql_query("SELECT COUNT(*) as cnt FROM engrams")
            .get_result(&mut self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;

        if fts_count.cnt > 0 || engram_count.cnt == 0 {
            return Ok(0);
        }

        // Backfill FTS5 from existing engrams
        let affected = diesel::sql_query(
            "INSERT INTO engram_fts(content, engram_id) SELECT content, id FROM engrams",
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
        result_ids: &[EngramId],
        recalled_ids: &[EngramId],
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
    fn save_and_load_engram() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let engram = Engram::with_tags("Test memory", vec!["work".into(), "rust".into()]);
        let id = engram.id;

        storage.save_engram(&engram).unwrap();

        let loaded = storage.load_engram(&id).unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.content, "Test memory");
        assert_eq!(loaded.tags, vec!["work", "rust"]);
    }

    #[test]
    fn save_and_load_identity() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let identity = Identity::new()
            .with_persona("Porter", "A test assistant")
            .with_trait("pragmatic")
            .with_value(Value::new("Test value"))
            .with_preference(Preference::new("Rust").over("JavaScript"));

        storage.save_identity(&identity).unwrap();

        let loaded = storage.load_identity().unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.persona.name, "Porter");
        assert_eq!(loaded.persona.traits, vec!["pragmatic"]);
    }

    #[test]
    fn save_and_load_association() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let e1 = Engram::new("Memory 1");
        let e2 = Engram::new("Memory 2");
        let assoc = Association::with_weight(e1.id, e2.id, 0.8);

        storage.save_association(&assoc).unwrap();

        let loaded = storage.load_associations_from(&e1.id).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].to, e2.id);
        assert!((loaded[0].weight - 0.8).abs() < 0.001);
    }

    #[test]
    fn batch_save_engrams() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let e1 = Engram::new("Memory 1");
        let e2 = Engram::new("Memory 2");
        let e3 = Engram::new("Memory 3");

        storage.save_engrams(&[&e1, &e2, &e3]).unwrap();

        let all = storage.load_all_engrams().unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn load_by_state() {
        let mut storage = EngramStorage::in_memory().unwrap();
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

        let archived_only = storage
            .load_engrams_by_state(MemoryState::Archived)
            .unwrap();
        assert_eq!(archived_only.len(), 1);
        assert_eq!(archived_only[0].content, "Archived memory");
    }

    #[test]
    fn load_by_tag() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let work = Engram::with_tags("Work memory", vec!["work".into()]);
        let personal = Engram::with_tags("Personal memory", vec!["personal".into()]);

        storage.save_engram(&work).unwrap();
        storage.save_engram(&personal).unwrap();

        let work_only = storage.load_engrams_by_tag("work").unwrap();
        assert_eq!(work_only.len(), 1);
        assert_eq!(work_only[0].content, "Work memory");
    }

    #[test]
    fn config_persistence() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let config = Config {
            decay_rate_per_day: 0.1,
            decay_interval_hours: 2.0,
            propagation_damping: 0.6,
            hebbian_learning_rate: 0.2,
            recall_strength: 0.3,
            ..Default::default()
        };

        storage.save_config(&config).unwrap();

        let loaded = storage.load_config().unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert!((loaded.decay_rate_per_day - 0.1).abs() < 0.001);
        assert!((loaded.recall_strength - 0.3).abs() < 0.001);
    }

    #[test]
    fn brain_with_diesel() {
        use crate::engram::Brain;

        // Create a brain with Diesel backend
        let storage = EngramStorage::in_memory().unwrap();
        let mut brain = Brain::open(storage).unwrap();

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
        assert_eq!(stats.total_engrams, 2);
    }

    // ==================
    // ORDINAL TESTS
    // ==================

    #[test]
    fn migration_adds_ordinal_column() {
        // Fresh DB should have ordinal column after initialize
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        // Verify by saving an association with ordinal
        let e1 = Engram::new("Step 1");
        let e2 = Engram::new("Step 2");
        let assoc = Association::with_ordinal(e1.id, e2.id, 0.8, Some(1));

        storage.save_association(&assoc).unwrap();
        let loaded = storage.load_associations_from(&e1.id).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].ordinal, Some(1));
    }

    #[test]
    fn ordinal_roundtrip_with_value() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let e1 = Engram::new("Anchor");
        let e2 = Engram::new("Step 1");
        let e3 = Engram::new("Step 2");
        let e4 = Engram::new("Step 3");

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
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let e1 = Engram::new("Memory A");
        let e2 = Engram::new("Memory B");

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
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let anchor = Engram::new("Procedure anchor");
        let step1 = Engram::new("Step 1");
        let step2 = Engram::new("Step 2");
        let related = Engram::new("Related note");

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
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let anchor = Engram::new("Deploy procedure");
        let steps: Vec<Engram> = (1..=5)
            .map(|i| Engram::new(format!("Step {}", i)))
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
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let e1 = Engram::new("A");
        let e2 = Engram::new("B");
        let e3 = Engram::new("C");

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
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let e1 = Engram::new("X");
        let e2 = Engram::new("Y");

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
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        // Verify the FTS5 table exists by querying it
        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }

        let result: CountResult = diesel::sql_query("SELECT COUNT(*) as cnt FROM engram_fts")
            .get_result(&mut storage.conn)
            .expect("engram_fts table should exist after init");
        assert_eq!(result.cnt, 0);
    }

    #[test]
    fn fts_populated_on_create() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let engram = Engram::new("Rust programming language features");
        storage.save_engram(&engram).unwrap();

        // Check FTS5 has the entry
        #[derive(QueryableByName)]
        struct FtsCheck {
            #[diesel(sql_type = diesel::sql_types::Text)]
            engram_id: String,
        }

        let rows: Vec<FtsCheck> =
            diesel::sql_query("SELECT engram_id FROM engram_fts WHERE engram_fts MATCH '\"Rust\"'")
                .load(&mut storage.conn)
                .unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].engram_id, engram.id.to_string());
    }

    #[test]
    fn fts_cleaned_on_delete() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let engram = Engram::new("Temporary memory for testing");
        let id = engram.id;
        storage.save_engram(&engram).unwrap();

        // Verify it's in FTS
        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }

        let before: CountResult = diesel::sql_query("SELECT COUNT(*) as cnt FROM engram_fts")
            .get_result(&mut storage.conn)
            .unwrap();
        assert_eq!(before.cnt, 1);

        // Delete the engram
        storage.delete_engram(&id).unwrap();

        // FTS should be cleaned
        let after: CountResult = diesel::sql_query("SELECT COUNT(*) as cnt FROM engram_fts")
            .get_result(&mut storage.conn)
            .unwrap();
        assert_eq!(after.cnt, 0);
    }

    #[test]
    fn keyword_search_basic() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        storage
            .save_engram(&Engram::new("Rust programming language"))
            .unwrap();
        storage
            .save_engram(&Engram::new("Python programming language"))
            .unwrap();
        storage
            .save_engram(&Engram::new("Cooking delicious pasta"))
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
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        storage
            .save_engram(&Engram::new("Researching machine learning techniques"))
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
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        storage
            .save_engram(&Engram::new("Rust programming"))
            .unwrap();

        let results = storage.keyword_search("quantum physics", 10).unwrap();
        assert!(results.is_empty(), "No memories about quantum physics");
    }

    #[test]
    fn keyword_search_empty_query() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        storage.save_engram(&Engram::new("Some content")).unwrap();

        let results = storage.keyword_search("", 10).unwrap();
        assert!(results.is_empty(), "Empty query should return no results");
    }

    #[test]
    fn fts_updated_on_content_change() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let mut engram = Engram::new("Original content about dogs");
        storage.save_engram(&engram).unwrap();

        // Should find "dogs"
        let results = storage.keyword_search("dogs", 10).unwrap();
        assert_eq!(results.len(), 1);

        // Update content
        engram.content = "Updated content about cats".to_string();
        storage.save_engram(&engram).unwrap();

        // Should no longer find "dogs"
        let results = storage.keyword_search("dogs", 10).unwrap();
        assert!(results.is_empty(), "Old content should not be findable");

        // Should find "cats"
        let results = storage.keyword_search("cats", 10).unwrap();
        assert_eq!(results.len(), 1, "New content should be findable");
    }

    #[test]
    fn ensure_fts_populated_backfills() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        // Create engrams directly in the engrams table without FTS sync
        // by using raw SQL to bypass the save_engram FTS logic
        let e1 = Engram::new("Memory about Rust");
        let e2 = Engram::new("Memory about Python");

        // Save normally (which populates FTS)
        storage.save_engram(&e1).unwrap();
        storage.save_engram(&e2).unwrap();

        // Clear FTS manually to simulate pre-FTS data
        diesel::sql_query("DELETE FROM engram_fts")
            .execute(&mut storage.conn)
            .unwrap();

        // Verify FTS is empty
        #[derive(QueryableByName)]
        struct CountResult {
            #[diesel(sql_type = diesel::sql_types::Integer)]
            cnt: i32,
        }

        let count: CountResult = diesel::sql_query("SELECT COUNT(*) as cnt FROM engram_fts")
            .get_result(&mut storage.conn)
            .unwrap();
        assert_eq!(count.cnt, 0, "FTS should be empty after manual clear");

        // Run backfill
        let backfilled = storage.ensure_fts_populated().unwrap();
        assert_eq!(backfilled, 2, "Should backfill 2 engrams");

        // Verify search works after backfill
        let results = storage.keyword_search("Rust", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Rust"));
    }

    #[test]
    fn ensure_fts_populated_noop_when_already_populated() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        storage
            .save_engram(&Engram::new("Already indexed"))
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
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let e1 = Engram::new("Batch item alpha");
        let e2 = Engram::new("Batch item beta");
        let e3 = Engram::new("Batch item gamma");

        storage.save_engrams(&[&e1, &e2, &e3]).unwrap();

        let results = storage.keyword_search("batch", 10).unwrap();
        assert_eq!(results.len(), 3, "All batch-saved engrams should be in FTS");
    }

    #[test]
    fn access_log_records_search_recall_cycle() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let e1 = Engram::new("Brandon prefers ale");
        let e2 = Engram::new("Brandon lives in Oregon City");
        let e3 = Engram::new("Brandon is an engineer");

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
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let e1 = Engram::new("Memory one");
        let e2 = Engram::new("Memory two");

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
}
