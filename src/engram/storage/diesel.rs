//! Diesel-backed storage implementation for Engrams
//!
//! This module provides database storage using Diesel ORM.
//! The actual database backend (SQLite, Postgres, MySQL) is selected
//! via feature flags at compile time.

use super::{Storage, StorageResult, StorageError, SimilarityResult};
use crate::engram::{
    EngramId, Engram, MemoryState, Config, Association,
    Identity,
};
use crate::storage::schema::{engrams, associations, config, identity, metadata};
use crate::storage::models::*;

use diesel::prelude::*;
use diesel::connection::SimpleConnection;
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
        conn.batch_execute(r#"
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA foreign_keys = ON;
            PRAGMA busy_timeout = 5000;
        "#).map_err(|e| StorageError::Database(e.to_string()))?;
        
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
    #[cfg(feature = "sqlite")]
    pub fn in_memory() -> StorageResult<Self> {
        let mut conn = DbConnection::establish(":memory:")
            .map_err(|e| StorageError::Database(e.to_string()))?;
        
        conn.batch_execute("PRAGMA foreign_keys = ON;")
            .map_err(|e| StorageError::Database(e.to_string()))?;
        
        Ok(Self { conn })
    }
    
    /// Get mutable reference to the underlying connection
    pub fn connection(&mut self) -> &mut DbConnection {
        &mut self.conn
    }
    
    /// Create the database schema (SQLite)
    #[cfg(feature = "sqlite")]
    fn create_schema(&mut self) -> StorageResult<()> {
        self.conn.batch_execute(r#"
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
        "#).map_err(|e| StorageError::Database(e.to_string()))?;
        
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
            "SELECT COUNT(*) as cnt FROM pragma_table_info('engrams') WHERE name = 'embedding'"
        )
        .get_result(&mut self.conn);
        
        let has_embedding = match result {
            Ok(r) => r.cnt > 0,
            Err(_) => false,
        };
        
        if !has_embedding {
            self.conn.batch_execute("ALTER TABLE engrams ADD COLUMN embedding BLOB")
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
        
        Ok(())
    }
    
    fn save_engrams(&mut self, engram_list: &[&Engram]) -> StorageResult<()> {
        self.conn.transaction::<_, diesel::result::Error, _>(|conn| {
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
        
        rows.into_iter()
            .map(|row| row.into_engram())
            .collect()
    }
    
    fn load_engrams_by_state(&mut self, state: MemoryState) -> StorageResult<Vec<Engram>> {
        let state_str = state_to_str(state);
        
        let rows: Vec<EngramRow> = engrams::table
            .filter(engrams::state.eq(state_str))
            .select(EngramRow::as_select())
            .load(&mut self.conn)?;
        
        rows.into_iter()
            .map(|row| row.into_engram())
            .collect()
    }
    
    fn load_engrams_by_tag(&mut self, tag: &str) -> StorageResult<Vec<Engram>> {
        // Tags are stored as JSON array, search with LIKE
        let pattern = format!("%\"{}%", tag.to_lowercase());
        
        let rows: Vec<EngramRow> = engrams::table
            .filter(diesel::dsl::sql::<diesel::sql_types::Bool>(&format!(
                "LOWER(tags) LIKE '{}'", pattern.replace("'", "''")
            )))
            .select(EngramRow::as_select())
            .load(&mut self.conn)?;
        
        rows.into_iter()
            .map(|row| row.into_engram())
            .collect()
    }
    
    fn delete_engram(&mut self, id: &EngramId) -> StorageResult<bool> {
        let id_str = id.to_string();
        
        // Delete the engram
        let deleted = diesel::delete(engrams::table.filter(engrams::id.eq(&id_str)))
            .execute(&mut self.conn)?;
        
        // Delete associations from/to this engram
        diesel::delete(associations::table.filter(
            associations::from_id.eq(&id_str).or(associations::to_id.eq(&id_str))
        ))
        .execute(&mut self.conn)?;
        
        Ok(deleted > 0)
    }
    
    fn save_engram_energies(&mut self, updates: &[(&EngramId, f64, MemoryState)]) -> StorageResult<()> {
        if updates.is_empty() {
            return Ok(());
        }
        
        self.conn.transaction::<_, diesel::result::Error, _>(|conn| {
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
            })
            .execute(&mut self.conn)?;
        
        Ok(())
    }
    
    fn save_associations(&mut self, assocs: &[&Association]) -> StorageResult<()> {
        self.conn.transaction::<_, diesel::result::Error, _>(|conn| {
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
        
        rows.into_iter()
            .map(|row| row.into_association())
            .collect()
    }
    
    fn load_all_associations(&mut self) -> StorageResult<Vec<Association>> {
        let rows: Vec<AssociationRow> = associations::table
            .select(AssociationRow::as_select())
            .load(&mut self.conn)?;
        
        rows.into_iter()
            .map(|row| row.into_association())
            .collect()
    }
    
    fn delete_all_associations(&mut self) -> StorageResult<()> {
        diesel::delete(associations::table)
            .execute(&mut self.conn)?;
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
                let timestamp: i64 = row.value.parse()
                    .map_err(|e: std::num::ParseIntError| StorageError::Serialization(e.to_string()))?;
                Ok(Some(timestamp))
            }
            None => Ok(None),
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::{Value, Preference};
    
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
        
        let archived_only = storage.load_engrams_by_state(MemoryState::Archived).unwrap();
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
        let rust_talk = brain.create_with_tags(
            "Discussed Rust FFI patterns",
            vec!["work".into(), "rust".into()]
        ).unwrap();
        
        let ffi_fact = brain.create_with_tags(
            "TwilioDataCore uses opaque handles",
            vec!["work".into(), "technical".into()]
        ).unwrap();
        
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
}
