//! SQLite storage implementation

use super::{Storage, StorageResult, StorageError, SimilarityResult, VectorSearch};
use crate::engram::{
    EngramId, Engram, MemoryState, Config, Association,
    Identity,
};
use crate::storage::Database;
use rusqlite::{Connection, params};
use std::path::Path;

/// SQLite-backed storage implementation
pub struct SqliteStorage {
    db: Database,
}

impl SqliteStorage {
    /// Create a new SQLite storage at the given path
    pub fn new<P: AsRef<Path>>(path: P) -> StorageResult<Self> {
        let db = Database::open(path)?;
        Ok(Self { db })
    }
    
    /// Create an in-memory SQLite database (for testing)
    pub fn in_memory() -> StorageResult<Self> {
        let db = Database::in_memory()?;
        Ok(Self { db })
    }
    
    /// Get a reference to the underlying connection (for VectorSearch)
    pub fn connection(&self) -> &Connection {
        self.db.conn()
    }
    
    /// Create the database schema
    fn create_schema(&self) -> StorageResult<()> {
        self.db.initialize_schema(r#"
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
                tags TEXT NOT NULL,  -- JSON array
                embedding BLOB       -- 384-dim f32 vector (1536 bytes)
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
            
            -- FTS5 virtual table for full-text search on engram content
            -- Stores engram_id as unindexed column so we can join back
            CREATE VIRTUAL TABLE IF NOT EXISTS engrams_fts USING fts5(
                content,
                tags,
                engram_id UNINDEXED
            );
        "#)
    }
}

impl Storage for SqliteStorage {
    fn initialize(&mut self) -> StorageResult<()> {
        self.create_schema()?;
        self.migrate_embeddings()?;
        Ok(())
    }
    
    // ==================
    // IDENTITY
    // ==================
    
    fn save_identity(&mut self, identity: &Identity) -> StorageResult<()> {
        let json = serde_json::to_string(identity)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        
        self.db.conn().execute(
            "INSERT OR REPLACE INTO identity (id, data) VALUES (1, ?1)",
            params![json]
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        Ok(())
    }
    
    fn load_identity(&self) -> StorageResult<Option<Identity>> {
        let result: Result<String, _> = self.db.conn().query_row(
            "SELECT data FROM identity WHERE id = 1",
            [],
            |row| row.get(0)
        );
        
        match result {
            Ok(json) => {
                let identity: Identity = serde_json::from_str(&json)
                    .map_err(|e| StorageError::Serialization(e.to_string()))?;
                Ok(Some(identity))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(StorageError::Database(e.to_string())),
        }
    }
    
    // ==================
    // ENGRAMS
    // ==================
    
    fn save_engram(&mut self, engram: &Engram) -> StorageResult<()> {
        let tags_json = serde_json::to_string(&engram.tags)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        
        let state_str = match engram.state {
            MemoryState::Active => "active",
            MemoryState::Dormant => "dormant",
            MemoryState::Deep => "deep",
            MemoryState::Archived => "archived",
        };
        
        let id_str = engram.id.to_string();
        
        self.db.conn().execute(
            r#"INSERT OR REPLACE INTO engrams 
               (id, content, energy, state, confidence, created_at, last_accessed, access_count, tags, embedding)
               VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)"#,
            params![
                &id_str,
                engram.content,
                engram.energy,
                state_str,
                engram.confidence,
                engram.created_at,
                engram.last_accessed,
                engram.access_count as i64,
                &tags_json,
                engram.embedding.as_ref().map(|e| embedding_to_bytes(e)),
            ]
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        // Update FTS index: delete existing entry (if any), then insert new
        self.db.conn().execute(
            "DELETE FROM engrams_fts WHERE engram_id = ?1",
            params![&id_str]
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        self.db.conn().execute(
            "INSERT INTO engrams_fts (content, tags, engram_id) VALUES (?1, ?2, ?3)",
            params![&engram.content, &tags_json, &id_str]
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        Ok(())
    }
    
    fn save_engrams(&mut self, engrams: &[&Engram]) -> StorageResult<()> {
        let tx = self.db.conn_mut().transaction()
            .map_err(|e| StorageError::Database(e.to_string()))?;
        
        {
            let mut engram_stmt = tx.prepare(
                r#"INSERT OR REPLACE INTO engrams 
                   (id, content, energy, state, confidence, created_at, last_accessed, access_count, tags, embedding)
                   VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)"#
            ).map_err(|e| StorageError::Database(e.to_string()))?;
            
            let mut fts_delete_stmt = tx.prepare(
                "DELETE FROM engrams_fts WHERE engram_id = ?1"
            ).map_err(|e| StorageError::Database(e.to_string()))?;
            
            let mut fts_insert_stmt = tx.prepare(
                "INSERT INTO engrams_fts (content, tags, engram_id) VALUES (?1, ?2, ?3)"
            ).map_err(|e| StorageError::Database(e.to_string()))?;
            
            for engram in engrams {
                let tags_json = serde_json::to_string(&engram.tags)
                    .map_err(|e| StorageError::Serialization(e.to_string()))?;
                
                let state_str = match engram.state {
                    MemoryState::Active => "active",
                    MemoryState::Dormant => "dormant",
                    MemoryState::Deep => "deep",
                    MemoryState::Archived => "archived",
                };
                
                let id_str = engram.id.to_string();
                
                engram_stmt.execute(params![
                    &id_str,
                    engram.content,
                    engram.energy,
                    state_str,
                    engram.confidence,
                    engram.created_at,
                    engram.last_accessed,
                    engram.access_count as i64,
                    &tags_json,
                    engram.embedding.as_ref().map(|e| embedding_to_bytes(e)),
                ]).map_err(|e| StorageError::Database(e.to_string()))?;
                
                // Update FTS index
                fts_delete_stmt.execute(params![&id_str])
                    .map_err(|e| StorageError::Database(e.to_string()))?;
                fts_insert_stmt.execute(params![&engram.content, &tags_json, &id_str])
                    .map_err(|e| StorageError::Database(e.to_string()))?;
            }
        }
        
        tx.commit().map_err(|e| StorageError::Database(e.to_string()))?;
        Ok(())
    }
    
    fn load_engram(&self, id: &EngramId) -> StorageResult<Option<Engram>> {
        let result = self.db.conn().query_row(
            "SELECT id, content, energy, state, confidence, created_at, last_accessed, access_count, tags, embedding FROM engrams WHERE id = ?1",
            params![id.to_string()],
            |row| {
                Ok(RowData {
                    id: row.get::<_, String>(0)?,
                    content: row.get(1)?,
                    energy: row.get(2)?,
                    state: row.get::<_, String>(3)?,
                    confidence: row.get(4)?,
                    created_at: row.get(5)?,
                    last_accessed: row.get(6)?,
                    access_count: row.get::<_, i64>(7)?,
                    tags: row.get::<_, String>(8)?,
                    embedding: row.get::<_, Option<Vec<u8>>>(9)?,
                })
            }
        );
        
        match result {
            Ok(data) => Ok(Some(row_to_engram(data)?)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(StorageError::Database(e.to_string())),
        }
    }
    
    fn load_all_engrams(&self) -> StorageResult<Vec<Engram>> {
        let mut stmt = self.db.conn().prepare(
            "SELECT id, content, energy, state, confidence, created_at, last_accessed, access_count, tags FROM engrams"
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        let rows = stmt.query_map([], |row| {
            Ok(RowData {
                id: row.get::<_, String>(0)?,
                content: row.get(1)?,
                energy: row.get(2)?,
                state: row.get::<_, String>(3)?,
                confidence: row.get(4)?,
                created_at: row.get(5)?,
                last_accessed: row.get(6)?,
                access_count: row.get::<_, i64>(7)?,
                tags: row.get::<_, String>(8)?,
                    embedding: None,
            })
        }).map_err(|e| StorageError::Database(e.to_string()))?;
        
        let mut engrams = Vec::new();
        for row in rows {
            let data = row.map_err(|e| StorageError::Database(e.to_string()))?;
            engrams.push(row_to_engram(data)?);
        }
        
        Ok(engrams)
    }
    
    fn load_engrams_by_state(&self, state: MemoryState) -> StorageResult<Vec<Engram>> {
        let state_str = match state {
            MemoryState::Active => "active",
            MemoryState::Dormant => "dormant",
            MemoryState::Deep => "deep",
            MemoryState::Archived => "archived",
        };
        
        let mut stmt = self.db.conn().prepare(
            "SELECT id, content, energy, state, confidence, created_at, last_accessed, access_count, tags FROM engrams WHERE state = ?1"
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        let rows = stmt.query_map(params![state_str], |row| {
            Ok(RowData {
                id: row.get::<_, String>(0)?,
                content: row.get(1)?,
                energy: row.get(2)?,
                state: row.get::<_, String>(3)?,
                confidence: row.get(4)?,
                created_at: row.get(5)?,
                last_accessed: row.get(6)?,
                access_count: row.get::<_, i64>(7)?,
                tags: row.get::<_, String>(8)?,
                embedding: None,
            })
        }).map_err(|e| StorageError::Database(e.to_string()))?;
        
        let mut engrams = Vec::new();
        for row in rows {
            let data = row.map_err(|e| StorageError::Database(e.to_string()))?;
            engrams.push(row_to_engram(data)?);
        }
        
        Ok(engrams)
    }
    
    fn load_engrams_by_tag(&self, tag: &str) -> StorageResult<Vec<Engram>> {
        // SQLite JSON search: tags column contains JSON array
        // We search for the tag in the JSON array
        let pattern = format!("%\"{}%", tag.to_lowercase());
        
        let mut stmt = self.db.conn().prepare(
            "SELECT id, content, energy, state, confidence, created_at, last_accessed, access_count, tags FROM engrams WHERE LOWER(tags) LIKE ?1"
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        let rows = stmt.query_map(params![pattern], |row| {
            Ok(RowData {
                id: row.get::<_, String>(0)?,
                content: row.get(1)?,
                energy: row.get(2)?,
                state: row.get::<_, String>(3)?,
                confidence: row.get(4)?,
                created_at: row.get(5)?,
                last_accessed: row.get(6)?,
                access_count: row.get::<_, i64>(7)?,
                tags: row.get::<_, String>(8)?,
                embedding: None,
            })
        }).map_err(|e| StorageError::Database(e.to_string()))?;
        
        let mut engrams = Vec::new();
        for row in rows {
            let data = row.map_err(|e| StorageError::Database(e.to_string()))?;
            engrams.push(row_to_engram(data)?);
        }
        
        Ok(engrams)
    }
    
    fn search_content(&self, query: &str) -> StorageResult<Vec<EngramId>> {
        self.search_fts(query)
    }
    
    fn delete_engram(&mut self, id: &EngramId) -> StorageResult<bool> {
        let id_str = id.to_string();
        
        // Delete the engram
        let deleted = self.db.conn().execute(
            "DELETE FROM engrams WHERE id = ?1",
            params![&id_str]
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        // Delete from FTS index
        self.db.conn().execute(
            "DELETE FROM engrams_fts WHERE engram_id = ?1",
            params![&id_str]
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        // Delete associations from/to this engram
        self.db.conn().execute(
            "DELETE FROM associations WHERE from_id = ?1 OR to_id = ?1",
            params![&id_str]
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        Ok(deleted > 0)
    }
    
    fn save_engram_energies(&mut self, updates: &[(&EngramId, f64, MemoryState)]) -> StorageResult<()> {
        if updates.is_empty() {
            return Ok(());
        }
        
        let tx = self.db.conn_mut().transaction()
            .map_err(|e| StorageError::Database(e.to_string()))?;
        
        {
            let mut stmt = tx.prepare(
                "UPDATE engrams SET energy = ?1, state = ?2 WHERE id = ?3"
            ).map_err(|e| StorageError::Database(e.to_string()))?;
            
            for (id, energy, state) in updates {
                let state_str = match state {
                    MemoryState::Active => "active",
                    MemoryState::Dormant => "dormant",
                    MemoryState::Deep => "deep",
                    MemoryState::Archived => "archived",
                };
                
                stmt.execute(params![energy, state_str, id.to_string()])
                    .map_err(|e| StorageError::Database(e.to_string()))?;
            }
        }
        
        tx.commit().map_err(|e| StorageError::Database(e.to_string()))?;
        Ok(())
    }
    
    // ==================
    // ASSOCIATIONS
    // ==================
    
    fn save_association(&mut self, assoc: &Association) -> StorageResult<()> {
        self.db.conn().execute(
            r#"INSERT OR REPLACE INTO associations 
               (from_id, to_id, weight, created_at, last_activated, co_activation_count)
               VALUES (?1, ?2, ?3, ?4, ?5, ?6)"#,
            params![
                assoc.from.to_string(),
                assoc.to.to_string(),
                assoc.weight,
                assoc.created_at,
                assoc.last_activated,
                assoc.co_activation_count as i64,
            ]
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        Ok(())
    }
    
    fn save_associations(&mut self, assocs: &[&Association]) -> StorageResult<()> {
        let tx = self.db.conn_mut().transaction()
            .map_err(|e| StorageError::Database(e.to_string()))?;
        
        {
            let mut stmt = tx.prepare(
                r#"INSERT OR REPLACE INTO associations 
                   (from_id, to_id, weight, created_at, last_activated, co_activation_count)
                   VALUES (?1, ?2, ?3, ?4, ?5, ?6)"#
            ).map_err(|e| StorageError::Database(e.to_string()))?;
            
            for assoc in assocs {
                stmt.execute(params![
                    assoc.from.to_string(),
                    assoc.to.to_string(),
                    assoc.weight,
                    assoc.created_at,
                    assoc.last_activated,
                    assoc.co_activation_count as i64,
                ]).map_err(|e| StorageError::Database(e.to_string()))?;
            }
        }
        
        tx.commit().map_err(|e| StorageError::Database(e.to_string()))?;
        Ok(())
    }
    
    fn load_associations_from(&self, from: &EngramId) -> StorageResult<Vec<Association>> {
        let mut stmt = self.db.conn().prepare(
            "SELECT from_id, to_id, weight, created_at, last_activated, co_activation_count FROM associations WHERE from_id = ?1"
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        let rows = stmt.query_map(params![from.to_string()], |row| {
            Ok(AssocRowData {
                from_id: row.get::<_, String>(0)?,
                to_id: row.get::<_, String>(1)?,
                weight: row.get(2)?,
                created_at: row.get(3)?,
                last_activated: row.get(4)?,
                co_activation_count: row.get::<_, i64>(5)?,
            })
        }).map_err(|e| StorageError::Database(e.to_string()))?;
        
        let mut assocs = Vec::new();
        for row in rows {
            let data = row.map_err(|e| StorageError::Database(e.to_string()))?;
            assocs.push(row_to_association(data)?);
        }
        
        Ok(assocs)
    }
    
    fn load_all_associations(&self) -> StorageResult<Vec<Association>> {
        let mut stmt = self.db.conn().prepare(
            "SELECT from_id, to_id, weight, created_at, last_activated, co_activation_count FROM associations"
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        let rows = stmt.query_map([], |row| {
            Ok(AssocRowData {
                from_id: row.get::<_, String>(0)?,
                to_id: row.get::<_, String>(1)?,
                weight: row.get(2)?,
                created_at: row.get(3)?,
                last_activated: row.get(4)?,
                co_activation_count: row.get::<_, i64>(5)?,
            })
        }).map_err(|e| StorageError::Database(e.to_string()))?;
        
        let mut assocs = Vec::new();
        for row in rows {
            let data = row.map_err(|e| StorageError::Database(e.to_string()))?;
            assocs.push(row_to_association(data)?);
        }
        
        Ok(assocs)
    }
    
    fn delete_all_associations(&mut self) -> StorageResult<()> {
        self.db.conn().execute("DELETE FROM associations", [])
            .map_err(|e| StorageError::Database(e.to_string()))?;
        Ok(())
    }
    
    // ==================
    // CONFIG
    // ==================
    
    fn save_config(&mut self, config: &Config) -> StorageResult<()> {
        let json = serde_json::to_string(config)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        
        self.db.conn().execute(
            "INSERT OR REPLACE INTO config (id, data) VALUES (1, ?1)",
            params![json]
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        Ok(())
    }
    
    fn load_config(&self) -> StorageResult<Option<Config>> {
        let result: Result<String, _> = self.db.conn().query_row(
            "SELECT data FROM config WHERE id = 1",
            [],
            |row| row.get(0)
        );
        
        match result {
            Ok(json) => {
                let config: Config = serde_json::from_str(&json)
                    .map_err(|e| StorageError::Serialization(e.to_string()))?;
                Ok(Some(config))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(StorageError::Database(e.to_string())),
        }
    }
    
    fn save_last_decay_at(&mut self, timestamp: i64) -> StorageResult<()> {
        self.db.conn().execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('last_decay_at', ?1)",
            params![timestamp.to_string()]
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        Ok(())
    }
    
    fn load_last_decay_at(&self) -> StorageResult<Option<i64>> {
        let result: Result<String, _> = self.db.conn().query_row(
            "SELECT value FROM metadata WHERE key = 'last_decay_at'",
            [],
            |row| row.get(0)
        );
        
        match result {
            Ok(value) => {
                let timestamp: i64 = value.parse()
                    .map_err(|e: std::num::ParseIntError| StorageError::Serialization(e.to_string()))?;
                Ok(Some(timestamp))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(StorageError::Database(e.to_string())),
        }
    }
    
    // ==================
    // LIFECYCLE
    // ==================
    
    fn flush(&mut self) -> StorageResult<()> {
        // SQLite auto-commits, but we can force a checkpoint for WAL mode
        self.db.conn().execute_batch("PRAGMA wal_checkpoint(PASSIVE);")
            .map_err(|e| StorageError::Database(e.to_string()))?;
        Ok(())
    }
    
    // ==================
    // VECTOR SEARCH
    // ==================
    
    fn find_similar_by_embedding(
        &self,
        query_embedding: &[f32],
        limit: usize,
        min_score: f32,
    ) -> StorageResult<Vec<SimilarityResult>> {
        let vs = VectorSearch::new(self.db.conn());
        vs.find_similar(query_embedding, limit, min_score)
    }
    
    fn count_with_embeddings(&self) -> StorageResult<usize> {
        let vs = VectorSearch::new(self.db.conn());
        vs.count_with_embeddings()
    }
    
    fn count_without_embeddings(&self) -> StorageResult<usize> {
        let vs = VectorSearch::new(self.db.conn());
        vs.count_without_embeddings()
    }
    
    fn get_ids_without_embeddings(&self, limit: usize) -> StorageResult<Vec<EngramId>> {
        let vs = VectorSearch::new(self.db.conn());
        vs.get_ids_without_embeddings(limit)
    }
    
    fn set_embedding(&mut self, id: &EngramId, embedding: &[f32]) -> StorageResult<()> {
        let vs = VectorSearch::new(self.db.conn());
        vs.set_embedding(id, embedding)
    }
    
    fn get_embedding(&self, id: &EngramId) -> StorageResult<Option<Vec<f32>>> {
        let vs = VectorSearch::new(self.db.conn());
        vs.get_embedding(id)
    }
}

// Additional methods specific to SqliteStorage (not part of Storage trait)
impl SqliteStorage {
    /// Search using FTS5 full-text search
    /// Returns engram IDs that match the query
    /// Query supports FTS5 syntax: AND, OR, NOT, phrase "quotes", prefix*
    pub fn search_fts(&self, query: &str) -> StorageResult<Vec<EngramId>> {
        // Tokenize the query for better matching
        // Split on whitespace and join with OR for flexible matching
        let tokens: Vec<&str> = query.split_whitespace().collect();
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        
        // Build FTS5 query: each token matches with OR logic
        // Use * suffix for prefix matching
        let fts_query = tokens.iter()
            .map(|t| format!("{}*", t))
            .collect::<Vec<_>>()
            .join(" OR ");
        
        let mut stmt = self.db.conn().prepare(
            "SELECT engram_id FROM engrams_fts WHERE engrams_fts MATCH ?1"
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        let rows = stmt.query_map(params![fts_query], |row| {
            row.get::<_, String>(0)
        }).map_err(|e| StorageError::Database(e.to_string()))?;
        
        let mut ids = Vec::new();
        for row in rows {
            let id_str = row.map_err(|e| StorageError::Database(e.to_string()))?;
            let id = uuid::Uuid::parse_str(&id_str)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
            ids.push(id);
        }
        
        Ok(ids)
    }
    
    /// Rebuild the FTS index from existing engrams
    /// Call this after migrating an existing database to add FTS support
    pub fn rebuild_fts_index(&mut self) -> StorageResult<usize> {
        // Clear existing FTS data
        self.db.conn().execute("DELETE FROM engrams_fts", [])
            .map_err(|e| StorageError::Database(e.to_string()))?;
        
        // Re-insert all engrams into FTS
        let count = self.db.conn().execute(
            "INSERT INTO engrams_fts (content, tags, engram_id) SELECT content, tags, id FROM engrams",
            []
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        Ok(count)
    }
    
    /// Check if FTS index needs rebuilding (exists but is empty while engrams exist)
    pub fn fts_needs_rebuild(&self) -> StorageResult<bool> {
        let engram_count: i64 = self.db.conn().query_row(
            "SELECT COUNT(*) FROM engrams",
            [],
            |row| row.get(0)
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        let fts_count: i64 = self.db.conn().query_row(
            "SELECT COUNT(*) FROM engrams_fts",
            [],
            |row| row.get(0)
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        Ok(engram_count > 0 && fts_count == 0)
    }
    
    /// Migrate existing database to add embedding column if missing
    fn migrate_embeddings(&self) -> StorageResult<()> {
        // Check if embedding column exists
        let has_embedding: bool = self.db.conn().query_row(
            "SELECT COUNT(*) FROM pragma_table_info('engrams') WHERE name = 'embedding'",
            [],
            |row| row.get::<_, i64>(0).map(|c| c > 0)
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        if !has_embedding {
            // Add the embedding column
            self.db.conn().execute(
                "ALTER TABLE engrams ADD COLUMN embedding BLOB",
                []
            ).map_err(|e| StorageError::Database(e.to_string()))?;
        }
        
        Ok(())
    }
}

// Helper structs for row data
struct RowData {
    id: String,
    content: String,
    energy: f64,
    state: String,
    confidence: f64,
    created_at: i64,
    last_accessed: i64,
    access_count: i64,
    tags: String,
    embedding: Option<Vec<u8>>,
}

struct AssocRowData {
    from_id: String,
    to_id: String,
    weight: f64,
    created_at: i64,
    last_activated: i64,
    co_activation_count: i64,
}

fn row_to_engram(data: RowData) -> StorageResult<Engram> {
    let id = uuid::Uuid::parse_str(&data.id)
        .map_err(|e| StorageError::Serialization(e.to_string()))?;
    
    let state = match data.state.as_str() {
        "active" => MemoryState::Active,
        "dormant" => MemoryState::Dormant,
        "deep" => MemoryState::Deep,
        "archived" => MemoryState::Archived,
        _ => return Err(StorageError::Serialization(format!("Unknown state: {}", data.state))),
    };
    
    let tags: Vec<String> = serde_json::from_str(&data.tags)
        .map_err(|e| StorageError::Serialization(e.to_string()))?;
    
    Ok(Engram {
        id,
        content: data.content,
        energy: data.energy,
        state,
        confidence: data.confidence,
        created_at: data.created_at,
        last_accessed: data.last_accessed,
        access_count: data.access_count as u64,
        tags,
        embedding: None,  // Loaded separately when needed
    })
}

fn row_to_association(data: AssocRowData) -> StorageResult<Association> {
    let from = uuid::Uuid::parse_str(&data.from_id)
        .map_err(|e| StorageError::Serialization(e.to_string()))?;
    let to = uuid::Uuid::parse_str(&data.to_id)
        .map_err(|e| StorageError::Serialization(e.to_string()))?;
    
    Ok(Association {
        from,
        to,
        weight: data.weight,
        created_at: data.created_at,
        last_activated: data.last_activated,
        co_activation_count: data.co_activation_count as u64,
    })
}


// Embedding serialization helpers
fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_embedding(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() % 4 != 0 { return None; }
    Some(bytes.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::{Value, Preference};
    
    #[test]
    fn save_and_load_engram() {
        let mut storage = SqliteStorage::in_memory().unwrap();
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
        let mut storage = SqliteStorage::in_memory().unwrap();
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
        let mut storage = SqliteStorage::in_memory().unwrap();
        storage.initialize().unwrap();
        
        let e1 = Engram::new("Memory 1");
        let e2 = Engram::new("Memory 2");
        let assoc = Association::with_weight(e1.id, e2.id, 0.8);
        
        storage.save_association(&assoc).unwrap();
        
        let loaded = storage.load_associations_from(&e1.id).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].to, e2.id);
        assert_eq!(loaded[0].weight, 0.8);
    }
    
    #[test]
    fn batch_save_engrams() {
        let mut storage = SqliteStorage::in_memory().unwrap();
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
        let mut storage = SqliteStorage::in_memory().unwrap();
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
        let mut storage = SqliteStorage::in_memory().unwrap();
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
        let mut storage = SqliteStorage::in_memory().unwrap();
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
        assert_eq!(loaded.decay_rate_per_day, 0.1);
        assert_eq!(loaded.recall_strength, 0.3);
    }
    
    #[test]
    fn brain_with_sqlite() {
        use crate::engram::Brain;
        
        // Create a brain with SQLite backend
        let storage = SqliteStorage::in_memory().unwrap();
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
        
        // Search
        let results = brain.search("rust");
        assert_eq!(results.len(), 1);
        
        // Find associated
        let associated = brain.find_associated(&rust_talk);
        assert_eq!(associated.len(), 1);
        
        // Stats
        let stats = brain.stats();
        assert_eq!(stats.total_engrams, 2);
    }
    
    #[test]
    fn fts_search() {
        let mut storage = SqliteStorage::in_memory().unwrap();
        storage.initialize().unwrap();
        
        // Create engrams with various content
        let rust_ffi = Engram::with_tags("engram crate structure: brain.rs (main interface), substrate.rs", vec!["engram".into(), "architecture".into()]);
        let memory_mcp = Engram::with_tags("memory MCP server location: /Users/bsneed/work/memoryco/memory", vec!["memory".into(), "memoryco".into()]);
        let swift_code = Engram::with_tags("Swift concurrency debugging notes", vec!["swift".into(), "debug".into()]);
        
        storage.save_engram(&rust_ffi).unwrap();
        storage.save_engram(&memory_mcp).unwrap();
        storage.save_engram(&swift_code).unwrap();
        
        // Test single token search
        let results = storage.search_fts("engram").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], rust_ffi.id);
        
        // Test multi-token search (should find both with OR logic)
        let results = storage.search_fts("memory MCP engram").unwrap();
        assert_eq!(results.len(), 2); // Should find both engram and memory MCP
        
        // Test prefix matching
        let results = storage.search_fts("sub").unwrap(); // Should match "substrate"
        assert_eq!(results.len(), 1);
        
        // Test tag search (tags are also indexed)
        let results = storage.search_fts("architecture").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], rust_ffi.id);
    }
    
    #[test]
    fn fts_rebuild_index() {
        let mut storage = SqliteStorage::in_memory().unwrap();
        storage.initialize().unwrap();
        
        // Generate valid UUIDs
        let id1 = uuid::Uuid::new_v4().to_string();
        let id2 = uuid::Uuid::new_v4().to_string();
        
        // Manually insert engrams without FTS (simulating legacy data)
        storage.connection().execute(
            r#"INSERT INTO engrams (id, content, energy, state, confidence, created_at, last_accessed, access_count, tags, embedding)
               VALUES (?1, 'Test content one', 1.0, 'active', 1.0, 0, 0, 0, '["test"]', NULL)"#,
            params![&id1]
        ).unwrap();
        storage.connection().execute(
            r#"INSERT INTO engrams (id, content, energy, state, confidence, created_at, last_accessed, access_count, tags, embedding)
               VALUES (?1, 'Test content two', 1.0, 'active', 1.0, 0, 0, 0, '["test"]', NULL)"#,
            params![&id2]
        ).unwrap();
        
        // FTS should need rebuild
        assert!(storage.fts_needs_rebuild().unwrap());
        
        // Search should find nothing before rebuild
        let results = storage.search_fts("content").unwrap();
        assert_eq!(results.len(), 0);
        
        // Rebuild index
        let count = storage.rebuild_fts_index().unwrap();
        assert_eq!(count, 2);
        
        // Now search should work
        let results = storage.search_fts("content").unwrap();
        assert_eq!(results.len(), 2);
        
        // FTS should not need rebuild anymore
        assert!(!storage.fts_needs_rebuild().unwrap());
    }
}
