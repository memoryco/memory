//! Vector similarity search operations
//! 
//! Provides cosine similarity search over engram embeddings stored in SQLite.
//! Kept separate from sqlite.rs to manage file size.

use crate::engram::{EngramId, Engram};
use crate::engram::storage::{StorageResult, StorageError};
use crate::embedding::cosine_similarity;
use rusqlite::{Connection, params};

/// Result of a vector similarity search
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    pub id: EngramId,
    pub score: f32,
    pub content: String,
}

/// Vector search operations on SQLite storage
pub struct VectorSearch<'a> {
    conn: &'a Connection,
}

impl<'a> VectorSearch<'a> {
    /// Create a new VectorSearch instance
    pub fn new(conn: &'a Connection) -> Self {
        Self { conn }
    }
    
    /// Find engrams most similar to the given embedding
    /// 
    /// Returns up to `limit` results sorted by descending similarity score.
    /// Only returns results with similarity >= min_score.
    pub fn find_similar(
        &self,
        query_embedding: &[f32],
        limit: usize,
        min_score: f32,
    ) -> StorageResult<Vec<SimilarityResult>> {
        // Load all engrams with embeddings
        let mut stmt = self.conn.prepare(
            "SELECT id, content, embedding FROM engrams WHERE embedding IS NOT NULL"
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Vec<u8>>(2)?,
            ))
        }).map_err(|e| StorageError::Database(e.to_string()))?;
        
        let mut results: Vec<SimilarityResult> = Vec::new();
        
        for row in rows {
            let (id_str, content, embedding_bytes) = row
                .map_err(|e| StorageError::Database(e.to_string()))?;
            
            // Deserialize embedding
            if let Some(embedding) = bytes_to_embedding(&embedding_bytes) {
                let score = cosine_similarity(query_embedding, &embedding);
                
                if score >= min_score {
                    let id = uuid::Uuid::parse_str(&id_str)
                        .map_err(|e| StorageError::Serialization(e.to_string()))?;
                    
                    results.push(SimilarityResult { id, score, content });
                }
            }
        }
        
        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit results
        results.truncate(limit);
        
        Ok(results)
    }
    
    /// Find engrams similar to a given engram by ID
    pub fn find_similar_to(
        &self,
        engram_id: &EngramId,
        limit: usize,
        min_score: f32,
    ) -> StorageResult<Vec<SimilarityResult>> {
        // First, get the embedding for the source engram
        let embedding_bytes: Option<Vec<u8>> = self.conn.query_row(
            "SELECT embedding FROM engrams WHERE id = ?1",
            params![engram_id.to_string()],
            |row| row.get(0)
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        let embedding_bytes = embedding_bytes
            .ok_or_else(|| StorageError::NotFound(format!("No embedding for engram {}", engram_id)))?;
        
        let query_embedding = bytes_to_embedding(&embedding_bytes)
            .ok_or_else(|| StorageError::Serialization("Invalid embedding format".into()))?;
        
        // Find similar, excluding the source engram
        let mut results = self.find_similar(&query_embedding, limit + 1, min_score)?;
        results.retain(|r| r.id != *engram_id);
        results.truncate(limit);
        
        Ok(results)
    }
    
    /// Count engrams that have embeddings
    pub fn count_with_embeddings(&self) -> StorageResult<usize> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM engrams WHERE embedding IS NOT NULL",
            [],
            |row| row.get(0)
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        Ok(count as usize)
    }
    
    /// Count engrams that need embeddings
    pub fn count_without_embeddings(&self) -> StorageResult<usize> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM engrams WHERE embedding IS NULL",
            [],
            |row| row.get(0)
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        Ok(count as usize)
    }
    
    /// Get IDs of engrams that need embeddings (for backfill)
    pub fn get_ids_without_embeddings(&self, limit: usize) -> StorageResult<Vec<EngramId>> {
        let mut stmt = self.conn.prepare(
            "SELECT id FROM engrams WHERE embedding IS NULL LIMIT ?1"
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        let rows = stmt.query_map(params![limit as i64], |row| {
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
    
    /// Update embedding for a single engram
    pub fn set_embedding(&self, id: &EngramId, embedding: &[f32]) -> StorageResult<()> {
        let bytes = embedding_to_bytes(embedding);
        
        self.conn.execute(
            "UPDATE engrams SET embedding = ?1 WHERE id = ?2",
            params![bytes, id.to_string()]
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        Ok(())
    }
    
    /// Get embedding for a single engram
    pub fn get_embedding(&self, id: &EngramId) -> StorageResult<Option<Vec<f32>>> {
        let embedding_bytes: Option<Vec<u8>> = self.conn.query_row(
            "SELECT embedding FROM engrams WHERE id = ?1",
            params![id.to_string()],
            |row| row.get(0)
        ).map_err(|e| StorageError::Database(e.to_string()))?;
        
        match embedding_bytes {
            Some(bytes) => Ok(bytes_to_embedding(&bytes)),
            None => Ok(None),
        }
    }
}

// Embedding serialization helpers (duplicated from sqlite.rs to avoid coupling)
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
    use crate::engram::storage::SqliteStorage;
    use crate::engram::Engram;
    use crate::engram::storage::Storage;
    
    #[test]
    fn vector_search_basics() {
        let mut storage = SqliteStorage::in_memory().unwrap();
        storage.initialize().unwrap();
        
        // Create engrams with embeddings
        let mut e1 = Engram::new("Rust programming language");
        e1.embedding = Some(vec![1.0, 0.0, 0.0]); // Unit vector along x
        
        let mut e2 = Engram::new("Python programming language");
        e2.embedding = Some(vec![0.9, 0.1, 0.0]); // Similar to e1
        
        let mut e3 = Engram::new("Cooking recipes");
        e3.embedding = Some(vec![0.0, 1.0, 0.0]); // Orthogonal to e1
        
        storage.save_engram(&e1).unwrap();
        storage.save_engram(&e2).unwrap();
        storage.save_engram(&e3).unwrap();
        
        // Search for similar to e1's embedding
        let vs = VectorSearch::new(storage.connection());
        let results = vs.find_similar(&[1.0, 0.0, 0.0], 10, 0.5).unwrap();
        
        assert_eq!(results.len(), 2); // e1 and e2, not e3 (orthogonal)
        assert_eq!(results[0].id, e1.id); // e1 is most similar (exact match)
        assert_eq!(results[1].id, e2.id); // e2 is second
    }
    
    #[test]
    fn find_similar_to_engram() {
        let mut storage = SqliteStorage::in_memory().unwrap();
        storage.initialize().unwrap();
        
        let mut e1 = Engram::new("Machine learning basics");
        e1.embedding = Some(vec![1.0, 0.0, 0.0]);
        
        let mut e2 = Engram::new("Deep learning neural networks");
        e2.embedding = Some(vec![0.95, 0.05, 0.0]);
        
        let mut e3 = Engram::new("Gardening tips");
        e3.embedding = Some(vec![0.0, 0.0, 1.0]);
        
        storage.save_engram(&e1).unwrap();
        storage.save_engram(&e2).unwrap();
        storage.save_engram(&e3).unwrap();
        
        let vs = VectorSearch::new(storage.connection());
        let results = vs.find_similar_to(&e1.id, 10, 0.5).unwrap();
        
        // Should find e2 but not e3, and not e1 itself
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, e2.id);
    }
    
    #[test]
    fn count_embeddings() {
        let mut storage = SqliteStorage::in_memory().unwrap();
        storage.initialize().unwrap();
        
        let mut e1 = Engram::new("With embedding");
        e1.embedding = Some(vec![1.0, 0.0, 0.0]);
        
        let e2 = Engram::new("Without embedding");
        
        storage.save_engram(&e1).unwrap();
        storage.save_engram(&e2).unwrap();
        
        let vs = VectorSearch::new(storage.connection());
        
        assert_eq!(vs.count_with_embeddings().unwrap(), 1);
        assert_eq!(vs.count_without_embeddings().unwrap(), 1);
    }
}
