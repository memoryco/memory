//! Vector similarity search operations
//! 
//! Provides cosine similarity search over engram embeddings stored in SQLite.
//! Uses Diesel's sql_query for raw SQL operations.

use crate::engram::EngramId;
use crate::engram::storage::{StorageResult, StorageError};
use crate::embedding::cosine_similarity;
use super::models::{embedding_to_bytes, bytes_to_embedding};

use diesel::prelude::*;
use diesel::sqlite::SqliteConnection;
use diesel::sql_query;
use diesel::sql_types::{Text, Nullable, Binary};

/// Result of a vector similarity search
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    pub id: EngramId,
    pub score: f32,
    pub content: String,
}

/// Row returned from embedding queries
#[derive(QueryableByName, Debug)]
struct EmbeddingRow {
    #[diesel(sql_type = Text)]
    id: String,
    #[diesel(sql_type = Text)]
    content: String,
    #[diesel(sql_type = Binary)]
    embedding: Vec<u8>,
}

/// Row for optional embedding
#[derive(QueryableByName, Debug)]
struct OptionalEmbeddingRow {
    #[diesel(sql_type = Nullable<Binary>)]
    embedding: Option<Vec<u8>>,
}

/// Row for ID only
#[derive(QueryableByName, Debug)]
struct IdRow {
    #[diesel(sql_type = Text)]
    id: String,
}

/// Row for count
#[derive(QueryableByName, Debug)]
struct CountRow {
    #[diesel(sql_type = diesel::sql_types::BigInt)]
    cnt: i64,
}

/// Vector search operations on SQLite storage
pub struct VectorSearch<'a> {
    conn: &'a mut SqliteConnection,
}

impl<'a> VectorSearch<'a> {
    /// Create a new VectorSearch instance
    pub fn new(conn: &'a mut SqliteConnection) -> Self {
        Self { conn }
    }
    
    /// Find engrams most similar to the given embedding
    /// 
    /// Returns up to `limit` results sorted by descending similarity score.
    /// Only returns results with similarity >= min_score.
    pub fn find_similar(
        &mut self,
        query_embedding: &[f32],
        limit: usize,
        min_score: f32,
    ) -> StorageResult<Vec<SimilarityResult>> {
        let rows: Vec<EmbeddingRow> = sql_query(
            "SELECT id, content, embedding FROM engrams WHERE embedding IS NOT NULL"
        )
        .load(self.conn)
        .map_err(|e| StorageError::Database(e.to_string()))?;
        
        let mut results: Vec<SimilarityResult> = Vec::new();
        
        for row in rows {
            if let Some(embedding) = bytes_to_embedding(&row.embedding) {
                let score = cosine_similarity(query_embedding, &embedding);
                
                if score >= min_score {
                    let id = uuid::Uuid::parse_str(&row.id)
                        .map_err(|e| StorageError::Serialization(e.to_string()))?;
                    
                    results.push(SimilarityResult { id, score, content: row.content });
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
        &mut self,
        engram_id: &EngramId,
        limit: usize,
        min_score: f32,
    ) -> StorageResult<Vec<SimilarityResult>> {
        // First, get the embedding for the source engram
        let id_str = engram_id.to_string();
        let row: OptionalEmbeddingRow = sql_query(
            "SELECT embedding FROM engrams WHERE id = ?"
        )
        .bind::<Text, _>(&id_str)
        .get_result(self.conn)
        .map_err(|e| StorageError::Database(e.to_string()))?;
        
        let embedding_bytes = row.embedding
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
    pub fn count_with_embeddings(&mut self) -> StorageResult<usize> {
        let row: CountRow = sql_query(
            "SELECT COUNT(*) as cnt FROM engrams WHERE embedding IS NOT NULL"
        )
        .get_result(self.conn)
        .map_err(|e| StorageError::Database(e.to_string()))?;
        
        Ok(row.cnt as usize)
    }
    
    /// Count engrams that need embeddings
    pub fn count_without_embeddings(&mut self) -> StorageResult<usize> {
        let row: CountRow = sql_query(
            "SELECT COUNT(*) as cnt FROM engrams WHERE embedding IS NULL"
        )
        .get_result(self.conn)
        .map_err(|e| StorageError::Database(e.to_string()))?;
        
        Ok(row.cnt as usize)
    }
    
    /// Get IDs of engrams that need embeddings (for backfill)
    pub fn get_ids_without_embeddings(&mut self, limit: usize) -> StorageResult<Vec<EngramId>> {
        let rows: Vec<IdRow> = sql_query(
            "SELECT id FROM engrams WHERE embedding IS NULL LIMIT ?"
        )
        .bind::<diesel::sql_types::BigInt, _>(limit as i64)
        .load(self.conn)
        .map_err(|e| StorageError::Database(e.to_string()))?;
        
        let mut ids = Vec::new();
        for row in rows {
            let id = uuid::Uuid::parse_str(&row.id)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
            ids.push(id);
        }
        
        Ok(ids)
    }
    
    /// Update embedding for a single engram
    pub fn set_embedding(&mut self, id: &EngramId, embedding: &[f32]) -> StorageResult<()> {
        let bytes = embedding_to_bytes(embedding);
        let id_str = id.to_string();
        
        sql_query("UPDATE engrams SET embedding = ? WHERE id = ?")
            .bind::<Nullable<Binary>, _>(Some(&bytes[..]))
            .bind::<Text, _>(&id_str)
            .execute(self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;
        
        Ok(())
    }
    
    /// Get embedding for a single engram
    pub fn get_embedding(&mut self, id: &EngramId) -> StorageResult<Option<Vec<f32>>> {
        let id_str = id.to_string();
        
        let row: OptionalEmbeddingRow = sql_query(
            "SELECT embedding FROM engrams WHERE id = ?"
        )
        .bind::<Text, _>(&id_str)
        .get_result(self.conn)
        .map_err(|e| StorageError::Database(e.to_string()))?;
        
        match row.embedding {
            Some(bytes) => Ok(bytes_to_embedding(&bytes)),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::storage::EngramStorage;
    use crate::engram::Engram;
    use crate::engram::storage::Storage;
    
    #[test]
    fn vector_search_basics() {
        let mut storage = EngramStorage::in_memory().unwrap();
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
        let mut vs = VectorSearch::new(storage.connection());
        let results = vs.find_similar(&[1.0, 0.0, 0.0], 10, 0.5).unwrap();
        
        assert_eq!(results.len(), 2); // e1 and e2, not e3 (orthogonal)
        assert_eq!(results[0].id, e1.id); // e1 is most similar (exact match)
        assert_eq!(results[1].id, e2.id); // e2 is second
    }
    
    #[test]
    fn find_similar_to_engram() {
        let mut storage = EngramStorage::in_memory().unwrap();
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
        
        let mut vs = VectorSearch::new(storage.connection());
        let results = vs.find_similar_to(&e1.id, 10, 0.5).unwrap();
        
        // Should find e2 but not e3, and not e1 itself
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, e2.id);
    }
    
    #[test]
    fn count_embeddings() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();
        
        let mut e1 = Engram::new("With embedding");
        e1.embedding = Some(vec![1.0, 0.0, 0.0]);
        
        let e2 = Engram::new("Without embedding");
        
        storage.save_engram(&e1).unwrap();
        storage.save_engram(&e2).unwrap();
        
        let mut vs = VectorSearch::new(storage.connection());
        
        assert_eq!(vs.count_with_embeddings().unwrap(), 1);
        assert_eq!(vs.count_without_embeddings().unwrap(), 1);
    }
}
