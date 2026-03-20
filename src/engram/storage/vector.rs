//! Vector similarity search operations
//!
//! Provides cosine similarity search over engram embeddings stored in SQLite.
//! Uses Diesel's sql_query for raw SQL operations.

use super::models::{bytes_to_embedding, embedding_to_bytes};
use crate::embedding::cosine_similarity;
use crate::engram::EngramId;
use crate::engram::storage::{StorageError, StorageResult};

use diesel::prelude::*;
use diesel::sql_query;
use diesel::sql_types::{Binary, Nullable, Text};
use diesel::sqlite::SqliteConnection;

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

/// Row returned from enrichment embedding queries
#[derive(QueryableByName, Debug)]
struct EnrichmentRow {
    #[diesel(sql_type = Text)]
    engram_id: String,
    #[diesel(sql_type = Text)]
    content: String,
    #[diesel(sql_type = Binary)]
    embedding: Vec<u8>,
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
    /// Searches BOTH the primary engram embeddings AND enrichment vectors,
    /// deduplicating by engram_id (keeping highest score per engram).
    pub fn find_similar(
        &mut self,
        query_embedding: &[f32],
        limit: usize,
        min_score: f32,
    ) -> StorageResult<Vec<SimilarityResult>> {
        // best_scores: engram_id -> (score, content)
        let mut best_scores: std::collections::HashMap<uuid::Uuid, (f32, String)> =
            std::collections::HashMap::new();

        // Search primary engram embeddings
        let rows: Vec<EmbeddingRow> =
            sql_query("SELECT id, content, embedding FROM engrams WHERE embedding IS NOT NULL")
                .load(self.conn)
                .map_err(|e| StorageError::Database(e.to_string()))?;

        for row in rows {
            if let Some(embedding) = bytes_to_embedding(&row.embedding) {
                if embedding.len() != query_embedding.len() {
                    continue;
                }
                let score = cosine_similarity(query_embedding, &embedding);
                if score >= min_score {
                    let id = uuid::Uuid::parse_str(&row.id)
                        .map_err(|e| StorageError::Serialization(e.to_string()))?;
                    let entry = best_scores.entry(id).or_insert((score, row.content.clone()));
                    if score > entry.0 {
                        *entry = (score, row.content.clone());
                    }
                }
            }
        }

        // Search enrichment embeddings (JOIN to get parent content)
        let enrichment_rows: Vec<EnrichmentRow> = sql_query(
            "SELECT e.engram_id, eng.content, e.embedding \
             FROM engram_enrichments e \
             JOIN engrams eng ON eng.id = e.engram_id",
        )
        .load(self.conn)
        .map_err(|e| StorageError::Database(e.to_string()))?;

        for row in enrichment_rows {
            if let Some(embedding) = bytes_to_embedding(&row.embedding) {
                if embedding.len() != query_embedding.len() {
                    continue;
                }
                let score = cosine_similarity(query_embedding, &embedding);
                if score >= min_score {
                    let id = uuid::Uuid::parse_str(&row.engram_id)
                        .map_err(|e| StorageError::Serialization(e.to_string()))?;
                    let entry = best_scores
                        .entry(id)
                        .or_insert((score, row.content.clone()));
                    if score > entry.0 {
                        *entry = (score, row.content.clone());
                    }
                }
            }
        }

        let mut results: Vec<SimilarityResult> = best_scores
            .into_iter()
            .map(|(id, (score, content))| SimilarityResult { id, score, content })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit results
        results.truncate(limit);

        Ok(results)
    }

    /// Find engrams similar to a given engram by ID
    #[allow(dead_code)]
    pub fn find_similar_to(
        &mut self,
        engram_id: &EngramId,
        limit: usize,
        min_score: f32,
    ) -> StorageResult<Vec<SimilarityResult>> {
        // First, get the embedding for the source engram
        let id_str = engram_id.to_string();
        let row: OptionalEmbeddingRow = sql_query("SELECT embedding FROM engrams WHERE id = ?")
            .bind::<Text, _>(&id_str)
            .get_result(self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;

        let embedding_bytes = row.embedding.ok_or_else(|| {
            StorageError::NotFound(format!("No embedding for engram {}", engram_id))
        })?;

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
        let row: CountRow =
            sql_query("SELECT COUNT(*) as cnt FROM engrams WHERE embedding IS NOT NULL")
                .get_result(self.conn)
                .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(row.cnt as usize)
    }

    /// Count engrams that need embeddings
    pub fn count_without_embeddings(&mut self) -> StorageResult<usize> {
        let row: CountRow =
            sql_query("SELECT COUNT(*) as cnt FROM engrams WHERE embedding IS NULL")
                .get_result(self.conn)
                .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(row.cnt as usize)
    }

    /// Get IDs of engrams that need embeddings (for backfill)
    pub fn get_ids_without_embeddings(&mut self, limit: usize) -> StorageResult<Vec<EngramId>> {
        let rows: Vec<IdRow> = sql_query("SELECT id FROM engrams WHERE embedding IS NULL LIMIT ?")
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

    /// Clear all embeddings (set to NULL) for model migration.
    /// Returns the number of affected rows.
    pub fn clear_all_embeddings(&mut self) -> StorageResult<usize> {
        let result = sql_query("UPDATE engrams SET embedding = NULL WHERE embedding IS NOT NULL")
            .execute(self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;
        Ok(result)
    }

    /// Get embedding for a single engram
    pub fn get_embedding(&mut self, id: &EngramId) -> StorageResult<Option<Vec<f32>>> {
        let id_str = id.to_string();

        let row: OptionalEmbeddingRow = sql_query("SELECT embedding FROM engrams WHERE id = ?")
            .bind::<Text, _>(&id_str)
            .get_result(self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;

        match row.embedding {
            Some(bytes) => Ok(bytes_to_embedding(&bytes)),
            None => Ok(None),
        }
    }

    /// Store enrichment embeddings for an engram.
    /// Replaces any existing enrichments for this engram.
    pub fn set_enrichment_embeddings(
        &mut self,
        id: &EngramId,
        embeddings: &[Vec<f32>],
        source: &str,
    ) -> StorageResult<()> {
        let id_str = id.to_string();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        // Delete existing enrichments for this engram
        sql_query("DELETE FROM engram_enrichments WHERE engram_id = ?")
            .bind::<Text, _>(&id_str)
            .execute(self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;

        // Insert new enrichments with seq 0, 1, 2...
        for (seq, embedding) in embeddings.iter().enumerate() {
            let bytes = embedding_to_bytes(embedding);
            sql_query(
                "INSERT INTO engram_enrichments (engram_id, seq, embedding, source, created_at) \
                 VALUES (?, ?, ?, ?, ?)",
            )
            .bind::<Text, _>(&id_str)
            .bind::<diesel::sql_types::BigInt, _>(seq as i64)
            .bind::<Binary, _>(&bytes[..])
            .bind::<Text, _>(source)
            .bind::<diesel::sql_types::BigInt, _>(now)
            .execute(self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;
        }

        Ok(())
    }

    /// Delete all enrichment embeddings for an engram.
    pub fn delete_enrichments(&mut self, id: &EngramId) -> StorageResult<()> {
        let id_str = id.to_string();
        sql_query("DELETE FROM engram_enrichments WHERE engram_id = ?")
            .bind::<Text, _>(&id_str)
            .execute(self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;
        Ok(())
    }

    /// Count total enrichment vectors across all engrams.
    pub fn count_enrichments(&mut self) -> StorageResult<usize> {
        let row: CountRow = sql_query("SELECT COUNT(*) as cnt FROM engram_enrichments")
            .get_result(self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;
        Ok(row.cnt as usize)
    }

    /// Clear all enrichment embeddings for migration purposes.
    /// Returns the number of affected rows.
    pub fn clear_all_enrichments(&mut self) -> StorageResult<usize> {
        let result = sql_query("DELETE FROM engram_enrichments")
            .execute(self.conn)
            .map_err(|e| StorageError::Database(e.to_string()))?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::Engram;
    use crate::engram::storage::EngramStorage;
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

    #[test]
    fn clear_all_embeddings() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let mut e1 = Engram::new("Has embedding 1");
        e1.embedding = Some(vec![1.0, 0.0, 0.0]);

        let mut e2 = Engram::new("Has embedding 2");
        e2.embedding = Some(vec![0.0, 1.0, 0.0]);

        let e3 = Engram::new("No embedding");

        storage.save_engram(&e1).unwrap();
        storage.save_engram(&e2).unwrap();
        storage.save_engram(&e3).unwrap();

        let mut vs = VectorSearch::new(storage.connection());
        assert_eq!(vs.count_with_embeddings().unwrap(), 2);

        let cleared = vs.clear_all_embeddings().unwrap();
        assert_eq!(
            cleared, 2,
            "Should clear exactly the 2 rows with embeddings"
        );

        assert_eq!(vs.count_with_embeddings().unwrap(), 0);
        assert_eq!(vs.count_without_embeddings().unwrap(), 3);
    }

    #[test]
    fn clear_all_embeddings_empty_db() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let mut vs = VectorSearch::new(storage.connection());
        let cleared = vs.clear_all_embeddings().unwrap();
        assert_eq!(cleared, 0);
    }

    #[test]
    fn no_hardcoded_dimension_in_storage() {
        // Verify that the storage layer (SQL, blob format) has no hardcoded
        // dimension assumptions. Both small and large embeddings are stored
        // and retrieved correctly via the same code paths.
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        // Simulate 384-dim embedding (old model)
        let small_emb: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
        let mut e1 = Engram::new("Small embedding");
        e1.embedding = Some(small_emb.clone());
        storage.save_engram(&e1).unwrap();

        // Simulate 1024-dim embedding (new model)
        let large_emb: Vec<f32> = (0..1024).map(|i| (i as f32) / 1024.0).collect();
        let mut e2 = Engram::new("Large embedding");
        e2.embedding = Some(large_emb.clone());
        storage.save_engram(&e2).unwrap();

        // Both should be stored and retrievable
        let mut vs = VectorSearch::new(storage.connection());
        assert_eq!(vs.count_with_embeddings().unwrap(), 2);

        // Roundtrip: get them back and check dimensions
        let got1 = vs.get_embedding(&e1.id).unwrap().unwrap();
        assert_eq!(got1.len(), 384);

        let got2 = vs.get_embedding(&e2.id).unwrap().unwrap();
        assert_eq!(got2.len(), 1024);

        // Search with a 1024-dim query — only e2 will match dimensions
        // (cosine_similarity asserts matching dimensions, so mismatched entries
        // will be skipped by the search since bytes_to_embedding returns all of them
        // but find_similar computes cosine_similarity which panics on mismatch.
        // This is correct: after migration, all embeddings have the same dimension.)

        // After clearing and re-embedding (as migration does), all are same dimension.
        let cleared = vs.clear_all_embeddings().unwrap();
        assert_eq!(cleared, 2);

        // Re-embed both with 1024 dims
        let new_emb1: Vec<f32> = (0..1024).map(|i| (i as f32 + 0.5) / 1024.0).collect();
        let new_emb2: Vec<f32> = (0..1024).map(|i| (i as f32 + 1.0) / 1024.0).collect();
        vs.set_embedding(&e1.id, &new_emb1).unwrap();
        vs.set_embedding(&e2.id, &new_emb2).unwrap();

        // Now both are 1024-dim, search should work
        let results = vs.find_similar(&new_emb1, 10, 0.0).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn enrichment_vectors_stored_and_retrieved() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let e1 = Engram::new("Test memory");
        storage.save_engram(&e1).unwrap();

        let mut vs = VectorSearch::new(storage.connection());
        assert_eq!(vs.count_enrichments().unwrap(), 0);

        vs.set_enrichment_embeddings(
            &e1.id,
            &[vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
            "llm",
        )
        .unwrap();

        assert_eq!(vs.count_enrichments().unwrap(), 2);
    }

    #[test]
    fn find_similar_includes_enrichment_vectors() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        // Engram whose primary embedding does NOT match the query
        let mut e1 = Engram::new("Memory about cooking");
        e1.embedding = Some(vec![0.0, 1.0, 0.0]); // orthogonal to query
        storage.save_engram(&e1).unwrap();

        let mut vs = VectorSearch::new(storage.connection());

        // Add an enrichment embedding that DOES match the query
        vs.set_enrichment_embeddings(&e1.id, &[vec![1.0, 0.0, 0.0]], "llm")
            .unwrap();

        // Query along x-axis — primary embedding won't match (min_score 0.5), enrichment will
        let results = vs.find_similar(&[1.0, 0.0, 0.0], 10, 0.5).unwrap();

        assert_eq!(results.len(), 1, "Should find e1 via its enrichment vector");
        assert_eq!(results[0].id, e1.id);
        // The enrichment vector is an exact match so score should be 1.0
        assert!((results[0].score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn find_similar_deduplicates_by_engram_id() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        // Engram whose primary embedding also matches the query
        let mut e1 = Engram::new("Memory that matches multiple ways");
        e1.embedding = Some(vec![0.9, 0.1, 0.0]); // primary: good but not perfect
        storage.save_engram(&e1).unwrap();

        let mut vs = VectorSearch::new(storage.connection());

        // Enrichment vector: even better match
        vs.set_enrichment_embeddings(&e1.id, &[vec![1.0, 0.0, 0.0]], "llm")
            .unwrap();

        // Query along x-axis — both primary and enrichment match
        let results = vs.find_similar(&[1.0, 0.0, 0.0], 10, 0.0).unwrap();

        // Should only return ONE result for e1
        assert_eq!(results.len(), 1, "Should deduplicate to one result per engram");
        assert_eq!(results[0].id, e1.id);
        // The enrichment vector is an exact match so score should be 1.0 (highest wins)
        assert!(
            (results[0].score - 1.0).abs() < 1e-5,
            "Should return the higher score (enrichment), got {}",
            results[0].score
        );
    }

    #[test]
    fn enrichment_cleanup_on_re_enrichment() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let e1 = Engram::new("Some memory");
        storage.save_engram(&e1).unwrap();

        let mut vs = VectorSearch::new(storage.connection());

        // First enrichment: 3 vectors
        vs.set_enrichment_embeddings(
            &e1.id,
            &[vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]],
            "llm",
        )
        .unwrap();
        assert_eq!(vs.count_enrichments().unwrap(), 3);

        // Re-enrich: 2 vectors — old ones should be replaced
        vs.set_enrichment_embeddings(
            &e1.id,
            &[vec![0.7, 0.3, 0.0], vec![0.3, 0.7, 0.0]],
            "llm",
        )
        .unwrap();
        assert_eq!(vs.count_enrichments().unwrap(), 2, "Old enrichments should be replaced");
    }

    #[test]
    fn delete_enrichments_removes_all() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let mut e1 = Engram::new("Memory with enrichment");
        e1.embedding = Some(vec![0.0, 1.0, 0.0]); // orthogonal to query
        storage.save_engram(&e1).unwrap();

        let mut vs = VectorSearch::new(storage.connection());

        // Add enrichment that matches query
        vs.set_enrichment_embeddings(&e1.id, &[vec![1.0, 0.0, 0.0]], "llm")
            .unwrap();
        assert_eq!(vs.count_enrichments().unwrap(), 1);

        // Verify search finds it via enrichment
        let results = vs.find_similar(&[1.0, 0.0, 0.0], 10, 0.5).unwrap();
        assert_eq!(results.len(), 1);

        // Delete enrichments
        vs.delete_enrichments(&e1.id).unwrap();
        assert_eq!(vs.count_enrichments().unwrap(), 0);

        // Search should no longer find via enrichment (primary doesn't match either)
        let results = vs.find_similar(&[1.0, 0.0, 0.0], 10, 0.5).unwrap();
        assert_eq!(results.len(), 0, "Should not find after enrichment deletion");
    }

    #[test]
    fn cascade_delete_removes_enrichments() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let e1 = Engram::new("Memory to delete");
        storage.save_engram(&e1).unwrap();

        let mut vs = VectorSearch::new(storage.connection());
        vs.set_enrichment_embeddings(&e1.id, &[vec![1.0, 0.0, 0.0]], "llm")
            .unwrap();
        assert_eq!(vs.count_enrichments().unwrap(), 1);

        // Delete the parent engram — FK CASCADE should remove enrichments too
        drop(vs);
        storage.delete_engram(&e1.id).unwrap();

        let mut vs = VectorSearch::new(storage.connection());
        assert_eq!(
            vs.count_enrichments().unwrap(),
            0,
            "Enrichments should be cascade-deleted with the parent engram"
        );
    }

    #[test]
    fn clear_all_enrichments() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let e1 = Engram::new("Memory one");
        let e2 = Engram::new("Memory two");
        storage.save_engram(&e1).unwrap();
        storage.save_engram(&e2).unwrap();

        let mut vs = VectorSearch::new(storage.connection());

        vs.set_enrichment_embeddings(&e1.id, &[vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]], "llm")
            .unwrap();
        vs.set_enrichment_embeddings(&e2.id, &[vec![0.0, 0.0, 1.0]], "llm")
            .unwrap();
        assert_eq!(vs.count_enrichments().unwrap(), 3);

        let cleared = vs.clear_all_enrichments().unwrap();
        assert_eq!(cleared, 3, "Should clear all 3 enrichment vectors");
        assert_eq!(vs.count_enrichments().unwrap(), 0);
    }

    #[test]
    fn clear_all_enrichments_empty_db() {
        let mut storage = EngramStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let mut vs = VectorSearch::new(storage.connection());
        let cleared = vs.clear_all_enrichments().unwrap();
        assert_eq!(cleared, 0);
    }
}
