//! Cross-encoder re-ranking using fastembed's TextRerank
//!
//! Provides a second-pass re-ranking step for search results. After cosine
//! similarity retrieves initial candidates, the cross-encoder scores each
//! (query, document) pair for much better relevance ordering.
//!
//! The reranker model is lazy-loaded on first use (~50MB download).

use crate::config;
use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
use std::sync::Mutex;

/// Global reranker model (lazy-loaded on first use)
static RERANKER: Mutex<Option<TextRerank>> = Mutex::new(None);

/// Result of re-ranking: original index and new score
#[derive(Debug, Clone)]
pub struct RerankScore {
    /// Index into the original documents slice
    pub index: usize,
    /// Cross-encoder relevance score
    pub score: f32,
}

/// Errors from re-ranking
#[derive(Debug, thiserror::Error)]
pub enum RerankError {
    #[error("Failed to load reranker model: {0}")]
    ModelLoad(String),
    #[error("Re-ranking failed: {0}")]
    Rerank(String),
    #[error("Lock poisoned: {0}")]
    Lock(String),
}

/// Ensure the global reranker model is loaded.
fn ensure_loaded() -> Result<(), RerankError> {
    let mut guard = RERANKER
        .lock()
        .map_err(|e| RerankError::Lock(e.to_string()))?;

    if guard.is_some() {
        return Ok(());
    }

    let cache_dir = config::get_model_cache_dir();
    std::fs::create_dir_all(&cache_dir).ok();
    eprintln!(
        "[reranker] Loading BGERerankerBase (cache: {})...",
        cache_dir.display()
    );

    let options = RerankInitOptions::new(RerankerModel::BGERerankerBase)
        .with_cache_dir(cache_dir)
        .with_show_download_progress(true);

    let reranker =
        TextRerank::try_new(options).map_err(|e| RerankError::ModelLoad(e.to_string()))?;

    *guard = Some(reranker);
    eprintln!("[reranker] BGERerankerBase loaded successfully");
    Ok(())
}

/// Re-rank documents against a query using a cross-encoder model.
///
/// Returns a vec of (original_index, rerank_score) sorted by score descending.
/// Documents are not modified — only their ordering changes.
///
/// # Arguments
/// * `query` - The search query
/// * `documents` - Slice of document strings to re-rank
///
/// # Returns
/// Vec of `RerankScore` sorted by score descending (most relevant first)
pub fn rerank(query: &str, documents: &[&str]) -> Result<Vec<RerankScore>, RerankError> {
    if documents.is_empty() {
        return Ok(Vec::new());
    }

    ensure_loaded()?;

    let mut guard = RERANKER
        .lock()
        .map_err(|e| RerankError::Lock(e.to_string()))?;

    let reranker = guard.as_mut().unwrap();

    let results = reranker
        .rerank(query, documents, false, None)
        .map_err(|e| RerankError::Rerank(e.to_string()))?;

    Ok(results
        .into_iter()
        .map(|r| RerankScore {
            index: r.index,
            score: r.score,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reranker_handles_empty_input() {
        let result = rerank("some query", &[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn reranker_loads_and_scores() {
        let query = "What is machine learning?";
        let documents = [
            "Machine learning is a subset of artificial intelligence that learns from data",
            "The weather today is sunny and warm",
            "Deep learning uses neural networks for pattern recognition",
        ];

        let results = rerank(query, &documents).unwrap();

        // Should return one result per document
        assert_eq!(results.len(), 3);

        // All indices should be valid
        for r in &results {
            assert!(r.index < documents.len());
        }
    }

    #[test]
    fn reranker_reorders_results() {
        let query = "Rust programming language memory safety";
        let documents = [
            "How to bake chocolate chip cookies at home",
            "Rust is a systems programming language focused on safety, speed, and concurrency",
            "The history of ancient Egyptian pyramids",
        ];

        let results = rerank(query, &documents).unwrap();

        // The Rust document (index 1) should be ranked first
        assert_eq!(
            results[0].index, 1,
            "The Rust programming document should be ranked highest, got index {}",
            results[0].index
        );

        // The Rust doc score should be higher than the others
        assert!(
            results[0].score > results[1].score,
            "Top result score ({}) should be > second ({})",
            results[0].score,
            results[1].score
        );
    }
}
