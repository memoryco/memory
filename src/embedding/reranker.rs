//! Cross-encoder re-ranking (stub)
//!
//! The fastembed/ONNX cross-encoder has been removed. This module provides
//! the same API surface so that search.rs compiles, but always returns an
//! error — causing the search pipeline to fall back gracefully.
//!
//! The nemotron-rerank-1b-v2 cross-encoder will replace this once wired
//! through memoryco-llm.

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

/// Re-rank documents against a query using a cross-encoder model.
///
/// Currently returns an error — the fastembed cross-encoder has been removed.
/// The search pipeline falls back gracefully to cosine ordering.
pub fn rerank(_query: &str, _documents: &[&str]) -> Result<Vec<RerankScore>, RerankError> {
    Err(RerankError::Rerank(
        "fastembed cross-encoder removed; nemotron reranker pending integration".to_string(),
    ))
}
