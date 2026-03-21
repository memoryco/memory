//! Cross-encoder re-ranking via nemotron-rerank-1b-v2 GGUF.
//!
//! Uses [`memoryco_llm::rerank::LocalRerankerService`] to score
//! (query, document) pairs through the Nemotron cross-encoder model.
//! The model is lazy-loaded on first call and shared via a global static.
//!
//! Only called when `rerank_mode` is `"cross-encoder"` or `"hybrid"`,
//! so the model is never loaded for `"off"` or `"llm"` modes.

use memoryco_llm::rerank::{LocalRerankerService, RerankConfig};
use std::sync::Mutex;

/// Hardcoded reranker model constants (appliance mode).
const RERANK_MODEL_FILENAME: &str = "llama-nemotron-rerank-1b-v2.Q4_K_M.gguf";

/// Global reranker service (lazy-loaded on first use).
static RERANK_SERVICE: Mutex<Option<LocalRerankerService>> = Mutex::new(None);

/// Result of re-ranking: original index and new score.
#[derive(Debug, Clone)]
pub struct RerankScore {
    /// Index into the original documents slice.
    pub index: usize,
    /// Cross-encoder relevance score.
    pub score: f32,
}

/// Errors from re-ranking.
#[derive(Debug, thiserror::Error)]
pub enum RerankError {
    #[error("Failed to load reranker model: {0}")]
    ModelLoad(String),
    #[error("Re-ranking failed: {0}")]
    Rerank(String),
    #[error("Lock poisoned: {0}")]
    Lock(String),
}

/// Re-rank documents against a query using the cross-encoder model.
///
/// The model is lazy-loaded on first call. Returns documents scored and
/// sorted by relevance (descending).
pub fn rerank(query: &str, documents: &[&str]) -> Result<Vec<RerankScore>, RerankError> {
    if documents.is_empty() {
        return Ok(Vec::new());
    }

    ensure_loaded()?;

    let guard = RERANK_SERVICE
        .lock()
        .map_err(|e| RerankError::Lock(format!("{e}")))?;

    let service = guard.as_ref().unwrap();

    let llm_scores = service
        .rerank(query, documents)
        .map_err(|e| RerankError::Rerank(e.to_string()))?;

    Ok(llm_scores
        .into_iter()
        .map(|s| RerankScore {
            index: s.index,
            score: s.score,
        })
        .collect())
}

/// Ensure the global reranker service is loaded.
fn ensure_loaded() -> Result<(), RerankError> {
    let mut guard = RERANK_SERVICE
        .lock()
        .map_err(|e| RerankError::Lock(format!("{e}")))?;

    if guard.is_some() {
        return Ok(());
    }

    let model_path = resolve_model_path().ok_or_else(|| {
        RerankError::ModelLoad(format!(
            "Reranker model '{}' not found. Searched:\n  \
             1. $MEMORY_HOME/cache/models/\n  \
             2. <repo>/gguf_models/llama-nemotron-rerank-1b-v2-GGUF/",
            RERANK_MODEL_FILENAME
        ))
    })?;

    eprintln!(
        "[reranker] Loading nemotron-rerank-1b from {}...",
        model_path.display()
    );

    let config = RerankConfig {
        model_path,
        n_gpu_layers: -1,
        threads: 4,
        context_length: 512,
    };

    let service = LocalRerankerService::new(config)
        .map_err(|e| RerankError::ModelLoad(format!("Failed to load reranker: {e}")))?;

    eprintln!("[reranker] nemotron-rerank-1b loaded");

    *guard = Some(service);
    Ok(())
}

/// Resolve the reranker model GGUF path.
///
/// Checks in order:
/// 1. Production: `$MEMORY_HOME/cache/models/<filename>`
/// 2. Dev/repo:   `<crate_root>/../gguf_models/llama-nemotron-rerank-1b-v2-GGUF/<filename>`
fn resolve_model_path() -> Option<std::path::PathBuf> {
    // 1. Production path
    let production = crate::config::get_model_cache_dir().join(RERANK_MODEL_FILENAME);
    if production.exists() {
        return Some(production);
    }

    // 2. Dev fallback
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let dev_path = std::path::PathBuf::from(manifest_dir)
            .join("../gguf_models/llama-nemotron-rerank-1b-v2-GGUF")
            .join(RERANK_MODEL_FILENAME);
        if dev_path.exists() {
            return Some(dev_path);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rerank_empty_returns_empty() {
        let result = rerank("test query", &[]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
}
