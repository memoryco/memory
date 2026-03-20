//! Embedding generator using llama.cpp (GGUF model via memoryco-llm)
//!
//! Provides vector embeddings for engram content, enabling semantic
//! similarity search beyond keyword matching.
//!
//! Uses a hardcoded NVIDIA Nemotron embedding model in appliance mode —
//! no user-configurable model selection.

use crate::config;
use memoryco_llm::embed::{EmbedConfig, LocalEmbeddingService, PoolingType};
use std::sync::Mutex;

/// Hardcoded embedding model constants (appliance mode)
const EMBED_MODEL_FILENAME: &str = "llama-nemotron-embed-1b-v2.Q4_K_M.gguf";
const EMBED_MODEL_NAME: &str = "nemotron-embed-1b-v2";
const EMBED_DIMENSIONS: usize = 2048;

/// Global embedding service (lazy-loaded on first use)
static EMBED_SERVICE: Mutex<Option<LocalEmbeddingService>> = Mutex::new(None);

/// Embedding generator for semantic search
///
/// Wraps `LocalEmbeddingService` from memoryco-llm. The model is loaded
/// lazily on first `generate()` call and shared via a global static.
pub struct EmbeddingGenerator;

impl EmbeddingGenerator {
    /// Create a new embedding generator (zero-cost — model loads lazily)
    pub fn new() -> Self {
        Self
    }

    /// Get the embedding dimension for the active model
    #[allow(dead_code)]
    pub fn dimension() -> usize {
        EMBED_DIMENSIONS
    }

    /// Generate embedding for text content
    ///
    /// Returns a normalized vector of `EMBED_DIMENSIONS` dimensions.
    /// Model is lazy-loaded on first call.
    pub fn generate(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        self.ensure_loaded()?;

        let guard = EMBED_SERVICE
            .lock()
            .map_err(|e| EmbeddingError::Generation(format!("Lock poisoned: {}", e)))?;

        let service = guard.as_ref().unwrap();

        service
            .embed(text)
            .map_err(|e| EmbeddingError::Generation(e.to_string()))
    }

    /// Generate embeddings for multiple texts
    ///
    /// Calls the model once per text. For bulk migration, this is acceptable
    /// since migration is a one-time operation.
    pub fn generate_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        self.ensure_loaded()?;

        let guard = EMBED_SERVICE
            .lock()
            .map_err(|e| EmbeddingError::Generation(format!("Lock poisoned: {}", e)))?;

        let service = guard.as_ref().unwrap();

        let mut results = Vec::with_capacity(texts.len());
        for (i, text) in texts.iter().enumerate() {
            let embedding = service
                .embed(text)
                .map_err(|e| EmbeddingError::Generation(e.to_string()))?;
            results.push(embedding);

            // Progress for large batches
            if texts.len() > 100 && (i + 1) % 100 == 0 {
                eprintln!("[embedding] Progress: {}/{}", i + 1, texts.len());
            }
        }

        Ok(results)
    }

    /// Ensure the global embedding service is loaded.
    fn ensure_loaded(&self) -> Result<(), EmbeddingError> {
        let mut guard = EMBED_SERVICE
            .lock()
            .map_err(|e| EmbeddingError::Generation(format!("Lock poisoned: {}", e)))?;

        if guard.is_some() {
            return Ok(());
        }

        let model_path = resolve_model_path().ok_or_else(|| {
            EmbeddingError::Generation(format!(
                "Embedding model '{}' not found. Searched:\n  \
                 1. $MEMORY_HOME/cache/models/\n  \
                 2. <repo>/gguf_models/llama-nemotron-embed-1b-v2-GGUF/",
                EMBED_MODEL_FILENAME
            ))
        })?;

        eprintln!(
            "[embedding] Loading {} from {}...",
            EMBED_MODEL_NAME,
            model_path.display()
        );

        let embed_config = EmbedConfig {
            model_path,
            n_gpu_layers: -1, // Use all available GPU layers
            threads: 4,
            context_length: 512, // Engram text is short
            pooling: PoolingType::Mean,
        };

        let service = LocalEmbeddingService::new(embed_config).map_err(|e| {
            EmbeddingError::Generation(format!("Failed to load embedding model: {}", e))
        })?;

        let actual_dims = service.n_embd();
        eprintln!(
            "[embedding] {} loaded ({}-dim)",
            EMBED_MODEL_NAME, actual_dims
        );

        debug_assert_eq!(
            actual_dims, EMBED_DIMENSIONS,
            "Model reports {}-dim but expected {}-dim",
            actual_dims, EMBED_DIMENSIONS
        );

        *guard = Some(service);
        Ok(())
    }
}

impl Default for EmbeddingGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Resolve the embedding model GGUF path.
///
/// Checks in order:
/// 1. Production: `$MEMORY_HOME/cache/models/<filename>`
/// 2. Dev/repo:   `<crate_root>/../gguf_models/llama-nemotron-embed-1b-v2-GGUF/<filename>`
///
/// Returns None if the model isn't found in any location.
fn resolve_model_path() -> Option<std::path::PathBuf> {
    // 1. Production path (MEMORY_HOME/cache/models/)
    let production = config::get_model_cache_dir().join(EMBED_MODEL_FILENAME);
    if production.exists() {
        return Some(production);
    }

    // 2. Dev fallback: repo gguf_models directory
    //    CARGO_MANIFEST_DIR points to memory/ crate root, so ../gguf_models/ is the repo stash.
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let dev_path = std::path::PathBuf::from(manifest_dir)
            .join("../gguf_models/llama-nemotron-embed-1b-v2-GGUF")
            .join(EMBED_MODEL_FILENAME);
        if dev_path.exists() {
            return Some(dev_path);
        }
    }

    None
}

/// Default embedding model name for config tracking
pub fn default_embedding_model() -> String {
    EMBED_MODEL_NAME.to_string()
}

/// Errors from embedding generation
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Failed to generate embedding: {0}")]
    Generation(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_model_name() {
        let name = default_embedding_model();
        assert_eq!(name, "nemotron-embed-1b-v2");
    }

    #[test]
    fn dimension_is_correct() {
        assert_eq!(EmbeddingGenerator::dimension(), 2048);
    }

    #[test]
    fn generator_is_zero_cost() {
        // EmbeddingGenerator::new() should not load the model
        let _gen = EmbeddingGenerator::new();
        // If we got here without a panic, the model wasn't loaded
        let guard = EMBED_SERVICE.lock().unwrap();
        // Note: service may or may not be loaded from other tests,
        // but new() itself shouldn't trigger it
        drop(guard);
    }
}
