//! Embedding generator using fastembed (all-MiniLM-L6-v2)

use crate::config;
use crate::embedding::EMBEDDING_DIM;
use std::sync::{OnceLock, Mutex};
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

/// Global embedding model (lazy-loaded on first use)
/// Wrapped in Mutex because embed() requires &mut self
static EMBEDDING_MODEL: OnceLock<Mutex<TextEmbedding>> = OnceLock::new();

/// Embedding generator for semantic search
pub struct EmbeddingGenerator;

impl EmbeddingGenerator {
    /// Create a new embedding generator
    pub fn new() -> Self {
        Self
    }

    /// Generate embedding for text content
    ///
    /// Returns a 384-dimensional vector representing the semantic content.
    /// Model is lazy-loaded on first call (~22MB download on first use).
    pub fn generate(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let model_mutex = EMBEDDING_MODEL.get_or_init(|| {
            let cache_dir = config::get_model_cache_dir();
            std::fs::create_dir_all(&cache_dir).ok();
            eprintln!("Loading embedding model (cache: {})...", cache_dir.display());
            
            let options = InitOptions::new(EmbeddingModel::AllMiniLML6V2)
                .with_cache_dir(cache_dir)
                .with_show_download_progress(true);
            Mutex::new(
                TextEmbedding::try_new(options)
                    .expect("Failed to load embedding model")
            )
        });

        let mut model = model_mutex.lock()
            .map_err(|e| EmbeddingError::Generation(format!("Lock poisoned: {}", e)))?;

        let embeddings = model
            .embed(vec![text], None)
            .map_err(|e| EmbeddingError::Generation(e.to_string()))?;

        let embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::Generation("No embedding returned".to_string()))?;

        debug_assert_eq!(embedding.len(), EMBEDDING_DIM);
        Ok(embedding)
    }

    /// Generate embeddings for multiple texts (batch)
    pub fn generate_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let model_mutex = EMBEDDING_MODEL.get_or_init(|| {
            let cache_dir = config::get_model_cache_dir();
            std::fs::create_dir_all(&cache_dir).ok();
            eprintln!("Loading embedding model (cache: {})...", cache_dir.display());
            
            let options = InitOptions::new(EmbeddingModel::AllMiniLML6V2)
                .with_cache_dir(cache_dir)
                .with_show_download_progress(true);
            Mutex::new(
                TextEmbedding::try_new(options)
                    .expect("Failed to load embedding model")
            )
        });

        let mut model = model_mutex.lock()
            .map_err(|e| EmbeddingError::Generation(format!("Lock poisoned: {}", e)))?;

        let embeddings = model
            .embed(texts.to_vec(), None)
            .map_err(|e| EmbeddingError::Generation(e.to_string()))?;

        Ok(embeddings)
    }
}

impl Default for EmbeddingGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors from embedding generation
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Failed to generate embedding: {0}")]
    Generation(String),
}
