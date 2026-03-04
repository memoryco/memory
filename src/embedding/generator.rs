//! Embedding generator using fastembed (configurable model)

use crate::config;
use std::sync::Mutex;
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

/// Global embedding model (lazy-loaded on first use, swappable on model change)
/// Stores (model_name, TextEmbedding) so we can detect when the model has changed.
static EMBEDDING_MODEL: Mutex<Option<(String, TextEmbedding)>> = Mutex::new(None);

/// Embedding generator for semantic search
pub struct EmbeddingGenerator;

impl EmbeddingGenerator {
    /// Create a new embedding generator
    pub fn new() -> Self {
        Self
    }

    /// Get the embedding dimension for the currently active model
    #[allow(dead_code)]
    pub fn dimension() -> usize {
        embedding_dimension(&active_model_name())
    }

    /// Generate embedding for text content
    ///
    /// Returns a vector representing the semantic content, with dimensionality
    /// determined by the active model. Model is lazy-loaded on first call.
    pub fn generate(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let model_name = active_model_name();
        self.ensure_model_loaded(&model_name)?;

        let mut guard = EMBEDDING_MODEL.lock()
            .map_err(|e| EmbeddingError::Generation(format!("Lock poisoned: {}", e)))?;

        let (_, model) = guard.as_mut().unwrap();

        let embeddings = model
            .embed(vec![text], None)
            .map_err(|e| EmbeddingError::Generation(e.to_string()))?;

        let embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::Generation("No embedding returned".to_string()))?;

        let expected_dim = embedding_dimension(&model_name);
        debug_assert_eq!(embedding.len(), expected_dim);
        Ok(embedding)
    }

    /// Generate embeddings for multiple texts (batch)
    pub fn generate_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let model_name = active_model_name();
        self.ensure_model_loaded(&model_name)?;

        let mut guard = EMBEDDING_MODEL.lock()
            .map_err(|e| EmbeddingError::Generation(format!("Lock poisoned: {}", e)))?;

        let (_, model) = guard.as_mut().unwrap();

        let embeddings = model
            .embed(texts, None)
            .map_err(|e| EmbeddingError::Generation(e.to_string()))?;

        Ok(embeddings)
    }

    /// Ensure the global model is loaded and matches the requested model name.
    /// If the model is stale (different name), drops the old one and loads the new one.
    fn ensure_model_loaded(&self, model_name: &str) -> Result<(), EmbeddingError> {
        let mut guard = EMBEDDING_MODEL.lock()
            .map_err(|e| EmbeddingError::Generation(format!("Lock poisoned: {}", e)))?;

        // Check if we already have the right model
        if let Some((ref current_name, _)) = *guard {
            if current_name == model_name {
                return Ok(());
            }
            // Model changed — drop the old one
            eprintln!("Embedding model changed from {} to {}, reloading...", current_name, model_name);
        }

        let cache_dir = config::get_model_cache_dir();
        std::fs::create_dir_all(&cache_dir).ok();
        eprintln!("Loading embedding model {} (cache: {})...", model_name, cache_dir.display());

        let fastembed_model = model_from_name(model_name)
            .ok_or_else(|| EmbeddingError::Generation(
                format!("Unknown embedding model: {}", model_name)
            ))?;

        let options = InitOptions::new(fastembed_model)
            .with_cache_dir(cache_dir)
            .with_show_download_progress(true);

        let text_embedding = TextEmbedding::try_new(options)
            .map_err(|e| EmbeddingError::Generation(
                format!("Failed to load embedding model {}: {}", model_name, e)
            ))?;

        *guard = Some((model_name.to_string(), text_embedding));
        Ok(())
    }
}

impl Default for EmbeddingGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the currently active model name from the global config.
///
/// This reads from a global state that Brain sets on startup.
/// Falls back to the default if not yet set or lock is poisoned.
pub fn active_model_name() -> String {
    let name = ACTIVE_MODEL.lock()
        .map(|g| g.clone())
        .unwrap_or_default();

    if name.is_empty() {
        default_embedding_model()
    } else {
        name
    }
}

/// Set the active model name. Called by Brain on startup after reading config.
pub fn set_active_model(name: &str) {
    if let Ok(mut guard) = ACTIVE_MODEL.lock() {
        *guard = name.to_string();
    }
}

/// Global active model name, set by Brain on startup
static ACTIVE_MODEL: Mutex<String> = Mutex::new(String::new());

/// Default embedding model name for fresh installs
pub fn default_embedding_model() -> String {
    "AllMiniLML6V2".to_string()
}

/// Map a config string to a fastembed EmbeddingModel enum variant.
/// Returns None for unknown model names.
pub fn model_from_name(name: &str) -> Option<EmbeddingModel> {
    match name {
        "AllMiniLML6V2" => Some(EmbeddingModel::AllMiniLML6V2),
        "AllMiniLML6V2Q" => Some(EmbeddingModel::AllMiniLML6V2Q),
        "AllMiniLML12V2" => Some(EmbeddingModel::AllMiniLML12V2),
        "AllMiniLML12V2Q" => Some(EmbeddingModel::AllMiniLML12V2Q),
        "BGEBaseENV15" => Some(EmbeddingModel::BGEBaseENV15),
        "BGEBaseENV15Q" => Some(EmbeddingModel::BGEBaseENV15Q),
        "BGELargeENV15" => Some(EmbeddingModel::BGELargeENV15),
        "BGELargeENV15Q" => Some(EmbeddingModel::BGELargeENV15Q),
        "BGESmallENV15" => Some(EmbeddingModel::BGESmallENV15),
        "BGESmallENV15Q" => Some(EmbeddingModel::BGESmallENV15Q),
        "NomicEmbedTextV1" => Some(EmbeddingModel::NomicEmbedTextV1),
        "NomicEmbedTextV15" => Some(EmbeddingModel::NomicEmbedTextV15),
        "NomicEmbedTextV15Q" => Some(EmbeddingModel::NomicEmbedTextV15Q),
        "SnowflakeArcticEmbedM" => Some(EmbeddingModel::SnowflakeArcticEmbedM),
        "SnowflakeArcticEmbedMQ" => Some(EmbeddingModel::SnowflakeArcticEmbedMQ),
        "SnowflakeArcticEmbedL" => Some(EmbeddingModel::SnowflakeArcticEmbedL),
        "SnowflakeArcticEmbedLQ" => Some(EmbeddingModel::SnowflakeArcticEmbedLQ),
        "MxbaiEmbedLargeV1" => Some(EmbeddingModel::MxbaiEmbedLargeV1),
        "MxbaiEmbedLargeV1Q" => Some(EmbeddingModel::MxbaiEmbedLargeV1Q),
        _ => None,
    }
}

/// Get the embedding dimension for a given model name.
/// Returns 0 for unknown model names.
pub fn embedding_dimension(name: &str) -> usize {
    match name {
        "AllMiniLML6V2" | "AllMiniLML6V2Q" => 384,
        "AllMiniLML12V2" | "AllMiniLML12V2Q" => 384,
        "BGEBaseENV15" | "BGEBaseENV15Q" => 768,
        "BGELargeENV15" | "BGELargeENV15Q" => 1024,
        "BGESmallENV15" | "BGESmallENV15Q" => 384,
        "NomicEmbedTextV1" => 768,
        "NomicEmbedTextV15" | "NomicEmbedTextV15Q" => 768,
        "SnowflakeArcticEmbedM" | "SnowflakeArcticEmbedMQ" => 768,
        "SnowflakeArcticEmbedL" | "SnowflakeArcticEmbedLQ" => 1024,
        "MxbaiEmbedLargeV1" | "MxbaiEmbedLargeV1Q" => 1024,
        _ => 0,
    }
}

/// Check if a model name is valid (known to the system)
pub fn is_valid_model(name: &str) -> bool {
    model_from_name(name).is_some()
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
    fn model_name_mapping_covers_all() {
        let names = [
            "AllMiniLML6V2", "AllMiniLML6V2Q",
            "AllMiniLML12V2", "AllMiniLML12V2Q",
            "BGEBaseENV15", "BGEBaseENV15Q",
            "BGELargeENV15", "BGELargeENV15Q",
            "BGESmallENV15", "BGESmallENV15Q",
            "NomicEmbedTextV1",
            "NomicEmbedTextV15", "NomicEmbedTextV15Q",
            "SnowflakeArcticEmbedM", "SnowflakeArcticEmbedMQ",
            "SnowflakeArcticEmbedL", "SnowflakeArcticEmbedLQ",
            "MxbaiEmbedLargeV1", "MxbaiEmbedLargeV1Q",
        ];
        for name in &names {
            assert!(model_from_name(name).is_some(), "model_from_name should recognize {}", name);
            assert!(embedding_dimension(name) > 0, "embedding_dimension should be > 0 for {}", name);
        }
    }

    #[test]
    fn unknown_model_returns_none() {
        assert!(model_from_name("NotARealModel").is_none());
        assert_eq!(embedding_dimension("NotARealModel"), 0);
        assert!(!is_valid_model("NotARealModel"));
    }

    #[test]
    fn dimension_values_correct() {
        assert_eq!(embedding_dimension("AllMiniLML6V2"), 384);
        assert_eq!(embedding_dimension("BGEBaseENV15"), 768);
        assert_eq!(embedding_dimension("BGELargeENV15"), 1024);
        assert_eq!(embedding_dimension("SnowflakeArcticEmbedL"), 1024);
        assert_eq!(embedding_dimension("SnowflakeArcticEmbedM"), 768);
        assert_eq!(embedding_dimension("NomicEmbedTextV15"), 768);
        assert_eq!(embedding_dimension("MxbaiEmbedLargeV1"), 1024);
    }

    #[test]
    fn default_model_is_valid() {
        let default = default_embedding_model();
        assert!(is_valid_model(&default));
        assert_eq!(embedding_dimension(&default), 384);
    }

    #[test]
    fn quantized_same_dimension_as_base() {
        let pairs = [
            ("AllMiniLML6V2", "AllMiniLML6V2Q"),
            ("BGELargeENV15", "BGELargeENV15Q"),
            ("SnowflakeArcticEmbedL", "SnowflakeArcticEmbedLQ"),
            ("MxbaiEmbedLargeV1", "MxbaiEmbedLargeV1Q"),
        ];
        for (base, quantized) in &pairs {
            assert_eq!(
                embedding_dimension(base), embedding_dimension(quantized),
                "{} and {} should have the same dimension", base, quantized
            );
        }
    }
}
