//! Embedding generation for semantic search
//!
//! Provides vector embeddings for engram content, enabling semantic
//! similarity search beyond keyword matching. The embedding model and
//! dimensionality are runtime-configurable via the substrate config.

mod generator;
pub mod reranker;
mod similarity;

pub use generator::EmbeddingGenerator;
#[allow(unused_imports)]
pub use generator::{
    active_model_name, set_active_model, default_embedding_model,
    model_from_name, embedding_dimension, is_valid_model,
};
pub use similarity::cosine_similarity;
