//! Embedding generation for semantic search
//!
//! Provides vector embeddings for engram content, enabling semantic
//! similarity search beyond keyword matching. Uses the hardcoded
//! Nemotron embedding model via llama.cpp (GGUF).

mod generator;
mod similarity;

pub use generator::EmbeddingGenerator;
pub use generator::default_embedding_model;
pub use similarity::cosine_similarity;
