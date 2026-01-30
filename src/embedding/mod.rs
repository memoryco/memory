//! Embedding generation for semantic search
//!
//! Provides vector embeddings (384-dim) for engram content,
//! enabling semantic similarity search beyond keyword matching.

mod generator;
mod similarity;

pub use generator::EmbeddingGenerator;
pub use similarity::cosine_similarity;

/// Embedding dimension (all-MiniLM-L6-v2 produces 384-dim vectors)
pub const EMBEDDING_DIM: usize = 384;

/// Minimum similarity threshold for vector search results
pub const SIMILARITY_THRESHOLD: f32 = 0.3;
