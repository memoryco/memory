//! Engram - A Neural Memory Substrate
//!
//! An experimental memory system that models organic decay,
//! associative links, and emergent behavior through feedback loops.
//!
//! Key concepts:
//! - **Engrams** never get deleted, they sink through states: Active → Dormant → Deep → Archived
//! - **Associations** form automatically through Hebbian learning (co-access)
//! - **Stimulation** propagates through the association network
//! - **Decay** is organic but never destroys data - memories just become harder to access

pub mod association;
pub mod bootstrap;
pub mod brain;
pub mod decompose;
#[allow(clippy::module_inception)]
pub mod engram;
pub mod persistence;
pub mod storage;
pub mod substrate;

pub use association::Association;
pub use brain::Brain;
pub use engram::{Engram, MemoryState};
#[cfg(test)]
pub use storage::memory::MemoryStorage;
pub use storage::{SimilarityResult, Storage, StorageResult};
pub use substrate::{RecallResult, SearchOptions, Substrate, SubstrateStats, TagMatchMode};

// Re-export identity types from the identity module for backwards compatibility
pub use crate::identity::Identity;

/// Unique identifier for an engram
pub type EngramId = uuid::Uuid;

use serde::{Deserialize, Serialize};

/// Configuration for the substrate behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Decay rate per day (0.0 - 1.0)
    /// e.g., 0.05 = 5% energy loss per day of non-use
    pub decay_rate_per_day: f64,

    /// Minimum hours between decay applications
    /// Prevents constant recalculation on every read
    pub decay_interval_hours: f64,

    /// How much energy propagates to neighbors (0.0 - 1.0)
    /// e.g., 0.5 means neighbor gets half the stimulation
    pub propagation_damping: f64,

    /// How much co-access strengthens associations
    pub hebbian_learning_rate: f64,

    /// Default stimulation amount when recalling a memory
    pub recall_strength: f64,

    /// Tags that are too generic and should be blocked
    /// These will be stripped from memories on create
    #[serde(default)]
    pub blocked_tags: Vec<String>,

    /// Maximum number of memories a tag can appear on before warning
    /// Tags exceeding this are likely too generic to be useful
    #[serde(default = "default_max_tag_cardinality")]
    pub max_tag_cardinality: usize,

    /// Association decay rate relative to memory decay (0.0 - 2.0)
    /// 1.0 = same rate as memories, 0.5 = half as fast, 2.0 = twice as fast
    #[serde(default = "default_association_decay_rate")]
    pub association_decay_rate: f64,

    /// Minimum association weight to keep (0.0 - 1.0)
    /// Associations below this threshold are pruned on startup
    #[serde(default = "default_min_association_weight")]
    pub min_association_weight: f64,

    /// Whether search should follow associations after the initial vector pass
    /// to discover related memories that vector search alone would miss
    #[serde(default = "default_search_follow_associations")]
    pub search_follow_associations: bool,

    /// How many hops to follow when search_follow_associations is true
    /// 1 = direct associations only, 2 = friends-of-friends, etc.
    #[serde(default = "default_search_association_depth")]
    pub search_association_depth: u8,

    /// The desired embedding model name (e.g. "SnowflakeArcticEmbedL")
    /// Changing this triggers re-embedding on next Brain startup.
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,

    /// Which model was actually used to generate current embeddings.
    /// When this differs from `embedding_model`, migration is triggered.
    /// None means no embeddings have been generated yet (or pre-config era).
    #[serde(default)]
    pub embedding_model_active: Option<String>,

    /// Whether to use cross-encoder re-ranking on search results.
    /// When enabled, cosine similarity results are re-scored by a cross-encoder
    /// model for significantly better relevance ordering.
    #[serde(default = "default_rerank_enabled")]
    pub rerank_enabled: bool,

    /// How many candidates to pull from cosine similarity before re-ranking.
    /// Higher = better recall but slower. Only used when rerank_enabled is true.
    #[serde(default = "default_rerank_candidates")]
    pub rerank_candidates: usize,

    /// Whether to use hybrid BM25+vector search with RRF fusion.
    /// When enabled, searches both FTS5 (keyword/BM25) and vector (cosine similarity)
    /// in parallel, then merges results using Reciprocal Rank Fusion.
    #[serde(default = "default_hybrid_search_enabled")]
    pub hybrid_search_enabled: bool,

    /// Whether to use query expansion before retrieval.
    /// When enabled, generates variant queries (stop-word-stripped, key terms)
    /// and runs each through the full retrieval pipeline before merging.
    #[serde(default = "default_query_expansion_enabled")]
    pub query_expansion_enabled: bool,
}

fn default_max_tag_cardinality() -> usize {
    20
}

fn default_association_decay_rate() -> f64 {
    1.0 // Same rate as memories
}

fn default_min_association_weight() -> f64 {
    0.05 // Prune associations below 5%
}

fn default_search_follow_associations() -> bool {
    true // Enabled by default
}

fn default_search_association_depth() -> u8 {
    1 // Direct associations only
}

fn default_embedding_model() -> String {
    crate::embedding::default_embedding_model()
}

fn default_rerank_enabled() -> bool {
    false
}

fn default_rerank_candidates() -> usize {
    30
}

fn default_hybrid_search_enabled() -> bool {
    true
}

fn default_query_expansion_enabled() -> bool {
    true
}

impl Default for Config {
    fn default() -> Self {
        Self {
            decay_rate_per_day: 0.05,   // 5% energy loss per day
            decay_interval_hours: 1.0,  // Check decay every hour minimum
            propagation_damping: 0.5,   // 50% signal to neighbors
            hebbian_learning_rate: 0.1, // 10% association boost on co-access
            recall_strength: 0.2,       // 20% energy boost on recall
            blocked_tags: vec![
                // Tags too generic to be useful
                "architecture".into(),
                "design".into(),
                "code".into(),
                "work".into(),
                "misc".into(),
                "general".into(),
                "other".into(),
            ],
            max_tag_cardinality: 20,          // Warn if tag on >20 memories
            association_decay_rate: 1.0,      // Same decay rate as memories
            min_association_weight: 0.05,     // Prune below 5%
            search_follow_associations: true, // Enabled by default (depth 1, cap 5)
            search_association_depth: 1,      // Direct associations only
            embedding_model: crate::embedding::default_embedding_model(),
            embedding_model_active: None,
            rerank_enabled: true,
            rerank_candidates: 30,
            hybrid_search_enabled: true,
            query_expansion_enabled: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_sane() {
        let config = Config::default();
        assert!(config.decay_rate_per_day > 0.0 && config.decay_rate_per_day < 1.0);
        assert!(config.decay_interval_hours > 0.0);
        assert!(config.propagation_damping > 0.0 && config.propagation_damping < 1.0);
    }
}
