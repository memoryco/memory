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
pub mod config_toml;
pub mod decompose;
#[allow(clippy::module_inception)]
pub mod engram;
pub mod persistence;
pub mod search;
pub mod session;
pub mod storage;
pub mod substrate;

pub use association::Association;
pub use brain::Brain;
pub use engram::{Engram, MemoryState};
#[allow(unused_imports)]
pub use session::SessionContext;
pub use session::generate_session_id;
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

    /// Reranking mode: "off" or "cross-encoder".
    /// off = cosine order only, cross-encoder = nemotron reranker.
    #[serde(default = "default_rerank_mode")]
    pub rerank_mode: String,

    /// How many candidates to pull from cosine similarity before re-ranking.
    /// Higher = better recall but slower. Only used when rerank_mode is not "off".
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


    /// Server-side default minimum similarity score (0.0-1.0).
    /// Used when caller doesn't specify min_score.
    #[serde(default = "default_search_min_score")]
    pub search_min_score: f64,

    /// Minimum effective_limit for composite/list-style queries.
    #[serde(default = "default_composite_limit_min")]
    pub composite_limit_min: usize,

    /// Maximum effective_limit for composite/list-style queries.
    #[serde(default = "default_composite_limit_max")]
    pub composite_limit_max: usize,

    /// Minimum association discoveries to merge into search results.
    #[serde(default = "default_association_cap_min")]
    pub association_cap_min: usize,

    /// Maximum association discoveries to merge into search results.
    #[serde(default = "default_association_cap_max")]
    pub association_cap_max: usize,

    // ── Session Context ─────────────────────────────────────────────────────

    /// How much session context biases retrieval (0.0 = off, higher = stronger bias).
    #[serde(default = "default_session_context_weight")]
    pub session_context_weight: f32,

    /// Maximum queries to retain per session history.
    #[serde(default = "default_session_max_queries")]
    pub session_max_queries: usize,

    /// EMA smoothing factor for session centroid (higher = more recency bias).
    #[serde(default = "default_session_centroid_smoothing")]
    pub session_centroid_smoothing: f32,

    /// Delete sessions not accessed in this many days (0 = never expire).
    #[serde(default = "default_session_expire_days")]
    pub session_expire_days: usize,

    // ── Diagnostics ─────────────────────────────────────────────────────

    /// When true, search responses include a debug section with pipeline diagnostics.
    #[serde(default)]
    pub debug: bool,
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

// Note: active_model_name, set_active_model, model_from_name, embedding_dimension,
// is_valid_model have been removed. The embedding model is now hardcoded in
// generator.rs (appliance mode — nemotron-embed-1b-v2 via llama.cpp).

fn default_rerank_mode() -> String {
    "cross-encoder".to_string()
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

fn default_search_min_score() -> f64 {
    0.3
}

fn default_composite_limit_min() -> usize {
    15
}

fn default_composite_limit_max() -> usize {
    30
}

fn default_association_cap_min() -> usize {
    5
}

fn default_association_cap_max() -> usize {
    12
}

fn default_session_context_weight() -> f32 {
    0.3
}

fn default_session_max_queries() -> usize {
    50
}

fn default_session_centroid_smoothing() -> f32 {
    0.1
}

fn default_session_expire_days() -> usize {
    90
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
            rerank_mode: default_rerank_mode(),
            rerank_candidates: 30,
            hybrid_search_enabled: true,
            query_expansion_enabled: true,
            search_min_score: 0.3,
            composite_limit_min: 15,
            composite_limit_max: 30,
            association_cap_min: 5,
            association_cap_max: 12,
            session_context_weight: 0.3,
            session_max_queries: 50,
            session_centroid_smoothing: 0.1,
            session_expire_days: 90,
            debug: false,
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

    #[test]
    fn rerank_mode_serializes_and_deserializes() {
        let mut config = Config::default();
        config.rerank_mode = "off".to_string();
        let json = serde_json::to_string(&config).unwrap();
        let loaded: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.rerank_mode, "off");
    }

    #[test]
    fn rerank_mode_defaults_to_cross_encoder_when_missing() {
        // Old configs missing rerank_mode should default to "cross-encoder"
        let json = r#"{
            "decay_rate_per_day": 0.05,
            "decay_interval_hours": 1.0,
            "propagation_damping": 0.5,
            "hebbian_learning_rate": 0.1,
            "recall_strength": 0.2,
            "blocked_tags": [],
            "max_tag_cardinality": 20,
            "association_decay_rate": 1.0,
            "min_association_weight": 0.05,
            "search_follow_associations": true,
            "search_association_depth": 1
        }"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.rerank_mode, "cross-encoder");
    }
}
