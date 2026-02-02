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


pub mod engram;
pub mod substrate;
pub mod association;
pub mod storage;
pub mod brain;
pub mod bootstrap;
pub mod persistence;

pub use engram::{Engram, MemoryState};
pub use substrate::{Substrate, SubstrateStats, StimulationResult, RecallResult, SearchOptions, StateFilter, TagMatchMode};
pub use association::Association;
pub use storage::{Storage, StorageError, StorageResult, MemoryStorage, EngramStorage, SimilarityResult};
pub use brain::{Brain, UpsertResult};

// Re-export identity types from the identity module for backwards compatibility
pub use crate::identity::{
    Identity, Persona, Value, Preference, Relationship, 
    Antipattern, CommunicationStyle, IdentitySearchResults,
};

/// Unique identifier for an engram
pub type EngramId = uuid::Uuid;

use serde::{Serialize, Deserialize};

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
}

fn default_max_tag_cardinality() -> usize {
    20
}

fn default_association_decay_rate() -> f64 {
    1.0  // Same rate as memories
}

fn default_min_association_weight() -> f64 {
    0.05  // Prune associations below 5%
}

impl Default for Config {
    fn default() -> Self {
        Self {
            decay_rate_per_day: 0.05,    // 5% energy loss per day
            decay_interval_hours: 1.0,    // Check decay every hour minimum
            propagation_damping: 0.5,     // 50% signal to neighbors
            hebbian_learning_rate: 0.1,   // 10% association boost on co-access
            recall_strength: 0.2,         // 20% energy boost on recall
            blocked_tags: vec![           // Tags too generic to be useful
                "architecture".into(),
                "design".into(),
                "code".into(),
                "work".into(),
                "misc".into(),
                "general".into(),
                "other".into(),
            ],
            max_tag_cardinality: 20,      // Warn if tag on >20 memories
            association_decay_rate: 1.0,  // Same decay rate as memories
            min_association_weight: 0.05, // Prune below 5%
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
