//! Engram - a single memory trace

use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use super::EngramId;

/// Get current unix timestamp in seconds
fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// The state of a memory in the substrate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryState {
    /// Normal operation - searchable, participates in propagation
    Active,
    /// Low energy but still searchable, flagged as fading
    Dormant,
    /// Very low energy - NOT in normal search results
    /// Can only be found by exact ID or strong association cascade
    Deep,
    /// Minimal energy - frozen, read-only, preserved but inaccessible
    /// Can be resurrected with strong enough stimulation
    Archived,
}

impl MemoryState {
    /// Determine state from energy level
    pub fn from_energy(energy: f64) -> Self {
        if energy >= 0.3 {
            MemoryState::Active
        } else if energy >= 0.1 {
            MemoryState::Dormant
        } else if energy >= 0.02 {
            MemoryState::Deep
        } else {
            MemoryState::Archived
        }
    }
    
    /// Emoji representation for display
    pub fn emoji(&self) -> &'static str {
        match self {
            MemoryState::Active => "✨",
            MemoryState::Dormant => "💤",
            MemoryState::Deep => "🌊",
            MemoryState::Archived => "🧊",
        }
    }
    
    /// Whether this state participates in normal searches
    pub fn is_searchable(&self) -> bool {
        matches!(self, MemoryState::Active | MemoryState::Dormant)
    }
    
    /// Whether this state can propagate energy to neighbors
    pub fn can_propagate(&self) -> bool {
        matches!(self, MemoryState::Active | MemoryState::Dormant)
    }
    
    /// Whether this state can be modified (content changes)
    pub fn is_writable(&self) -> bool {
        // Archived is frozen, everything else is writable
        !matches!(self, MemoryState::Archived)
    }
}

/// A single memory unit in the substrate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Engram {
    /// Unique identifier
    pub id: EngramId,
    
    /// The actual content of the memory
    pub content: String,
    
    /// Current energy level (0.0 - 1.0)
    /// Decays over time, strengthened by access
    pub energy: f64,
    
    /// Current state based on energy
    pub state: MemoryState,
    
    /// Confidence in this memory's accuracy (0.0 - 1.0)
    /// Can drift over time if not reinforced
    pub confidence: f64,
    
    /// When this engram was created (unix timestamp)
    pub created_at: i64,
    
    /// When this engram was last accessed/stimulated (unix timestamp)
    pub last_accessed: i64,
    
    /// Total number of times this engram has been accessed
    pub access_count: u64,
    
    /// Optional tags/entities for categorization
    pub tags: Vec<String>,
    
    /// Semantic embedding vector (384-dim) for similarity search
    /// None if not yet computed or embeddings disabled
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub embedding: Option<Vec<f32>>,
}

impl Engram {
    /// Create a new engram with full energy
    pub fn new(content: impl Into<String>) -> Self {
        let now = now_unix();
        Self {
            id: uuid::Uuid::new_v4(),
            content: content.into(),
            energy: 1.0,
            state: MemoryState::Active,
            confidence: 1.0,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            tags: Vec::new(),
            embedding: None,
        }
    }
    
    /// Create a new engram with an explicit creation timestamp
    pub fn new_with_timestamp(content: impl Into<String>, created_at: i64) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            content: content.into(),
            energy: 1.0,
            state: MemoryState::Active,
            confidence: 1.0,
            created_at,
            last_accessed: created_at,
            access_count: 0,
            tags: Vec::new(),
            embedding: None,
        }
    }

    /// Create an engram with specific tags
    pub fn with_tags(content: impl Into<String>, tags: Vec<String>) -> Self {
        let mut engram = Self::new(content);
        engram.tags = tags;
        engram
    }
    
    /// Update state based on current energy level
    fn update_state(&mut self) {
        self.state = MemoryState::from_energy(self.energy);
    }
    
    /// Stimulate this engram - increases energy and updates access time
    /// Returns the previous state if resurrection occurred
    pub fn stimulate(&mut self, amount: f64) -> Option<MemoryState> {
        let old_state = self.state;
        
        self.energy = (self.energy + amount).min(1.0);
        self.last_accessed = now_unix();
        self.access_count += 1;
        self.update_state();
        
        // Return old state if we resurrected from Deep or Archived
        if (old_state == MemoryState::Deep || old_state == MemoryState::Archived)
            && self.state.is_searchable() 
        {
            Some(old_state)
        } else {
            None
        }
    }
    
    /// Apply decay to this engram
    pub fn decay(&mut self, rate: f64) {
        // Archived memories don't decay further - they're frozen
        if self.state == MemoryState::Archived {
            return;
        }
        
        // Deep memories decay slower
        let effective_rate = if self.state == MemoryState::Deep {
            rate * 0.25  // 75% slower decay in deep storage
        } else {
            rate
        };
        
        self.energy = (self.energy - effective_rate).max(0.001); // Never hit true zero
        self.update_state();
    }
    
    /// Check if this engram is dormant (low energy but still searchable)
    pub fn is_dormant(&self) -> bool {
        self.state == MemoryState::Dormant
    }
    
    /// Check if this engram is in deep storage
    pub fn is_deep(&self) -> bool {
        self.state == MemoryState::Deep
    }
    
    /// Check if this engram is archived
    pub fn is_archived(&self) -> bool {
        self.state == MemoryState::Archived
    }
    
    /// Check if this engram is searchable (Active or Dormant)
    pub fn is_searchable(&self) -> bool {
        self.state.is_searchable()
    }
    
    /// Seconds since last access
    pub fn seconds_since_access(&self) -> i64 {
        now_unix() - self.last_accessed
    }
    
    /// Age of this engram in seconds
    pub fn age_seconds(&self) -> i64 {
        now_unix() - self.created_at
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_engram_has_full_energy() {
        let e = Engram::new("test memory");
        assert_eq!(e.energy, 1.0);
        assert_eq!(e.confidence, 1.0);
        assert_eq!(e.state, MemoryState::Active);
        assert_eq!(e.access_count, 0);
    }

    #[test]
    fn stimulate_increases_energy() {
        let mut e = Engram::new("test");
        e.energy = 0.5;
        e.update_state();
        e.stimulate(0.3);
        assert_eq!(e.energy, 0.8);
        assert_eq!(e.access_count, 1);
    }

    #[test]
    fn stimulate_caps_at_one() {
        let mut e = Engram::new("test");
        e.stimulate(0.5);
        assert_eq!(e.energy, 1.0);
    }

    #[test]
    fn decay_reduces_energy() {
        let mut e = Engram::new("test");
        e.decay(0.3);
        assert!((e.energy - 0.7).abs() < 0.001);
    }

    #[test]
    fn decay_never_hits_zero() {
        let mut e = Engram::new("test");
        e.decay(1.5);
        assert!(e.energy > 0.0);
        assert_eq!(e.energy, 0.001);
    }

    #[test]
    fn state_transitions_on_decay() {
        let mut e = Engram::new("test");
        assert_eq!(e.state, MemoryState::Active);
        
        e.energy = 0.25;
        e.update_state();
        assert_eq!(e.state, MemoryState::Dormant);
        
        e.energy = 0.05;
        e.update_state();
        assert_eq!(e.state, MemoryState::Deep);
        
        e.energy = 0.01;
        e.update_state();
        assert_eq!(e.state, MemoryState::Archived);
    }

    #[test]
    fn archived_doesnt_decay() {
        let mut e = Engram::new("test");
        e.energy = 0.01;
        e.update_state();
        assert_eq!(e.state, MemoryState::Archived);
        
        e.decay(0.5);
        assert_eq!(e.energy, 0.01); // Unchanged
    }

    #[test]
    fn resurrection_from_archived() {
        let mut e = Engram::new("test");
        e.energy = 0.01;
        e.update_state();
        assert_eq!(e.state, MemoryState::Archived);
        
        let old_state = e.stimulate(0.5);
        assert_eq!(old_state, Some(MemoryState::Archived));
        assert_eq!(e.state, MemoryState::Active);
        assert!(e.energy > 0.5);
    }

    #[test]
    fn deep_decays_slower() {
        let mut active = Engram::new("active");
        active.energy = 0.5;
        active.update_state();
        
        let mut deep = Engram::new("deep");
        deep.energy = 0.05;
        deep.update_state();
        assert_eq!(deep.state, MemoryState::Deep);
        
        let rate = 0.04;
        let active_before = active.energy;
        let deep_before = deep.energy;
        
        active.decay(rate);
        deep.decay(rate);
        
        let active_loss = active_before - active.energy;
        let deep_loss = deep_before - deep.energy;
        
        // Deep should lose 75% less
        assert!((deep_loss - active_loss * 0.25).abs() < 0.001);
    }
}
