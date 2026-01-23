//! Association - weighted connections between engrams

use super::EngramId;
use serde::{Serialize, Deserialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Get current unix timestamp in seconds
fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// A weighted connection between two engrams
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Association {
    /// Source engram
    pub from: EngramId,
    
    /// Target engram
    pub to: EngramId,
    
    /// Strength of the association (0.0 - 1.0)
    /// Strengthens with co-activation (Hebbian learning)
    pub weight: f64,
    
    /// When this association was formed (unix timestamp)
    pub created_at: i64,
    
    /// When this association was last activated (unix timestamp)
    pub last_activated: i64,
    
    /// Number of times both engrams were co-accessed
    pub co_activation_count: u64,
}

impl Association {
    /// Create a new association with default weight
    pub fn new(from: EngramId, to: EngramId) -> Self {
        let now = now_unix();
        Self {
            from,
            to,
            weight: 0.5, // Start at neutral
            created_at: now,
            last_activated: now,
            co_activation_count: 0,
        }
    }
    
    /// Create an association with a specific initial weight
    pub fn with_weight(from: EngramId, to: EngramId, weight: f64) -> Self {
        let mut assoc = Self::new(from, to);
        assoc.weight = weight.clamp(0.0, 1.0);
        assoc
    }
    
    /// Strengthen this association (Hebbian learning)
    pub fn strengthen(&mut self, amount: f64) {
        self.weight = (self.weight + amount).min(1.0);
        self.last_activated = now_unix();
        self.co_activation_count += 1;
    }
    
    /// Weaken this association
    pub fn weaken(&mut self, amount: f64) {
        self.weight = (self.weight - amount).max(0.0);
    }
    
    /// Calculate propagation energy based on weight
    pub fn propagation_energy(&self, source_energy: f64, damping: f64) -> f64 {
        source_energy * self.weight * damping
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;
    
    #[test]
    fn new_association_has_neutral_weight() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let assoc = Association::new(a, b);
        
        assert_eq!(assoc.weight, 0.5);
        assert_eq!(assoc.co_activation_count, 0);
    }
    
    #[test]
    fn strengthen_increases_weight() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let mut assoc = Association::new(a, b);
        
        assoc.strengthen(0.2);
        
        assert_eq!(assoc.weight, 0.7);
        assert_eq!(assoc.co_activation_count, 1);
    }
    
    #[test]
    fn strengthen_caps_at_one() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let mut assoc = Association::with_weight(a, b, 0.9);
        
        assoc.strengthen(0.5);
        
        assert_eq!(assoc.weight, 1.0);
    }
    
    #[test]
    fn weaken_decreases_weight() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let mut assoc = Association::new(a, b);
        
        assoc.weaken(0.3);
        
        assert_eq!(assoc.weight, 0.2);
    }
    
    #[test]
    fn propagation_calculation() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let assoc = Association::with_weight(a, b, 0.8);
        
        let energy = assoc.propagation_energy(1.0, 0.5);
        
        // 1.0 * 0.8 * 0.5 = 0.4
        assert!((energy - 0.4).abs() < 0.001);
    }
}
