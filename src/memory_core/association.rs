//! Association - weighted connections between memories

use super::MemoryId;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Get current unix timestamp in seconds
fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// A weighted connection between two memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Association {
    /// Source memory
    pub from: MemoryId,

    /// Target memory
    pub to: MemoryId,

    /// Strength of the association (0.0 - 1.0)
    /// Strengthens with co-activation (Hebbian learning)
    pub weight: f64,

    /// When this association was formed (unix timestamp)
    pub created_at: i64,

    /// When this association was last activated (unix timestamp)
    pub last_activated: i64,

    /// Number of times both memories were co-accessed
    pub co_activation_count: u64,

    /// Position in an ordered chain (e.g., procedure steps).
    /// None = unordered (legacy/Hebbian association).
    /// Some(n) = position in a sequence.
    pub ordinal: Option<u32>,
}

impl Association {
    /// Create a new association with default weight
    pub fn new(from: MemoryId, to: MemoryId) -> Self {
        let now = now_unix();
        Self {
            from,
            to,
            weight: 0.5, // Start at neutral
            created_at: now,
            last_activated: now,
            co_activation_count: 0,
            ordinal: None,
        }
    }

    /// Create an association with a specific initial weight
    pub fn with_weight(from: MemoryId, to: MemoryId, weight: f64) -> Self {
        let mut assoc = Self::new(from, to);
        assoc.weight = weight.clamp(0.0, 1.0);
        assoc
    }

    /// Create an association with weight and ordinal (for ordered chains)
    pub fn with_ordinal(from: MemoryId, to: MemoryId, weight: f64, ordinal: Option<u32>) -> Self {
        let mut assoc = Self::with_weight(from, to, weight);
        assoc.ordinal = ordinal;
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

    #[test]
    fn new_association_has_no_ordinal() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let assoc = Association::new(a, b);
        assert_eq!(assoc.ordinal, None);
    }

    #[test]
    fn with_weight_has_no_ordinal() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let assoc = Association::with_weight(a, b, 0.7);
        assert_eq!(assoc.ordinal, None);
    }

    #[test]
    fn with_ordinal_sets_position() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let assoc = Association::with_ordinal(a, b, 0.8, Some(3));
        assert_eq!(assoc.ordinal, Some(3));
        assert!((assoc.weight - 0.8).abs() < 0.001);
    }

    #[test]
    fn with_ordinal_none_is_unordered() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let assoc = Association::with_ordinal(a, b, 0.5, None);
        assert_eq!(assoc.ordinal, None);
    }

    #[test]
    fn strengthen_preserves_ordinal() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let mut assoc = Association::with_ordinal(a, b, 0.5, Some(2));
        assoc.strengthen(0.2);
        // Ordinal should not change — it's structural metadata
        assert_eq!(assoc.ordinal, Some(2));
        assert!((assoc.weight - 0.7).abs() < 0.001);
    }

    #[test]
    fn weaken_preserves_ordinal() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let mut assoc = Association::with_ordinal(a, b, 0.8, Some(5));
        assoc.weaken(0.3);
        assert_eq!(assoc.ordinal, Some(5));
        assert!((assoc.weight - 0.5).abs() < 0.001);
    }
}
