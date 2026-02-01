//! Plans - A simple planning system
//!
//! Plans are working documents for tracking multi-step work.
//! They're transient - when you're done (finished or abandoned),
//! delete them with stop().
//!
//! Key concepts:
//! - **Plan** - A goal with a description and ordered steps
//! - **Step** - An individual action item within a plan
//! - Plans exist until explicitly stopped (no decay)
//! - Steps are append-only (indices auto-increment)

pub mod bootstrap;
pub mod storage;
pub mod store;

pub use store::PlanStore;
pub use storage::{PlanStorage, DieselPlanStorage, StorageResult, StorageError};

use serde::{Serialize, Deserialize};

/// Unique identifier for a plan
pub type PlanId = uuid::Uuid;

/// A plan with its steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    /// Unique identifier
    pub id: PlanId,
    
    /// What this plan is for
    pub description: String,
    
    /// When the plan was created (unix timestamp)
    pub created_at: i64,
    
    /// Ordered list of steps
    pub steps: Vec<Step>,
}

impl Plan {
    /// Create a new plan with no steps
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            id: PlanId::new_v4(),
            description: description.into(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
            steps: Vec::new(),
        }
    }
    
    /// Check if all steps are completed
    pub fn is_complete(&self) -> bool {
        !self.steps.is_empty() && self.steps.iter().all(|s| s.completed)
    }
    
    /// Count completed steps
    pub fn completed_count(&self) -> usize {
        self.steps.iter().filter(|s| s.completed).count()
    }
    
    /// Get progress as a fraction (0.0 - 1.0)
    pub fn progress(&self) -> f64 {
        if self.steps.is_empty() {
            0.0
        } else {
            self.completed_count() as f64 / self.steps.len() as f64
        }
    }
}

/// A single step within a plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    /// Step index (1-based)
    pub index: usize,
    
    /// What needs to be done
    pub description: String,
    
    /// Whether this step is done
    pub completed: bool,
}

impl Step {
    /// Create a new incomplete step
    pub fn new(index: usize, description: impl Into<String>) -> Self {
        Self {
            index,
            description: description.into(),
            completed: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn plan_progress() {
        let mut plan = Plan::new("Test plan");
        
        assert_eq!(plan.progress(), 0.0);
        assert!(!plan.is_complete());
        
        plan.steps.push(Step::new(1, "Step 1"));
        plan.steps.push(Step::new(2, "Step 2"));
        
        assert_eq!(plan.progress(), 0.0);
        assert!(!plan.is_complete());
        
        plan.steps[0].completed = true;
        assert_eq!(plan.progress(), 0.5);
        assert!(!plan.is_complete());
        
        plan.steps[1].completed = true;
        assert_eq!(plan.progress(), 1.0);
        assert!(plan.is_complete());
    }
}
