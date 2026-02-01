//! PlanStore - The coordination layer for plans
//!
//! PlanStore wraps storage and provides the high-level API
//! for managing plans and steps.

use super::{Plan, PlanId, Step};
use super::storage::{PlanStorage, StorageResult};

/// PlanStore - high-level API for plan management
/// 
/// This is the main entry point for the plans system.
/// All operations persist to storage immediately.
pub struct PlanStore {
    storage: Box<dyn PlanStorage>,
}

impl PlanStore {
    /// Create a new PlanStore with the given storage backend
    pub fn new<S: PlanStorage + 'static>(mut storage: S) -> StorageResult<Self> {
        storage.initialize()?;
        Ok(Self {
            storage: Box::new(storage),
        })
    }
    
    /// List all active plans (id + description)
    pub fn list(&mut self) -> StorageResult<Vec<(PlanId, String)>> {
        self.storage.list_plans()
    }
    
    /// Get a single plan with all steps
    pub fn get(&mut self, id: &PlanId) -> StorageResult<Option<Plan>> {
        self.storage.get_plan(id)
    }
    
    /// Start a new plan
    /// Returns the plan ID
    pub fn start(&mut self, description: &str) -> StorageResult<PlanId> {
        let id = PlanId::new_v4();
        self.storage.create_plan(&id, description)?;
        Ok(id)
    }
    
    /// Add a step to a plan
    /// Returns the step index
    pub fn add_step(&mut self, plan_id: &PlanId, description: &str) -> StorageResult<usize> {
        self.storage.add_step(plan_id, description)
    }
    
    /// Mark a step as completed
    /// Returns true if the step existed and was updated
    pub fn complete_step(&mut self, plan_id: &PlanId, step: usize) -> StorageResult<bool> {
        self.storage.complete_step(plan_id, step)
    }
    
    /// Stop (delete) a plan
    /// Returns true if the plan existed and was deleted
    pub fn stop(&mut self, id: &PlanId) -> StorageResult<bool> {
        self.storage.delete_plan(id)
    }
    
    /// Flush any pending writes
    pub fn flush(&mut self) -> StorageResult<()> {
        self.storage.flush()
    }
    
    /// Close the store cleanly
    pub fn close(&mut self) -> StorageResult<()> {
        self.storage.close()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::storage::DieselPlanStorage;
    
    #[test]
    fn full_workflow() {
        let storage = DieselPlanStorage::in_memory().unwrap();
        let mut store = PlanStore::new(storage).unwrap();
        
        // Start a plan
        let plan_id = store.start("Build the plans feature").unwrap();
        
        // Add steps
        let s1 = store.add_step(&plan_id, "Create storage layer").unwrap();
        let s2 = store.add_step(&plan_id, "Create PlanStore").unwrap();
        let s3 = store.add_step(&plan_id, "Create tools").unwrap();
        
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
        assert_eq!(s3, 3);
        
        // Complete first step
        store.complete_step(&plan_id, 1).unwrap();
        
        // Check status
        let plan = store.get(&plan_id).unwrap().unwrap();
        assert_eq!(plan.steps.len(), 3);
        assert!(plan.steps[0].completed);
        assert!(!plan.steps[1].completed);
        assert!(!plan.steps[2].completed);
        
        // List plans
        let plans = store.list().unwrap();
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].1, "Build the plans feature");
        
        // Stop the plan
        let stopped = store.stop(&plan_id).unwrap();
        assert!(stopped);
        
        // Plan is gone
        let plans = store.list().unwrap();
        assert!(plans.is_empty());
    }
}
