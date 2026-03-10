//! Storage trait for plans
//!
//! This module defines the contract for storing and retrieving plan data.
//! Implementations can be SQLite, Postgres, etc.

mod diesel;
pub mod models;
pub mod schema;

pub use diesel::DieselPlanStorage;

// Re-export unified storage types from the foundation
pub use crate::storage::{StorageError, StorageResult};

use crate::plans::{Plan, PlanId};

/// The storage contract for plans
///
/// Implementations of this trait provide persistence for the plans system.
/// All operations are synchronous.
pub trait PlanStorage: Send {
    /// Initialize storage (create tables, etc.)
    fn initialize(&mut self) -> StorageResult<()>;

    /// List all active plans (id + description)
    fn list_plans(&mut self) -> StorageResult<Vec<(PlanId, String)>>;

    /// Get a single plan with all its steps
    fn get_plan(&mut self, id: &PlanId) -> StorageResult<Option<Plan>>;

    /// Create a new plan
    fn create_plan(&mut self, id: &PlanId, description: &str) -> StorageResult<()>;

    /// Delete a plan and all its steps
    fn delete_plan(&mut self, id: &PlanId) -> StorageResult<bool>;

    /// Add a step to a plan, returns the step index
    fn add_step(&mut self, plan_id: &PlanId, description: &str) -> StorageResult<usize>;

    /// Mark a step as completed
    fn complete_step(&mut self, plan_id: &PlanId, step: usize) -> StorageResult<bool>;

    /// Flush any pending writes
    fn flush(&mut self) -> StorageResult<()> {
        Ok(())
    }

    /// Close the storage cleanly
    fn close(&mut self) -> StorageResult<()> {
        self.flush()
    }
}
