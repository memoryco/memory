//! Diesel models for plans database
//!
//! Row types for database operations and conversions to domain types.

use diesel::prelude::*;
use super::schema::{plans, steps};
use crate::plans::{Plan, Step, PlanId};
use crate::storage::StorageResult;

// ==================
// PLAN
// ==================

#[derive(Queryable, Selectable, Debug)]
#[diesel(table_name = plans)]
#[diesel(check_for_backend(diesel::sqlite::Sqlite))]
pub struct PlanRow {
    pub id: String,
    pub description: String,
    pub created_at: i64,
}

#[derive(Insertable)]
#[diesel(table_name = plans)]
pub struct NewPlan<'a> {
    pub id: &'a str,
    pub description: &'a str,
    pub created_at: i64,
}

impl PlanRow {
    pub fn into_plan(self, steps: Vec<Step>) -> StorageResult<Plan> {
        let id = PlanId::parse_str(&self.id)
            .map_err(|e| crate::storage::StorageError::Serialization(e.to_string()))?;
        
        Ok(Plan {
            id,
            description: self.description,
            created_at: self.created_at,
            steps,
        })
    }
}

// ==================
// STEP
// ==================

#[allow(dead_code)] // Fields mapped from DB schema
#[derive(Queryable, Selectable, Debug)]
#[diesel(table_name = steps)]
#[diesel(check_for_backend(diesel::sqlite::Sqlite))]
pub struct StepRow {
    pub plan_id: String,
    pub step: i32,
    pub description: String,
    pub completed: i32,
}

#[derive(Insertable)]
#[diesel(table_name = steps)]
pub struct NewStep<'a> {
    pub plan_id: &'a str,
    pub step: i32,
    pub description: &'a str,
    pub completed: i32,
}

impl StepRow {
    pub fn into_step(self) -> Step {
        Step {
            index: self.step as usize,
            description: self.description,
            completed: self.completed != 0,
        }
    }
}
