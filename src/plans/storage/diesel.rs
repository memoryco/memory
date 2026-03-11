//! Diesel-backed storage implementation for Plans
//!
//! This module provides database storage using Diesel ORM.
//! The actual database backend (SQLite, Postgres) is selected
//! via feature flags at compile time.

use super::models::*;
use super::schema::{plans, steps};
use super::{PlanStorage, StorageError, StorageResult};
use crate::plans::{Plan, PlanId, Step};

use diesel::connection::SimpleConnection;
use diesel::prelude::*;
use std::path::Path;

// Conditional imports based on feature flags
#[cfg(feature = "sqlite")]
use diesel::sqlite::SqliteConnection as DbConnection;

#[cfg(feature = "postgres")]
use diesel::pg::PgConnection as DbConnection;

/// Database-backed storage implementation for Plans
///
/// The underlying database is selected at compile time via feature flags:
/// - `sqlite` (default) - Uses SQLite, suitable for local/embedded use
/// - `postgres` - Uses PostgreSQL, suitable for server/SaaS deployments
pub struct DieselPlanStorage {
    conn: DbConnection,
}

impl DieselPlanStorage {
    /// Open storage at the given path (SQLite)
    #[cfg(feature = "sqlite")]
    pub fn open<P: AsRef<Path>>(path: P) -> StorageResult<Self> {
        let path_str = path.as_ref().to_string_lossy();
        let mut conn = DbConnection::establish(&path_str)
            .map_err(|e| StorageError::Database(e.to_string()))?;

        // Apply SQLite pragmas for performance
        conn.batch_execute(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA foreign_keys = ON;
            PRAGMA busy_timeout = 5000;
        "#,
        )
        .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(Self { conn })
    }

    /// Open storage at the given connection URL (Postgres)
    #[cfg(feature = "postgres")]
    pub fn open(connection_url: &str) -> StorageResult<Self> {
        let conn = DbConnection::establish(connection_url)
            .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(Self { conn })
    }

    /// Create an in-memory database (for testing) - SQLite only
    #[cfg(test)]
    pub fn in_memory() -> StorageResult<Self> {
        let mut conn = DbConnection::establish(":memory:")
            .map_err(|e| StorageError::Database(e.to_string()))?;

        conn.batch_execute("PRAGMA foreign_keys = ON;")
            .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(Self { conn })
    }

    /// Create the database schema (SQLite)
    #[cfg(feature = "sqlite")]
    fn create_schema(&mut self) -> StorageResult<()> {
        self.conn
            .batch_execute(
                r#"
            -- Plans table
            CREATE TABLE IF NOT EXISTS plans (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            
            -- Steps table
            CREATE TABLE IF NOT EXISTS steps (
                plan_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                description TEXT NOT NULL,
                completed INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (plan_id, step),
                FOREIGN KEY (plan_id) REFERENCES plans(id) ON DELETE CASCADE
            );
            
            -- Index for step lookups
            CREATE INDEX IF NOT EXISTS idx_steps_plan ON steps(plan_id);
        "#,
            )
            .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(())
    }

    /// Load steps for a plan
    fn load_steps(&mut self, plan_id: &str) -> StorageResult<Vec<Step>> {
        let rows: Vec<StepRow> = steps::table
            .filter(steps::plan_id.eq(plan_id))
            .order(steps::step.asc())
            .select(StepRow::as_select())
            .load(&mut self.conn)?;

        Ok(rows.into_iter().map(|r| r.into_step()).collect())
    }
}

impl PlanStorage for DieselPlanStorage {
    #[cfg(feature = "sqlite")]
    fn initialize(&mut self) -> StorageResult<()> {
        self.create_schema()
    }

    #[cfg(feature = "postgres")]
    fn initialize(&mut self) -> StorageResult<()> {
        // Postgres schema is managed via migrations (diesel_cli)
        Ok(())
    }

    fn list_plans(&mut self) -> StorageResult<Vec<(PlanId, String)>> {
        let rows: Vec<PlanRow> = plans::table
            .select(PlanRow::as_select())
            .load(&mut self.conn)?;

        rows.into_iter()
            .map(|row| {
                let id = PlanId::parse_str(&row.id)
                    .map_err(|e| StorageError::Serialization(e.to_string()))?;
                Ok((id, row.description))
            })
            .collect()
    }

    fn get_plan(&mut self, id: &PlanId) -> StorageResult<Option<Plan>> {
        let id_str = id.to_string();

        let result: Option<PlanRow> = plans::table
            .filter(plans::id.eq(&id_str))
            .select(PlanRow::as_select())
            .first(&mut self.conn)
            .optional()?;

        match result {
            Some(row) => {
                let steps = self.load_steps(&id_str)?;
                Ok(Some(row.into_plan(steps)?))
            }
            None => Ok(None),
        }
    }

    fn create_plan(&mut self, id: &PlanId, description: &str) -> StorageResult<()> {
        let id_str = id.to_string();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        diesel::insert_into(plans::table)
            .values(NewPlan {
                id: &id_str,
                description,
                created_at: now,
            })
            .execute(&mut self.conn)?;

        Ok(())
    }

    fn delete_plan(&mut self, id: &PlanId) -> StorageResult<bool> {
        let id_str = id.to_string();

        // Steps are deleted via CASCADE, but let's be explicit
        diesel::delete(steps::table.filter(steps::plan_id.eq(&id_str))).execute(&mut self.conn)?;

        let deleted =
            diesel::delete(plans::table.filter(plans::id.eq(&id_str))).execute(&mut self.conn)?;

        Ok(deleted > 0)
    }

    fn add_step(&mut self, plan_id: &PlanId, description: &str) -> StorageResult<usize> {
        let plan_id_str = plan_id.to_string();

        // Get the next step index
        let max_step: Option<i32> = steps::table
            .filter(steps::plan_id.eq(&plan_id_str))
            .select(diesel::dsl::max(steps::step))
            .first(&mut self.conn)?;

        let next_step = max_step.map(|s| s + 1).unwrap_or(1);

        diesel::insert_into(steps::table)
            .values(NewStep {
                plan_id: &plan_id_str,
                step: next_step,
                description,
                completed: 0,
            })
            .execute(&mut self.conn)?;

        Ok(next_step as usize)
    }

    fn complete_step(&mut self, plan_id: &PlanId, step: usize) -> StorageResult<bool> {
        let plan_id_str = plan_id.to_string();

        let updated = diesel::update(
            steps::table
                .filter(steps::plan_id.eq(&plan_id_str))
                .filter(steps::step.eq(step as i32)),
        )
        .set(steps::completed.eq(1))
        .execute(&mut self.conn)?;

        Ok(updated > 0)
    }

    #[cfg(feature = "sqlite")]
    fn flush(&mut self) -> StorageResult<()> {
        self.conn.batch_execute("PRAGMA wal_checkpoint(PASSIVE);")?;
        Ok(())
    }

    #[cfg(feature = "postgres")]
    fn flush(&mut self) -> StorageResult<()> {
        // Postgres doesn't need explicit flush
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_get_plan() {
        let mut storage = DieselPlanStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let id = PlanId::new_v4();
        storage.create_plan(&id, "Test plan").unwrap();

        let plan = storage.get_plan(&id).unwrap();
        assert!(plan.is_some());
        let plan = plan.unwrap();
        assert_eq!(plan.description, "Test plan");
        assert!(plan.steps.is_empty());
    }

    #[test]
    fn add_and_complete_steps() {
        let mut storage = DieselPlanStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let id = PlanId::new_v4();
        storage.create_plan(&id, "Multi-step plan").unwrap();

        let step1 = storage.add_step(&id, "First step").unwrap();
        let step2 = storage.add_step(&id, "Second step").unwrap();

        assert_eq!(step1, 1);
        assert_eq!(step2, 2);

        // Complete step 1
        let completed = storage.complete_step(&id, 1).unwrap();
        assert!(completed);

        // Verify
        let plan = storage.get_plan(&id).unwrap().unwrap();
        assert_eq!(plan.steps.len(), 2);
        assert!(plan.steps[0].completed);
        assert!(!plan.steps[1].completed);
    }

    #[test]
    fn list_plans() {
        let mut storage = DieselPlanStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let id1 = PlanId::new_v4();
        let id2 = PlanId::new_v4();

        storage.create_plan(&id1, "Plan A").unwrap();
        storage.create_plan(&id2, "Plan B").unwrap();

        let all = storage.list_plans().unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn delete_plan() {
        let mut storage = DieselPlanStorage::in_memory().unwrap();
        storage.initialize().unwrap();

        let id = PlanId::new_v4();
        storage.create_plan(&id, "To be deleted").unwrap();
        storage.add_step(&id, "Step 1").unwrap();

        let deleted = storage.delete_plan(&id).unwrap();
        assert!(deleted);

        let plan = storage.get_plan(&id).unwrap();
        assert!(plan.is_none());
    }
}
