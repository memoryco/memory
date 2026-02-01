//! Foundation storage module for SQLite database operations.
//!
//! This module provides common database infrastructure used by engram, identity,
//! plans, reference, and other modules that need persistent storage.
//!
//! # Overview
//!
//! - [`Database`] - SQLite connection wrapper with sensible defaults
//! - [`StorageError`] - Unified error type for all storage operations
//! - [`StorageResult`] - Result type alias
//!
//! # Example
//!
//! ```ignore
//! use memory::storage::{Database, StorageResult};
//!
//! fn setup_my_storage() -> StorageResult<Database> {
//!     let db = Database::open("my_module.db")?;
//!     db.initialize_schema(r#"
//!         CREATE TABLE IF NOT EXISTS my_data (
//!             id TEXT PRIMARY KEY,
//!             value TEXT NOT NULL,
//!             created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
//!         );
//!     "#)?;
//!     Ok(db)
//! }
//! ```

mod database;
mod error;

pub use database::Database;
pub use error::{StorageError, StorageResult};
