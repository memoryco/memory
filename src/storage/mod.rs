//! Foundation storage module for database operations.
//!
//! This module provides common database infrastructure used by engram, identity,
//! and other modules that need persistent storage.
//!
//! # Overview
//!
//! - [`StorageError`] - Unified error type for all storage operations
//! - [`StorageResult`] - Result type alias
//! - [`schema`] - Diesel table definitions
//! - [`models`] - Diesel model structs (Queryable, Insertable)
//!
//! The engram storage system uses Diesel ORM for type-safe database access.
//! Backend selection (SQLite, Postgres, MySQL) is done via Cargo features:
//!
//! ```toml
//! # SQLite (default, local development)
//! diesel = { version = "2.2", features = ["sqlite"] }
//!
//! # PostgreSQL (production)
//! diesel = { version = "2.2", features = ["postgres"] }
//! ```

mod error;
pub mod models;
pub mod schema;

pub use error::{StorageError, StorageResult};
