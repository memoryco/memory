//! Foundation storage module for database operations.
//!
//! This module provides common database infrastructure shared across
//! all storage backends.
//!
//! # Overview
//!
//! - [`StorageError`] - Unified error type for all storage operations
//! - [`StorageResult`] - Result type alias
//!
//! Each domain (engram, identity) owns its own schema and models.
//! See `engram::storage` for domain-specific types.

mod error;

pub use error::{StorageError, StorageResult};
