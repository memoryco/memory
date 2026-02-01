//! Unified storage error type for all database operations.

use std::fmt;

/// Unified error type for all storage operations.
#[derive(Debug)]
pub enum StorageError {
    /// Database connection or query error
    Database(String),
    /// Requested item not found
    NotFound(String),
    /// Data serialization/deserialization error
    Serialization(String),
    /// Schema or migration error
    Schema(String),
    /// I/O error (file operations)
    Io(String),
    /// Configuration error
    Config(String),
    /// Generic error with context
    Other(String),
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageError::Database(msg) => write!(f, "Database error: {}", msg),
            StorageError::NotFound(msg) => write!(f, "Not found: {}", msg),
            StorageError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            StorageError::Schema(msg) => write!(f, "Schema error: {}", msg),
            StorageError::Io(msg) => write!(f, "I/O error: {}", msg),
            StorageError::Config(msg) => write!(f, "Config error: {}", msg),
            StorageError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for StorageError {}

impl From<rusqlite::Error> for StorageError {
    fn from(err: rusqlite::Error) -> Self {
        StorageError::Database(err.to_string())
    }
}

impl From<std::io::Error> for StorageError {
    fn from(err: std::io::Error) -> Self {
        StorageError::Io(err.to_string())
    }
}

impl From<serde_json::Error> for StorageError {
    fn from(err: serde_json::Error) -> Self {
        StorageError::Serialization(err.to_string())
    }
}

/// Result type alias for storage operations.
pub type StorageResult<T> = Result<T, StorageError>;
