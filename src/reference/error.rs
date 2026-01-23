//! Error types for the reference crate.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ReferenceError {
    #[error("PDF extraction failed: {0}")]
    Extraction(String),

    #[error("Index error: {0}")]
    Index(String),

    #[error("Search error: {0}")]
    Search(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Source not found: {0}")]
    SourceNotFound(String),
}

pub type Result<T> = std::result::Result<T, ReferenceError>;
