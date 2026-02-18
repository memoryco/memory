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

    #[error("PDF validation failed: {0}")]
    Validation(#[from] super::sanitize::SanitizeError),
}

pub type Result<T> = std::result::Result<T, ReferenceError>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::sanitize::SanitizeError;
    use std::path::PathBuf;

    #[test]
    fn sanitize_error_not_a_pdf_converts_to_reference_error() {
        let sanitize_err = SanitizeError::NotAPdf;
        let ref_err: ReferenceError = sanitize_err.into();
        assert!(matches!(ref_err, ReferenceError::Validation(_)));
        assert!(ref_err.to_string().contains("not a PDF file"));
    }

    #[test]
    fn sanitize_error_invalid_magic_converts_to_reference_error() {
        let sanitize_err = SanitizeError::InvalidMagicBytes;
        let ref_err: ReferenceError = sanitize_err.into();
        assert!(matches!(ref_err, ReferenceError::Validation(_)));
        assert!(ref_err.to_string().contains("magic bytes"));
    }

    #[test]
    fn sanitize_error_file_not_found_converts_to_reference_error() {
        let sanitize_err = SanitizeError::FileNotFound(PathBuf::from("/missing/file.pdf"));
        let ref_err: ReferenceError = sanitize_err.into();
        assert!(matches!(ref_err, ReferenceError::Validation(_)));
        assert!(ref_err.to_string().contains("/missing/file.pdf"));
    }

    #[test]
    fn validation_error_display_includes_context() {
        let sanitize_err = SanitizeError::NotAPdf;
        let ref_err: ReferenceError = sanitize_err.into();
        let msg = ref_err.to_string();
        assert!(msg.starts_with("PDF validation failed:"), "got: {}", msg);
    }
}
