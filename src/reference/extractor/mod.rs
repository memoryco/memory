//! PDF text extraction abstraction layer.
//!
//! Provides a trait for extracting text from PDFs, allowing different
//! backends to be swapped in.

mod pdf_extract_impl;

pub use pdf_extract_impl::PdfExtractBackend;

use super::error::Result;
use std::path::Path;

/// Text content from a single page.
#[derive(Debug, Clone)]
pub struct PageText {
    /// 1-indexed page number
    pub page_number: usize,
    /// Extracted text content
    pub text: String,
}

/// Trait for PDF text extraction backends.
pub trait PdfExtractor: Send + Sync {
    /// Extract text from all pages of a PDF.
    fn extract(&self, path: &Path) -> Result<Vec<PageText>>;

    /// Extract text from a specific page range (inclusive, 1-indexed).
    fn extract_range(&self, path: &Path, start: usize, end: usize) -> Result<Vec<PageText>> {
        let all = self.extract(path)?;
        Ok(all
            .into_iter()
            .filter(|p| p.page_number >= start && p.page_number <= end)
            .collect())
    }

    /// Extract a sample of text for profile detection.
    /// Default: first 5 pages.
    fn extract_sample(&self, path: &Path) -> Result<String> {
        let pages = self.extract_range(path, 1, 5)?;
        Ok(pages.into_iter().map(|p| p.text).collect::<Vec<_>>().join("\n"))
    }
}

/// Returns the default PDF extractor.
pub fn default_extractor() -> Box<dyn PdfExtractor> {
    Box::new(PdfExtractBackend::new())
}
