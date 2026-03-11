//! Reference - Authoritative document indexing and search.
//!
//! This crate provides full-text search over PDF documents, with optional
//! curated profiles for richer section-level parsing of known document types.
//!
//! # Usage
//!
//! ```no_run
//! use reference::ReferenceManager;
//!
//! // Create a manager and load sources from a directory
//! let mut manager = ReferenceManager::new();
//! manager.load_directory("/path/to/references")?;
//!
//! // Search across all sources
//! let results = manager.search("major depressive disorder", 10)?;
//!
//! // Search a specific source
//! let results = manager.search_source("dsm5tr", "bipolar", 10)?;
//! # Ok::<(), reference::ReferenceError>(())
//! ```

pub mod bootstrap;
pub mod citation;
pub mod error;
pub mod extractor;
pub mod indexer;
pub mod profiles;
pub mod sanitize;
pub mod searcher;
pub mod source;

#[cfg(test)]
mod harness;

pub use citation::{Citation, SourceMeta};
pub use error::{ReferenceError, Result};
pub use extractor::PdfExtractor;
pub use profiles::{DocumentProfile, ProfileRegistry};
pub use sanitize::{sanitize_and_copy, validate_pdf};
pub use searcher::SearchResult;
pub use source::ReferenceSource;

use std::collections::HashMap;
use std::path::Path;

/// Manages multiple reference sources.
pub struct ReferenceManager {
    sources: HashMap<String, ReferenceSource>,
    extractor: Box<dyn PdfExtractor>,
    profiles: ProfileRegistry,
}

impl ReferenceManager {
    /// Create a new reference manager with default settings.
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
            extractor: extractor::default_extractor(),
            profiles: ProfileRegistry::new(),
        }
    }

    /// Create with a custom PDF extractor.
    pub fn with_extractor(extractor: Box<dyn PdfExtractor>) -> Self {
        Self {
            sources: HashMap::new(),
            extractor,
            profiles: ProfileRegistry::new(),
        }
    }

    /// Register a custom document profile.
    pub fn register_profile(&mut self, profile: Box<dyn DocumentProfile>) {
        self.profiles.register(profile);
    }

    /// Add a reference source from a PDF path.
    /// Validates the PDF first, then builds the index if it doesn't exist or is outdated.
    pub fn add_source(&mut self, pdf_path: impl AsRef<Path>) -> Result<&str> {
        let path = pdf_path.as_ref();
        validate_pdf(path)?;
        let mut source = ReferenceSource::new(path);
        source.ensure_index(self.extractor.as_ref(), &self.profiles)?;
        let name = source.name().to_string();
        self.sources.insert(name.clone(), source);
        Ok(self.sources.get(&name).unwrap().name())
    }

    /// Load all PDFs from a directory.
    pub fn load_directory(&mut self, dir: impl AsRef<Path>) -> Result<Vec<String>> {
        let dir = dir.as_ref();
        let mut loaded = Vec::new();

        if !dir.exists() {
            return Ok(loaded);
        }

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("pdf") {
                // Validate before attempting full ingestion
                if let Err(e) = validate_pdf(&path) {
                    eprintln!("Warning: skipping invalid PDF {}: {}", path.display(), e);
                    continue;
                }

                match self.add_source(&path) {
                    Ok(name) => loaded.push(name.to_string()),
                    Err(e) => {
                        eprintln!("Warning: failed to load {}: {}", path.display(), e);
                    }
                }
            }
        }

        Ok(loaded)
    }

    /// List all loaded sources.
    pub fn sources(&self) -> Vec<&str> {
        self.sources.keys().map(|s| s.as_str()).collect()
    }

    /// Get a source by name.
    pub fn get_source(&mut self, name: &str) -> Option<&mut ReferenceSource> {
        self.sources.get_mut(name)
    }

    /// Search across all sources.
    pub fn search(&mut self, query: &str, limit: usize) -> Result<Vec<(String, SearchResult)>> {
        let mut all_results = Vec::new();

        for (name, source) in &mut self.sources {
            match source.search(query, limit) {
                Ok(results) => {
                    for result in results {
                        all_results.push((name.clone(), result));
                    }
                }
                Err(e) => {
                    eprintln!("Warning: search failed for {}: {}", name, e);
                }
            }
        }

        // Sort by rank across all sources
        all_results.sort_by(|a, b| {
            a.1.rank
                .partial_cmp(&b.1.rank)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_results.truncate(limit);

        Ok(all_results)
    }

    /// Search a specific source.
    pub fn search_source(
        &mut self,
        source_name: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let source = self
            .sources
            .get_mut(source_name)
            .ok_or_else(|| ReferenceError::SourceNotFound(source_name.to_string()))?;

        source.search(query, limit)
    }

    /// Get a section from a specific source.
    pub fn get_section(&mut self, source_name: &str, title: &str) -> Result<Option<SearchResult>> {
        let source = self
            .sources
            .get_mut(source_name)
            .ok_or_else(|| ReferenceError::SourceNotFound(source_name.to_string()))?;

        source.get_section(title)
    }

    /// List sections in a source.
    pub fn list_sections(&mut self, source_name: &str) -> Result<Vec<String>> {
        let source = self
            .sources
            .get_mut(source_name)
            .ok_or_else(|| ReferenceError::SourceNotFound(source_name.to_string()))?;

        source.list_sections()
    }

    /// Get the citation for a source.
    pub fn get_citation(&self, source_name: &str) -> Option<&Citation> {
        self.sources.get(source_name)?.citation()
    }

    /// Get the full source metadata.
    pub fn get_meta(&self, source_name: &str) -> Option<&SourceMeta> {
        self.sources.get(source_name)?.meta()
    }

    /// Get related sources for a source.
    pub fn get_related_sources(&self, source_name: &str) -> Vec<&str> {
        self.get_meta(source_name)
            .map(|m| m.related_sources.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Search a source and optionally its related sources.
    /// Returns results tagged with source name for proper citation.
    pub fn search_source_with_related(
        &mut self,
        source_name: &str,
        query: &str,
        limit: usize,
        include_related: bool,
    ) -> Result<Vec<(String, SearchResult)>> {
        let mut all_results = Vec::new();

        // Get related sources first (before mutable borrow)
        let related: Vec<String> = if include_related {
            self.get_related_sources(source_name)
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            Vec::new()
        };

        // Search primary source
        if let Some(source) = self.sources.get_mut(source_name) {
            match source.search(query, limit) {
                Ok(results) => {
                    for result in results {
                        all_results.push((source_name.to_string(), result));
                    }
                }
                Err(e) => {
                    eprintln!("Warning: search failed for {}: {}", source_name, e);
                }
            }
        } else {
            return Err(ReferenceError::SourceNotFound(source_name.to_string()));
        }

        // Search related sources
        for related_name in related {
            if let Some(source) = self.sources.get_mut(&related_name) {
                match source.search(query, limit) {
                    Ok(results) => {
                        for result in results {
                            all_results.push((related_name.clone(), result));
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: search failed for related source {}: {}",
                            related_name, e
                        );
                    }
                }
            }
        }

        // Sort by rank across all sources
        all_results.sort_by(|a, b| {
            a.1.rank
                .partial_cmp(&b.1.rank)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_results.truncate(limit);

        Ok(all_results)
    }
}

impl Default for ReferenceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_fake_pdf(dir: &Path, name: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"%PDF-1.4 fake pdf content").unwrap();
        path
    }

    fn create_text_file(dir: &Path, name: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"This is not a PDF").unwrap();
        path
    }

    #[test]
    fn add_source_rejects_nonexistent_path() {
        let mut mgr = ReferenceManager::new();
        let result = mgr.add_source("/nonexistent/missing.pdf");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, ReferenceError::Validation(_)),
            "expected Validation error, got: {:?}",
            err
        );
    }

    #[test]
    fn add_source_rejects_non_pdf_extension() {
        let dir = TempDir::new().unwrap();
        let path = create_text_file(dir.path(), "notes.txt");
        let mut mgr = ReferenceManager::new();
        let result = mgr.add_source(&path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, ReferenceError::Validation(_)),
            "expected Validation error, got: {:?}",
            err
        );
    }

    #[test]
    fn add_source_rejects_invalid_magic_bytes() {
        let dir = TempDir::new().unwrap();
        let path = create_text_file(dir.path(), "fake.pdf");
        let mut mgr = ReferenceManager::new();
        let result = mgr.add_source(&path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, ReferenceError::Validation(_)),
            "expected Validation error, got: {:?}",
            err
        );
    }

    #[test]
    fn load_directory_skips_invalid_pdfs() {
        let dir = TempDir::new().unwrap();

        // Create a file with .pdf extension but invalid magic bytes
        create_text_file(dir.path(), "bad_magic.pdf");

        // Create a non-pdf file
        create_text_file(dir.path(), "readme.txt");

        let mut mgr = ReferenceManager::new();
        let loaded = mgr.load_directory(dir.path()).unwrap();
        // Neither file should have been loaded
        assert!(
            loaded.is_empty(),
            "expected no sources loaded, got: {:?}",
            loaded
        );
    }

    #[test]
    fn load_directory_nonexistent_dir_returns_empty() {
        let mut mgr = ReferenceManager::new();
        let loaded = mgr.load_directory("/nonexistent/dir").unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn load_directory_empty_dir_returns_empty() {
        let dir = TempDir::new().unwrap();
        let mut mgr = ReferenceManager::new();
        let loaded = mgr.load_directory(dir.path()).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn add_source_valid_fake_pdf_fails_at_extraction_not_validation() {
        // A file with valid PDF magic bytes but no real structure should
        // pass validation but fail during extraction/indexing
        let dir = TempDir::new().unwrap();
        let path = create_fake_pdf(dir.path(), "valid_header.pdf");
        let mut mgr = ReferenceManager::new();
        let result = mgr.add_source(&path);
        // Should fail at extraction, not validation
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            !matches!(err, ReferenceError::Validation(_)),
            "should have passed validation but failed later, got: {:?}",
            err
        );
    }
}
