//! Reference source management - a single PDF + index pair.

use super::citation::{Citation, SourceMeta, meta_path_for};
use super::error::Result;
use super::extractor::PdfExtractor;
use super::indexer::{Indexer, index_path_for};
use super::profiles::ProfileRegistry;
use super::searcher::{SearchResult, Searcher};
use std::path::{Path, PathBuf};

/// A single reference source (PDF + FTS5 index).
pub struct ReferenceSource {
    /// Source name (derived from filename without extension)
    name: String,
    /// Path to the PDF file
    pdf_path: PathBuf,
    /// Path to the index file
    index_path: PathBuf,
    /// Source metadata (citation + bootstrap memories)
    meta: Option<SourceMeta>,
    /// Searcher instance (lazy-loaded)
    searcher: Option<Searcher>,
}

impl ReferenceSource {
    /// Create a new reference source from a PDF path.
    pub fn new(pdf_path: impl AsRef<Path>) -> Self {
        let pdf_path = pdf_path.as_ref().to_path_buf();
        let index_path = index_path_for(&pdf_path);
        let meta_path = meta_path_for(&pdf_path);
        let name = pdf_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Try to load source metadata
        let meta = SourceMeta::load(&meta_path);

        Self {
            name,
            pdf_path,
            index_path,
            meta,
            searcher: None,
        }
    }

    /// Get the source name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the PDF path.
    pub fn pdf_path(&self) -> &Path {
        &self.pdf_path
    }

    /// Get the index path.
    pub fn index_path(&self) -> &Path {
        &self.index_path
    }

    /// Get the source metadata if available.
    pub fn meta(&self) -> Option<&SourceMeta> {
        self.meta.as_ref()
    }

    /// Get the citation metadata if available.
    pub fn citation(&self) -> Option<&Citation> {
        self.meta.as_ref().and_then(|m| m.citation.as_ref())
    }

    /// Check if the index exists.
    pub fn has_index(&self) -> bool {
        self.index_path.exists()
    }

    /// Build or rebuild the index.
    pub fn build_index(
        &mut self,
        extractor: &dyn PdfExtractor,
        profiles: &ProfileRegistry,
    ) -> Result<()> {
        let indexer = Indexer::new(extractor, profiles);
        indexer.build(&self.pdf_path)?;
        // Reset searcher so it reloads
        self.searcher = None;
        Ok(())
    }

    /// Ensure the index exists, building if necessary.
    pub fn ensure_index(
        &mut self,
        extractor: &dyn PdfExtractor,
        profiles: &ProfileRegistry,
    ) -> Result<()> {
        let indexer = Indexer::new(extractor, profiles);

        if !indexer.index_is_current(&self.pdf_path) {
            self.build_index(extractor, profiles)?;
        }

        Ok(())
    }

    /// Get a searcher for this source.
    pub fn searcher(&mut self) -> Result<&Searcher> {
        if self.searcher.is_none() {
            self.searcher = Some(Searcher::open(&self.index_path)?);
        }
        Ok(self.searcher.as_ref().unwrap())
    }

    /// Search this source.
    pub fn search(&mut self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        self.searcher()?.search(query, limit)
    }

    /// Get a section by title.
    pub fn get_section(&mut self, title: &str) -> Result<Option<SearchResult>> {
        self.searcher()?.get_section(title)
    }

    /// List all top-level sections.
    pub fn list_sections(&mut self) -> Result<Vec<String>> {
        self.searcher()?.list_sections()
    }
}
