//! PDF to FTS5 index generation.

use super::error::{ReferenceError, Result};
use super::extractor::{PageText, PdfExtractor};
use super::profiles::{ProfileRegistry, Section, SectionType};
use rusqlite::Connection;
use std::path::Path;

/// Builds an FTS5 index from a PDF file.
pub struct Indexer<'a> {
    extractor: &'a dyn PdfExtractor,
    profiles: &'a ProfileRegistry,
}

impl<'a> Indexer<'a> {
    pub fn new(extractor: &'a dyn PdfExtractor, profiles: &'a ProfileRegistry) -> Self {
        Self { extractor, profiles }
    }

    /// Build an index for a PDF file.
    /// Returns the path to the generated index file.
    pub fn build(&self, pdf_path: &Path) -> Result<()> {
        let index_path = index_path_for(pdf_path);

        // Extract text from PDF
        let pages = self.extractor.extract(pdf_path)?;

        if pages.is_empty() {
            return Err(ReferenceError::Extraction(
                "No extractable text (PDF may be scanned images without OCR)".to_string(),
            ));
        }

        // Try to find a matching profile
        let filename = pdf_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let sample_text = self.extractor.extract_sample(pdf_path)?;
        let profile = self.profiles.find_profile(filename, &sample_text);

        // Create/open the index database
        let conn = Connection::open(&index_path)?;
        init_schema(&conn)?;

        // Try profile-based section parsing first
        if let Some(profile) = profile {
            if let Some(sections) = profile.parse_sections(&pages) {
                index_sections(&conn, &sections, profile.id())?;
                return Ok(());
            }
        }

        // Fall back to page-level indexing
        index_pages(&conn, &pages)?;

        Ok(())
    }

    /// Check if an index exists and is newer than the PDF.
    pub fn index_is_current(&self, pdf_path: &Path) -> bool {
        let index_path = index_path_for(pdf_path);

        if !index_path.exists() {
            return false;
        }

        // Check modification times
        let pdf_modified = pdf_path
            .metadata()
            .and_then(|m| m.modified())
            .ok();
        let index_modified = index_path
            .metadata()
            .and_then(|m| m.modified())
            .ok();

        match (pdf_modified, index_modified) {
            (Some(pdf_time), Some(index_time)) => index_time >= pdf_time,
            _ => false,
        }
    }
}

/// Get the index path for a PDF (same name, .idx extension).
pub fn index_path_for(pdf_path: &Path) -> std::path::PathBuf {
    pdf_path.with_extension("idx")
}

/// Initialize the database schema.
fn init_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        -- Metadata table
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        -- Sections table (for profile-parsed content)
        CREATE TABLE IF NOT EXISTS sections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            parent TEXT,
            page_start INTEGER NOT NULL,
            page_end INTEGER NOT NULL,
            codes TEXT,  -- JSON array of ICD codes
            section_type TEXT NOT NULL,
            content TEXT NOT NULL
        );

        -- Pages table (for generic page-level indexing)
        CREATE TABLE IF NOT EXISTS pages (
            page_number INTEGER PRIMARY KEY,
            content TEXT NOT NULL
        );

        -- FTS5 virtual table for full-text search
        CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
            title,
            content,
            codes,
            tokenize='porter unicode61'
        );

        -- Clear existing data
        DELETE FROM sections;
        DELETE FROM pages;
        DELETE FROM fts;
        "#,
    )?;

    Ok(())
}

/// Index sections from a profile parser.
fn index_sections(conn: &Connection, sections: &[Section], profile_id: &str) -> Result<()> {
    // Store profile metadata
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('profile', ?1)",
        [profile_id],
    )?;
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('index_type', 'sections')",
        [],
    )?;

    let mut insert_section = conn.prepare(
        r#"
        INSERT INTO sections (title, parent, page_start, page_end, codes, section_type, content)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
        "#,
    )?;

    let mut insert_fts = conn.prepare(
        "INSERT INTO fts (rowid, title, content, codes) VALUES (?1, ?2, ?3, ?4)",
    )?;

    for section in sections {
        let codes_json = serde_json::to_string(&section.codes).unwrap_or_default();
        let section_type_str = section_type_to_str(&section.section_type);

        insert_section.execute(rusqlite::params![
            section.title,
            section.parent,
            section.page_start,
            section.page_end,
            codes_json,
            section_type_str,
            section.content,
        ])?;

        let rowid = conn.last_insert_rowid();
        insert_fts.execute(rusqlite::params![
            rowid,
            section.title,
            section.content,
            codes_json,
        ])?;
    }

    Ok(())
}

/// Index pages (generic fallback).
fn index_pages(conn: &Connection, pages: &[PageText]) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('index_type', 'pages')",
        [],
    )?;

    let mut insert_page = conn.prepare(
        "INSERT INTO pages (page_number, content) VALUES (?1, ?2)",
    )?;

    let mut insert_fts = conn.prepare(
        "INSERT INTO fts (rowid, title, content, codes) VALUES (?1, ?2, ?3, ?4)",
    )?;

    for page in pages {
        insert_page.execute(rusqlite::params![page.page_number, page.text])?;

        // Use negative rowids for pages to avoid collision with sections
        let rowid = -(page.page_number as i64);
        let title = format!("Page {}", page.page_number);

        insert_fts.execute(rusqlite::params![rowid, title, page.text, ""])?;
    }

    Ok(())
}

fn section_type_to_str(st: &SectionType) -> &'static str {
    match st {
        SectionType::Disorder => "disorder",
        SectionType::DiagnosticCriteria => "diagnostic_criteria",
        SectionType::DiagnosticFeatures => "diagnostic_features",
        SectionType::DifferentialDiagnosis => "differential_diagnosis",
        SectionType::DevelopmentAndCourse => "development_and_course",
        SectionType::AssociatedFeatures => "associated_features",
        SectionType::Comorbidity => "comorbidity",
        SectionType::Other => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    /// A mock extractor that returns empty pages (simulating a scanned-image PDF).
    struct EmptyExtractor;

    impl PdfExtractor for EmptyExtractor {
        fn extract(&self, _path: &Path) -> Result<Vec<PageText>> {
            Ok(vec![])
        }
    }

    /// A mock extractor that returns some pages.
    struct FakeExtractor {
        pages: Vec<PageText>,
    }

    impl PdfExtractor for FakeExtractor {
        fn extract(&self, _path: &Path) -> Result<Vec<PageText>> {
            Ok(self.pages.clone())
        }
    }

    fn create_fake_pdf(dir: &Path, name: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"%PDF-1.4 fake").unwrap();
        path
    }

    #[test]
    fn build_empty_pages_returns_descriptive_error() {
        let dir = TempDir::new().unwrap();
        let pdf = create_fake_pdf(dir.path(), "scanned.pdf");

        let extractor = EmptyExtractor;
        let profiles = ProfileRegistry::new();
        let indexer = Indexer::new(&extractor, &profiles);

        let result = indexer.build(&pdf);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("scanned images without OCR"),
            "expected OCR hint in error message, got: {}",
            msg
        );
    }

    #[test]
    fn build_with_pages_creates_index() {
        let dir = TempDir::new().unwrap();
        let pdf = create_fake_pdf(dir.path(), "valid.pdf");

        let extractor = FakeExtractor {
            pages: vec![PageText {
                page_number: 1,
                text: "Hello world, this is some test content.".to_string(),
            }],
        };
        let profiles = ProfileRegistry::new();
        let indexer = Indexer::new(&extractor, &profiles);

        let result = indexer.build(&pdf);
        assert!(result.is_ok(), "build failed: {:?}", result.unwrap_err());

        // Index file should exist
        let idx = index_path_for(&pdf);
        assert!(idx.exists(), "index file not created");
    }

    #[test]
    fn index_is_current_false_when_no_index() {
        let dir = TempDir::new().unwrap();
        let pdf = create_fake_pdf(dir.path(), "test.pdf");

        let extractor = EmptyExtractor;
        let profiles = ProfileRegistry::new();
        let indexer = Indexer::new(&extractor, &profiles);

        assert!(!indexer.index_is_current(&pdf));
    }

    #[test]
    fn index_is_current_true_after_build() {
        let dir = TempDir::new().unwrap();
        let pdf = create_fake_pdf(dir.path(), "test.pdf");

        let extractor = FakeExtractor {
            pages: vec![PageText {
                page_number: 1,
                text: "Content for indexing.".to_string(),
            }],
        };
        let profiles = ProfileRegistry::new();
        let indexer = Indexer::new(&extractor, &profiles);

        indexer.build(&pdf).unwrap();
        assert!(indexer.index_is_current(&pdf));
    }
}
