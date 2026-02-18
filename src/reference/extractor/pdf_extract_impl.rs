//! pdf-extract based PDF text extraction.

use super::super::error::{ReferenceError, Result};
use super::{PageText, PdfExtractor};
use std::panic;
use std::path::Path;

/// PDF extractor using the pdf-extract crate (pure Rust).
pub struct PdfExtractBackend {
    // Future: could add config options here
}

impl PdfExtractBackend {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for PdfExtractBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl PdfExtractor for PdfExtractBackend {
    fn extract(&self, path: &Path) -> Result<Vec<PageText>> {
        let bytes = std::fs::read(path)?;

        // pdf-extract returns all text as one string.
        // We'll try to detect page breaks via form feed characters,
        // or fall back to treating it as a single page.
        //
        // Wrap in catch_unwind because pdf-extract panics on certain PDFs
        // (e.g. "unhandled function type 4", "DeviceN" color spaces).
        let raw_text = panic::catch_unwind(|| {
            pdf_extract::extract_text_from_mem(&bytes)
        })
        .map_err(|panic_payload| {
            let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic in pdf-extract".to_string()
            };
            ReferenceError::Extraction(format!("pdf-extract panicked: {}", msg))
        })?
        .map_err(|e| ReferenceError::Extraction(e.to_string()))?;

        // Post-process: collapse multiple spaces (common pdf-extract issue)
        let text = normalize_whitespace(&raw_text);

        // Check for form feed characters (common page separator)
        let pages: Vec<PageText> = if text.contains('\x0C') {
            text.split('\x0C')
                .enumerate()
                .map(|(i, page_text)| PageText {
                    page_number: i + 1,
                    text: page_text.to_string(),
                })
                .filter(|p| !p.text.trim().is_empty())
                .collect()
        } else {
            // No page breaks detected - split by rough page size
            // Average page is ~3000 chars, but this is a fallback
            split_by_size(&text, 3000)
        };

        Ok(pages)
    }
}

/// Normalize whitespace: collapse multiple spaces, fix common pdf-extract issues.
fn normalize_whitespace(text: &str) -> String {
    // Collapse runs of spaces (but preserve newlines for structure)
    let mut result = String::with_capacity(text.len());
    let mut prev_was_space = false;
    
    for ch in text.chars() {
        if ch == ' ' || ch == '\t' {
            if !prev_was_space {
                result.push(' ');
                prev_was_space = true;
            }
        } else {
            result.push(ch);
            prev_was_space = false;
        }
    }
    
    result
}

/// Fallback: split text into chunks of approximately `size` characters,
/// breaking at paragraph boundaries where possible.
fn split_by_size(text: &str, target_size: usize) -> Vec<PageText> {
    let mut pages = Vec::new();
    let mut current_page = String::new();
    let mut page_num = 1;

    for paragraph in text.split("\n\n") {
        if current_page.len() + paragraph.len() > target_size && !current_page.is_empty() {
            pages.push(PageText {
                page_number: page_num,
                text: std::mem::take(&mut current_page),
            });
            page_num += 1;
        }

        if !current_page.is_empty() {
            current_page.push_str("\n\n");
        }
        current_page.push_str(paragraph);
    }

    if !current_page.trim().is_empty() {
        pages.push(PageText {
            page_number: page_num,
            text: current_page,
        });
    }

    pages
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn extract_nonexistent_file_returns_io_error() {
        let backend = PdfExtractBackend::new();
        let result = backend.extract(Path::new("/nonexistent/fake.pdf"));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ReferenceError::Io(_)));
    }

    #[test]
    fn extract_garbage_bytes_returns_extraction_error_not_panic() {
        // Write garbage bytes that look nothing like a PDF.
        // pdf-extract should either error or panic — either way we should get
        // a clean ReferenceError::Extraction, not a process crash.
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(b"this is definitely not a pdf file at all").unwrap();
        tmp.flush().unwrap();

        let backend = PdfExtractBackend::new();
        let result = backend.extract(tmp.path());
        // We don't care whether it's Ok([]) or Err — just that it didn't panic
        // and crash the process. If it errors, it should be Extraction.
        if let Err(e) = result {
            assert!(
                matches!(e, ReferenceError::Extraction(_)),
                "expected Extraction error, got: {:?}",
                e
            );
        }
    }

    #[test]
    fn extract_truncated_pdf_header_returns_error_not_panic() {
        // A file that starts with %PDF- but has no valid structure
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(b"%PDF-1.4\n% truncated garbage").unwrap();
        tmp.flush().unwrap();

        let backend = PdfExtractBackend::new();
        let result = backend.extract(tmp.path());
        if let Err(e) = result {
            assert!(
                matches!(e, ReferenceError::Extraction(_)),
                "expected Extraction error, got: {:?}",
                e
            );
        }
    }

    #[test]
    fn extract_empty_file_returns_error_not_panic() {
        let tmp = NamedTempFile::new().unwrap();

        let backend = PdfExtractBackend::new();
        let result = backend.extract(tmp.path());
        if let Err(e) = result {
            assert!(
                matches!(e, ReferenceError::Extraction(_)),
                "expected Extraction error, got: {:?}",
                e
            );
        }
    }

    #[test]
    fn normalize_whitespace_collapses_spaces() {
        assert_eq!(normalize_whitespace("hello   world"), "hello world");
    }

    #[test]
    fn normalize_whitespace_preserves_newlines() {
        assert_eq!(normalize_whitespace("hello\nworld"), "hello\nworld");
    }

    #[test]
    fn normalize_whitespace_collapses_tabs_and_spaces() {
        assert_eq!(normalize_whitespace("a \t\t b"), "a b");
    }

    #[test]
    fn split_by_size_single_chunk() {
        let pages = split_by_size("short text", 3000);
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].page_number, 1);
        assert_eq!(pages[0].text, "short text");
    }

    #[test]
    fn split_by_size_empty_text() {
        let pages = split_by_size("", 3000);
        assert!(pages.is_empty());
    }

    #[test]
    fn split_by_size_respects_paragraphs() {
        // Two paragraphs, each ~10 chars, with target_size=15
        let text = "paragraph one\n\nparagraph two";
        let pages = split_by_size(text, 15);
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[0].text, "paragraph one");
        assert_eq!(pages[1].text, "paragraph two");
    }
}
