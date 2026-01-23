//! pdf-extract based PDF text extraction.

use super::super::error::{ReferenceError, Result};
use super::{PageText, PdfExtractor};
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
        let raw_text = pdf_extract::extract_text_from_mem(&bytes)
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
