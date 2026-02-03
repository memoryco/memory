//! ICD-11 MMS (Mortality and Morbidity Statistics) document profile.
//!
//! Parses the WHO International Classification of Diseases, 11th Revision
//! with its chapter/block/category hierarchy and alphanumeric codes.

use super::super::extractor::PageText;
use super::{DocumentProfile, Section, SectionType};
use regex::Regex;

/// First page where actual clinical content begins (skips TOC/front matter).
/// The PDF has ~43 pages of front matter before Chapter 01 starts.
const MIN_CONTENT_PAGE: usize = 44;

/// Profile for parsing ICD-11 MMS documents.
pub struct Icd11Profile {
    // Regex patterns compiled once
    chapter_pattern: Regex,
    block_pattern: Regex,
    category_pattern: Regex,
    subcategory_pattern: Regex,
}

impl Icd11Profile {
    pub fn new() -> Self {
        Self {
            // Match Chapter headers: "CHAPTER 01" or "CHAPTER 06" etc
            // Title appears on the following line
            chapter_pattern: Regex::new(
                r"(?m)^CHAPTER\s+(\d{2})\s*\n+([A-Z][^\n]+?)$"
            ).unwrap(),
            
            // Match Block headers: "Block name (CODE-CODE)" with code range
            // e.g., "Gastroenteritis or colitis of infectious origin (1A00-1A40.Z)"
            block_pattern: Regex::new(
                r"(?m)^([A-Z][A-Za-z0-9\s,\-']+?)\s+\(([A-Z0-9]{2,5})-([A-Z0-9]{2,5}(?:\.[A-Z0-9])?)\)\s*$"
            ).unwrap(),
            
            // Match Category entries: 4-char code + title on same line
            // e.g., "1A00 Cholera" or "6A70 Depressive disorders"
            // Code format: digit + letter + 2 alphanumeric (e.g., 1A00, 6A70)
            // Note: whitespace is normalized to single spaces by extractor
            category_pattern: Regex::new(
                r"(?m)^([0-9][A-Z][A-Z0-9]{2})\s+([A-Z][^\n]+?)$"
            ).unwrap(),
            
            // Match Subcategory entries: code with decimal + title
            // e.g., "1A03.0 Enteropathogenic Escherichia coli infection"
            subcategory_pattern: Regex::new(
                r"(?m)^([0-9][A-Z][A-Z0-9]{2}\.[A-Z0-9]+)\s+([A-Z][^\n]+?)$"
            ).unwrap(),
        }
    }
}

impl Default for Icd11Profile {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentProfile for Icd11Profile {
    fn id(&self) -> &str {
        "icd11"
    }

    fn name(&self) -> &str {
        "ICD-11 MMS"
    }

    fn matches(&self, filename: &str, _sample_text: &str) -> bool {
        let filename_lower = filename.to_lowercase();
        
        // Match ICD-11 MMS documents
        (filename_lower.contains("icd") && filename_lower.contains("11")) ||
        filename_lower.contains("icd-11") ||
        filename_lower.contains("icd11")
    }

    fn parse_sections(&self, pages: &[PageText]) -> Option<Vec<Section>> {
        // Combine all pages into one text for parsing
        let full_text: String = pages
            .iter()
            .map(|p| format!("<<<PAGE {}>>>\n{}", p.page_number, p.text))
            .collect::<Vec<_>>()
            .join("\n");

        let mut sections = Vec::new();

        // Collect all structural elements with their positions
        let mut elements: Vec<(usize, ElementType, String, Vec<String>)> = Vec::new();

        // Find Chapters
        for cap in self.chapter_pattern.captures_iter(&full_text) {
            let pos = cap.get(0).unwrap().start();
            let num = cap.get(1).unwrap().as_str();
            let title = cap.get(2).unwrap().as_str().trim();
            elements.push((
                pos, 
                ElementType::Chapter(num.to_string()), 
                format!("Chapter {}: {}", num, title),
                Vec::new()
            ));
        }

        // Find Blocks (code ranges)
        for cap in self.block_pattern.captures_iter(&full_text) {
            let pos = cap.get(0).unwrap().start();
            let title = cap.get(1).unwrap().as_str().trim();
            let code_start = cap.get(2).unwrap().as_str();
            let code_end = cap.get(3).unwrap().as_str();
            elements.push((
                pos,
                ElementType::Block,
                format!("{} ({}-{})", title, code_start, code_end),
                vec![code_start.to_string(), code_end.to_string()]
            ));
        }

        // Find Categories (main diagnostic codes)
        for cap in self.category_pattern.captures_iter(&full_text) {
            let pos = cap.get(0).unwrap().start();
            let code = cap.get(1).unwrap().as_str();
            let title = cap.get(2).unwrap().as_str().trim();
            elements.push((
                pos,
                ElementType::Category,
                format!("{} {}", code, title),
                vec![code.to_string()]
            ));
        }

        // Find Subcategories (specific codes with decimals)
        for cap in self.subcategory_pattern.captures_iter(&full_text) {
            let pos = cap.get(0).unwrap().start();
            let code = cap.get(1).unwrap().as_str();
            let title = cap.get(2).unwrap().as_str().trim();
            elements.push((
                pos,
                ElementType::Subcategory,
                format!("{} {}", code, title),
                vec![code.to_string()]
            ));
        }

        // Sort by position
        elements.sort_by_key(|(pos, _, _, _)| *pos);

        // Track current parent context
        let mut current_chapter: Option<String> = None;
        let mut current_block: Option<String> = None;
        let mut current_category: Option<String> = None;

        // Build sections with content
        for (i, (pos, elem_type, title, codes)) in elements.iter().enumerate() {
            // Calculate end position (next element or end of text)
            let end_pos = elements
                .get(i + 1)
                .map(|(p, _, _, _)| *p)
                .unwrap_or(full_text.len());

            let content = &full_text[*pos..end_pos];
            let page_start = extract_page_number(&full_text[..*pos]);
            let page_end = extract_page_number(&full_text[..end_pos]);

            let (parent, section_type) = match elem_type {
                ElementType::Chapter(_) => {
                    current_chapter = Some(title.clone());
                    current_block = None;
                    current_category = None;
                    (None, SectionType::Other)
                }
                ElementType::Block => {
                    current_block = Some(title.clone());
                    current_category = None;
                    (current_chapter.clone(), SectionType::Other)
                }
                ElementType::Category => {
                    current_category = Some(title.clone());
                    // Categories are the main disorder entries
                    (current_block.clone().or(current_chapter.clone()), SectionType::Disorder)
                }
                ElementType::Subcategory => {
                    // Subcategories are specific variants
                    (current_category.clone().or(current_block.clone()), SectionType::Disorder)
                }
            };

            sections.push(Section {
                title: title.clone(),
                parent,
                page_start,
                page_end,
                content: clean_page_markers(content),
                codes: codes.clone(),
                section_type,
            });
        }

        // Filter out TOC/front matter entries
        let sections: Vec<Section> = sections
            .into_iter()
            .filter(|s| s.page_start >= MIN_CONTENT_PAGE)
            .collect();

        if sections.is_empty() {
            None // Fall back to page-level indexing
        } else {
            Some(sections)
        }
    }
}

/// Type of structural element in ICD-11.
#[allow(dead_code)] // Variant data used for structural context
#[derive(Debug, Clone)]
enum ElementType {
    Chapter(String),  // chapter number
    Block,            // code range block
    Category,         // 4-char diagnostic code
    Subcategory,      // code with decimal
}

/// Extract the most recent page number from text with our markers.
fn extract_page_number(text: &str) -> usize {
    let pattern = Regex::new(r"<<<PAGE (\d+)>>>").unwrap();
    pattern
        .captures_iter(text)
        .last()
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse().ok())
        .unwrap_or(1)
}

/// Remove our page markers from text.
fn clean_page_markers(text: &str) -> String {
    let pattern = Regex::new(r"<<<PAGE \d+>>>\n?").unwrap();
    pattern.replace_all(text, "").to_string()
}
