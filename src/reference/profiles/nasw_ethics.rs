//! NASW Code of Ethics document profile.
//!
//! Parses the National Association of Social Workers Code of Ethics
//! with its section/standard hierarchy.

use super::super::extractor::PageText;
use super::{DocumentProfile, Section, SectionType};
use regex::Regex;

/// First page where actual content begins (skips cover/TOC).
const MIN_CONTENT_PAGE: usize = 3;

/// Profile for parsing NASW Code of Ethics documents.
pub struct NaswEthicsProfile {
    // Regex patterns compiled once
    major_section_pattern: Regex,
    standard_pattern: Regex,
    preamble_pattern: Regex,
}

impl NaswEthicsProfile {
    pub fn new() -> Self {
        Self {
            // Match major section headers (ALL CAPS):
            // "1. SOCIAL WORKERS' ETHICAL RESPONSIBILITIES TO CLIENTS"
            major_section_pattern: Regex::new(
                r"(?m)^(\d)\.\s+(SOCIAL WORKERS['']?\s+ETHICAL RESPONSIBILITIES[^\n]+)$"
            ).unwrap(),
            
            // Match individual standards: "1.01 Commitment to Clients"
            // Format: N.NN Title (where N is 1-6, NN is 01-99)
            standard_pattern: Regex::new(
                r"(?m)^(\d)\.(\d{2})\s+([A-Z][^\n]+?)$"
            ).unwrap(),
            
            // Match top-level sections: Preamble, Purpose, Ethical Principles, etc.
            preamble_pattern: Regex::new(
                r"(?m)^(Preamble|Purpose Of The NASW Code Of Ethics|Ethical Principles|Ethical Standards|Overview)\s*$"
            ).unwrap(),
        }
    }
}

impl Default for NaswEthicsProfile {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentProfile for NaswEthicsProfile {
    fn id(&self) -> &str {
        "nasw_ethics"
    }

    fn name(&self) -> &str {
        "NASW Code of Ethics"
    }

    fn matches(&self, filename: &str, _sample_text: &str) -> bool {
        let filename_lower = filename.to_lowercase();
        
        // Match NASW ethics documents
        (filename_lower.contains("nasw") && filename_lower.contains("ethics")) ||
        (filename_lower.contains("code") && filename_lower.contains("ethics") && 
         filename_lower.contains("social"))
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
        let mut elements: Vec<(usize, ElementType, String)> = Vec::new();

        // Find top-level sections (Preamble, Purpose, etc.)
        for cap in self.preamble_pattern.captures_iter(&full_text) {
            let pos = cap.get(0).unwrap().start();
            let title = cap.get(1).unwrap().as_str().trim();
            elements.push((pos, ElementType::TopLevel, title.to_string()));
        }

        // Find major sections (1. SOCIAL WORKERS' ETHICAL RESPONSIBILITIES...)
        for cap in self.major_section_pattern.captures_iter(&full_text) {
            let pos = cap.get(0).unwrap().start();
            let num = cap.get(1).unwrap().as_str();
            let title = cap.get(2).unwrap().as_str().trim();
            // Convert to title case for readability
            let title_cased = title_case(title);
            elements.push((
                pos,
                ElementType::MajorSection(num.to_string()),
                format!("{}. {}", num, title_cased)
            ));
        }

        // Find individual standards (1.01 Commitment to Clients)
        for cap in self.standard_pattern.captures_iter(&full_text) {
            let pos = cap.get(0).unwrap().start();
            let section_num = cap.get(1).unwrap().as_str();
            let standard_num = cap.get(2).unwrap().as_str();
            let title = cap.get(3).unwrap().as_str().trim();
            elements.push((
                pos,
                ElementType::Standard(section_num.to_string()),
                format!("{}.{} {}", section_num, standard_num, title)
            ));
        }

        // Sort by position
        elements.sort_by_key(|(pos, _, _)| *pos);

        // Track current parent context
        let mut current_major_section: Option<String> = None;

        // Build sections with content
        for (i, (pos, elem_type, title)) in elements.iter().enumerate() {
            // Calculate end position (next element or end of text)
            let end_pos = elements
                .get(i + 1)
                .map(|(p, _, _)| *p)
                .unwrap_or(full_text.len());

            let content = &full_text[*pos..end_pos];
            let page_start = extract_page_number(&full_text[..*pos]);
            let page_end = extract_page_number(&full_text[..end_pos]);

            let parent = match elem_type {
                ElementType::TopLevel => {
                    current_major_section = None;
                    None
                }
                ElementType::MajorSection(_) => {
                    current_major_section = Some(title.clone());
                    None
                }
                ElementType::Standard(_) => {
                    current_major_section.clone()
                }
            };

            sections.push(Section {
                title: title.clone(),
                parent,
                page_start,
                page_end,
                content: clean_page_markers(content),
                codes: Vec::new(),
                section_type: SectionType::Other,
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

/// Type of structural element in the Code of Ethics.
#[allow(dead_code)] // Variant data used for structural context
#[derive(Debug, Clone)]
enum ElementType {
    TopLevel,                // Preamble, Purpose, etc.
    MajorSection(String),    // 1. Social Workers' Ethical Responsibilities...
    Standard(String),        // 1.01 Commitment to Clients
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

/// Convert ALL CAPS text to Title Case.
fn title_case(s: &str) -> String {
    s.split_whitespace()
        .map(|word| {
            let lower = word.to_lowercase();
            // Keep small words lowercase (except first word handled by join logic)
            if ["to", "the", "in", "as", "of", "and", "or"].contains(&lower.as_str()) {
                lower
            } else {
                let mut chars = lower.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().chain(chars).collect(),
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
