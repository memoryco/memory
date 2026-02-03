//! DSM-5-TR Handbook document profile.
//!
//! Parses the companion study guide with its chapter/section hierarchy.

use super::super::extractor::PageText;
use super::{DocumentProfile, Section, SectionType};
use regex::Regex;

/// First page where actual content begins (skips TOC/front matter).
/// TOC entries use the same section numbering format as real content,
/// so we filter them out by page number to avoid duplicate index entries.
const MIN_CONTENT_PAGE: usize = 11;

/// Profile for parsing DSM-5-TR Handbook documents.
pub struct Dsm5HandbookProfile {
    // Regex patterns compiled once
    part_pattern: Regex,
    chapter_pattern: Regex,
    section_pattern: Regex,
    subsection_pattern: Regex,
}

impl Dsm5HandbookProfile {
    pub fn new() -> Self {
        Self {
            // Match Part headers: "Part 1: Title" or "PART 1: TITLE"
            part_pattern: Regex::new(
                r"(?mi)^(?:PART|Part)\s+(\d+)\s*:\s*(.+?)$"
            ).unwrap(),
            
            // Match Chapter headers: "Chapter 4: Title" or "CHAPTER 4: TITLE"
            chapter_pattern: Regex::new(
                r"(?mi)^(?:CHAPTER|Chapter)\s+(\d+)\s*:\s*(.+?)$"
            ).unwrap(),
            
            // Match section headers: "4.1 Title" (number.number at line start)
            section_pattern: Regex::new(
                r"(?m)^(\d+)\.(\d+)\s+([A-Z][^\n]+?)$"
            ).unwrap(),
            
            // Match subsection headers: "4.1.1 Title" (number.number.number at line start)
            subsection_pattern: Regex::new(
                r"(?m)^(\d+)\.(\d+)\.(\d+)\s+([A-Z][^\n]+?)$"
            ).unwrap(),
        }
    }
}

impl Default for Dsm5HandbookProfile {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentProfile for Dsm5HandbookProfile {
    fn id(&self) -> &str {
        "dsm5trhandbook"
    }

    fn name(&self) -> &str {
        "DSM-5-TR Handbook"
    }

    fn matches(&self, filename: &str, _sample_text: &str) -> bool {
        let filename_lower = filename.to_lowercase();
        
        // Must have "handbook" to distinguish from main DSM-5-TR
        filename_lower.contains("handbook") && 
        (filename_lower.contains("dsm") || filename_lower.contains("diagnostic"))
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

        // Find Parts
        for cap in self.part_pattern.captures_iter(&full_text) {
            let pos = cap.get(0).unwrap().start();
            let num = cap.get(1).unwrap().as_str();
            let title = cap.get(2).unwrap().as_str().trim();
            elements.push((pos, ElementType::Part, format!("Part {}: {}", num, title)));
        }

        // Find Chapters
        for cap in self.chapter_pattern.captures_iter(&full_text) {
            let pos = cap.get(0).unwrap().start();
            let num = cap.get(1).unwrap().as_str();
            let title = cap.get(2).unwrap().as_str().trim();
            elements.push((pos, ElementType::Chapter(num.to_string()), format!("Chapter {}: {}", num, title)));
        }

        // Find Sections (e.g., "4.1 Autism Spectrum Disorder")
        for cap in self.section_pattern.captures_iter(&full_text) {
            let pos = cap.get(0).unwrap().start();
            let chapter_num = cap.get(1).unwrap().as_str();
            let section_num = cap.get(2).unwrap().as_str();
            let title = cap.get(3).unwrap().as_str().trim();
            elements.push((
                pos, 
                ElementType::Section(chapter_num.to_string(), section_num.to_string()), 
                format!("{}.{} {}", chapter_num, section_num, title)
            ));
        }

        // Find Subsections (e.g., "4.1.1 Diagnostic Criteria")
        for cap in self.subsection_pattern.captures_iter(&full_text) {
            let pos = cap.get(0).unwrap().start();
            let chapter_num = cap.get(1).unwrap().as_str();
            let section_num = cap.get(2).unwrap().as_str();
            let subsection_num = cap.get(3).unwrap().as_str();
            let title = cap.get(4).unwrap().as_str().trim();
            elements.push((
                pos,
                ElementType::Subsection(chapter_num.to_string(), section_num.to_string(), subsection_num.to_string()),
                format!("{}.{}.{} {}", chapter_num, section_num, subsection_num, title)
            ));
        }

        // Sort by position
        elements.sort_by_key(|(pos, _, _)| *pos);

        // Track current parent context
        let mut current_part: Option<String> = None;
        let mut current_chapter: Option<String> = None;
        let mut current_section: Option<String> = None;

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

            let (parent, section_type) = match elem_type {
                ElementType::Part => {
                    current_part = Some(title.clone());
                    current_chapter = None;
                    current_section = None;
                    (None, SectionType::Other)
                }
                ElementType::Chapter(_) => {
                    current_chapter = Some(title.clone());
                    current_section = None;
                    (current_part.clone(), SectionType::Other)
                }
                ElementType::Section(_, _) => {
                    current_section = Some(title.clone());
                    (current_chapter.clone(), SectionType::Other)
                }
                ElementType::Subsection(_, _, _) => {
                    (current_section.clone(), SectionType::Other)
                }
            };

            sections.push(Section {
                title: title.clone(),
                parent,
                page_start,
                page_end,
                content: clean_page_markers(content),
                codes: Vec::new(), // Handbook doesn't have ICD codes
                section_type,
            });
        }

        // Filter out TOC/front matter entries - they have the same format as
        // real content but appear on early pages with no actual content
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

/// Type of structural element in the handbook.
#[allow(dead_code)] // Variant data used for structural context
#[derive(Debug, Clone)]
enum ElementType {
    Part,
    Chapter(String),                    // chapter_num
    Section(String, String),            // chapter_num, section_num
    Subsection(String, String, String), // chapter_num, section_num, subsection_num
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
