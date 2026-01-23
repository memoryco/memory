//! DSM-5-TR document profile.

use super::super::extractor::PageText;
use super::{DocumentProfile, Section, SectionType};
use regex::Regex;

/// Profile for parsing DSM-5-TR documents.
pub struct Dsm5Profile {
    // Regex patterns compiled once
    disorder_pattern: Regex,
    icd_code_pattern: Regex,
    section_headers: Vec<(&'static str, SectionType)>,
}

impl Dsm5Profile {
    pub fn new() -> Self {
        Self {
            // Match disorder titles (Title Case followed by newline and "Diagnostic Criteria")
            disorder_pattern: Regex::new(
                r"(?m)^([A-Z][a-zA-Z\-\s,()]+(?:Disorder|Syndrome|Episode|Specifier))\s*$"
            ).unwrap(),
            
            // Match ICD-10 codes (F##.## pattern)
            icd_code_pattern: Regex::new(r"F\d{2}(?:\.\d{1,2})?").unwrap(),
            
            // Standard DSM section headers
            section_headers: vec![
                ("Diagnostic Criteria", SectionType::DiagnosticCriteria),
                ("Diagnostic Features", SectionType::DiagnosticFeatures),
                ("Associated Features", SectionType::AssociatedFeatures),
                ("Development and Course", SectionType::DevelopmentAndCourse),
                ("Differential Diagnosis", SectionType::DifferentialDiagnosis),
                ("Comorbidity", SectionType::Comorbidity),
            ],
        }
    }

    /// Extract ICD codes from text.
    fn extract_codes(&self, text: &str) -> Vec<String> {
        self.icd_code_pattern
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect()
    }

    /// Find section boundaries within disorder text.
    fn find_section_boundaries(&self, text: &str) -> Vec<(usize, &'static str, SectionType)> {
        let mut boundaries: Vec<(usize, &'static str, SectionType)> = Vec::new();

        for (header, section_type) in &self.section_headers {
            if let Some(pos) = text.find(header) {
                boundaries.push((pos, header, section_type.clone()));
            }
        }

        boundaries.sort_by_key(|(pos, _, _)| *pos);
        boundaries
    }
}

impl Default for Dsm5Profile {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentProfile for Dsm5Profile {
    fn id(&self) -> &str {
        "dsm5tr"
    }

    fn name(&self) -> &str {
        "DSM-5-TR"
    }

    fn matches(&self, filename: &str, sample_text: &str) -> bool {
        let filename_lower = filename.to_lowercase();
        
        // Match by filename
        if filename_lower.contains("dsm") && 
           (filename_lower.contains("5") || filename_lower.contains("tr")) {
            return true;
        }

        // Match by content
        sample_text.contains("Diagnostic and Statistical Manual") ||
        sample_text.contains("DSM-5-TR") ||
        sample_text.contains("American Psychiatric Association")
    }

    fn parse_sections(&self, pages: &[PageText]) -> Option<Vec<Section>> {
        // Combine all pages into one text for parsing
        let full_text: String = pages
            .iter()
            .map(|p| format!("<<<PAGE {}>>>\n{}", p.page_number, p.text))
            .collect::<Vec<_>>()
            .join("\n");

        let mut sections = Vec::new();

        // Find all disorder entries
        let disorder_matches: Vec<_> = self.disorder_pattern
            .find_iter(&full_text)
            .collect();

        for (i, disorder_match) in disorder_matches.iter().enumerate() {
            let disorder_name = disorder_match.as_str().trim().to_string();
            let start_pos = disorder_match.start();
            
            // End position is either next disorder or end of text
            let end_pos = disorder_matches
                .get(i + 1)
                .map(|m| m.start())
                .unwrap_or(full_text.len());

            let disorder_text = &full_text[start_pos..end_pos];

            // Find page numbers from our markers
            let page_start = extract_page_number(&full_text[..start_pos]);
            let page_end = extract_page_number(&full_text[..end_pos]);

            // Extract ICD codes
            let codes = self.extract_codes(disorder_text);

            // Create main disorder section
            sections.push(Section {
                title: disorder_name.clone(),
                parent: None,
                page_start,
                page_end,
                content: clean_page_markers(disorder_text),
                codes: codes.clone(),
                section_type: SectionType::Disorder,
            });

            // Parse subsections within this disorder
            let boundaries = self.find_section_boundaries(disorder_text);
            
            for (j, (pos, header, section_type)) in boundaries.iter().enumerate() {
                let section_end = boundaries
                    .get(j + 1)
                    .map(|(p, _, _)| *p)
                    .unwrap_or(disorder_text.len());

                let section_text = &disorder_text[*pos..section_end];
                let section_page_start = extract_page_number(&full_text[..start_pos + pos]);
                let section_page_end = extract_page_number(&full_text[..start_pos + section_end]);

                sections.push(Section {
                    title: format!("{} - {}", disorder_name, header),
                    parent: Some(disorder_name.clone()),
                    page_start: section_page_start,
                    page_end: section_page_end,
                    content: clean_page_markers(section_text),
                    codes: codes.clone(),
                    section_type: section_type.clone(),
                });
            }
        }

        if sections.is_empty() {
            None // Fall back to page-level indexing
        } else {
            Some(sections)
        }
    }
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
