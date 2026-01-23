//! Document profiles for structured parsing.
//!
//! Profiles allow curated documents to be parsed with richer section-level
//! granularity instead of just page-level indexing.

mod dsm5;
mod dsm5_handbook;
mod icd11;
mod nasw_ethics;

pub use dsm5::Dsm5Profile;
pub use dsm5_handbook::Dsm5HandbookProfile;
pub use icd11::Icd11Profile;
pub use nasw_ethics::NaswEthicsProfile;

use super::extractor::PageText;

/// A parsed section from a document.
#[derive(Debug, Clone)]
pub struct Section {
    /// Section title/name (e.g., "Major Depressive Disorder")
    pub title: String,
    /// Parent section if nested (e.g., "Depressive Disorders")
    pub parent: Option<String>,
    /// Starting page number
    pub page_start: usize,
    /// Ending page number
    pub page_end: usize,
    /// Full text content of this section
    pub content: String,
    /// Optional ICD code(s) associated with this section
    pub codes: Vec<String>,
    /// Section type for filtering
    pub section_type: SectionType,
}

/// Type of section for filtering searches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SectionType {
    /// Full disorder entry
    Disorder,
    /// Diagnostic criteria specifically
    DiagnosticCriteria,
    /// Diagnostic features prose
    DiagnosticFeatures,
    /// Differential diagnosis
    DifferentialDiagnosis,
    /// Development and course
    DevelopmentAndCourse,
    /// Associated features
    AssociatedFeatures,
    /// Comorbidity
    Comorbidity,
    /// Other/generic section
    Other,
}

/// Trait for document-specific parsing profiles.
pub trait DocumentProfile: Send + Sync {
    /// Unique identifier for this profile (e.g., "dsm5tr")
    fn id(&self) -> &str;

    /// Human-readable name
    fn name(&self) -> &str;

    /// Check if this profile matches a document.
    /// Called with filename and sample text from first few pages.
    fn matches(&self, filename: &str, sample_text: &str) -> bool;

    /// Parse pages into structured sections.
    /// Returns None to fall back to page-level indexing.
    fn parse_sections(&self, pages: &[PageText]) -> Option<Vec<Section>>;
}

/// Registry of available document profiles.
pub struct ProfileRegistry {
    profiles: Vec<Box<dyn DocumentProfile>>,
}

impl ProfileRegistry {
    /// Create a new registry with default profiles.
    pub fn new() -> Self {
        Self {
            profiles: vec![
                Box::new(Dsm5HandbookProfile::new()),
                Box::new(Dsm5Profile::new()),
                Box::new(Icd11Profile::new()),
                Box::new(NaswEthicsProfile::new()),
            ],
        }
    }

    /// Find a matching profile for a document.
    pub fn find_profile(&self, filename: &str, sample_text: &str) -> Option<&dyn DocumentProfile> {
        self.profiles
            .iter()
            .find(|p| p.matches(filename, sample_text))
            .map(|p| p.as_ref())
    }

    /// Register a custom profile.
    pub fn register(&mut self, profile: Box<dyn DocumentProfile>) {
        self.profiles.push(profile);
    }
}

impl Default for ProfileRegistry {
    fn default() -> Self {
        Self::new()
    }
}
