//! Citation metadata for reference sources.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Source metadata loaded from .meta.json sidecar file.
/// Contains citation info plus optional bootstrap memories.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceMeta {
    /// Citation metadata for APA 7 format
    #[serde(default)]
    pub citation: Option<Citation>,
    /// Bootstrap memories to create when this source is first loaded
    #[serde(default)]
    pub memories: Vec<String>,
    /// Related sources to include in searches (e.g., companion guides)
    #[serde(default)]
    pub related_sources: Vec<String>,
}

impl SourceMeta {
    /// Load source metadata from a .meta.json sidecar file.
    pub fn load(meta_path: &Path) -> Option<Self> {
        let content = std::fs::read_to_string(meta_path).ok()?;
        serde_json::from_str(&content).ok()
    }
}

/// Citation metadata for APA 7 format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// Author or organization
    pub author: String,
    /// Publication year
    pub year: u32,
    /// Title of the work
    pub title: String,
    /// Edition (e.g., "5th ed., text rev.")
    #[serde(default)]
    pub edition: Option<String>,
    /// Publisher (if no DOI)
    #[serde(default)]
    pub publisher: Option<String>,
    /// DOI (preferred for APA 7)
    #[serde(default)]
    pub doi: Option<String>,
    /// URL (if no DOI)
    #[serde(default)]
    pub url: Option<String>,
}

impl Citation {

    /// Format as APA 7 full reference.
    pub fn format_reference(&self) -> String {
        let mut parts = vec![
            format!("{}. ({}).", self.author, self.year),
        ];

        // Title in italics (represented with asterisks for plain text)
        parts.push(format!("*{}*", self.title));

        // Edition if present
        if let Some(ref ed) = self.edition {
            parts.push(format!("({}).", ed));
        }

        // Publisher or DOI/URL
        if let Some(ref doi) = self.doi {
            parts.push(format!("https://doi.org/{}", doi));
        } else if let Some(ref url) = self.url {
            parts.push(url.clone());
        } else if let Some(ref pub_) = self.publisher {
            parts.push(format!("{}.", pub_));
        }

        parts.join(" ")
    }

    /// Format as APA 7 in-text citation with page number(s).
    pub fn format_inline(&self, page_start: usize, page_end: usize) -> String {
        if page_start == page_end {
            format!("({}, {}, p. {})", self.author, self.year, page_start)
        } else {
            format!("({}, {}, pp. {}-{})", self.author, self.year, page_start, page_end)
        }
    }

    /// Format as APA 7 in-text citation without page numbers.
    pub fn format_inline_short(&self) -> String {
        format!("({}, {})", self.author, self.year)
    }
}



/// Get the meta.json path for a PDF.
pub fn meta_path_for(pdf_path: &Path) -> std::path::PathBuf {
    let stem = pdf_path.file_stem().unwrap_or_default();
    let parent = pdf_path.parent().unwrap_or(Path::new("."));
    parent.join(format!("{}.meta.json", stem.to_string_lossy()))
}
