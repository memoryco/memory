//! Reference bootstrap - reference instructions and citation generation

use super::ReferenceManager;

/// General instructions for using reference tools — imported by the instructions tool.
pub const INSTRUCTIONS: &str = r#"## References

References are authoritative documents (PDFs) indexed for full-text search.
Use for clinical guidelines, standards, legal codes - sources requiring citation.

**When to use references:**
- Looking up diagnostic criteria, standards, or guidelines
- Need accurate citations for professional/academic work
- Querying large documents you can't hold in context

**Tools:**
- `reference_list` - see loaded sources
- `reference_search` - FTS5 query with snippets and citations
- `reference_get` - fetch full section by exact title
- `reference_sections` - browse available sections in a source
- `reference_citation` - get APA 7 in-text and full reference formats

**Key distinction:**
- Memories: personal memory, decays without use
- Lenses: hold entire guide in context for a task
- References: query authoritative docs, always cite sources
"#;

/// Generate per-source citation instructions from loaded references.
/// Called by the instructions tool at request time.
pub fn generate_citation_instructions(references: &ReferenceManager) -> String {
    let mut parts: Vec<String> = Vec::new();

    for source_name in references.sources() {
        let meta = match references.get_meta(source_name) {
            Some(m) => m,
            None => continue,
        };

        if let Some(citation) = &meta.citation {
            parts.push(format!(
                "When presenting information from {}, always include the APA 7 citation.\n\
                 Full reference: {}",
                citation.title,
                citation.format_reference()
            ));
        }
    }

    parts.join("\n\n")
}
