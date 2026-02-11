//! Reference bootstrap - add instructions and per-source citations to identity

use crate::identity::{IdentityStore, UpsertResult};
use super::ReferenceManager;

/// General instructions for using reference tools
const INSTRUCTIONS: &str = r#"## References

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
- Engrams: personal memory, decays without use
- Lenses: hold entire guide in context for a task
- References: query authoritative docs, always cite sources
"#;

/// Marker to detect reference instructions
const MARKER: &str = "## References";

/// Bootstrap references: add general instructions and per-source citations to identity
/// Adds if missing, updates if changed, skips if identical
pub fn bootstrap(identity: &mut IdentityStore, references: &ReferenceManager) -> Result<(), Box<dyn std::error::Error>> {
    // Upsert general instructions
    match identity.upsert_instruction(INSTRUCTIONS, MARKER)? {
        UpsertResult::Added => {
            eprintln!("  Reference instructions added to identity");
        }
        UpsertResult::Updated => {
            eprintln!("  Reference instructions updated in identity");
        }
        UpsertResult::Unchanged => {}
    }

    // Upsert per-source citation instructions
    for source_name in references.sources() {
        let marker = format!("reference:{}", source_name);
        
        // Get source metadata
        let meta = match references.get_meta(source_name) {
            Some(m) => m,
            None => {
                eprintln!("  Reference '{}' has no metadata, skipping bootstrap", source_name);
                continue;
            }
        };
        
        // Build citation instruction if citation info is available
        if let Some(citation) = &meta.citation {
            let instruction = format!(
                "<!-- reference:{} -->\n\
                When presenting information from {}, always include the APA 7 citation.\n\n\
                Full reference: {}",
                source_name,
                citation.title,
                citation.format_reference()
            );
            
            match identity.upsert_instruction(&instruction, &marker)? {
                UpsertResult::Added => {
                    eprintln!("  Citation instructions for '{}' added to identity", source_name);
                }
                UpsertResult::Updated => {
                    eprintln!("  Citation instructions for '{}' updated in identity", source_name);
                }
                UpsertResult::Unchanged => {}
            }
        }
    }
    
    Ok(())
}
