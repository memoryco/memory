//! Reference bootstrap - add per-source citation instructions to identity

use crate::engram::Brain;
use super::ReferenceManager;

/// Bootstrap reference sources: add citation instructions to identity for each source
pub fn bootstrap(brain: &mut Brain, references: &ReferenceManager) -> Result<(), Box<dyn std::error::Error>> {
    for source_name in references.sources() {
        // Check if this source already has citation instructions
        let marker = format!("reference:{}", source_name);
        let already_present = brain.identity().instructions.iter()
            .any(|i| i.contains(&marker));
        
        if already_present {
            continue;
        }
        
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
            eprintln!("  Bootstrapping citation instructions for '{}'", source_name);
            
            let instruction = format!(
                "<!-- reference:{} -->\n\
                When presenting information from {}, always include the APA 7 citation.\n\n\
                Full reference: {}",
                source_name,
                citation.title,
                citation.format_reference()
            );
            
            let mut identity = brain.identity().clone();
            identity = identity.with_instruction(&instruction);
            brain.set_identity(identity)?;
            
            eprintln!("  Citation instructions for '{}' added to identity", source_name);
        }
    }
    
    Ok(())
}
