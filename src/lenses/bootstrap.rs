//! Lenses bootstrap - create directory and add instructions to identity

use crate::identity::{IdentityStore, UpsertResult};
use std::path::Path;

const INSTRUCTIONS: &str = r#"## Lenses

Lenses are task-specific context guides loaded whole into working memory.
Unlike engrams (searched/recalled) or references (queried), lenses are
meant to be held in context during an entire task.

**When to use lenses:**
- Before writing/editing tasks → check for style guides
- Before code review → check for review checklists  
- Before documentation work → check for doc standards
- Any task where consistent application of rules matters

**How to use (AI-initiated via tools):**
1. `lenses_list` - see available lenses
2. `lenses_get` with `name: "lens-name"` - load into context
3. Apply the lens guidance throughout the task

**How to use (user-initiated via UI):**
1. Click "Add from memory" in Claude Desktop
2. Select the desired lens
3. Lens content is injected into conversation

**Key distinction:**
- Engrams: "What do I know about X?" → search, recall
- References: "What are the DSM criteria for Y?" → query, cite
- Lenses: "Load the style guide" → hold entire guide while working
"#;

const SAMPLE_LENS: &str = r#"# Sample Lens

This is a sample lens to demonstrate the format.

## How Lenses Work

Lenses are markdown files that get loaded whole into context for specific tasks.
Unlike searchable references (where you query for specific information), lenses
are meant to be held in working memory during an entire task.

## Good Use Cases for Lenses

- Style guides (load before writing/editing)
- Code review checklists (load before reviewing PRs)
- Documentation standards (load before writing docs)
- Process checklists (load before specific workflows)

## Creating Your Own Lens

1. Create a `.md` file in `$MEMORY_HOME/lenses`
2. The filename becomes the lens name (e.g., `humanizer.md` → `humanizer`)
3. If the first line starts with `#`, it becomes the description
4. The rest of the file is the lens content

## Using Lenses

**AI-initiated (via tools):**
1. `lenses_list` - see available lenses
2. `lenses_get` with `name: "lens-name"` - load a lens into context
3. Apply the lens guidance to your task

**User-initiated (via UI):**
1. Click "Add from memory" in Claude Desktop
2. Select the desired lens from the submenu
3. Lens content is injected into conversation

Delete this file once you've created your own lenses!
"#;

/// Marker to detect lenses instructions
const MARKER: &str = "## Lenses";

/// Bootstrap lenses: add instructions to identity and create directory
/// Adds if missing, updates if changed, skips if identical
pub fn bootstrap(
    identity: &mut IdentityStore,
    lenses_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Upsert instructions to identity
    match identity.upsert_instruction(INSTRUCTIONS, MARKER)? {
        UpsertResult::Added => {
            eprintln!("  Lenses instructions added to identity");
        }
        UpsertResult::Updated => {
            eprintln!("  Lenses instructions updated in identity");
        }
        UpsertResult::Unchanged => {}
    }

    // Create directory if it doesn't exist
    if !lenses_dir.exists() {
        eprintln!("  Creating lenses directory: {}", lenses_dir.display());
        std::fs::create_dir_all(lenses_dir)?;
    }

    // Check if directory is empty (no .md files)
    let is_empty = std::fs::read_dir(lenses_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("md"))
        .count()
        == 0;

    if is_empty {
        eprintln!("  Creating sample lens...");
        let sample_path = lenses_dir.join("sample.md");
        std::fs::write(&sample_path, SAMPLE_LENS)?;
        eprintln!("  Created sample lens at: {}", sample_path.display());
    }

    Ok(())
}
