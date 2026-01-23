//! Lenses bootstrap - create directory and add instructions to identity

use crate::engram::Brain;
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

**How to use:**
1. `prompts/list` - see available lenses
2. `prompts/get` with `name: "lens-name"` - load into context
3. Apply the lens guidance throughout the task

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

1. Create a `.md` file in `~/.lenses/`
2. The filename becomes the lens name (e.g., `humanizer.md` → `humanizer`)
3. If the first line starts with `#`, it becomes the description
4. The rest of the file is the lens content

## Using Lenses

1. `prompts/list` - see available lenses
2. `prompts/get` with `name: "lens-name"` - load a lens into context
3. Apply the lens guidance to your task

Delete this file once you've created your own lenses!
"#;

/// Marker to detect if lenses instructions already exist
const MARKER: &str = "## Lenses";

/// Bootstrap lenses: add instructions to identity and create directory
pub fn bootstrap(brain: &mut Brain, lenses_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // Add instructions to identity if not present
    let already_present = brain.identity().instructions.iter()
        .any(|i| i.contains(MARKER));
    
    if !already_present {
        eprintln!("  Bootstrapping lenses instructions...");
        let mut identity = brain.identity().clone();
        identity = identity.with_instruction(INSTRUCTIONS);
        brain.set_identity(identity)?;
        eprintln!("  Lenses instructions added to identity");
    }
    
    // Create directory if it doesn't exist
    if !lenses_dir.exists() {
        eprintln!("  Creating lenses directory: {}", lenses_dir.display());
        std::fs::create_dir_all(lenses_dir)?;
    }

    // Check if directory is empty (no .md files)
    let is_empty = std::fs::read_dir(lenses_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|s| s.to_str())
                == Some("md")
        })
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
