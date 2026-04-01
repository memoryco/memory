//! Lenses bootstrap - create directory and expose instructions

use std::path::Path;

/// Lenses usage instructions — imported by the instructions tool.
pub const INSTRUCTIONS: &str = r#"## Lenses

Task-specific context guides loaded whole into working memory.
Use before writing, code review, documentation — any task needing consistent rule application.

1. `lenses_list` — see available lenses
2. `lenses_get` with `name` — load into context
3. Apply guidance throughout the task
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

/// Bootstrap lenses: create directory and sample lens if empty
pub fn bootstrap(lenses_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
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
