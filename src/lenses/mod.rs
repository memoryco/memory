//! Lenses - Task-specific context for AI assistants
//!
//! Lenses are markdown files loaded whole into context for specific tasks.
//! Unlike searchable references, lenses are meant to be held in working
//! memory during a task (e.g., a style guide while editing).
//!
//! # Directory Structure
//!
//! ```text
//! ~/.lenses/
//! ├── humanizer.md      # Writing style guide
//! ├── code-review.md    # Code review checklist
//! └── technical-docs.md # Documentation standards
//! ```
//!
//! Each `.md` file becomes an MCP prompt. The filename (minus extension)
//! is the prompt name. If the first line starts with `#`, it becomes
//! the description.

mod bootstrap;

pub use bootstrap::{bootstrap_if_needed, IDENTITY_INSTRUCTIONS};

use sml_mcps::server::PromptDef;
use sml_mcps::types::{Content, PromptArgument, PromptMessage, Result, Role};
use std::collections::HashMap;
use std::path::PathBuf;

/// A lens loaded from a markdown file.
pub struct Lens {
    name: String,
    description: Option<String>,
    content: String,
}

impl Lens {
    /// Load a lens from a markdown file.
    pub fn from_file(path: &PathBuf) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Extract description from first line if it's a heading
        let (description, content) = if content.starts_with('#') {
            if let Some(first_newline) = content.find('\n') {
                let first_line = &content[..first_newline];
                let desc = first_line.trim_start_matches('#').trim().to_string();
                let rest = content[first_newline..].trim_start().to_string();
                (Some(desc), rest)
            } else {
                (None, content)
            }
        } else {
            (None, content)
        };

        Ok(Self {
            name,
            description,
            content,
        })
    }
}

impl PromptDef for Lens {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    fn arguments(&self) -> Vec<PromptArgument> {
        // Lenses don't take arguments - they're loaded as-is
        vec![]
    }

    fn get_messages(&self, _args: &HashMap<String, String>) -> Result<Vec<PromptMessage>> {
        Ok(vec![PromptMessage {
            role: Role::User,
            content: Content::text(&self.content),
        }])
    }
}

/// Get the default lenses directory (~/.lenses)
pub fn get_default_lenses_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".lenses")
}

/// Load all lenses from a directory
pub fn load_lenses(dir: &PathBuf) -> Vec<Lens> {
    let mut lenses = Vec::new();

    if !dir.exists() {
        return lenses;
    }

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Warning: failed to read lenses directory: {}", e);
            return lenses;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("md") {
            match Lens::from_file(&path) {
                Ok(lens) => {
                    eprintln!("  Loaded lens: {}", lens.name);
                    lenses.push(lens);
                }
                Err(e) => {
                    eprintln!("Warning: failed to load {}: {}", path.display(), e);
                }
            }
        }
    }

    lenses
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_lens(dir: &std::path::Path, name: &str, content: &str) {
        let path = dir.join(format!("{}.md", name));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
    }

    #[test]
    fn test_lens_from_file_with_heading() {
        let dir = TempDir::new().unwrap();
        create_test_lens(dir.path(), "test", "# My Test Lens\n\nThis is the content.");
        
        let lens = Lens::from_file(&dir.path().join("test.md")).unwrap();
        
        assert_eq!(lens.name, "test");
        assert_eq!(lens.description, Some("My Test Lens".to_string()));
        assert_eq!(lens.content, "This is the content.");
    }

    #[test]
    fn test_lens_from_file_without_heading() {
        let dir = TempDir::new().unwrap();
        create_test_lens(dir.path(), "plain", "Just some content without a heading.");
        
        let lens = Lens::from_file(&dir.path().join("plain.md")).unwrap();
        
        assert_eq!(lens.name, "plain");
        assert_eq!(lens.description, None);
        assert_eq!(lens.content, "Just some content without a heading.");
    }

    #[test]
    fn test_lens_from_file_multiline_heading() {
        let dir = TempDir::new().unwrap();
        create_test_lens(dir.path(), "multi", "## Secondary Heading\n\nParagraph one.\n\nParagraph two.");
        
        let lens = Lens::from_file(&dir.path().join("multi.md")).unwrap();
        
        assert_eq!(lens.name, "multi");
        assert_eq!(lens.description, Some("Secondary Heading".to_string()));
        assert!(lens.content.contains("Paragraph one"));
        assert!(lens.content.contains("Paragraph two"));
    }

    #[test]
    fn test_load_lenses_empty_dir() {
        let dir = TempDir::new().unwrap();
        let lenses = load_lenses(&dir.path().to_path_buf());
        assert!(lenses.is_empty());
    }

    #[test]
    fn test_load_lenses_multiple_files() {
        let dir = TempDir::new().unwrap();
        create_test_lens(dir.path(), "one", "# First\n\nContent one.");
        create_test_lens(dir.path(), "two", "# Second\n\nContent two.");
        
        // Also create a non-md file that should be ignored
        std::fs::write(dir.path().join("ignore.txt"), "ignored").unwrap();
        
        let lenses = load_lenses(&dir.path().to_path_buf());
        
        assert_eq!(lenses.len(), 2);
        let names: Vec<&str> = lenses.iter().map(|l| l.name.as_str()).collect();
        assert!(names.contains(&"one"));
        assert!(names.contains(&"two"));
    }

    #[test]
    fn test_load_lenses_nonexistent_dir() {
        let lenses = load_lenses(&PathBuf::from("/nonexistent/path/12345"));
        assert!(lenses.is_empty());
    }

    #[test]
    fn test_lens_prompt_def_name() {
        let lens = Lens {
            name: "test-lens".to_string(),
            description: Some("A test".to_string()),
            content: "Content here".to_string(),
        };
        
        assert_eq!(lens.name(), "test-lens");
    }

    #[test]
    fn test_lens_prompt_def_description() {
        let lens = Lens {
            name: "test".to_string(),
            description: Some("My description".to_string()),
            content: "Content".to_string(),
        };
        
        assert_eq!(lens.description(), Some("My description"));
    }

    #[test]
    fn test_lens_prompt_def_no_arguments() {
        let lens = Lens {
            name: "test".to_string(),
            description: None,
            content: "Content".to_string(),
        };
        
        assert!(lens.arguments().is_empty());
    }

    #[test]
    fn test_lens_prompt_def_get_messages() {
        let lens = Lens {
            name: "test".to_string(),
            description: None,
            content: "The full lens content here.".to_string(),
        };
        
        let messages = lens.get_messages(&HashMap::new()).unwrap();
        
        assert_eq!(messages.len(), 1);
        assert!(matches!(messages[0].role, Role::User));
    }
}
