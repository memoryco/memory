//! CLAUDE.md management for Claude Code.
//!
//! Appends/removes a MemoryCo instruction block in ~/.claude/CLAUDE.md
//! so Claude Code automatically calls identity_get on every new conversation.
//!
//! Uses HTML comment markers for clean idempotent install/uninstall.

use std::path::PathBuf;

const MARKER_START: &str = "<!-- memoryco:start -->";
const MARKER_END: &str = "<!-- memoryco:end -->";

const MEMORYCO_BLOCK: &str = "\
<!-- memoryco:start -->
## MemoryCo
You have access to a cognitive memory system via MCP tools (`memoryco:*`).

As your first action in every conversation, call `identity_get` from your memoryco MCP tools.
This contains your persona, values, preferences, and operational instructions. Follow what you find.
<!-- memoryco:end -->";

/// Get the path to the global CLAUDE.md file.
pub fn claude_md_path() -> PathBuf {
    dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".claude")
        .join("CLAUDE.md")
}

/// Check if the MemoryCo block is present in CLAUDE.md.
pub fn is_installed() -> bool {
    let path = claude_md_path();
    if !path.exists() {
        return false;
    }
    match std::fs::read_to_string(&path) {
        Ok(content) => content.contains(MARKER_START),
        Err(_) => false,
    }
}

/// Install the MemoryCo block into CLAUDE.md.
///
/// - If the file doesn't exist, creates it with just our block.
/// - If it exists but has no marker, appends our block.
/// - If it exists with our marker, replaces the existing block (idempotent update).
pub fn install() -> Result<(), std::io::Error> {
    let path = claude_md_path();

    // Ensure ~/.claude/ directory exists
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    if !path.exists() {
        // Fresh file — just our block
        std::fs::write(&path, MEMORYCO_BLOCK)?;
        return Ok(());
    }

    let content = std::fs::read_to_string(&path)?;

    if content.contains(MARKER_START) {
        // Replace existing block
        let new_content = replace_between_markers(&content);
        std::fs::write(&path, new_content)?;
    } else {
        // Append with a blank line separator
        let mut new_content = content.clone();
        if !new_content.ends_with('\n') {
            new_content.push('\n');
        }
        new_content.push('\n');
        new_content.push_str(MEMORYCO_BLOCK);
        new_content.push('\n');
        std::fs::write(&path, new_content)?;
    }

    Ok(())
}

/// Remove the MemoryCo block from CLAUDE.md.
///
/// Strips everything between markers (inclusive) and cleans up extra blank lines.
pub fn uninstall() -> Result<(), std::io::Error> {
    let path = claude_md_path();

    if !path.exists() {
        return Ok(());
    }

    let content = std::fs::read_to_string(&path)?;

    if !content.contains(MARKER_START) {
        return Ok(());
    }

    let new_content = remove_between_markers(&content);
    let trimmed = new_content.trim();

    if trimmed.is_empty() {
        // File would be empty — remove it
        std::fs::remove_file(&path)?;
    } else {
        std::fs::write(&path, format!("{}\n", trimmed))?;
    }

    Ok(())
}

/// Replace content between markers with the current block.
fn replace_between_markers(content: &str) -> String {
    let Some(start) = content.find(MARKER_START) else {
        return content.to_string();
    };
    let Some(end) = content.find(MARKER_END) else {
        return content.to_string();
    };

    let Some(before) = content.get(..start) else {
        return content.to_string();
    };
    let after_start = end + MARKER_END.len();
    let Some(after) = content.get(after_start..) else {
        return content.to_string();
    };

    format!(
        "{}{}{}{}",
        before.trim_end_matches('\n'),
        if before.is_empty() { "" } else { "\n\n" },
        MEMORYCO_BLOCK,
        after
    )
}

/// Remove content between markers (inclusive).
fn remove_between_markers(content: &str) -> String {
    let Some(start) = content.find(MARKER_START) else {
        return content.to_string();
    };
    let Some(end) = content.find(MARKER_END) else {
        return content.to_string();
    };

    let Some(before_raw) = content.get(..start) else {
        return content.to_string();
    };
    let after_start = end + MARKER_END.len();
    let Some(after_raw) = content.get(after_start..) else {
        return content.to_string();
    };

    let before = before_raw.trim_end();
    let after = after_raw.trim_start();

    if before.is_empty() {
        after.to_string()
    } else if after.is_empty() {
        before.to_string()
    } else {
        format!("{}\n\n{}", before, after)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup_temp_claude_md(dir: &TempDir, content: &str) -> PathBuf {
        let claude_dir = dir.path().join(".claude");
        std::fs::create_dir_all(&claude_dir).unwrap();
        let path = claude_dir.join("CLAUDE.md");
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn memoryco_block_has_markers() {
        assert!(MEMORYCO_BLOCK.starts_with(MARKER_START));
        assert!(MEMORYCO_BLOCK.ends_with(MARKER_END));
    }

    #[test]
    fn replace_between_markers_updates_block() {
        let content = "# My Config\n\n<!-- memoryco:start -->\nold stuff\n<!-- memoryco:end -->\n\n# Other stuff\n";
        let result = replace_between_markers(content);

        assert!(result.contains(MEMORYCO_BLOCK));
        assert!(!result.contains("old stuff"));
        assert!(result.contains("# My Config"));
        assert!(result.contains("# Other stuff"));
    }

    #[test]
    fn replace_preserves_surrounding_content() {
        let content = "before\n\n<!-- memoryco:start -->\nold\n<!-- memoryco:end -->\n\nafter\n";
        let result = replace_between_markers(content);

        assert!(result.contains("before"));
        assert!(result.contains("after"));
        assert!(result.contains(MEMORYCO_BLOCK));
    }

    #[test]
    fn remove_between_markers_strips_block() {
        let content =
            "# Config\n\n<!-- memoryco:start -->\nstuff\n<!-- memoryco:end -->\n\n# Other\n";
        let result = remove_between_markers(content);

        assert!(!result.contains(MARKER_START));
        assert!(!result.contains(MARKER_END));
        assert!(result.contains("# Config"));
        assert!(result.contains("# Other"));
    }

    #[test]
    fn remove_when_block_is_only_content() {
        let content = format!("{}\n", MEMORYCO_BLOCK);
        let result = remove_between_markers(&content);
        assert!(result.trim().is_empty());
    }

    #[test]
    fn remove_block_at_end() {
        let content = format!("# My stuff\n\n{}\n", MEMORYCO_BLOCK);
        let result = remove_between_markers(&content);

        assert!(result.contains("# My stuff"));
        assert!(!result.contains(MARKER_START));
    }

    #[test]
    fn remove_block_at_start() {
        let content = format!("{}\n\n# My stuff\n", MEMORYCO_BLOCK);
        let result = remove_between_markers(&content);

        assert!(result.contains("# My stuff"));
        assert!(!result.contains(MARKER_START));
    }

    #[test]
    fn append_to_existing_content() {
        let existing = "# My Project\n\nSome instructions here.\n";
        let mut content = existing.to_string();
        content.push('\n');
        content.push_str(MEMORYCO_BLOCK);
        content.push('\n');

        assert!(content.contains("# My Project"));
        assert!(content.contains("Some instructions here"));
        assert!(content.contains(MEMORYCO_BLOCK));
    }

    #[test]
    fn idempotent_replace() {
        // Installing twice should produce the same result
        let content = format!("# Stuff\n\n{}\n", MEMORYCO_BLOCK);
        let result1 = replace_between_markers(&content);
        let result2 = replace_between_markers(&result1);
        assert_eq!(result1, result2);
    }
}
