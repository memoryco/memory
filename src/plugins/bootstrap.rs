//! Plugin bootstrap — scan `$MEMORY_HOME/bootstrap.d/` for TOML manifests
//! and return their instruction content.

use std::path::Path;
use toml_edit::DocumentMut;

const BOOTSTRAP_DIR: &str = "bootstrap.d";

/// Load plugin instructions from TOML manifests in `bootstrap.d/`.
/// Called by the instructions tool at request time.
///
/// Returns a Vec of instruction content strings, one per valid manifest.
/// Invalid files are logged and skipped — they never crash.
pub fn load_plugin_instructions(memory_home: &Path) -> Vec<String> {
    let bootstrap_dir = memory_home.join(BOOTSTRAP_DIR);

    if !bootstrap_dir.exists() {
        return Vec::new();
    }

    let entries: Vec<_> = match std::fs::read_dir(&bootstrap_dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("toml"))
            .collect(),
        Err(_) => return Vec::new(),
    };

    let mut results = Vec::new();

    for entry in entries {
        let path = entry.path();
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("<unknown>");

        match load_instruction(&path) {
            Ok(content) => results.push(content),
            Err(e) => {
                eprintln!("  Warning: Skipping plugin manifest '{}': {}", filename, e);
            }
        }
    }

    results
}

/// Ensure the bootstrap.d/ directory exists. Called during server startup.
pub fn ensure_bootstrap_dir(memory_home: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let bootstrap_dir = memory_home.join(BOOTSTRAP_DIR);
    if !bootstrap_dir.exists() {
        std::fs::create_dir_all(&bootstrap_dir)?;
        eprintln!(
            "  Created plugin bootstrap directory: {}",
            bootstrap_dir.display()
        );
    }
    Ok(())
}

/// Parse a single TOML manifest and extract its instruction content.
fn load_instruction(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    let doc: DocumentMut = raw.parse()?;

    // Extract [meta] fields
    let meta = doc
        .get("meta")
        .and_then(|v| v.as_table())
        .ok_or("missing [meta] table")?;

    let _name = meta
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or("missing meta.name")?;

    let _version = meta
        .get("version")
        .and_then(|v| v.as_str())
        .ok_or("missing meta.version")?;

    // Extract [instructions] fields
    let instructions = doc
        .get("instructions")
        .and_then(|v| v.as_table())
        .ok_or("missing [instructions] table")?;

    let content = instructions
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or("missing instructions.content")?;

    if content.trim().is_empty() {
        return Err("instructions.content is empty".into());
    }

    Ok(content.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn valid_manifest(name: &str) -> String {
        format!(
            r#"[meta]
name = "{name}"
version = "0.1.0"

[instructions]
content = """
## {name_cap} Tools

When exploring codebases, prefer search_semantic over grep.
"""
"#,
            name = name,
            name_cap = name.chars().next().unwrap().to_uppercase().to_string() + &name[1..],
        )
    }

    #[test]
    fn ensure_creates_directory() {
        let tmp = TempDir::new().unwrap();
        let memory_home = tmp.path();
        let bootstrap_dir = memory_home.join("bootstrap.d");

        assert!(!bootstrap_dir.exists());
        ensure_bootstrap_dir(memory_home).unwrap();
        assert!(bootstrap_dir.exists());
    }

    #[test]
    fn empty_directory_returns_empty() {
        let tmp = TempDir::new().unwrap();
        let memory_home = tmp.path();
        std::fs::create_dir_all(memory_home.join("bootstrap.d")).unwrap();

        let result = load_plugin_instructions(memory_home);
        assert!(result.is_empty());
    }

    #[test]
    fn valid_manifest_loads() {
        let tmp = TempDir::new().unwrap();
        let memory_home = tmp.path();
        let bootstrap_dir = memory_home.join("bootstrap.d");
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        std::fs::write(
            bootstrap_dir.join("filesystem.toml"),
            valid_manifest("filesystem"),
        )
        .unwrap();

        let result = load_plugin_instructions(memory_home);
        assert_eq!(result.len(), 1);
        assert!(result[0].contains("Filesystem Tools"));
    }

    #[test]
    fn invalid_toml_skipped() {
        let tmp = TempDir::new().unwrap();
        let memory_home = tmp.path();
        let bootstrap_dir = memory_home.join("bootstrap.d");
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        std::fs::write(
            bootstrap_dir.join("broken.toml"),
            "this is not valid toml {{{",
        )
        .unwrap();

        let result = load_plugin_instructions(memory_home);
        assert!(result.is_empty());
    }

    #[test]
    fn missing_meta_name_skipped() {
        let tmp = TempDir::new().unwrap();
        let memory_home = tmp.path();
        let bootstrap_dir = memory_home.join("bootstrap.d");
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        let manifest = r#"[meta]
version = "0.1.0"

[instructions]
content = "Some instructions"
"#;
        std::fs::write(bootstrap_dir.join("noname.toml"), manifest).unwrap();

        let result = load_plugin_instructions(memory_home);
        assert!(result.is_empty(), "Manifest missing meta.name should be skipped");
    }

    #[test]
    fn missing_instructions_table_skipped() {
        let tmp = TempDir::new().unwrap();
        let memory_home = tmp.path();
        let bootstrap_dir = memory_home.join("bootstrap.d");
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        let manifest = r#"[meta]
name = "incomplete"
version = "0.1.0"
"#;
        std::fs::write(bootstrap_dir.join("incomplete.toml"), manifest).unwrap();

        let result = load_plugin_instructions(memory_home);
        assert!(result.is_empty());
    }

    #[test]
    fn empty_content_skipped() {
        let tmp = TempDir::new().unwrap();
        let memory_home = tmp.path();
        let bootstrap_dir = memory_home.join("bootstrap.d");
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        let manifest = r#"[meta]
name = "empty"
version = "0.1.0"

[instructions]
content = ""
"#;
        std::fs::write(bootstrap_dir.join("empty.toml"), manifest).unwrap();

        let result = load_plugin_instructions(memory_home);
        assert!(result.is_empty());
    }

    #[test]
    fn non_toml_files_ignored() {
        let tmp = TempDir::new().unwrap();
        let memory_home = tmp.path();
        let bootstrap_dir = memory_home.join("bootstrap.d");
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        std::fs::write(bootstrap_dir.join("readme.txt"), "not a manifest").unwrap();
        std::fs::write(bootstrap_dir.join("config.json"), "{}").unwrap();
        std::fs::write(bootstrap_dir.join(".hidden"), "nope").unwrap();

        let result = load_plugin_instructions(memory_home);
        assert!(result.is_empty());
    }

    #[test]
    fn invalid_manifest_doesnt_poison_batch() {
        let tmp = TempDir::new().unwrap();
        let memory_home = tmp.path();
        let bootstrap_dir = memory_home.join("bootstrap.d");
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        std::fs::write(bootstrap_dir.join("aaa_broken.toml"), "not valid toml {{{").unwrap();
        std::fs::write(
            bootstrap_dir.join("zzz_valid.toml"),
            valid_manifest("validplugin"),
        )
        .unwrap();

        let result = load_plugin_instructions(memory_home);
        assert_eq!(result.len(), 1, "Valid manifest should still load despite broken sibling");
        assert!(result[0].contains("Validplugin Tools"));
    }

    #[test]
    fn multiple_manifests() {
        let tmp = TempDir::new().unwrap();
        let memory_home = tmp.path();
        let bootstrap_dir = memory_home.join("bootstrap.d");
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        std::fs::write(
            bootstrap_dir.join("filesystem.toml"),
            valid_manifest("filesystem"),
        )
        .unwrap();
        std::fs::write(
            bootstrap_dir.join("gittools.toml"),
            valid_manifest("gittools"),
        )
        .unwrap();

        let result = load_plugin_instructions(memory_home);
        assert_eq!(result.len(), 2);

        let all_text: String = result.join("\n");
        assert!(all_text.contains("Filesystem Tools"));
        assert!(all_text.contains("Gittools Tools"));
    }

    #[test]
    fn no_bootstrap_dir_returns_empty() {
        let tmp = TempDir::new().unwrap();
        // Don't create bootstrap.d/
        let result = load_plugin_instructions(tmp.path());
        assert!(result.is_empty());
    }
}
