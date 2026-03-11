//! Plugin bootstrap — scan `$MEMORY_HOME/bootstrap.d/` for TOML manifests
//! and upsert their instructions into identity.

use crate::identity::{IdentityStore, UpsertResult};
use std::path::Path;
use toml_edit::DocumentMut;

const BOOTSTRAP_DIR: &str = "bootstrap.d";
const MARKER_PREFIX: &str = "<!-- plugin:";

/// Bootstrap plugin instructions from TOML manifests in `bootstrap.d/`.
///
/// For each `*.toml` file in `$MEMORY_HOME/bootstrap.d/`:
/// 1. Parse the TOML manifest
/// 2. Validate the marker starts with `<!-- plugin:`
/// 3. Upsert the instruction content via the existing identity codepath
///
/// Invalid files are logged and skipped — they never crash the server.
pub fn bootstrap(
    identity: &mut IdentityStore,
    memory_home: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let bootstrap_dir = memory_home.join(BOOTSTRAP_DIR);

    // Create directory if missing
    if !bootstrap_dir.exists() {
        std::fs::create_dir_all(&bootstrap_dir)?;
        eprintln!(
            "  Created plugin bootstrap directory: {}",
            bootstrap_dir.display()
        );
    }

    // Glob for *.toml files
    let entries: Vec<_> = std::fs::read_dir(&bootstrap_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("toml"))
        .collect();

    if entries.is_empty() {
        return Ok(());
    }

    for entry in entries {
        let path = entry.path();
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("<unknown>");

        match load_and_upsert(identity, &path) {
            Ok(result) => match result {
                UpsertResult::Added => {
                    eprintln!("  Plugin '{}' instructions added to identity", filename);
                }
                UpsertResult::Updated => {
                    eprintln!("  Plugin '{}' instructions updated in identity", filename);
                }
                UpsertResult::Unchanged => {}
            },
            Err(e) => {
                eprintln!("  Warning: Skipping plugin manifest '{}': {}", filename, e);
            }
        }
    }

    Ok(())
}

/// Parse a single TOML manifest and upsert its instruction.
fn load_and_upsert(
    identity: &mut IdentityStore,
    path: &Path,
) -> Result<UpsertResult, Box<dyn std::error::Error>> {
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

    let marker = meta
        .get("marker")
        .and_then(|v| v.as_str())
        .ok_or("missing meta.marker")?;

    // Validate marker format to avoid collisions with internal markers
    if !marker.starts_with(MARKER_PREFIX) {
        return Err(format!(
            "marker must start with '{}', got: {}",
            MARKER_PREFIX, marker
        )
        .into());
    }

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

    Ok(identity.upsert_instruction(content, marker)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::DieselIdentityStorage;
    use tempfile::TempDir;

    fn test_store() -> IdentityStore {
        let storage = DieselIdentityStorage::in_memory().unwrap();
        IdentityStore::new(storage).unwrap()
    }

    fn valid_manifest(name: &str) -> String {
        format!(
            r#"[meta]
name = "{name}"
version = "0.1.0"
marker = "<!-- plugin:{name} -->"

[instructions]
content = """
<!-- plugin:{name} -->
## {name_cap} Tools

When exploring codebases, prefer search_semantic over grep.
"""
"#,
            name = name,
            name_cap = name.chars().next().unwrap().to_uppercase().to_string() + &name[1..],
        )
    }

    #[test]
    fn bootstrap_creates_directory() {
        let mut identity = test_store();
        let tmp = TempDir::new().unwrap();
        let memory_home = tmp.path();
        let bootstrap_dir = memory_home.join("bootstrap.d");

        assert!(!bootstrap_dir.exists());
        bootstrap(&mut identity, memory_home).unwrap();
        assert!(bootstrap_dir.exists());
    }

    #[test]
    fn bootstrap_empty_directory() {
        let mut identity = test_store();
        let tmp = TempDir::new().unwrap();
        let memory_home = tmp.path();
        std::fs::create_dir_all(memory_home.join("bootstrap.d")).unwrap();

        // No manifests — should succeed with no instructions added
        bootstrap(&mut identity, memory_home).unwrap();
        let result = identity.get().unwrap();
        assert!(result.instructions.is_empty());
    }

    #[test]
    fn bootstrap_valid_manifest() {
        let mut identity = test_store();
        let tmp = TempDir::new().unwrap();
        let memory_home = tmp.path();
        let bootstrap_dir = memory_home.join("bootstrap.d");
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        // Write a valid manifest
        std::fs::write(
            bootstrap_dir.join("filesystem.toml"),
            valid_manifest("filesystem"),
        )
        .unwrap();

        bootstrap(&mut identity, memory_home).unwrap();

        let result = identity.get().unwrap();
        assert_eq!(result.instructions.len(), 1);
        assert!(result.instructions[0].contains("<!-- plugin:filesystem -->"));
        assert!(result.instructions[0].contains("Filesystem Tools"));
    }

    #[test]
    fn bootstrap_idempotent() {
        let mut identity = test_store();
        let tmp = TempDir::new().unwrap();
        let memory_home = tmp.path();
        let bootstrap_dir = memory_home.join("bootstrap.d");
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        std::fs::write(
            bootstrap_dir.join("filesystem.toml"),
            valid_manifest("filesystem"),
        )
        .unwrap();

        bootstrap(&mut identity, memory_home).unwrap();
        let count_first = identity.get().unwrap().instructions.len();

        bootstrap(&mut identity, memory_home).unwrap();
        let count_second = identity.get().unwrap().instructions.len();

        assert_eq!(
            count_first, count_second,
            "Running plugin bootstrap twice should not duplicate instructions"
        );
    }

    #[test]
    fn bootstrap_invalid_toml_skipped() {
        let mut identity = test_store();
        let tmp = TempDir::new().unwrap();
        let memory_home = tmp.path();
        let bootstrap_dir = memory_home.join("bootstrap.d");
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        // Write garbage TOML
        std::fs::write(
            bootstrap_dir.join("broken.toml"),
            "this is not valid toml {{{",
        )
        .unwrap();

        // Should not crash
        bootstrap(&mut identity, memory_home).unwrap();
        let result = identity.get().unwrap();
        assert!(result.instructions.is_empty());
    }

    #[test]
    fn bootstrap_invalid_marker_skipped() {
        let mut identity = test_store();
        let tmp = TempDir::new().unwrap();
        let memory_home = tmp.path();
        let bootstrap_dir = memory_home.join("bootstrap.d");
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        // Marker doesn't start with <!-- plugin:
        let bad_manifest = "[meta]\n\
            name = \"sneaky\"\n\
            version = \"0.1.0\"\n\
            marker = \"## Memory Workflow\"\n\
            \n\
            [instructions]\n\
            content = \"Hijacking internal markers!\"\n";
        std::fs::write(bootstrap_dir.join("sneaky.toml"), bad_manifest).unwrap();

        bootstrap(&mut identity, memory_home).unwrap();
        let result = identity.get().unwrap();
        assert!(
            result.instructions.is_empty(),
            "Manifest with non-plugin marker should be rejected"
        );
    }

    #[test]
    fn bootstrap_missing_meta_name_skipped() {
        let mut identity = test_store();
        let tmp = TempDir::new().unwrap();
        let memory_home = tmp.path();
        let bootstrap_dir = memory_home.join("bootstrap.d");
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        let manifest = r#"[meta]
version = "0.1.0"
marker = "<!-- plugin:noname -->"

[instructions]
content = "Some instructions"
"#;
        std::fs::write(bootstrap_dir.join("noname.toml"), manifest).unwrap();

        bootstrap(&mut identity, memory_home).unwrap();
        let result = identity.get().unwrap();
        assert!(
            result.instructions.is_empty(),
            "Manifest missing meta.name should be skipped"
        );
    }

    #[test]
    fn bootstrap_missing_instructions_table_skipped() {
        let mut identity = test_store();
        let tmp = TempDir::new().unwrap();
        let memory_home = tmp.path();
        let bootstrap_dir = memory_home.join("bootstrap.d");
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        let manifest = r#"[meta]
name = "incomplete"
version = "0.1.0"
marker = "<!-- plugin:incomplete -->"
"#;
        std::fs::write(bootstrap_dir.join("incomplete.toml"), manifest).unwrap();

        bootstrap(&mut identity, memory_home).unwrap();
        let result = identity.get().unwrap();
        assert!(
            result.instructions.is_empty(),
            "Manifest missing [instructions] table should be skipped"
        );
    }

    #[test]
    fn bootstrap_missing_instructions_content_skipped() {
        let mut identity = test_store();
        let tmp = TempDir::new().unwrap();
        let memory_home = tmp.path();
        let bootstrap_dir = memory_home.join("bootstrap.d");
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        let manifest = r#"[meta]
name = "nocontent"
version = "0.1.0"
marker = "<!-- plugin:nocontent -->"

[instructions]
"#;
        std::fs::write(bootstrap_dir.join("nocontent.toml"), manifest).unwrap();

        bootstrap(&mut identity, memory_home).unwrap();
        let result = identity.get().unwrap();
        assert!(
            result.instructions.is_empty(),
            "Manifest missing instructions.content should be skipped"
        );
    }

    #[test]
    fn bootstrap_empty_content_skipped() {
        let mut identity = test_store();
        let tmp = TempDir::new().unwrap();
        let memory_home = tmp.path();
        let bootstrap_dir = memory_home.join("bootstrap.d");
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        let manifest = r#"[meta]
name = "empty"
version = "0.1.0"
marker = "<!-- plugin:empty -->"

[instructions]
content = ""
"#;
        std::fs::write(bootstrap_dir.join("empty.toml"), manifest).unwrap();

        bootstrap(&mut identity, memory_home).unwrap();
        let result = identity.get().unwrap();
        assert!(
            result.instructions.is_empty(),
            "Manifest with empty content should be skipped"
        );
    }

    #[test]
    fn bootstrap_non_toml_files_ignored() {
        let mut identity = test_store();
        let tmp = TempDir::new().unwrap();
        let memory_home = tmp.path();
        let bootstrap_dir = memory_home.join("bootstrap.d");
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        // Drop non-TOML files
        std::fs::write(bootstrap_dir.join("readme.txt"), "not a manifest").unwrap();
        std::fs::write(bootstrap_dir.join("config.json"), "{}").unwrap();
        std::fs::write(bootstrap_dir.join(".hidden"), "nope").unwrap();

        bootstrap(&mut identity, memory_home).unwrap();
        let result = identity.get().unwrap();
        assert!(
            result.instructions.is_empty(),
            "Non-TOML files should be silently ignored"
        );
    }

    #[test]
    fn bootstrap_updated_content_replaces() {
        let mut identity = test_store();
        let tmp = TempDir::new().unwrap();
        let memory_home = tmp.path();
        let bootstrap_dir = memory_home.join("bootstrap.d");
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        // First boot with v1 content
        let v1 = r#"[meta]
name = "evolving"
version = "0.1.0"
marker = "<!-- plugin:evolving -->"

[instructions]
content = "<!-- plugin:evolving -->\nVersion 1 instructions"
"#;
        std::fs::write(bootstrap_dir.join("evolving.toml"), v1).unwrap();
        bootstrap(&mut identity, memory_home).unwrap();

        let result = identity.get().unwrap();
        assert_eq!(result.instructions.len(), 1);
        assert!(result.instructions[0].contains("Version 1"));

        // Second boot with v2 content (same marker, different payload)
        let v2 = r#"[meta]
name = "evolving"
version = "0.2.0"
marker = "<!-- plugin:evolving -->"

[instructions]
content = "<!-- plugin:evolving -->\nVersion 2 instructions with new features"
"#;
        std::fs::write(bootstrap_dir.join("evolving.toml"), v2).unwrap();
        bootstrap(&mut identity, memory_home).unwrap();

        let result = identity.get().unwrap();
        assert_eq!(
            result.instructions.len(),
            1,
            "Updated manifest should replace, not duplicate"
        );
        assert!(
            result.instructions[0].contains("Version 2"),
            "Content should reflect the updated manifest"
        );
        assert!(
            !result.instructions[0].contains("Version 1"),
            "Old content should be gone"
        );
    }

    #[test]
    fn bootstrap_invalid_manifest_doesnt_poison_batch() {
        let mut identity = test_store();
        let tmp = TempDir::new().unwrap();
        let memory_home = tmp.path();
        let bootstrap_dir = memory_home.join("bootstrap.d");
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        // One broken, one valid
        std::fs::write(bootstrap_dir.join("aaa_broken.toml"), "not valid toml {{{").unwrap();
        std::fs::write(
            bootstrap_dir.join("zzz_valid.toml"),
            valid_manifest("validplugin"),
        )
        .unwrap();

        bootstrap(&mut identity, memory_home).unwrap();

        let result = identity.get().unwrap();
        assert_eq!(
            result.instructions.len(),
            1,
            "Valid manifest should still be processed despite broken sibling"
        );
        assert!(result.instructions[0].contains("<!-- plugin:validplugin -->"));
    }

    #[test]
    fn bootstrap_multiple_manifests() {
        let mut identity = test_store();
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

        bootstrap(&mut identity, memory_home).unwrap();

        let result = identity.get().unwrap();
        assert_eq!(result.instructions.len(), 2);

        let all_text: String = result.instructions.join("\n");
        assert!(all_text.contains("<!-- plugin:filesystem -->"));
        assert!(all_text.contains("<!-- plugin:gittools -->"));
    }
}
