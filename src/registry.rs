//! Shared registry — publishes this server's MEMORY_HOME to a well-known location.
//!
//! The registry file at `~/.memoryco/registry.toml` is the coordination point
//! between independent MCP servers. Memory servers register their home paths;
//! external plugins read the registry to discover where to drop manifests.

use std::path::{Path, PathBuf};
use toml_edit::DocumentMut;

const REGISTRY_FILENAME: &str = "registry.toml";
const DEFAULT_DIR_NAME: &str = ".memoryco";

/// Get the well-known registry path: `~/.memoryco/registry.toml`
///
/// This is always at the default location, NOT inside a custom MEMORY_HOME.
/// It's the shared coordination point regardless of MEMORY_HOME configuration.
pub fn get_registry_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(DEFAULT_DIR_NAME).join(REGISTRY_FILENAME))
}

/// Ensure the current MEMORY_HOME is registered in `~/.memoryco/registry.toml`.
///
/// - Creates `~/.memoryco/` and `registry.toml` if they don't exist
/// - Reads the existing file, checks if `memory_home` is in the `homes` array
/// - Adds it if missing, writes back
/// - Logs and continues on any error — never crashes the server
pub fn ensure_registered(memory_home: &Path) {
    match try_ensure_registered(memory_home) {
        Ok(true) => eprintln!("Registered in shared registry"),
        Ok(false) => {} // already registered, silent
        Err(e) => eprintln!("Warning: Failed to update registry: {}", e),
    }
}

fn try_ensure_registered(memory_home: &Path) -> Result<bool, Box<dyn std::error::Error>> {
    let registry_path = get_registry_path()
        .ok_or("Could not determine home directory")?;
    try_register_at(&registry_path, memory_home)
}

/// Inner logic with explicit registry path (testable without touching ~/.memoryco/).
///
/// Returns `Ok(true)` if we added ourselves, `Ok(false)` if already present.
fn try_register_at(
    registry_path: &Path,
    memory_home: &Path,
) -> Result<bool, Box<dyn std::error::Error>> {
    // Ensure parent dir exists
    if let Some(parent) = registry_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Read existing or start fresh
    let raw = match std::fs::read_to_string(registry_path) {
        Ok(content) => content,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(e.into()),
    };

    let mut doc: DocumentMut = raw.parse()?;

    let home_str = memory_home.to_string_lossy().to_string();

    // Check if already present
    let already_present = doc
        .get("homes")
        .and_then(|v| v.as_array())
        .is_some_and(|homes| homes.iter().any(|v| v.as_str() == Some(&home_str)));

    if already_present {
        return Ok(false);
    }

    // Add our path
    let arr = doc
        .entry("homes")
        .or_insert_with(|| {
            toml_edit::Item::Value(toml_edit::Value::Array(Default::default()))
        })
        .as_array_mut()
        .ok_or("'homes' is not an array")?;

    arr.push(home_str);
    std::fs::write(registry_path, doc.to_string())?;

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn creates_registry_when_missing() {
        let tmp = TempDir::new().unwrap();
        let registry_path = tmp.path().join("registry.toml");
        let memory_home = tmp.path().join("my_memory");

        try_register_at(&registry_path, &memory_home).unwrap();

        let content = std::fs::read_to_string(&registry_path).unwrap();
        let doc: DocumentMut = content.parse().unwrap();
        let homes = doc["homes"].as_array().unwrap();
        assert_eq!(homes.len(), 1);
        assert_eq!(homes.get(0).unwrap().as_str().unwrap(), memory_home.to_string_lossy());
    }

    #[test]
    fn does_not_duplicate_existing_entry() {
        let tmp = TempDir::new().unwrap();
        let registry_path = tmp.path().join("registry.toml");
        let memory_home = tmp.path().join("my_memory");

        let added_first = try_register_at(&registry_path, &memory_home).unwrap();
        assert!(added_first, "First call should add");

        let added_second = try_register_at(&registry_path, &memory_home).unwrap();
        assert!(!added_second, "Second call should be a no-op");

        let content = std::fs::read_to_string(&registry_path).unwrap();
        let doc: DocumentMut = content.parse().unwrap();
        let homes = doc["homes"].as_array().unwrap();
        assert_eq!(homes.len(), 1, "Should not duplicate");
    }

    #[test]
    fn adds_second_home() {
        let tmp = TempDir::new().unwrap();
        let registry_path = tmp.path().join("registry.toml");
        let home1 = tmp.path().join("home1");
        let home2 = tmp.path().join("home2");

        try_register_at(&registry_path, &home1).unwrap();
        try_register_at(&registry_path, &home2).unwrap();

        let content = std::fs::read_to_string(&registry_path).unwrap();
        let doc: DocumentMut = content.parse().unwrap();
        let homes = doc["homes"].as_array().unwrap();
        assert_eq!(homes.len(), 2);
    }

    #[test]
    fn preserves_existing_content() {
        let tmp = TempDir::new().unwrap();
        let registry_path = tmp.path().join("registry.toml");

        // Pre-populate with existing home
        std::fs::write(&registry_path, "homes = [\"/existing/home\"]\n").unwrap();

        let new_home = tmp.path().join("new_home");
        try_register_at(&registry_path, &new_home).unwrap();

        let content = std::fs::read_to_string(&registry_path).unwrap();
        let doc: DocumentMut = content.parse().unwrap();
        let homes = doc["homes"].as_array().unwrap();
        assert_eq!(homes.len(), 2);
        assert_eq!(homes.get(0).unwrap().as_str().unwrap(), "/existing/home");
    }

    #[test]
    fn handles_corrupt_toml_gracefully() {
        let tmp = TempDir::new().unwrap();
        let registry_path = tmp.path().join("registry.toml");
        std::fs::write(&registry_path, "this is not valid {{{").unwrap();

        let memory_home = tmp.path().join("home");
        let result = try_register_at(&registry_path, &memory_home);
        assert!(result.is_err(), "Should return error on corrupt TOML");
    }
}
