//! Bootstrap coordinator - orchestrates directory creation on first run
//!
//! Instructions are no longer written to identity at bootstrap.
//! They are served on demand by the `instructions` tool.

use std::path::Path;

/// Bootstrap all modules: create directories and seed sample files.
///
/// This no longer writes to identity — instructions are served by the
/// `instructions` tool at call time.
pub fn bootstrap_all(
    lenses_dir: &Path,
    memory_home: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Lenses (creates directory + sample lens if empty)
    crate::lenses::bootstrap(lenses_dir)?;

    // Plugin bootstrap.d/ directory
    crate::plugins::bootstrap::ensure_bootstrap_dir(memory_home)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn bootstrap_creates_directories() {
        let tmp = TempDir::new().unwrap();
        let lenses_dir = tmp.path().join("lenses");
        let memory_home = tmp.path();

        bootstrap_all(&lenses_dir, memory_home).unwrap();

        assert!(lenses_dir.exists(), "Lenses directory should be created");
        assert!(
            memory_home.join("bootstrap.d").exists(),
            "bootstrap.d directory should be created"
        );
    }

    #[test]
    fn bootstrap_is_idempotent() {
        let tmp = TempDir::new().unwrap();
        let lenses_dir = tmp.path().join("lenses");
        let memory_home = tmp.path();

        bootstrap_all(&lenses_dir, memory_home).unwrap();
        bootstrap_all(&lenses_dir, memory_home).unwrap();

        // Should not crash or duplicate anything
        assert!(lenses_dir.exists());
    }

    #[test]
    fn bootstrap_creates_sample_lens() {
        let tmp = TempDir::new().unwrap();
        let lenses_dir = tmp.path().join("lenses");
        let memory_home = tmp.path();

        bootstrap_all(&lenses_dir, memory_home).unwrap();

        let sample = lenses_dir.join("sample.md");
        assert!(sample.exists(), "Sample lens should be created");
    }
}
