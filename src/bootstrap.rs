//! Bootstrap coordinator - orchestrates all module bootstrapping
//!
//! Each module has its own bootstrap function that:
//! - Checks if its instructions already exist in identity
//! - Adds instructions to identity.instructions if not present
//! - Performs any module-specific setup (directories, etc.)

use crate::identity::IdentityStore;
use crate::reference::ReferenceManager;
use std::path::Path;

/// Bootstrap all modules in the correct order.
///
/// All instructions are written to the IdentityStore (identity.db),
/// not the Brain (brain.db). This ensures `identity_get` returns
/// instructions even on a fresh install.
pub fn bootstrap_all(
    identity: &mut IdentityStore,
    lenses_dir: &Path,
    references: &ReferenceManager,
) -> Result<(), Box<dyn std::error::Error>> {
    // Engram first (core memory instructions)
    crate::engram::bootstrap::bootstrap(identity)?;
    
    // Lenses (adds instructions + creates directory)
    crate::lenses::bootstrap(identity, lenses_dir)?;
    
    // Reference (adds per-source citation instructions)
    crate::reference::bootstrap::bootstrap(identity, references)?;
    
    // Plans (task tracking instructions)
    crate::plans::bootstrap::bootstrap(identity)?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{IdentityStore, DieselIdentityStorage};
    use crate::reference::ReferenceManager;
    use tempfile::TempDir;

    fn test_store() -> IdentityStore {
        let storage = DieselIdentityStorage::in_memory().unwrap();
        IdentityStore::new(storage).unwrap()
    }

    #[test]
    fn bootstrap_fresh_identity_has_instructions() {
        let mut identity = test_store();
        let tmp = TempDir::new().unwrap();
        let lenses_dir = tmp.path().join("lenses");
        let references = ReferenceManager::new();

        // Before bootstrap: identity should be empty
        let before = identity.get().unwrap();
        assert!(before.instructions.is_empty(),
            "Fresh IdentityStore should have no instructions");

        // Run bootstrap
        bootstrap_all(&mut identity, &lenses_dir, &references).unwrap();

        // After bootstrap: identity must have instructions
        let after = identity.get().unwrap();
        assert!(!after.instructions.is_empty(),
            "Bootstrap must populate instructions in IdentityStore");

        // Verify each module's marker is present
        let all_text: String = after.instructions.join("\n");
        assert!(all_text.contains("## Memory Workflow"),
            "Missing engram instructions");
        assert!(all_text.contains("## Lenses"),
            "Missing lenses instructions");
        assert!(all_text.contains("## References"),
            "Missing reference instructions");
        assert!(all_text.contains("## Plans"),
            "Missing plans instructions");
    }

    #[test]
    fn bootstrap_is_idempotent() {
        let mut identity = test_store();
        let tmp = TempDir::new().unwrap();
        let lenses_dir = tmp.path().join("lenses");
        let references = ReferenceManager::new();

        // Run bootstrap twice
        bootstrap_all(&mut identity, &lenses_dir, &references).unwrap();
        let first = identity.get().unwrap();
        let count_after_first = first.instructions.len();

        bootstrap_all(&mut identity, &lenses_dir, &references).unwrap();
        let second = identity.get().unwrap();
        let count_after_second = second.instructions.len();

        assert_eq!(count_after_first, count_after_second,
            "Running bootstrap twice should not duplicate instructions");
    }

    #[test]
    fn bootstrap_identity_get_returns_instructions() {
        // This is THE test for the bug we caught:
        // On a fresh install, identity_get must return instructions,
        // not "No identity configured yet."
        let mut identity = test_store();
        let tmp = TempDir::new().unwrap();
        let lenses_dir = tmp.path().join("lenses");
        let references = ReferenceManager::new();

        bootstrap_all(&mut identity, &lenses_dir, &references).unwrap();

        let result = identity.get().unwrap();

        // The identity_get tool checks these three fields to decide
        // whether to return "No identity configured yet."
        // After bootstrap, instructions must be non-empty.
        let has_name = !result.persona.name.is_empty();
        let has_values = !result.values.is_empty();
        let has_instructions = !result.instructions.is_empty();

        assert!(has_name || has_values || has_instructions,
            "identity_get would return 'No identity configured yet.' \n\
             but bootstrap should ensure at least instructions are present.\n\
             persona.name.empty={}, values.empty={}, instructions.empty={}",
            result.persona.name.is_empty(),
            result.values.is_empty(),
            result.instructions.is_empty());
    }
}
