//! Bootstrap coordinator - orchestrates all module bootstrapping
//!
//! Each module has its own bootstrap function that:
//! - Checks if its instructions already exist in identity
//! - Adds instructions to identity.instructions if not present
//! - Performs any module-specific setup (directories, etc.)

use crate::engram::Brain;
use crate::reference::ReferenceManager;
use std::path::Path;

/// Bootstrap all modules in the correct order
pub fn bootstrap_all(
    brain: &mut Brain,
    lenses_dir: &Path,
    references: &ReferenceManager,
) -> Result<(), Box<dyn std::error::Error>> {
    // Engram first (core memory instructions)
    crate::engram::bootstrap::bootstrap(brain)?;
    
    // Lenses (adds instructions + creates directory)
    crate::lenses::bootstrap(brain, lenses_dir)?;
    
    // Reference (adds per-source citation instructions)
    crate::reference::bootstrap::bootstrap(brain, references)?;
    
    Ok(())
}
