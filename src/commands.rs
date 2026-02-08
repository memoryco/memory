//! CLI command implementations.
//!
//! Each subcommand gets its own function. Most are stubs for now.

use crate::config;
use std::io::{self, Write};

/// `memoryco setup` — full first-run experience.
pub fn setup(_yes: bool) {
    eprintln!("memoryco setup");
    eprintln!("──────────────");

    // Step 1: Cache embedding model
    cache();

    // Step 2: Install into detected clients
    install(_yes);

    // Step 3: Start serving
    eprintln!("\nStarting server...");
    crate::server::run();
}

/// `memoryco cache` — download and verify the embedding model.
pub fn cache() {
    let model_dir = config::get_model_cache_dir();
    eprintln!("Embedding model cache: {}", model_dir.display());

    // The embedding generator auto-downloads on first use.
    // Here we just trigger that eagerly so the user sees progress.
    eprintln!("Checking embedding model...");
    let generator = crate::embedding::EmbeddingGenerator::new();
    match generator.generate("warmup") {
        Ok(_) => eprintln!("✓ Embedding model ready"),
        Err(e) => {
            eprintln!("✗ Failed to load embedding model: {}", e);
            std::process::exit(1);
        }
    }
}

/// `memoryco install` — detect MCP clients and inject config.
pub fn install(_yes: bool) {
    eprintln!("Client auto-detection not yet implemented.");
    eprintln!("For now, add this to your MCP client config manually:\n");

    let exe = std::env::current_exe()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "memoryco".to_string());

    let memory_home = config::get_memory_home();

    eprintln!("  {{");
    eprintln!("    \"mcpServers\": {{");
    eprintln!("      \"memoryco\": {{");
    eprintln!("        \"command\": \"{}\",", exe);
    eprintln!("        \"args\": [\"serve\"],");
    eprintln!("        \"env\": {{");
    eprintln!("          \"MEMORY_HOME\": \"{}\"", memory_home.display());
    eprintln!("        }}");
    eprintln!("      }}");
    eprintln!("    }}");
    eprintln!("  }}");
}

/// `memoryco uninstall` — remove from MCP client configs.
pub fn uninstall(_yes: bool) {
    eprintln!("Client uninstall not yet implemented.");
}

/// `memoryco doctor` — health check.
pub fn doctor() {
    let memory_home = config::get_memory_home();

    eprintln!("MemoryCo Doctor");
    eprintln!("────────────────");

    // Home directory
    check("Home directory", &memory_home.display().to_string(),
        memory_home.exists());

    // Databases
    let brain_path = memory_home.join("brain.db");
    check("Brain database", "brain.db", brain_path.exists());

    let identity_path = memory_home.join("identity.db");
    check("Identity database", "identity.db", identity_path.exists());

    let plans_path = memory_home.join("plans.db");
    check("Plans database", "plans.db", plans_path.exists());

    // Embedding model
    let model_dir = config::get_model_cache_dir();
    let model_exists = model_dir.exists() && model_dir.read_dir()
        .map(|mut d| d.next().is_some())
        .unwrap_or(false);
    check("Embedding model", &model_dir.display().to_string(), model_exists);

    // Lenses
    let lenses_dir = memory_home.join("lenses");
    let lens_count = lenses_dir.read_dir()
        .map(|d| d.filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "md"))
            .count())
        .unwrap_or(0);
    check("Lenses", &format!("{} loaded", lens_count), lenses_dir.exists());

    // References
    let refs_dir = memory_home.join("references");
    let ref_count = refs_dir.read_dir()
        .map(|d| d.filter_map(|e| e.ok()).count())
        .unwrap_or(0);
    check("References", &format!("{} sources", ref_count), refs_dir.exists());

    eprintln!();
}

/// `memoryco reset` — delete databases with confirmation.
pub fn reset() {
    let memory_home = config::get_memory_home();

    eprintln!("⚠️  This will permanently delete all memories, identity, and plans.");
    eprintln!("   Cache, lenses, and references will be preserved.");
    eprintln!();
    eprintln!("   Files to delete:");
    for name in &["brain.db", "brain.db-wal", "brain.db-shm",
                   "identity.db", "identity.db-wal", "identity.db-shm",
                   "plans.db", "plans.db-wal", "plans.db-shm"] {
        let path = memory_home.join(name);
        if path.exists() {
            eprintln!("     {}", path.display());
        }
    }
    eprintln!();
    eprint!("Type 'reset' to confirm: ");
    io::stderr().flush().ok();

    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_ok() && input.trim() == "reset" {
        for name in &["brain.db", "brain.db-wal", "brain.db-shm",
                       "identity.db", "identity.db-wal", "identity.db-shm",
                       "plans.db", "plans.db-wal", "plans.db-shm"] {
            let path = memory_home.join(name);
            if path.exists() {
                if let Err(e) = std::fs::remove_file(&path) {
                    eprintln!("  ✗ Failed to delete {}: {}", name, e);
                } else {
                    eprintln!("  ✓ Deleted {}", name);
                }
            }
        }
        eprintln!("\nReset complete. Run `memoryco serve` to start fresh.");
    } else {
        eprintln!("Reset cancelled.");
    }
}

/// Print a doctor check line.
fn check(label: &str, detail: &str, ok: bool) {
    let icon = if ok { "✓" } else { "✗" };
    eprintln!("{} {:20} {}", icon, label, detail);
}
