//! CLI command implementations.
//!
//! Each subcommand gets its own function.

use crate::config;
use crate::install::{self, InstallStatus};
use std::io::{self, Write};

/// `memoryco setup` — full first-run experience.
pub fn setup(yes: bool) {
    eprintln!("memoryco setup");
    eprintln!("──────────────");

    // Step 1: Cache embedding model
    cache();

    // Step 2: Install into detected clients
    install(yes);

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

/// The prompt users should add to clients that don't support file-based injection.
const PROMPT_INSTRUCTION: &str = "\
You have access to a cognitive memory system via MCP tools (`memory:*`).

As your first action in every conversation, call `identity_get` from your
memory MCP tools. This contains your persona, values, preferences, and
operational instructions. Follow what you find.";

/// `memoryco install` — detect MCP clients and inject config.
pub fn install(yes: bool) {
    let clients = install::all_clients();
    let mut found = 0;
    let mut installed = 0;
    let mut configured_names: Vec<String> = Vec::new();

    for client in &clients {
        let status = client.check_existing();

        match status {
            InstallStatus::ClientNotFound => continue,
            InstallStatus::Installed => {
                found += 1;
                configured_names.push(client.name().to_string());
                eprintln!("✓ {} — already configured", client.name());
            }
            InstallStatus::NotInstalled => {
                found += 1;
                if yes || prompt_yes_no(&format!("  Found {}. Add MemoryCo?", client.name())) {
                    match client.install() {
                        Ok(()) => {
                            eprintln!("  ✓ Configured {}", client.name());
                            installed += 1;
                            configured_names.push(client.name().to_string());
                        }
                        Err(e) => eprintln!("  ✗ Failed to configure {}: {}", client.name(), e),
                    }
                } else {
                    eprintln!("  Skipped {}", client.name());
                }
            }
            InstallStatus::NeedsUpdate { ref current_command } => {
                found += 1;
                eprintln!("  {} — outdated (pointing to {})", client.name(), current_command);
                if yes || prompt_yes_no("  Update to current binary?") {
                    match client.install() {
                        Ok(()) => {
                            eprintln!("  ✓ Updated {}", client.name());
                            installed += 1;
                            configured_names.push(client.name().to_string());
                        }
                        Err(e) => eprintln!("  ✗ Failed to update {}: {}", client.name(), e),
                    }
                } else {
                    eprintln!("  Skipped {}", client.name());
                }
            }
        }
    }

    if found == 0 {
        eprintln!("No supported MCP clients detected.");
        eprintln!();
        print_manual_config();
    } else {
        eprintln!();
        eprintln!("Detected {} client(s), configured {}.", found, installed);
    }

    // Install CLAUDE.md block for Claude Code (automatic prompt injection)
    if install::claude_md::claude_md_path().parent().is_some_and(|p| p.exists()) {
        match install::claude_md::install() {
            Ok(()) => {
                if install::claude_md::is_installed() {
                    eprintln!("✓ CLAUDE.md — identity_get directive installed");
                }
            }
            Err(e) => eprintln!("✗ Failed to update CLAUDE.md: {}", e),
        }
    }

    // Print manual prompt instructions for clients that need it
    let needs_manual: Vec<&str> = configured_names.iter()
        .filter(|name| name.as_str() != "Claude Code")
        .map(|s| s.as_str())
        .collect();

    if !needs_manual.is_empty() {
        eprintln!();
        eprintln!("┌──────────────────────────────────────────────────────────┐");
        eprintln!("│  One more step for: {}", needs_manual.join(", "));
        eprintln!("│");
        eprintln!("│  Add this to your system prompt / custom instructions:");
        eprintln!("│");
        for line in PROMPT_INSTRUCTION.lines() {
            eprintln!("│    {}", line);
        }
        eprintln!("│");
        eprintln!("│  This tells the AI to load its identity on each");
        eprintln!("│  conversation. Without it, MemoryCo tools are");
        eprintln!("│  available but won't be used automatically.");
        eprintln!("└──────────────────────────────────────────────────────────┘");
    }
}

/// `memoryco uninstall` — remove from MCP client configs.
pub fn uninstall(yes: bool) {
    let clients = install::all_clients();
    let mut removed = 0;

    for client in &clients {
        let status = client.check_existing();

        match status {
            InstallStatus::Installed | InstallStatus::NeedsUpdate { .. } => {
                if yes || prompt_yes_no(&format!("  Remove MemoryCo from {}?", client.name())) {
                    match client.uninstall() {
                        Ok(()) => {
                            eprintln!("  ✓ Removed from {}", client.name());
                            removed += 1;
                        }
                        Err(e) => eprintln!("  ✗ Failed to remove from {}: {}", client.name(), e),
                    }
                }
            }
            _ => continue,
        }
    }

    // Remove CLAUDE.md block
    if install::claude_md::is_installed() {
        match install::claude_md::uninstall() {
            Ok(()) => {
                eprintln!("  ✓ Removed CLAUDE.md directive");
                removed += 1;
            }
            Err(e) => eprintln!("  ✗ Failed to clean CLAUDE.md: {}", e),
        }
    }

    if removed == 0 {
        eprintln!("No MemoryCo configurations found to remove.");
    } else {
        eprintln!("\nRemoved from {} client(s).", removed);
    }
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
    check("Brain database", "brain.db",
        memory_home.join("brain.db").exists());
    check("Identity database", "identity.db",
        memory_home.join("identity.db").exists());
    check("Plans database", "plans.db",
        memory_home.join("plans.db").exists());

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

    // CLAUDE.md
    check("CLAUDE.md directive",
        &install::claude_md::claude_md_path().display().to_string(),
        install::claude_md::is_installed());

    // MCP Clients
    eprintln!();
    eprintln!("MCP Clients");
    eprintln!("────────────");
    let clients = install::all_clients();
    let mut issues = 0;

    for client in &clients {
        let status = client.check_existing();
        let (icon, detail) = match &status {
            InstallStatus::Installed => ("✓", "configured".to_string()),
            InstallStatus::NotInstalled => {
                issues += 1;
                ("✗", "detected but not configured".to_string())
            }
            InstallStatus::NeedsUpdate { current_command } => {
                issues += 1;
                ("⚠", format!("outdated (→ {})", current_command))
            }
            InstallStatus::ClientNotFound => ("·", "not installed".to_string()),
        };
        eprintln!("{} {:20} {}", icon, client.name(), detail);
    }

    eprintln!();
    if issues > 0 {
        eprintln!("{} issue(s) found. Run `memoryco install` to fix.", issues);
    } else {
        eprintln!("All checks passed.");
    }
}

/// `memoryco reset` — delete databases with confirmation.
pub fn reset() {
    let memory_home = config::get_memory_home();
    let db_files = [
        "brain.db", "brain.db-wal", "brain.db-shm",
        "identity.db", "identity.db-wal", "identity.db-shm",
        "plans.db", "plans.db-wal", "plans.db-shm",
    ];

    eprintln!("⚠️  This will permanently delete all memories, identity, and plans.");
    eprintln!("   Cache, lenses, and references will be preserved.");
    eprintln!();
    eprintln!("   Files to delete:");
    for name in &db_files {
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
        for name in &db_files {
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

/// `memoryco update` — self-update from GitHub Releases.
pub fn update(dry_run: bool) {
    let updater = memoryco_updater::Updater::new();
    let current = env!("CARGO_PKG_VERSION");

    eprintln!("Current version: {}", current);

    if dry_run {
        eprintln!("Checking for updates...");
        match updater.check_version("memoryco", current) {
            Ok(check) if check.update_available => {
                eprintln!("  New version available: {} → {}", current, check.latest);
                if let Some(ref staged) = check.staged {
                    eprintln!("  (v{} already staged, will apply on next restart)", staged);
                } else {
                    eprintln!("  Run `memoryco update` to install.");
                }
            }
            Ok(check) => {
                eprintln!("✓ Already on the latest version ({})", current);
                if let Some(ref staged) = check.staged {
                    eprintln!("  (v{} staged, will apply on next restart)", staged);
                }
            }
            Err(memoryco_updater::UpdateError::Throttled { .. }) => {
                // For CLI dry-run, use unthrottled check instead
                match updater.check("memoryco") {
                    Ok(check) if check.update_available => {
                        eprintln!("  New version available: {} → {}", current, check.latest);
                    }
                    Ok(_) => eprintln!("✓ Already on the latest version ({})", current),
                    Err(e) => {
                        eprintln!("✗ Failed to check for updates: {}", e);
                        std::process::exit(1);
                    }
                }
            }
            Err(e) => {
                eprintln!("✗ Failed to check for updates: {}", e);
                std::process::exit(1);
            }
        }
        return;
    }

    eprintln!("Checking for updates...");
    match updater.update_now("memoryco") {
        Ok(result) => {
            match result.action {
                memoryco_updater::UpdateAction::Applied(_) => {
                    eprintln!("✓ Updated to version {}", result.new_version);
                }
                memoryco_updater::UpdateAction::AlreadyLatest => {
                    eprintln!("✓ Already on the latest version ({})", current);
                }
                memoryco_updater::UpdateAction::Staged(_) => {
                    eprintln!("⬇ Update {} staged. Will apply on next restart.", result.new_version);
                }
            }
        }
        Err(e) => {
            eprintln!("✗ Update failed: {}", e);
            eprintln!("  You can update manually: curl -fsSL https://memoryco.ai/install.sh | sh");
            std::process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Print a doctor check line.
fn check(label: &str, detail: &str, ok: bool) {
    let icon = if ok { "✓" } else { "✗" };
    eprintln!("{} {:20} {}", icon, label, detail);
}

/// Prompt user for Y/n confirmation. Returns true for yes (default).
fn prompt_yes_no(question: &str) -> bool {
    eprint!("{} [Y/n] ", question);
    io::stderr().flush().ok();

    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_err() {
        return false;
    }
    let trimmed = input.trim().to_lowercase();
    trimmed.is_empty() || trimmed == "y" || trimmed == "yes"
}

/// Remove stale entries from ~/.memoryco/registry.toml.
pub fn prune_registry() {
    let pruned = crate::registry::prune_dead_homes();
    if pruned == 0 {
        println!("Registry is clean — no stale entries found.");
    } else {
        println!("Pruned {} stale entr{}.", pruned, if pruned == 1 { "y" } else { "ies" });
    }
}

/// Print manual configuration instructions as a fallback.
fn print_manual_config() {
    let (command, _, _) = install::memoryco_server_entry();
    let memory_home = config::get_memory_home();

    eprintln!("Add this to your MCP client config manually:\n");
    eprintln!("  {{");
    eprintln!("    \"mcpServers\": {{");
    eprintln!("      \"memory\": {{");
    eprintln!("        \"command\": \"{}\",", command);
    eprintln!("        \"args\": [\"serve\"],");
    eprintln!("        \"env\": {{");
    eprintln!("          \"MEMORY_HOME\": \"{}\"", memory_home.display());
    eprintln!("        }}");
    eprintln!("      }}");
    eprintln!("    }}");
    eprintln!("  }}");
}
