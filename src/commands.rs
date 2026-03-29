//! CLI command implementations.
//!
//! Each subcommand gets its own function.

use crate::cli::LlmCommand;
use crate::config;
use crate::engram::Brain;
use crate::engram::storage::Storage as _;
use crate::install::{self, InstallStatus};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// `memoryco setup` — full first-run experience.
pub fn setup(yes: bool) {
    eprintln!("memoryco setup");
    eprintln!("──────────────");

    // Install into detected clients (also prompts for MEMORY_HOME and downloads models)
    install(yes);

    // Start serving
    eprintln!("\nStarting server...");
    crate::server::run();
}

/// The prompt users should add to clients that don't support file-based injection.
const PROMPT_INSTRUCTION: &str = "\
You have access to a cognitive memory system via MCP tools (`memory:*`).

As your first action in every conversation, call `identity_get` from your
memory MCP tools. This contains your persona, values, preferences, and
operational instructions. Follow what you find.";

/// `memoryco install` — detect MCP clients and inject config.
pub fn install(yes: bool) {
    // 1. Determine MEMORY_HOME
    let memory_home = prompt_memory_home(yes);
    // SAFETY: single-threaded CLI command; no other threads reading this env var yet.
    unsafe {
        std::env::set_var("MEMORY_HOME", memory_home.display().to_string());
    }

    // 2. Download bundled models
    download_bundled_models(&memory_home);

    // 3. Detect and configure MCP clients
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
            InstallStatus::NeedsUpdate {
                ref current_command,
            } => {
                found += 1;
                eprintln!(
                    "  {} — outdated (pointing to {})",
                    client.name(),
                    current_command
                );
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
    if install::claude_md::claude_md_path()
        .parent()
        .is_some_and(|p| p.exists())
    {
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
    let needs_manual: Vec<&str> = configured_names
        .iter()
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

/// `memoryco generate` — generate embeddings and/or enrichments.
pub fn generate(embeddings_only: bool, enrichments_only: bool) {
    let exit_code = generate_inner(embeddings_only, enrichments_only);
    // Explicitly release GPU models before process exit. Rust does not drop
    // statics, so without this the Metal/CUDA device finalizer asserts on
    // leaked resource sets during C++ __cxa_finalize.
    crate::embedding::EmbeddingGenerator::shutdown();
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}

fn generate_inner(embeddings_only: bool, enrichments_only: bool) -> i32 {
    let memory_home = config::get_memory_home();
    let db_path = memory_home.join("brain.db");

    if !db_path.exists() {
        eprintln!("✗ Brain database not found: {}", db_path.display());
        eprintln!("  Run `memoryco setup` to get started.");
        return 1;
    }

    let brain_config = crate::engram::config_toml::load_config_from_toml(&memory_home);
    let brain = match Brain::open_path(&db_path, brain_config) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("✗ Failed to open brain: {}", e);
            return 1;
        }
    };

    let do_embeddings = !enrichments_only;
    let do_enrichments = !embeddings_only;

    // Warn if enrichments-only but most engrams lack embeddings.
    if enrichments_only {
        let without = brain.count_without_embeddings().unwrap_or(0);
        let with_emb = brain.count_with_embeddings().unwrap_or(0);
        let total = without + with_emb;
        if total > 0 && without > total / 2 {
            eprintln!(
                "⚠ Most engrams have no embeddings. \
                 Run `memoryco generate` (without flags) to generate both."
            );
        }
    }

    let brain = Arc::new(RwLock::new(brain));
    let mut total_embeddings = 0usize;
    let mut total_enrichments = 0usize;
    let mut total_vectors = 0usize;

    if do_embeddings {
        let stats = crate::generate::generate_embeddings(Arc::clone(&brain), true);
        total_embeddings = stats.generated;
        total_vectors += stats.generated;
        if stats.errors > 0 {
            eprintln!("⚠ {} embedding error(s) encountered", stats.errors);
        }
    }

    if do_enrichments {
        let llm = match crate::llm::build_llm_service(&memory_home) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("✗ Failed to initialize LLM: {}", e);
                return 1;
            }
        };
        if !llm.available() {
            eprintln!(
                "✗ Local LLM unavailable — enrichments require an LLM.\n  \
                 Configure `llm_enable = true` in {}",
                memory_home.join("config.toml").display()
            );
            if do_embeddings && total_embeddings > 0 {
                eprintln!("  {} embedding(s) were generated successfully.", total_embeddings);
            }
            return 1;
        }
        let stats = crate::generate::generate_enrichments(Arc::clone(&brain), llm, true);
        total_enrichments = stats.generated;
        total_vectors += stats.vectors;
    }

    eprintln!();
    match (do_embeddings, do_enrichments) {
        (true, true) => eprintln!(
            "Generated {} embedding(s), {} enrichment(s) ({} vector(s))",
            total_embeddings, total_enrichments, total_vectors
        ),
        (true, false) => eprintln!("Generated {} embedding(s)", total_embeddings),
        (false, true) => eprintln!(
            "Generated {} enrichment(s) ({} vector(s))",
            total_enrichments, total_vectors
        ),
        (false, false) => unreachable!(),
    }

    0
}

/// `memoryco doctor` — health check.
pub fn doctor() {
    let memory_home = config::get_memory_home();

    eprintln!("MemoryCo Doctor");
    eprintln!("────────────────");

    // Home directory
    check(
        "Home directory",
        &memory_home.display().to_string(),
        memory_home.exists(),
    );

    // Databases
    check(
        "Brain database",
        "brain.db",
        memory_home.join("brain.db").exists(),
    );
    check(
        "Identity database",
        "identity.db",
        memory_home.join("identity.db").exists(),
    );


    // Embedding model
    let model_dir = config::get_model_cache_dir();
    let model_exists = model_dir.exists()
        && model_dir
            .read_dir()
            .map(|mut d| d.next().is_some())
            .unwrap_or(false);
    check(
        "Embedding model",
        &model_dir.display().to_string(),
        model_exists,
    );

    // Lenses
    let lenses_dir = memory_home.join("lenses");
    let lens_count = lenses_dir
        .read_dir()
        .map(|d| {
            d.filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "md"))
                .count()
        })
        .unwrap_or(0);
    check(
        "Lenses",
        &format!("{} loaded", lens_count),
        lenses_dir.exists(),
    );

    // References
    let refs_dir = memory_home.join("references");
    let ref_count = refs_dir
        .read_dir()
        .map(|d| d.filter_map(|e| e.ok()).count())
        .unwrap_or(0);
    check(
        "References",
        &format!("{} sources", ref_count),
        refs_dir.exists(),
    );

    // CLAUDE.md
    check(
        "CLAUDE.md directive",
        &install::claude_md::claude_md_path().display().to_string(),
        install::claude_md::is_installed(),
    );

    // Embeddings health — open storage directly (read-only, no migrations)
    let db_path = memory_home.join("brain.db");
    if db_path.exists() {
        eprintln!();
        eprintln!("Embeddings");
        eprintln!("────────────");

        use crate::engram::storage::EngramStorage;
        match EngramStorage::open(&db_path) {
            Err(e) => {
                eprintln!("✗ {:20} {}", "Brain DB", e);
            }
            Ok(mut storage) => {
                // initialize() is idempotent — just ensures tables exist
                if storage.initialize().is_ok() {
                    let brain_config =
                        crate::engram::config_toml::load_config_from_toml(&memory_home);
                    let desired_model = &brain_config.embedding_model;
                    let active_model = storage
                        .get_metadata("embedding_model_active")
                        .unwrap_or(None);

                    // Model status
                    match active_model.as_deref() {
                        None => {
                            check_warn(
                                "Model active",
                                "none — run `memoryco generate` to build embeddings",
                            );
                        }
                        Some(active) if active != desired_model.as_str() => {
                            check_warn(
                                "Model mismatch",
                                &format!(
                                    "config says {:?} but active is {:?} \
                                     — run `memoryco generate`",
                                    desired_model, active
                                ),
                            );
                        }
                        Some(active) => {
                            check("Model active", active, true);
                        }
                    }

                    let with_emb = storage.count_with_embeddings().unwrap_or(0);
                    let without_emb = storage.count_without_embeddings().unwrap_or(0);
                    let total_engrams = with_emb + without_emb;
                    let enrichments = storage.count_enrichments().unwrap_or(0);

                    // Embedding coverage
                    let embed_ok = without_emb == 0 && total_engrams > 0;
                    let embed_detail = if total_engrams == 0 {
                        "no engrams yet".to_string()
                    } else {
                        format!("{}/{} engrams", with_emb, total_engrams)
                    };
                    let embed_suffix = if without_emb > 0 {
                        format!(
                            "{} — run `memoryco generate --embeddings`",
                            embed_detail
                        )
                    } else {
                        embed_detail
                    };
                    check("Embeddings", &embed_suffix, embed_ok || total_engrams == 0);

                    // Enrichment coverage
                    let enrich_ok = enrichments > 0 || total_engrams == 0;
                    let enrich_detail = if total_engrams == 0 {
                        "no engrams yet".to_string()
                    } else if enrichments == 0 {
                        format!(
                            "0/{} engrams — run `memoryco generate --enrichments`",
                            total_engrams
                        )
                    } else {
                        format!("{} vector(s) across {} engram(s)", enrichments, with_emb)
                    };
                    check("Enrichments", &enrich_detail, enrich_ok);
                }
            }
        }
    }

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
        "brain.db",
        "brain.db-wal",
        "brain.db-shm",
        "identity.db",
        "identity.db-wal",
        "identity.db-shm",
    ];

    eprintln!("⚠️  This will permanently delete all memories and identity.");
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
        Ok(result) => match result.action {
            memoryco_updater::UpdateAction::Applied(_) => {
                eprintln!("✓ Updated to version {}", result.new_version);
            }
            memoryco_updater::UpdateAction::AlreadyLatest => {
                eprintln!("✓ Already on the latest version ({})", current);
            }
            memoryco_updater::UpdateAction::Staged(_) => {
                eprintln!(
                    "⬇ Update {} staged. Will apply on next restart.",
                    result.new_version
                );
            }
        },
        Err(e) => {
            eprintln!("✗ Update failed: {}", e);
            eprintln!("  You can update manually: curl -fsSL https://memoryco.ai/install.sh | sh");
            std::process::exit(1);
        }
    }
}

/// `memoryco llm` — local LLM debug commands.
pub fn llm(command: LlmCommand) {
    match command {
        LlmCommand::Status => llm_status(),
        LlmCommand::Expand {
            query,
            max_variants,
        } => llm_expand(&query, max_variants),
        LlmCommand::GeneratePairs { memory, count } => llm_generate_pairs(&memory, count),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn llm_status() {
    let memory_home = config::get_memory_home();
    match crate::llm::LlmConfig::load(&memory_home) {
        Ok(llm_config) => {
            println!("enabled: {}", llm_config.enabled);
            match llm_config.model_path {
                Some(model_path) => println!("configured_model: {}", model_path.display()),
                None => println!("configured_model: (none)"),
            }
        }
        Err(error) => {
            println!("enabled: unknown");
            println!("config_error: {error:?}");
        }
    }

    match crate::llm::build_llm_service(&memory_home) {
        Ok(service) => {
            println!("available: {}", service.available());
            println!("tier: {:?}", service.tier());
            println!("model: {}", service.model_name());
        }
        Err(error) => {
            println!("available: false");
            println!("init_error: {error:?}");
        }
    }
}

fn llm_expand(query: &str, max_variants: usize) {
    let service = load_llm_service();
    if !service.available() {
        eprintln!("{}", llm_unavailable_hint(&config::get_memory_home()));
        std::process::exit(1);
    }

    match service.expand_query(query, max_variants) {
        Ok(variants) => {
            eprintln!("model: {}", service.model_name());
            for variant in variants {
                println!("{variant}");
            }
        }
        Err(error) => {
            eprintln!("Failed to expand query: {error:?}");
            std::process::exit(1);
        }
    }
}

fn llm_generate_pairs(memory: &str, count: usize) {
    let service = load_llm_service();
    if !service.available() {
        eprintln!("{}", llm_unavailable_hint(&config::get_memory_home()));
        std::process::exit(1);
    }

    match service.generate_training_queries(memory, count) {
        Ok(queries) => {
            eprintln!("model: {}", service.model_name());
            for query in queries {
                println!("{query}");
            }
        }
        Err(error) => {
            eprintln!("Failed to generate query pairs: {error:?}");
            std::process::exit(1);
        }
    }
}

fn load_llm_service() -> crate::llm::SharedLlmService {
    let memory_home = config::get_memory_home();
    match crate::llm::build_llm_service(&memory_home) {
        Ok(service) => service,
        Err(error) => {
            eprintln!("Failed to initialize local LLM: {error:?}");
            std::process::exit(1);
        }
    }
}

fn llm_unavailable_hint(memory_home: &Path) -> String {
    format!(
        "Local LLM unavailable. Check {config} and set llm_enable = true plus llm_model = \"<path or filename>\". Bare filenames resolve under {models}.",
        config = memory_home.join("config.toml").display(),
        models = crate::llm::LlmConfig::managed_model_dir(memory_home).display(),
    )
}

/// Print a doctor check line.
fn check(label: &str, detail: &str, ok: bool) {
    let icon = if ok { "✓" } else { "✗" };
    eprintln!("{} {:20} {}", icon, label, detail);
}

/// Print a doctor warning line (neither pass nor fail, just advisory).
fn check_warn(label: &str, detail: &str) {
    eprintln!("⚠ {:20} {}", label, detail);
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
        println!(
            "Pruned {} stale entr{}.",
            pruned,
            if pruned == 1 { "y" } else { "ies" }
        );
    }
}

/// Prompt the user for their MEMORY_HOME location.
///
/// Shows the default (`~/.memoryco`) and accepts it on Enter.
/// Loops until a valid path is obtained or created.
/// With `--yes`, returns the default without prompting.
fn prompt_memory_home(yes: bool) -> PathBuf {
    let default = dirs::home_dir()
        .map(|h| h.join(".memoryco"))
        .unwrap_or_else(|| PathBuf::from("/root/.memoryco"));

    if yes {
        return default;
    }

    loop {
        eprint!("Memory storage location [{}]: ", default.display());
        io::stderr().flush().ok();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            return default;
        }

        let trimmed = input.trim();
        let path = if trimmed.is_empty() {
            default.clone()
        } else {
            expand_tilde_path(trimmed)
        };

        if path.exists() {
            return path;
        }

        if prompt_yes_no(&format!("  {} does not exist. Create it?", path.display())) {
            match std::fs::create_dir_all(&path) {
                Ok(()) => {
                    eprintln!("  ✓ Created {}", path.display());
                    return path;
                }
                Err(e) => {
                    eprintln!("  ✗ Failed to create {}: {}", path.display(), e);
                    // loop back and ask again
                }
            }
        }
        // user said no or creation failed — prompt again
    }
}

/// Expand a leading `~/` to the user's home directory.
fn expand_tilde_path(s: &str) -> PathBuf {
    if s == "~" {
        return dirs::home_dir().unwrap_or_else(|| PathBuf::from(s));
    }
    if let Some(rest) = s.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(s)
}

/// Download all bundled GGUF models into `memory_home/cache/models/`.
///
/// Skips models whose file already exists with the expected size.
/// Prints an error and continues on failure.
fn download_bundled_models(memory_home: &Path) {
    let model_dir = memory_home.join("cache").join("models");

    if let Err(e) = std::fs::create_dir_all(&model_dir) {
        eprintln!("✗ Failed to create model cache dir {}: {}", model_dir.display(), e);
        return;
    }

    eprintln!("Model cache: {}", model_dir.display());

    for model in memoryco_llm::BUNDLED_MODELS {
        let dest = model_dir.join(model.filename);

        // Skip if already downloaded with correct size
        if dest.exists() {
            if let Ok(meta) = dest.metadata() {
                if meta.len() == model.size_bytes {
                    eprintln!("✓ {} — already downloaded", model.filename);
                    continue;
                }
            }
            // File exists but size is wrong — re-download
        }

        let size_mb = model.size_bytes / 1_000_000;
        eprint!("  Downloading {} ({} MB)...", model.filename, size_mb);
        io::stderr().flush().ok();

        if let Err(e) = download_file(model.url, &dest, model.size_bytes) {
            eprintln!(" ✗ {}", e);
        } else {
            eprintln!(" ✓ done");
        }
    }
}

/// Download `url` to `dest`, printing percentage progress to stderr.
///
/// Downloads to a `.tmp` file first, then renames on success to avoid
/// leaving partial files at the final path.
fn download_file(url: &str, dest: &Path, expected_bytes: u64) -> Result<(), String> {
    let mut response = reqwest::blocking::get(url)
        .map_err(|e| format!("request failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("HTTP {}", response.status()));
    }

    let tmp = dest.with_extension("gguf.tmp");
    let mut file = std::fs::File::create(&tmp)
        .map_err(|e| format!("create temp file: {e}"))?;

    let mut buf = [0u8; 65536];
    let mut downloaded: u64 = 0;
    let mut last_pct: u64 = 101; // force first print

    loop {
        let n = response
            .read(&mut buf)
            .map_err(|e| format!("read: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| format!("write: {e}"))?;
        downloaded += n as u64;

        let pct = if expected_bytes > 0 {
            (downloaded * 100) / expected_bytes
        } else {
            0
        };
        if pct != last_pct {
            eprint!("\r  Downloading {} ({} MB)... {}%   ",
                dest.file_name().and_then(|n| n.to_str()).unwrap_or(""),
                expected_bytes / 1_000_000,
                pct);
            io::stderr().flush().ok();
            last_pct = pct;
        }
    }

    // Final newline is printed by caller after this returns Ok
    drop(file);
    std::fs::rename(&tmp, dest).map_err(|e| format!("rename: {e}"))?;
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_tmp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("memoryco-{prefix}-{nanos}"))
    }

    // ── expand_tilde_path ──────────────────────────────────────────────────────

    #[test]
    fn tilde_alone_expands_to_home() {
        let result = expand_tilde_path("~");
        let home = dirs::home_dir().expect("home dir must exist");
        assert_eq!(result, home);
    }

    #[test]
    fn tilde_slash_expands_to_home_subdir() {
        let result = expand_tilde_path("~/.memoryco");
        let home = dirs::home_dir().expect("home dir must exist");
        assert_eq!(result, home.join(".memoryco"));
    }

    #[test]
    fn tilde_slash_nested_path() {
        let result = expand_tilde_path("~/foo/bar/baz");
        let home = dirs::home_dir().expect("home dir must exist");
        assert_eq!(result, home.join("foo/bar/baz"));
    }

    #[test]
    fn absolute_path_unchanged() {
        let result = expand_tilde_path("/tmp/memoryco");
        assert_eq!(result, PathBuf::from("/tmp/memoryco"));
    }

    #[test]
    fn relative_path_returned_as_is() {
        let result = expand_tilde_path("relative/path");
        assert_eq!(result, PathBuf::from("relative/path"));
    }

    #[test]
    fn tilde_in_middle_is_not_expanded() {
        let result = expand_tilde_path("/foo/~/bar");
        assert_eq!(result, PathBuf::from("/foo/~/bar"));
    }

    // ── download skip logic ────────────────────────────────────────────────────

    /// The skip predicate (file exists AND size matches) should trigger correctly.
    ///
    /// Uses a small synthetic size to avoid allocating hundreds of MB in a test.
    #[test]
    fn skip_predicate_triggers_when_size_matches() {
        let dir = unique_tmp_dir("skip");
        std::fs::create_dir_all(&dir).expect("create dir");

        let dest = dir.join("fake-model.gguf");
        let expected_size: u64 = 1024;
        let fake_data = vec![0u8; expected_size as usize];
        std::fs::write(&dest, &fake_data).expect("write fake model");

        // Replicate the skip predicate from download_bundled_models
        let should_skip = dest.exists()
            && dest
                .metadata()
                .map(|m| m.len() == expected_size)
                .unwrap_or(false);

        assert!(should_skip, "skip predicate must trigger for correct-size file");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A model file that exists but has the wrong size should NOT be skipped.
    #[test]
    fn skip_predicate_does_not_trigger_on_wrong_size() {
        let dir = unique_tmp_dir("wrong-size");
        std::fs::create_dir_all(&dir).expect("create dir");

        let dest = dir.join("fake-model.gguf");
        let expected_size: u64 = 1_000_000;

        // Write a partial (wrong-size) file
        std::fs::write(&dest, b"partial download data").expect("write partial");

        // Replicate the skip predicate from download_bundled_models
        let should_skip = dest.exists()
            && dest
                .metadata()
                .map(|m| m.len() == expected_size)
                .unwrap_or(false);

        assert!(!should_skip, "skip predicate must not trigger for wrong-size file");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The cache/models directory should be created when it is absent.
    ///
    /// Exercises only the mkdir logic, not any HTTP calls.
    #[test]
    fn cache_models_dir_created_when_missing() {
        let home = unique_tmp_dir("mkdir");
        std::fs::create_dir_all(&home).expect("create home");

        let model_dir = home.join("cache").join("models");
        assert!(!model_dir.exists(), "model dir should not exist yet");

        // Replicate the mkdir step from download_bundled_models
        std::fs::create_dir_all(&model_dir).expect("create model dir");

        assert!(model_dir.exists(), "cache/models dir must exist after create_dir_all");
        std::fs::remove_dir_all(&home).ok();
    }

    /// If the cache/models dir already exists, create_dir_all is a no-op.
    #[test]
    fn cache_models_dir_existing_is_noop() {
        let home = unique_tmp_dir("mkdir-existing");
        let model_dir = home.join("cache").join("models");
        std::fs::create_dir_all(&model_dir).expect("pre-create model dir");

        // Should not panic or error
        std::fs::create_dir_all(&model_dir).expect("create_dir_all on existing dir");
        assert!(model_dir.exists());
        std::fs::remove_dir_all(&home).ok();
    }

    // ── progress percentage calculation ───────────────────────────────────────

    #[test]
    fn progress_pct_at_zero_bytes() {
        let downloaded: u64 = 0;
        let total: u64 = 1_000_000;
        let pct = (downloaded * 100) / total;
        assert_eq!(pct, 0);
    }

    #[test]
    fn progress_pct_at_half() {
        let downloaded: u64 = 500_000;
        let total: u64 = 1_000_000;
        let pct = (downloaded * 100) / total;
        assert_eq!(pct, 50);
    }

    #[test]
    fn progress_pct_at_full() {
        let downloaded: u64 = 1_000_000;
        let total: u64 = 1_000_000;
        let pct = (downloaded * 100) / total;
        assert_eq!(pct, 100);
    }

    #[test]
    fn progress_pct_guards_zero_denominator() {
        let downloaded: u64 = 42;
        let total: u64 = 0;
        // Guard used in download_file: if expected_bytes > 0 { ... } else { 0 }
        let pct = if total > 0 { (downloaded * 100) / total } else { 0 };
        assert_eq!(pct, 0);
    }
}
