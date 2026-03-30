//! Server initialization and startup.
//!
//! Contains all the heavy lifting: opening databases, applying decay,
//! registering tools, and running the MCP server.
//!
//! Embedding and enrichment generation is intentionally NOT done here.
//! Run `memoryco generate` to rebuild vectors after a model change.

use crate::config;
use crate::memory_core::Brain;
use crate::identity::{DieselIdentityStorage, IdentityStore};

use crate::reference::ReferenceManager;
use crate::{Context, bootstrap, lenses, tools};
use sml_mcps::{Server, ServerConfig, StdioTransport};
use std::sync::{Arc, Mutex, RwLock};

/// Initialize all state and run the MCP server over stdio.
///
/// This is the main entry point for `memoryco serve` (and the default command).
/// It opens databases, applies maintenance, registers tools, and blocks on stdio.
pub fn run() {
    // ── Auto-update: apply staged + background check ─────────────────────────────────
    {
        let updater = memoryco_updater::Updater::new();
        let my_name = "memoryco";
        let my_version = env!("CARGO_PKG_VERSION");

        // Apply any previously-staged update before we start.
        if let Ok(my_path) = std::env::current_exe() {
            match updater.apply_staged(my_name, my_version, &my_path) {
                Ok(Some(result)) => {
                    eprintln!(
                        "✓ Applied staged update: {} → {}",
                        my_version, result.new_version
                    );
                }
                Ok(None) => {} // nothing staged
                Err(e) => eprintln!("⚠ Staged update failed to apply: {}", e),
            }
        }

        // Background check for new updates (throttled to every 4 hours).
        let my_version = my_version.to_string();
        std::thread::spawn(move || {
            let updater = memoryco_updater::Updater::new();
            match updater.check_version("memoryco", &my_version) {
                Ok(check) if check.update_available => match updater.stage("memoryco") {
                    Ok(r) => eprintln!(
                        "⬇ Update v{} staged. Will apply on next restart.",
                        r.new_version
                    ),
                    Err(e) => eprintln!("⚠ Failed to stage update: {}", e),
                },
                Ok(_) => {}
                Err(memoryco_updater::UpdateError::Throttled { .. }) => {}
                Err(e) => eprintln!("⚠ Update check failed: {}", e),
            }
        });
    }

    // ── Normal server startup continues below ─────────────────────────────────────
    let memory_home = config::get_memory_home();
    let db_path = memory_home.join("brain.db");
    let identity_db_path = memory_home.join("identity.db");

    let lenses_dir = memory_home.join("lenses");
    let references_dir = memory_home.join("references");

    // Ensure directories exist
    std::fs::create_dir_all(&memory_home).ok();
    std::fs::create_dir_all(&lenses_dir).ok();
    std::fs::create_dir_all(&references_dir).ok();
    std::fs::create_dir_all(config::get_model_cache_dir()).ok();
    std::fs::create_dir_all(crate::llm::LlmConfig::managed_model_dir(&memory_home)).ok();

    // Register our MEMORY_HOME in the shared registry
    crate::registry::ensure_registered(&memory_home);

    eprintln!("Memory home: {}", memory_home.display());
    eprintln!("  Database: {}", db_path.display());
    eprintln!("  Identity DB: {}", identity_db_path.display());

    eprintln!("  Lenses: {}", lenses_dir.display());
    eprintln!("  References: {}", references_dir.display());

    let llm = match crate::llm::build_llm_service(&memory_home) {
        Ok(service) => {
            if service.available() {
                eprintln!(
                    "  Local LLM: {} ({:?})",
                    service.model_name(),
                    service.tier()
                );
            } else {
                eprintln!("  Local LLM: disabled");
            }
            service
        }
        Err(e) => {
            eprintln!("Warning: failed to initialize local LLM: {:?}", e);
            Arc::new(crate::llm::NoLlmService)
        }
    };

    // --- Config ---
    // Write default config.toml if missing, then load brain config from it.
    let config_toml_path = memory_home.join("config.toml");
    if !config_toml_path.exists() {
        if let Err(e) = crate::memory_core::config_toml::ensure_default_config_toml(&memory_home) {
            eprintln!("Warning: Failed to write default config.toml: {}", e);
        } else {
            eprintln!("  Config: wrote default config.toml");
        }
    }
    // Ensure LLM config keys exist (appended by the llm crate, not brain).
    if let Err(e) = crate::llm::ensure_llm_config_defaults(&memory_home) {
        eprintln!("Warning: Failed to write LLM config defaults: {}", e);
    }

    let mut brain_config = crate::memory_core::config_toml::load_config_from_toml(&memory_home);
    validate_config(&mut brain_config, &llm);
    eprintln!(
        "  Config: embedding_model={}, rerank_mode={}, hybrid_search={}",
        brain_config.embedding_model, brain_config.rerank_mode, brain_config.hybrid_search_enabled
    );

    // --- Brain ---
    let mut brain = Brain::open_path(&db_path, brain_config).expect("Failed to open brain");

    apply_maintenance(&mut brain);
    expire_sessions(&mut brain);

    // --- Identity ---
    let identity_storage =
        DieselIdentityStorage::open(&identity_db_path).expect("Failed to open identity database");
    let mut identity = IdentityStore::new(identity_storage).expect("Failed to open identity store");

    migrate_identity(&brain, &mut identity);

    // --- References ---
    let mut references = ReferenceManager::new();
    match references.load_directory(&references_dir) {
        Ok(loaded) if !loaded.is_empty() => {
            eprintln!(
                "Loaded {} reference source(s): {}",
                loaded.len(),
                loaded.join(", ")
            );
        }
        Ok(_) => {}
        Err(e) => eprintln!("Warning: Failed to load references: {}", e),
    }

    // --- Bootstrap ---
    // Instructions go into identity.db so identity_get always returns them,
    // even on a fresh install before the user has configured anything.
    if let Err(e) = bootstrap::bootstrap_all(&mut identity, &lenses_dir, &references, &memory_home)
    {
        eprintln!("Warning: Bootstrap failed: {}", e);
    }

    // --- Build shared state ---
    let brain = Arc::new(RwLock::new(brain));
    let identity = Arc::new(Mutex::new(identity));
    let references = Arc::new(Mutex::new(references));

    // --- Build context ---
    let context = Context {
        brain: Arc::clone(&brain),
        llm,
        identity: Arc::clone(&identity),

        references: Arc::clone(&references),
        lenses_dir: lenses_dir.clone(),
        memory_home: memory_home.clone(),
        last_search_query: Mutex::new(None),
        last_search_result_ids: Mutex::new(Vec::new()),
    };

    // --- Build and start server ---
    let mut server = build_server();

    let lenses_list = lenses::load_lenses(&lenses_dir);
    eprintln!("Loaded {} lens(es)", lenses_list.len());
    for lens in lenses_list {
        if let Err(e) = server.add_prompt(lens) {
            eprintln!("Warning: Failed to add lens prompt: {}", e);
        }
    }

    // --- Dashboard (shares state with MCP server) ---
    crate::dashboard::start_dashboard(brain, identity, references, &memory_home);

    eprintln!("Memory server starting...");

    if let Err(e) = server.start(StdioTransport::new(), context) {
        eprintln!("Server error: {}", e);
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Delete sessions that have not been accessed within `session_expire_days`.
///
/// Skipped if `session_expire_days` is 0 (expiry disabled).
fn expire_sessions(brain: &mut Brain) {
    let expire_days = brain.config().session_expire_days;
    if expire_days == 0 {
        return;
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let cutoff = now - (expire_days as i64 * 86400);

    match brain.delete_expired_sessions(cutoff) {
        Ok(0) => {}
        Ok(count) => eprintln!(
            "Expired {} session(s) not accessed in {} day(s)",
            count, expire_days
        ),
        Err(e) => eprintln!("Warning: Failed to expire sessions: {}", e),
    }
}

/// Apply time-based decay and prune weak associations.
fn apply_maintenance(brain: &mut Brain) {
    match brain.apply_time_decay() {
        Ok(true) => eprintln!("Applied time-based decay"),
        Ok(false) => eprintln!("No decay needed (interval not elapsed)"),
        Err(e) => eprintln!("Warning: Failed to apply decay: {}", e),
    }

    match brain.prune_weak_associations() {
        Ok(0) => {}
        Ok(count) => eprintln!(
            "Pruned {} weak associations (below {} threshold)",
            count,
            brain.config().min_association_weight
        ),
        Err(e) => eprintln!("Warning: Failed to prune associations: {}", e),
    }

    match brain.prune_orphan_associations() {
        Ok(0) => {}
        Ok(count) => eprintln!("Pruned {} orphan associations (dangling references)", count),
        Err(e) => eprintln!("Warning: Failed to prune orphan associations: {}", e),
    }
}


/// Migrate identity from old JSON blob to flat storage (one-time).
fn migrate_identity(brain: &Brain, identity: &mut IdentityStore) {
    let old_identity = brain.identity();
    match identity.migrate_from_identity(old_identity) {
        Ok(crate::identity::MigrationResult::Migrated { items }) => {
            eprintln!("Migrated {} identity items to new storage", items);
        }
        Ok(crate::identity::MigrationResult::AlreadyMigrated) => {}
        Err(e) => eprintln!("Warning: Failed to migrate identity: {}", e),
    }
}

/// Validate cross-crate config dependencies and degrade gracefully if needed.
///
/// Called after both the LLM service and brain config are loaded, before Brain::open_path().
/// Degrades LLM-dependent rerank mode to safe fallback when no LLM is available.
fn validate_config(config: &mut crate::memory_core::Config, llm: &crate::llm::SharedLlmService) {
    if !llm.available() && config.rerank_mode == "llm" {
        eprintln!(
            "⚠ rerank_mode=\"llm\" requires [llm] but none is available — degrading to \"off\""
        );
        config.rerank_mode = "off".to_string();
    }
}

/// Create the MCP server with all tools registered.
fn build_server() -> Server<Context> {
    let mut server = Server::new(ServerConfig {
        name: "memory".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        ..Default::default()
    });

    // Memory tools
    server
        .add_tool(tools::MemoryCreateTool)
        .expect("memory_create");
    server
        .add_tool(tools::MemoryRecallTool)
        .expect("memory_recall");
    server
        .add_tool(tools::MemorySearchTool)
        .expect("memory_search");
    server.add_tool(tools::MemoryGetTool).expect("memory_get");
    server
        .add_tool(tools::MemoryDeleteTool)
        .expect("memory_delete");
    server
        .add_tool(tools::MemoryAssociateTool)
        .expect("memory_associate");
    server
        .add_tool(tools::MemoryStatsTool)
        .expect("memory_stats");
    server
        .add_tool(tools::MemoryAssociationsTool)
        .expect("memory_associations");
    server
        .add_tool(tools::MemoryGraphTool)
        .expect("memory_graph");

    // Identity tools
    server
        .add_tool(tools::IdentityGetTool)
        .expect("identity_get");
    server
        .add_tool(tools::IdentitySearchTool)
        .expect("identity_search");
    server
        .add_tool(tools::IdentityListTool)
        .expect("identity_list");
    server
        .add_tool(tools::IdentityRemoveTool)
        .expect("identity_remove");
    server
        .add_tool(tools::IdentitySetPersonaNameTool)
        .expect("identity_set_persona_name");
    server
        .add_tool(tools::IdentitySetPersonaDescriptionTool)
        .expect("identity_set_persona_description");
    server
        .add_tool(tools::IdentityAddTraitTool)
        .expect("identity_add_trait");
    server
        .add_tool(tools::IdentityAddExpertiseTool)
        .expect("identity_add_expertise");
    server
        .add_tool(tools::IdentityAddInstructionTool)
        .expect("identity_add_instruction_v2");
    server
        .add_tool(tools::IdentityAddToneTool)
        .expect("identity_add_tone");
    server
        .add_tool(tools::IdentityAddDirectiveTool)
        .expect("identity_add_directive");
    server
        .add_tool(tools::IdentityAddValueTool)
        .expect("identity_add_value");
    server
        .add_tool(tools::IdentityAddPreferenceTool)
        .expect("identity_add_preference");
    server
        .add_tool(tools::IdentityAddRelationshipTool)
        .expect("identity_add_relationship");
    server
        .add_tool(tools::IdentityAddAntipatternTool)
        .expect("identity_add_antipattern");
    server
        .add_tool(tools::IdentitySetupTool)
        .expect("identity_setup");

    // Config tools
    server.add_tool(tools::ConfigGetTool).expect("config_get");
    server.add_tool(tools::ConfigSetTool).expect("config_set");

    // Lens tools
    server.add_tool(tools::LensesListTool).expect("lenses_list");
    server.add_tool(tools::LensesGetTool).expect("lenses_get");

    // Reference tools
    server
        .add_tool(tools::ReferenceListTool)
        .expect("reference_list");
    server
        .add_tool(tools::ReferenceSearchTool)
        .expect("reference_search");
    server
        .add_tool(tools::ReferenceGetTool)
        .expect("reference_get");
    server
        .add_tool(tools::ReferenceSectionsTool)
        .expect("reference_sections");
    server
        .add_tool(tools::ReferenceCitationTool)
        .expect("reference_citation");

    // Date tools
    server
        .add_tool(tools::DateResolveTool)
        .expect("date_resolve");

    // UI tools
    server
        .add_tool(tools::OpenDashboardTool)
        .expect("open_dashboard");

    server
}

#[cfg(test)]
mod tests {
    use super::*;
    use memoryco_llm::{LlmError, LlmService, LlmTier};
    use std::sync::Arc;

    struct MockLlmAvailable;

    impl LlmService for MockLlmAvailable {
        fn available(&self) -> bool {
            true
        }
        fn tier(&self) -> LlmTier {
            LlmTier::Standard
        }
        fn model_name(&self) -> &str {
            "mock"
        }
        fn expand_query(&self, _: &str, _: usize) -> Result<Vec<String>, LlmError> {
            Ok(vec![])
        }
        fn generate_training_queries(&self, _: &str, _: usize) -> Result<Vec<String>, LlmError> {
            Ok(vec![])
        }
        fn rerank(&self, _: &str, _: &[&str], _: usize) -> Result<Vec<usize>, LlmError> {
            Ok(vec![])
        }
    }

    fn no_llm() -> crate::llm::SharedLlmService {
        Arc::new(crate::llm::NoLlmService)
    }

    fn available_llm() -> crate::llm::SharedLlmService {
        Arc::new(MockLlmAvailable)
    }

    fn config_with_rerank(mode: &str) -> crate::memory_core::Config {
        let mut c = crate::memory_core::Config::default();
        c.rerank_mode = mode.to_string();
        c
    }

    #[test]
    fn llm_mode_degrades_to_off_when_no_llm() {
        let mut config = config_with_rerank("llm");
        validate_config(&mut config, &no_llm());
        assert_eq!(config.rerank_mode, "off");
    }

    #[test]
    fn off_unchanged_when_no_llm() {
        let mut config = config_with_rerank("off");
        validate_config(&mut config, &no_llm());
        assert_eq!(config.rerank_mode, "off");
    }

    #[test]
    fn llm_mode_unchanged_when_llm_available() {
        let mut config = config_with_rerank("llm");
        validate_config(&mut config, &available_llm());
        assert_eq!(config.rerank_mode, "llm");
    }
}
