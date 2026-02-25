//! Server initialization and startup.
//!
//! Contains all the heavy lifting: opening databases, applying decay,
//! backfilling embeddings, registering tools, and running the MCP server.

use crate::config;
use crate::embedding::EmbeddingGenerator;
use crate::engram::Brain;
use crate::identity::{IdentityStore, DieselIdentityStorage};
use crate::plans::{PlanStore, DieselPlanStorage};
use crate::reference::ReferenceManager;
use crate::{bootstrap, lenses, tools, Context};
use sml_mcps::{Server, ServerConfig, StdioTransport};
use std::sync::{Arc, Mutex};

/// Initialize all state and run the MCP server over stdio.
///
/// This is the main entry point for `memoryco serve` (and the default command).
/// It opens databases, applies maintenance, registers tools, and blocks on stdio.
pub fn run() {
    let memory_home = config::get_memory_home();
    let db_path = memory_home.join("brain.db");
    let identity_db_path = memory_home.join("identity.db");
    let plans_db_path = memory_home.join("plans.db");
    let lenses_dir = memory_home.join("lenses");
    let references_dir = memory_home.join("references");

    // Ensure directories exist
    std::fs::create_dir_all(&memory_home).ok();
    std::fs::create_dir_all(&lenses_dir).ok();
    std::fs::create_dir_all(&references_dir).ok();
    std::fs::create_dir_all(config::get_model_cache_dir()).ok();

    // Register our MEMORY_HOME in the shared registry
    crate::registry::ensure_registered(&memory_home);

    eprintln!("Memory home: {}", memory_home.display());
    eprintln!("  Database: {}", db_path.display());
    eprintln!("  Identity DB: {}", identity_db_path.display());
    eprintln!("  Plans DB: {}", plans_db_path.display());
    eprintln!("  Lenses: {}", lenses_dir.display());
    eprintln!("  References: {}", references_dir.display());

    // --- Brain ---
    let mut brain = Brain::open_path(&db_path)
        .expect("Failed to open brain");

    apply_maintenance(&mut brain);
    backfill_embeddings(&mut brain);
    bootstrap_associations(&mut brain);

    // --- Identity ---
    let identity_storage = DieselIdentityStorage::open(&identity_db_path)
        .expect("Failed to open identity database");
    let mut identity = IdentityStore::new(identity_storage)
        .expect("Failed to open identity store");

    migrate_identity(&brain, &mut identity);

    // --- Plans ---
    let plans_storage = DieselPlanStorage::open(&plans_db_path)
        .expect("Failed to open plans database");
    let plans = PlanStore::new(plans_storage)
        .expect("Failed to open plans store");

    // --- References ---
    let mut references = ReferenceManager::new();
    match references.load_directory(&references_dir) {
        Ok(loaded) if !loaded.is_empty() => {
            eprintln!("Loaded {} reference source(s): {}", loaded.len(), loaded.join(", "));
        }
        Ok(_) => {}
        Err(e) => eprintln!("Warning: Failed to load references: {}", e),
    }

    // --- Bootstrap ---
    // Instructions go into identity.db so identity_get always returns them,
    // even on a fresh install before the user has configured anything.
    if let Err(e) = bootstrap::bootstrap_all(&mut identity, &lenses_dir, &references, &memory_home) {
        eprintln!("Warning: Bootstrap failed: {}", e);
    }

    // --- Build shared state ---
    let brain = Arc::new(Mutex::new(brain));
    let identity = Arc::new(Mutex::new(identity));
    let references = Arc::new(Mutex::new(references));

    // --- Build context ---
    let context = Context {
        brain: Arc::clone(&brain),
        identity: Arc::clone(&identity),
        plans: Arc::new(Mutex::new(plans)),
        references: Arc::clone(&references),
        lenses_dir: lenses_dir.clone(),
        memory_home: memory_home.clone(),
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

/// Apply time-based decay and prune weak associations.
fn apply_maintenance(brain: &mut Brain) {
    match brain.apply_time_decay() {
        Ok(true) => eprintln!("Applied time-based decay"),
        Ok(false) => eprintln!("No decay needed (interval not elapsed)"),
        Err(e) => eprintln!("Warning: Failed to apply decay: {}", e),
    }

    match brain.prune_weak_associations() {
        Ok(0) => {}
        Ok(count) => eprintln!("Pruned {} weak associations (below {} threshold)",
            count, brain.config().min_association_weight),
        Err(e) => eprintln!("Warning: Failed to prune associations: {}", e),
    }
}

/// Generate embeddings for any memories missing them.
fn backfill_embeddings(brain: &mut Brain) {
    match brain.count_without_embeddings() {
        Ok(0) => {}
        Ok(count) => {
            eprintln!("Generating embeddings for {} memories...", count);
            let generator = EmbeddingGenerator::new();
            let mut processed = 0;
            let mut errors = 0;

            loop {
                match brain.get_ids_without_embeddings(50) {
                    Ok(ids) if ids.is_empty() => break,
                    Ok(ids) => {
                        let items: Vec<_> = ids.iter()
                            .filter_map(|id| brain.get(id).map(|e| (*id, e.content.clone())))
                            .collect();
                        let texts: Vec<&str> = items.iter().map(|(_, c)| c.as_str()).collect();

                        match generator.generate_batch(&texts) {
                            Ok(embeddings) => {
                                for ((id, _), embedding) in items.iter().zip(embeddings.iter()) {
                                    if brain.set_embedding(id, embedding).is_ok() {
                                        processed += 1;
                                    } else {
                                        errors += 1;
                                    }
                                }
                            }
                            Err(_) => errors += items.len(),
                        }
                        eprint!("\r  Processed {}/{} memories...", processed, count);
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to get memories for embedding: {}", e);
                        break;
                    }
                }
            }
            eprintln!("\r  Generated {} embeddings ({} errors)        ", processed, errors);
        }
        Err(e) => eprintln!("Warning: Failed to check embedding status: {}", e),
    }
}

/// Create semantic associations between similar memories.
fn bootstrap_associations(brain: &mut Brain) {
    match brain.bootstrap_semantic_associations(0.5, 5) {
        Ok((0, _)) => {}
        Ok((created, _)) => eprintln!("Created {} semantic associations", created),
        Err(e) => eprintln!("Warning: Failed to bootstrap semantic associations: {}", e),
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

/// Create the MCP server with all tools registered.
fn build_server() -> Server<Context> {
    let mut server = Server::new(ServerConfig {
        name: "memory".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        ..Default::default()
    });

    // Engram tools
    server.add_tool(tools::EngramCreateTool).expect("engram_create");
    server.add_tool(tools::EngramRecallTool).expect("engram_recall");
    server.add_tool(tools::EngramSearchTool).expect("engram_search");
    server.add_tool(tools::EngramGetTool).expect("engram_get");
    server.add_tool(tools::EngramDeleteTool).expect("engram_delete");
    server.add_tool(tools::EngramAssociateTool).expect("engram_associate");
    server.add_tool(tools::EngramStatsTool).expect("engram_stats");
    server.add_tool(tools::EngramAssociationsTool).expect("engram_associations");
    server.add_tool(tools::EngramGraphTool).expect("engram_graph");

    // Identity tools
    server.add_tool(tools::IdentityGetTool).expect("identity_get");
    server.add_tool(tools::IdentitySearchTool).expect("identity_search");
    server.add_tool(tools::IdentityListTool).expect("identity_list");
    server.add_tool(tools::IdentityRemoveTool).expect("identity_remove");
    server.add_tool(tools::IdentitySetPersonaNameTool).expect("identity_set_persona_name");
    server.add_tool(tools::IdentitySetPersonaDescriptionTool).expect("identity_set_persona_description");
    server.add_tool(tools::IdentityAddTraitTool).expect("identity_add_trait");
    server.add_tool(tools::IdentityAddExpertiseTool).expect("identity_add_expertise");
    server.add_tool(tools::IdentityAddInstructionTool).expect("identity_add_instruction_v2");
    server.add_tool(tools::IdentityAddToneTool).expect("identity_add_tone");
    server.add_tool(tools::IdentityAddDirectiveTool).expect("identity_add_directive");
    server.add_tool(tools::IdentityAddValueTool).expect("identity_add_value");
    server.add_tool(tools::IdentityAddPreferenceTool).expect("identity_add_preference");
    server.add_tool(tools::IdentityAddRelationshipTool).expect("identity_add_relationship");
    server.add_tool(tools::IdentityAddAntipatternTool).expect("identity_add_antipattern");
    server.add_tool(tools::IdentitySetupTool).expect("identity_setup");

    // Config tools
    server.add_tool(tools::ConfigGetTool).expect("config_get");
    server.add_tool(tools::ConfigSetTool).expect("config_set");

    // Lens tools
    server.add_tool(tools::LensesListTool).expect("lenses_list");
    server.add_tool(tools::LensesGetTool).expect("lenses_get");

    // Reference tools
    server.add_tool(tools::ReferenceListTool).expect("reference_list");
    server.add_tool(tools::ReferenceSearchTool).expect("reference_search");
    server.add_tool(tools::ReferenceGetTool).expect("reference_get");
    server.add_tool(tools::ReferenceSectionsTool).expect("reference_sections");
    server.add_tool(tools::ReferenceCitationTool).expect("reference_citation");

    // Plan tools
    server.add_tool(tools::PlansListTool).expect("plans");
    server.add_tool(tools::PlanGetTool).expect("plan_get");
    server.add_tool(tools::PlanStartTool).expect("plan_start");
    server.add_tool(tools::PlanStopTool).expect("plan_stop");
    server.add_tool(tools::StepAddTool).expect("step_add");
    server.add_tool(tools::StepCompleteTool).expect("step_complete");

    // UI tools
    server.add_tool(tools::OpenDashboardTool).expect("open_dashboard");

    server
}
