//! Memory MCP Server - Cognitive AI memory powered by engram
//!
//! This server provides MCP tools for a neural memory system with:
//! - Organic decay (memories fade without use)
//! - Associative learning (Hebbian "neurons that fire together wire together")
//! - Identity layer (persona, values, preferences - never decays)
//! - Substrate layer (episodic/semantic memories with energy states)

mod tools;
mod bootstrap;
mod embedding;
mod engram;
mod identity;
mod plans;
mod reference;
mod lenses;
mod storage;

use crate::embedding::EmbeddingGenerator;
use crate::engram::Brain;
use crate::identity::{IdentityStore, DieselIdentityStorage};
use crate::plans::{PlanStore, DieselPlanStorage};
use crate::reference::ReferenceManager;
use sml_mcps::{Server, ServerConfig, StdioTransport};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Server context containing the Brain and ReferenceManager.
/// Brain is Arc<Mutex<>> to allow background tasks (like embedding generation).
/// References is plain Mutex since it's only used synchronously.
pub struct Context {
    pub brain: Arc<Mutex<Brain>>,
    pub identity: Arc<Mutex<IdentityStore>>,
    pub plans: Arc<Mutex<PlanStore>>,
    pub references: Mutex<ReferenceManager>,
    pub lenses_dir: PathBuf,
    pub memory_home: PathBuf,
}

/// Get the memory home directory from MEMORY_HOME env var or default
fn get_memory_home() -> PathBuf {
    std::env::var("MEMORY_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("memory")
        })
}

fn main() {
    // All paths derived from MEMORY_HOME
    let memory_home = get_memory_home();
    let db_path = memory_home.join("brain.db");
    let identity_db_path = memory_home.join("identity.db");
    let plans_db_path = memory_home.join("plans.db");
    let lenses_dir = memory_home.join("lenses");
    let references_dir = memory_home.join("references");

    // Ensure directories exist
    std::fs::create_dir_all(&memory_home).ok();
    std::fs::create_dir_all(&lenses_dir).ok();
    std::fs::create_dir_all(&references_dir).ok();

    eprintln!("Memory home: {}", memory_home.display());
    eprintln!("  Database: {}", db_path.display());
    eprintln!("  Identity DB: {}", identity_db_path.display());
    eprintln!("  Plans DB: {}", plans_db_path.display());
    eprintln!("  Lenses: {}", lenses_dir.display());
    eprintln!("  References: {}", references_dir.display());

    // Open or create the brain (with async persistence support)
    let mut brain = Brain::open_path(&db_path)
        .expect("Failed to open brain");

    // Apply any decay that accumulated while server was offline
    match brain.apply_time_decay() {
        Ok(true) => eprintln!("Applied time-based decay"),
        Ok(false) => eprintln!("No decay needed (interval not elapsed)"),
        Err(e) => eprintln!("Warning: Failed to apply decay: {}", e),
    }

    // Prune weak associations (cleanup from decay)
    match brain.prune_weak_associations() {
        Ok(0) => {} // Nothing pruned, stay quiet
        Ok(count) => eprintln!("Pruned {} weak associations (below {} threshold)", 
            count, brain.config().min_association_weight),
        Err(e) => eprintln!("Warning: Failed to prune associations: {}", e),
    }

    // Backfill embeddings for memories that don't have them
    match brain.count_without_embeddings() {
        Ok(0) => {} // All memories have embeddings
        Ok(count) => {
            eprintln!("Generating embeddings for {} memories...", count);
            let generator = EmbeddingGenerator::new();
            let mut processed = 0;
            let mut errors = 0;
            
            // Process in batches of 50
            loop {
                match brain.get_ids_without_embeddings(50) {
                    Ok(ids) if ids.is_empty() => break,
                    Ok(ids) => {
                        // Collect contents for batch embedding
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

    // Bootstrap semantic associations for memories that have embeddings
    // This creates associations between semantically similar memories
    match brain.bootstrap_semantic_associations(0.5, 5) {
        Ok((0, _)) => {} // No new associations needed
        Ok((created, _)) => eprintln!("Created {} semantic associations", created),
        Err(e) => eprintln!("Warning: Failed to bootstrap semantic associations: {}", e),
    }

    // Open or create the identity store
    let identity_storage = DieselIdentityStorage::open(&identity_db_path)
        .expect("Failed to open identity database");
    let mut identity = IdentityStore::new(identity_storage)
        .expect("Failed to open identity store");

    // Migrate identity from old JSON blob to new flat storage (one-time)
    // brain.identity() returns the Identity loaded from old JSON blob storage
    let old_identity = brain.identity();
    match identity.migrate_from_identity(old_identity) {
        Ok(crate::identity::MigrationResult::Migrated { items }) => {
            eprintln!("Migrated {} identity items to new storage", items);
        }
        Ok(crate::identity::MigrationResult::AlreadyMigrated) => {
            // Already migrated, nothing to do
        }
        Err(e) => eprintln!("Warning: Failed to migrate identity: {}", e),
    }

    // Open or create the plans store
    let plans_storage = DieselPlanStorage::open(&plans_db_path)
        .expect("Failed to open plans database");
    let plans = PlanStore::new(plans_storage)
        .expect("Failed to open plans store");

    // Load reference sources
    let mut references = ReferenceManager::new();
    match references.load_directory(&references_dir) {
        Ok(loaded) if !loaded.is_empty() => {
            eprintln!("Loaded {} reference source(s): {}", loaded.len(), loaded.join(", "));
        }
        Ok(_) => {} // No references, that's fine
        Err(e) => eprintln!("Warning: Failed to load references: {}", e),
    }

    // Bootstrap all modules (adds instructions to identity if not present)
    if let Err(e) = bootstrap::bootstrap_all(&mut brain, &lenses_dir, &references, &plans) {
        eprintln!("Warning: Bootstrap failed: {}", e);
    }

    let context = Context {
        brain: Arc::new(Mutex::new(brain)),
        identity: Arc::new(Mutex::new(identity)),
        plans: Arc::new(Mutex::new(plans)),
        references: Mutex::new(references),
        lenses_dir: lenses_dir.clone(),
        memory_home: memory_home.clone(),
    };

    // Create MCP server
    let mut server = Server::new(ServerConfig {
        name: "memory".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        ..Default::default()
    });

    // Register engram tools
    server.add_tool(tools::EngramCreateTool).expect("Failed to add engram_create tool");
    server.add_tool(tools::EngramRecallTool).expect("Failed to add engram_recall tool");
    server.add_tool(tools::EngramSearchTool).expect("Failed to add engram_search tool");
    server.add_tool(tools::EngramGetTool).expect("Failed to add engram_get tool");
    server.add_tool(tools::EngramDeleteTool).expect("Failed to add engram_delete tool");
    server.add_tool(tools::EngramAssociateTool).expect("Failed to add engram_associate tool");
    server.add_tool(tools::EngramStatsTool).expect("Failed to add engram_stats tool");
    server.add_tool(tools::EngramAssociationsTool).expect("Failed to add engram_associations tool");
    server.add_tool(tools::EngramGraphTool).expect("Failed to add engram_graph tool");

    // Register identity tools
    server.add_tool(tools::IdentityGetTool).expect("Failed to add identity_get tool");
    server.add_tool(tools::IdentitySearchTool).expect("Failed to add identity_search tool");
    server.add_tool(tools::IdentityListTool).expect("Failed to add identity_list tool");
    server.add_tool(tools::IdentityRemoveTool).expect("Failed to add identity_remove tool");
    server.add_tool(tools::IdentitySetPersonaNameTool).expect("Failed to add identity_set_persona_name tool");
    server.add_tool(tools::IdentitySetPersonaDescriptionTool).expect("Failed to add identity_set_persona_description tool");
    server.add_tool(tools::IdentityAddTraitTool).expect("Failed to add identity_add_trait tool");
    server.add_tool(tools::IdentityAddExpertiseTool).expect("Failed to add identity_add_expertise tool");
    server.add_tool(tools::IdentityAddInstructionTool).expect("Failed to add identity_add_instruction tool");
    server.add_tool(tools::IdentityAddToneTool).expect("Failed to add identity_add_tone tool");
    server.add_tool(tools::IdentityAddDirectiveTool).expect("Failed to add identity_add_directive tool");
    server.add_tool(tools::IdentityAddValueTool).expect("Failed to add identity_add_value tool");
    server.add_tool(tools::IdentityAddPreferenceTool).expect("Failed to add identity_add_preference tool");
    server.add_tool(tools::IdentityAddRelationshipTool).expect("Failed to add identity_add_relationship tool");
    server.add_tool(tools::IdentityAddAntipatternTool).expect("Failed to add identity_add_antipattern tool");

    // Register config tools
    server.add_tool(tools::ConfigGetTool).expect("Failed to add config_get tool");
    server.add_tool(tools::ConfigSetTool).expect("Failed to add config_set tool");

    // Register lens tools
    server.add_tool(tools::LensesListTool).expect("Failed to add lenses_list tool");
    server.add_tool(tools::LensesGetTool).expect("Failed to add lenses_get tool");

    // Register reference tools
    server.add_tool(tools::ReferenceListTool).expect("Failed to add reference_list tool");
    server.add_tool(tools::ReferenceSearchTool).expect("Failed to add reference_search tool");
    server.add_tool(tools::ReferenceGetTool).expect("Failed to add reference_get tool");
    server.add_tool(tools::ReferenceSectionsTool).expect("Failed to add reference_sections tool");
    server.add_tool(tools::ReferenceCitationTool).expect("Failed to add reference_citation tool");

    // Register plan tools
    server.add_tool(tools::PlansListTool).expect("Failed to add plans tool");
    server.add_tool(tools::PlanGetTool).expect("Failed to add plan_get tool");
    server.add_tool(tools::PlanStartTool).expect("Failed to add plan_start tool");
    server.add_tool(tools::PlanStopTool).expect("Failed to add plan_stop tool");
    server.add_tool(tools::StepAddTool).expect("Failed to add step_add tool");
    server.add_tool(tools::StepCompleteTool).expect("Failed to add step_complete tool");

    // Load and register lenses as prompts
    let lenses_list = lenses::load_lenses(&lenses_dir);
    eprintln!("Loaded {} lens(es)", lenses_list.len());
    for lens in lenses_list {
        if let Err(e) = server.add_prompt(lens) {
            eprintln!("Warning: Failed to add lens prompt: {}", e);
        }
    }

    eprintln!("Memory server starting...");

    // Start the server (blocks forever)
    if let Err(e) = server.start(StdioTransport::new(), context) {
        eprintln!("Server error: {}", e);
        std::process::exit(1);
    }
}
