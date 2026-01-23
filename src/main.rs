//! Memory MCP Server - Cognitive AI memory powered by engram
//!
//! This server provides MCP tools for a neural memory system with:
//! - Organic decay (memories fade without use)
//! - Associative learning (Hebbian "neurons that fire together wire together")
//! - Identity layer (persona, values, preferences - never decays)
//! - Substrate layer (episodic/semantic memories with energy states)

mod tools;
mod bootstrap;
mod engram;
mod reference;
mod lenses;

use crate::engram::{Brain, SqliteStorage, Storage};
use crate::reference::ReferenceManager;
use sml_mcps::{Server, ServerConfig, StdioTransport};
use std::path::PathBuf;
use std::sync::Mutex;

/// Server context containing the Brain.
/// Wrapped in Mutex because Brain is not Sync.
pub struct Context {
    pub brain: Mutex<Brain>,
    pub lenses_dir: PathBuf,
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
    let lenses_dir = memory_home.join("lenses");
    let references_dir = memory_home.join("references");

    // Ensure directories exist
    std::fs::create_dir_all(&memory_home).ok();
    std::fs::create_dir_all(&lenses_dir).ok();
    std::fs::create_dir_all(&references_dir).ok();

    eprintln!("Memory home: {}", memory_home.display());
    eprintln!("  Database: {}", db_path.display());
    eprintln!("  Lenses: {}", lenses_dir.display());
    eprintln!("  References: {}", references_dir.display());

    // Open or create the brain
    let mut storage = SqliteStorage::new(&db_path)
        .expect("Failed to open database");
    
    // Initialize schema (creates FTS table if needed)
    storage.initialize().expect("Failed to initialize database");
    
    // Check if FTS index needs rebuilding (for databases migrating to FTS)
    if storage.fts_needs_rebuild().unwrap_or(false) {
        eprintln!("Rebuilding FTS search index...");
        match storage.rebuild_fts_index() {
            Ok(count) => eprintln!("Rebuilt FTS index with {} memories", count),
            Err(e) => eprintln!("Warning: Failed to rebuild FTS index: {}", e),
        }
    }
    
    let mut brain = Brain::open(storage)
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
    if let Err(e) = bootstrap::bootstrap_all(&mut brain, &lenses_dir, &references) {
        eprintln!("Warning: Bootstrap failed: {}", e);
    }

    let context = Context {
        brain: Mutex::new(brain),
        lenses_dir: lenses_dir.clone(),
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
    server.add_tool(tools::IdentitySetTool).expect("Failed to add identity_set tool");
    server.add_tool(tools::IdentitySearchTool).expect("Failed to add identity_search tool");

    // Register config tools
    server.add_tool(tools::ConfigGetTool).expect("Failed to add config_get tool");
    server.add_tool(tools::ConfigSetTool).expect("Failed to add config_set tool");

    // Register lens tools
    server.add_tool(tools::LensesListTool).expect("Failed to add lenses_list tool");
    server.add_tool(tools::LensesGetTool).expect("Failed to add lenses_get tool");

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
