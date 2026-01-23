//! Memory MCP Server - Cognitive AI memory powered by engram
//!
//! This server provides MCP tools for a neural memory system with:
//! - Organic decay (memories fade without use)
//! - Associative learning (Hebbian "neurons that fire together wire together")
//! - Identity layer (persona, values, preferences - never decays)
//! - Substrate layer (episodic/semantic memories with energy states)

mod tools;
mod bootstrap;
mod plugin;
mod engram;
mod reference;
mod lenses;

pub use plugin::{MemoryPlugin, combine_instructions};

use crate::engram::{Brain, SqliteStorage, Storage};
use sml_mcps::{Server, ServerConfig, StdioTransport};
use std::path::PathBuf;
use std::sync::Mutex;

/// Server context containing the Brain.
/// Wrapped in Mutex because Brain is not Sync.
pub struct Context {
    pub brain: Mutex<Brain>,
}

fn get_default_db_path() -> PathBuf {
    // Use platform-appropriate data directory
    let data_dir = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."));
    
    let memory_dir = data_dir.join("memory");
    std::fs::create_dir_all(&memory_dir).ok();
    
    memory_dir.join("brain.db")
}

fn main() {
    // Get database path from env or use default
    let db_path = std::env::var("MEMORY_DB")
        .map(PathBuf::from)
        .unwrap_or_else(|_| get_default_db_path());

    eprintln!("Memory database: {}", db_path.display());

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

    // Bootstrap identity and memories if this is a fresh database
    if let Err(e) = bootstrap::bootstrap_if_needed(&mut brain) {
        eprintln!("Warning: Bootstrap failed: {}", e);
    }

    let context = Context {
        brain: Mutex::new(brain),
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

    eprintln!("Memory server starting...");

    // Start the server (blocks forever)
    if let Err(e) = server.start(StdioTransport::new(), context) {
        eprintln!("Server error: {}", e);
        std::process::exit(1);
    }
}
