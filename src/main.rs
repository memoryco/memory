//! Memory MCP Server — Cognitive AI memory powered by memory
//!
//! This server provides MCP tools for a neural memory system with:
//! - Organic decay (memories fade without use)
//! - Associative learning (Hebbian "neurons that fire together wire together")
//! - Identity layer (persona, values, preferences — never decays)
//! - Substrate layer (episodic/semantic memories with energy states)

mod bootstrap;
mod cli;
mod commands;
pub mod config;
mod dashboard;
mod embedding;
mod memory_core;
mod generate;
mod identity;
mod install;
pub mod lang;
mod lenses;
mod llm;

mod plugins;
mod reference;
mod registry;
mod server;
mod storage;
mod tools;

use crate::memory_core::{Brain, MemoryId};
use crate::identity::IdentityStore;

use crate::reference::ReferenceManager;
use clap::Parser;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

/// Server context shared across all tool invocations.
pub struct Context {
    pub brain: Arc<RwLock<Brain>>,
    pub llm: crate::llm::SharedLlmService,
    pub identity: Arc<Mutex<IdentityStore>>,

    pub references: Arc<Mutex<ReferenceManager>>,
    pub lenses_dir: PathBuf,
    pub memory_home: PathBuf,
    /// Last search query text (for access log correlation with recall)
    pub last_search_query: Mutex<Option<String>>,
    /// Result IDs from the last search (for access log)
    pub last_search_result_ids: Mutex<Vec<MemoryId>>,
}

fn main() {
    // Register sqlite-vec as an auto-extension so every SQLite connection
    // (Diesel and rusqlite) gets vec0 virtual-table support automatically.
    unsafe {
        libsqlite3_sys::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    }

    let cli = cli::Cli::parse();

    match cli.command {
        None | Some(cli::Command::Serve) => server::run(),
        Some(cli::Command::Setup { yes }) => commands::setup(yes),
        Some(cli::Command::Install { yes }) => commands::install(yes),
        Some(cli::Command::Uninstall { yes }) => commands::uninstall(yes),
        Some(cli::Command::Doctor) => commands::doctor(),
        Some(cli::Command::Reset) => commands::reset(),
        Some(cli::Command::Update { dry_run }) => commands::update(dry_run),
        Some(cli::Command::PruneRegistry) => commands::prune_registry(),
        Some(cli::Command::Llm { command }) => commands::llm(command),
        Some(cli::Command::Generate {
            embeddings,
            enrichments,
        }) => commands::generate(embeddings, enrichments),
    }
}
