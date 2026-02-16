//! Memory MCP Server — Cognitive AI memory powered by engram
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
mod engram;
mod identity;
mod install;
mod lenses;
mod plans;
mod plugins;
mod registry;
mod reference;
mod server;
mod storage;
mod tools;

use crate::engram::Brain;
use crate::identity::IdentityStore;
use crate::plans::PlanStore;
use crate::reference::ReferenceManager;
use clap::Parser;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Server context shared across all tool invocations.
pub struct Context {
    pub brain: Arc<Mutex<Brain>>,
    pub identity: Arc<Mutex<IdentityStore>>,
    pub plans: Arc<Mutex<PlanStore>>,
    pub references: Mutex<ReferenceManager>,
    pub lenses_dir: PathBuf,
    pub memory_home: PathBuf,
}

fn main() {
    let cli = cli::Cli::parse();

    match cli.command {
        None | Some(cli::Command::Serve) => server::run(),
        Some(cli::Command::Setup { yes })    => commands::setup(yes),
        Some(cli::Command::Cache)            => commands::cache(),
        Some(cli::Command::Install { yes })  => commands::install(yes),
        Some(cli::Command::Uninstall { yes })=> commands::uninstall(yes),
        Some(cli::Command::Doctor)           => commands::doctor(),
        Some(cli::Command::Reset)            => commands::reset(),
        Some(cli::Command::Update { dry_run })=> commands::update(dry_run),
    }
}
