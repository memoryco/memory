//! CLI argument parsing for the memory server.

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "memoryco",
    about = "MemoryCo — local-first AI memory system",
    version,
    after_help = "Run `memoryco setup` to get started, or `memoryco serve` to start the MCP server."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand)]
pub enum Command {
    /// Start the MCP server (default if no command given)
    Serve,

    /// Full first-run setup: download model, configure clients, start server
    Setup {
        /// Skip confirmation prompts (for scripted installs)
        #[arg(long)]
        yes: bool,
    },

    /// Download and cache the embedding model
    Cache,

    /// Auto-detect MCP clients and add MemoryCo to their config
    Install {
        /// Skip confirmation prompts
        #[arg(long)]
        yes: bool,
    },

    /// Remove MemoryCo from MCP client configs
    Uninstall {
        /// Skip confirmation prompts
        #[arg(long)]
        yes: bool,
    },

    /// Check system health: databases, model, client configs, identity
    Doctor,

    /// Delete all databases (keeps cache, lenses, references)
    Reset,
}
