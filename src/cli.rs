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

#[derive(Debug, Subcommand)]
pub enum LlmCommand {
    /// Show local LLM availability and configured model
    Status,

    /// Expand a search query with the local LLM
    Expand {
        /// Query text to expand
        query: String,

        /// Maximum number of variants to request
        #[arg(long, default_value_t = 3)]
        max_variants: usize,
    },

    /// Generate synthetic search queries for a memory string
    GeneratePairs {
        /// Memory text to generate queries for
        memory: String,

        /// Maximum number of queries to request
        #[arg(long, default_value_t = 6)]
        count: usize,
    },
}

#[derive(Debug, Subcommand)]
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

    /// Update memoryco to the latest version
    Update {
        /// Show what would be updated without actually updating
        #[arg(long)]
        dry_run: bool,
    },

    /// Remove stale entries from the shared registry (~/.memoryco/registry.toml)
    PruneRegistry,

    /// Local LLM debug commands
    Llm {
        #[command(subcommand)]
        command: LlmCommand,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_llm_expand_command() {
        let cli = Cli::parse_from(["memoryco", "llm", "expand", "activity with dad"]);
        match cli.command {
            Some(Command::Llm {
                command:
                    LlmCommand::Expand {
                        query,
                        max_variants,
                    },
            }) => {
                assert_eq!(query, "activity with dad");
                assert_eq!(max_variants, 3);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn parses_llm_generate_pairs_command() {
        let cli = Cli::parse_from([
            "memoryco",
            "llm",
            "generate-pairs",
            "memory text",
            "--count",
            "4",
        ]);
        match cli.command {
            Some(Command::Llm {
                command: LlmCommand::GeneratePairs { memory, count },
            }) => {
                assert_eq!(memory, "memory text");
                assert_eq!(count, 4);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }
}
