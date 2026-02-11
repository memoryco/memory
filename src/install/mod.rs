//! MCP client detection and configuration.
//!
//! Discovers installed MCP clients, checks for existing MemoryCo config,
//! and injects/updates configuration as needed.

mod json_client;
mod codex;
mod clients;
pub mod claude_md;

pub use clients::all_clients;

use crate::config;
use std::fmt;
use std::path::PathBuf;

/// Result of checking whether MemoryCo is already configured in a client.
#[derive(Debug, PartialEq)]
pub enum InstallStatus {
    /// Client config exists but has no MemoryCo entry.
    NotInstalled,
    /// MemoryCo entry exists and matches current binary/args.
    Installed,
    /// MemoryCo entry exists but points to a different binary or has stale config.
    NeedsUpdate {
        current_command: String,
    },
    /// Client does not appear to be installed (no config file or parent dir).
    ClientNotFound,
}

impl fmt::Display for InstallStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotInstalled => write!(f, "not configured"),
            Self::Installed => write!(f, "configured"),
            Self::NeedsUpdate { current_command } => {
                write!(f, "outdated (pointing to {})", current_command)
            }
            Self::ClientNotFound => write!(f, "client not installed"),
        }
    }
}

/// Trait for MCP client configuration management.
pub trait McpClient {
    /// Human-readable client name (e.g., "Claude Desktop").
    fn name(&self) -> &str;

    /// Path to the client's MCP configuration file.
    fn config_path(&self) -> PathBuf;

    /// Whether the client appears to be installed on this machine.
    fn detect(&self) -> bool {
        let path = self.config_path();
        // Client is "detected" if config file exists OR its parent dir exists
        // (parent dir means client is installed but no MCP config yet)
        path.exists() || path.parent().is_some_and(|p| p.exists())
    }

    /// Check if MemoryCo is already configured in this client.
    fn check_existing(&self) -> InstallStatus;

    /// Install MemoryCo into this client's configuration.
    /// Preserves any existing config entries.
    fn install(&self) -> Result<(), InstallError>;

    /// Remove MemoryCo from this client's configuration.
    /// Preserves all other config entries.
    fn uninstall(&self) -> Result<(), InstallError>;
}

/// Generate the MemoryCo server entry for injection into client configs.
///
/// Returns (command, args, env) tuple.
pub fn memoryco_server_entry() -> (String, Vec<String>, Vec<(String, String)>) {
    let command = std::env::current_exe()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "memoryco".to_string());

    let args = vec!["serve".to_string()];

    let memory_home = config::get_memory_home();
    let env = vec![
        ("MEMORY_HOME".to_string(), memory_home.display().to_string()),
    ];

    (command, args, env)
}

/// Errors during client installation.
#[derive(Debug, thiserror::Error)]
pub enum InstallError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Failed to parse config: {0}")]
    Parse(String),

    #[error("Failed to serialize config: {0}")]
    Serialize(String),

}
