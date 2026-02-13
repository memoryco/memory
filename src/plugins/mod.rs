//! Plugin bootstrap system — external MCP servers register instructions via TOML manifests.
//!
//! External MCP servers can drop a `.toml` manifest into `$MEMORY_HOME/bootstrap.d/`
//! and MemoryCo will pick it up at boot, upserting the instructions into identity
//! via the same codepath as internal modules.

pub mod bootstrap;
