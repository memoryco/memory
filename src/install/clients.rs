//! Factory for all known MCP clients.
//!
//! Returns the full list of supported clients with their platform-specific
//! config paths. Currently macOS only; Linux/Windows paths can be added later.

use super::McpClient;
use super::json_client::JsonClient;
use super::codex::CodexClient;

/// Return all known MCP clients for this platform.
pub fn all_clients() -> Vec<Box<dyn McpClient>> {
    let home = dirs::home_dir().expect("Could not determine home directory");

    let mut clients: Vec<Box<dyn McpClient>> = Vec::new();

    // --- Tier 1: JSON-based clients ---

    // Claude Desktop
    //   macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
    #[cfg(target_os = "macos")]
    clients.push(Box::new(JsonClient::new(
        "Claude Desktop",
        home.join("Library/Application Support/Claude/claude_desktop_config.json"),
        "mcpServers",
    )));

    // Claude Code
    //   Config: ~/.claude.json  (detect via ~/.claude/ directory)
    clients.push(Box::new(JsonClient::new(
        "Claude Code",
        home.join(".claude.json"),
        "mcpServers",
    ).with_detect_path(home.join(".claude"))));

    // Cursor
    //   All platforms: ~/.cursor/mcp.json
    clients.push(Box::new(JsonClient::new(
        "Cursor",
        home.join(".cursor/mcp.json"),
        "mcpServers",
    )));

    // Windsurf
    //   All platforms: ~/.codeium/windsurf/mcp_config.json
    clients.push(Box::new(JsonClient::new(
        "Windsurf",
        home.join(".codeium/windsurf/mcp_config.json"),
        "mcpServers",
    )));

    // VS Code (Copilot)
    //   macOS: ~/Library/Application Support/Code/User/mcp.json
    //   NOTE: uses "servers" not "mcpServers"
    #[cfg(target_os = "macos")]
    clients.push(Box::new(JsonClient::new(
        "VS Code",
        home.join("Library/Application Support/Code/User/mcp.json"),
        "servers",
    )));

    // --- Tier 2: TOML-based clients ---

    // Codex (OpenAI)
    //   All platforms: ~/.codex/config.toml
    clients.push(Box::new(CodexClient::new(
        home.join(".codex/config.toml"),
    )));

    clients
}
