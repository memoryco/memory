//! Factory for all known MCP clients.
//!
//! Returns the full list of supported clients with their platform-specific
//! config paths. Currently macOS only; Linux/Windows paths can be added later.

use super::McpClient;
use super::codex::CodexClient;
use super::json_client::JsonClient;

/// Return all known MCP clients for this platform.
pub fn all_clients() -> Vec<Box<dyn McpClient>> {
    let home = dirs::home_dir().expect("Could not determine home directory");

    // --- Tier 1: JSON-based clients ---
    let mut clients: Vec<Box<dyn McpClient>> = vec![
        // Claude Code
        //   Config: ~/.claude.json  (detect via ~/.claude/ directory)
        Box::new(
            JsonClient::new("Claude Code", home.join(".claude.json"), "mcpServers")
                .with_detect_path(home.join(".claude")),
        ),
        // Cursor
        //   All platforms: ~/.cursor/mcp.json
        Box::new(JsonClient::new(
            "Cursor",
            home.join(".cursor/mcp.json"),
            "mcpServers",
        )),
        // Windsurf
        //   All platforms: ~/.codeium/windsurf/mcp_config.json
        Box::new(JsonClient::new(
            "Windsurf",
            home.join(".codeium/windsurf/mcp_config.json"),
            "mcpServers",
        )),
        // --- Tier 2: TOML-based clients ---
        // Codex (OpenAI)
        //   All platforms: ~/.codex/config.toml
        Box::new(CodexClient::new(home.join(".codex/config.toml"))),
    ];

    #[cfg(target_os = "macos")]
    {
        // Claude Desktop
        //   macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
        clients.insert(
            0,
            Box::new(JsonClient::new(
                "Claude Desktop",
                home.join("Library/Application Support/Claude/claude_desktop_config.json"),
                "mcpServers",
            )),
        );

        // VS Code (Copilot)
        //   macOS: ~/Library/Application Support/Code/User/mcp.json
        //   NOTE: uses "servers" not "mcpServers"
        clients.insert(
            4,
            Box::new(JsonClient::new(
                "VS Code",
                home.join("Library/Application Support/Code/User/mcp.json"),
                "servers",
            )),
        );
    }

    clients
}
