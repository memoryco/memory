//! Factory for all known MCP clients.
//!
//! Returns the full list of supported clients with their platform-specific
//! config paths. Supports macOS and Linux.

use super::McpClient;
use super::codex::CodexClient;
use super::json_client::JsonClient;

/// Return all known MCP clients for this platform.
pub fn all_clients() -> Vec<Box<dyn McpClient>> {
    let home = dirs::home_dir().expect("Could not determine home directory");

    // --- Claude Desktop ---
    //   macOS:  ~/Library/Application Support/Claude/claude_desktop_config.json
    //   Linux:  ~/.config/Claude/claude_desktop_config.json
    #[cfg(target_os = "macos")]
    let claude_desktop_path = home.join("Library/Application Support/Claude/claude_desktop_config.json");
    #[cfg(target_os = "linux")]
    let claude_desktop_path = home.join(".config/Claude/claude_desktop_config.json");

    // --- VS Code (Copilot) ---
    //   macOS:  ~/Library/Application Support/Code/User/mcp.json
    //   Linux:  ~/.config/Code/User/mcp.json
    //   NOTE: uses "servers" not "mcpServers"
    #[cfg(target_os = "macos")]
    let vscode_path = home.join("Library/Application Support/Code/User/mcp.json");
    #[cfg(target_os = "linux")]
    let vscode_path = home.join(".config/Code/User/mcp.json");

    let clients: Vec<Box<dyn McpClient>> = vec![
        // Claude Desktop
        Box::new(JsonClient::new(
            "Claude Desktop",
            claude_desktop_path,
            "mcpServers",
        )),
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
        // VS Code (Copilot)
        Box::new(JsonClient::new(
            "VS Code",
            vscode_path,
            "servers",
        )),
        // --- Tier 2: TOML-based clients ---
        // Codex (OpenAI)
        //   All platforms: ~/.codex/config.toml
        Box::new(CodexClient::new(home.join(".codex/config.toml"))),
    ];

    clients
}
