//! JSON-based MCP client configuration.
//!
//! Handles clients that store MCP server config in a JSON file with the structure:
//! ```json
//! { "<top_level_key>": { "memoryco": { "command": "...", "args": [...], "env": {...} } } }
//! ```
//!
//! Covers: Claude Desktop, Claude Code, Cursor, Windsurf, VS Code (Copilot).

use super::{McpClient, InstallStatus, InstallError, memoryco_server_entry};
use serde_json::{json, Map, Value};
use std::path::PathBuf;

/// An MCP client that uses JSON configuration files.
pub struct JsonClient {
    /// Human-readable name.
    name: String,
    /// Path to the JSON config file.
    config_path: PathBuf,
    /// Top-level key that holds MCP server entries (e.g., "mcpServers" or "servers").
    servers_key: String,
}

impl JsonClient {
    pub fn new(name: impl Into<String>, config_path: PathBuf, servers_key: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            config_path,
            servers_key: servers_key.into(),
        }
    }

    /// Read and parse the config file. Returns empty object if file doesn't exist.
    fn read_config(&self) -> Result<Value, InstallError> {
        if !self.config_path.exists() {
            return Ok(json!({}));
        }
        let contents = std::fs::read_to_string(&self.config_path)?;
        // Handle empty files
        if contents.trim().is_empty() {
            return Ok(json!({}));
        }
        serde_json::from_str(&contents)
            .map_err(|e| InstallError::Parse(format!("{}: {}", self.config_path.display(), e)))
    }

    /// Write config back to file with pretty formatting.
    fn write_config(&self, config: &Value) -> Result<(), InstallError> {
        // Ensure parent directory exists
        if let Some(parent) = self.config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let contents = serde_json::to_string_pretty(config)
            .map_err(|e| InstallError::Serialize(e.to_string()))?;
        std::fs::write(&self.config_path, contents)?;
        Ok(())
    }

    /// Get the memoryco entry from the config, if it exists.
    fn get_memoryco_entry<'a>(&self, config: &'a Value) -> Option<&'a Value> {
        config
            .get(&self.servers_key)?
            .get("memoryco")
    }

    /// Build the JSON value for our server entry.
    fn build_entry(&self) -> Value {
        let (command, args, env_pairs) = memoryco_server_entry();
        let mut env = Map::new();
        for (k, v) in env_pairs {
            env.insert(k, Value::String(v));
        }

        json!({
            "command": command,
            "args": args,
            "env": env
        })
    }
}

impl McpClient for JsonClient {
    fn name(&self) -> &str {
        &self.name
    }

    fn config_path(&self) -> PathBuf {
        self.config_path.clone()
    }

    fn check_existing(&self) -> InstallStatus {
        if !self.detect() {
            return InstallStatus::ClientNotFound;
        }

        let config = match self.read_config() {
            Ok(c) => c,
            Err(_) => return InstallStatus::NotInstalled,
        };

        let Some(existing) = self.get_memoryco_entry(&config) else {
            return InstallStatus::NotInstalled;
        };

        // Check if the command matches our current binary
        let (our_command, _, _) = memoryco_server_entry();
        let their_command = existing
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if their_command == our_command {
            InstallStatus::Installed
        } else {
            InstallStatus::NeedsUpdate {
                current_command: their_command.to_string(),
            }
        }
    }

    fn install(&self) -> Result<(), InstallError> {
        let mut config = self.read_config()?;
        let entry = self.build_entry();

        // Ensure the top-level object exists
        let obj = config.as_object_mut()
            .ok_or_else(|| InstallError::Parse("Config is not a JSON object".to_string()))?;

        // Ensure the servers key exists
        if !obj.contains_key(&self.servers_key) {
            obj.insert(self.servers_key.clone(), json!({}));
        }

        // Insert/update the memoryco entry
        let servers = obj
            .get_mut(&self.servers_key)
            .and_then(|v| v.as_object_mut())
            .ok_or_else(|| InstallError::Parse(
                format!("'{}' is not a JSON object", self.servers_key)
            ))?;

        servers.insert("memoryco".to_string(), entry);

        self.write_config(&config)?;
        Ok(())
    }

    fn uninstall(&self) -> Result<(), InstallError> {
        if !self.config_path.exists() {
            return Ok(());
        }

        let mut config = self.read_config()?;

        // Remove the memoryco entry if it exists
        if let Some(servers) = config
            .get_mut(&self.servers_key)
            .and_then(|v| v.as_object_mut())
        {
            servers.remove("memoryco");
        }

        self.write_config(&config)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_client(dir: &TempDir, servers_key: &str) -> JsonClient {
        let config_path = dir.path().join("config.json");
        JsonClient::new("Test Client", config_path, servers_key)
    }

    #[test]
    fn detect_returns_true_when_parent_exists() {
        let dir = TempDir::new().unwrap();
        let client = test_client(&dir, "mcpServers");
        // Parent dir (tempdir) exists, config file doesn't
        assert!(client.detect());
    }

    #[test]
    fn check_existing_not_installed_when_no_file() {
        let dir = TempDir::new().unwrap();
        let client = test_client(&dir, "mcpServers");
        assert_eq!(client.check_existing(), InstallStatus::NotInstalled);
    }

    #[test]
    fn install_creates_config_from_scratch() {
        let dir = TempDir::new().unwrap();
        let client = test_client(&dir, "mcpServers");

        client.install().unwrap();

        let config = client.read_config().unwrap();
        assert!(config["mcpServers"]["memoryco"]["command"].is_string());
        assert_eq!(config["mcpServers"]["memoryco"]["args"][0], "serve");
    }

    #[test]
    fn install_preserves_existing_servers() {
        let dir = TempDir::new().unwrap();
        let client = test_client(&dir, "mcpServers");

        // Pre-populate with another server
        let initial = json!({
            "mcpServers": {
                "other-server": {
                    "command": "/usr/bin/other",
                    "args": ["run"]
                }
            }
        });
        std::fs::write(client.config_path(), serde_json::to_string_pretty(&initial).unwrap()).unwrap();

        client.install().unwrap();

        let config = client.read_config().unwrap();
        // Our entry was added
        assert!(config["mcpServers"]["memoryco"]["command"].is_string());
        // Other entry preserved
        assert_eq!(config["mcpServers"]["other-server"]["command"], "/usr/bin/other");
    }

    #[test]
    fn install_preserves_non_mcp_config() {
        let dir = TempDir::new().unwrap();
        let client = test_client(&dir, "mcpServers");

        // Config with non-MCP settings
        let initial = json!({
            "theme": "dark",
            "fontSize": 14,
            "mcpServers": {}
        });
        std::fs::write(client.config_path(), serde_json::to_string_pretty(&initial).unwrap()).unwrap();

        client.install().unwrap();

        let config = client.read_config().unwrap();
        assert_eq!(config["theme"], "dark");
        assert_eq!(config["fontSize"], 14);
        assert!(config["mcpServers"]["memoryco"]["command"].is_string());
    }

    #[test]
    fn check_existing_installed_when_matching() {
        let dir = TempDir::new().unwrap();
        let client = test_client(&dir, "mcpServers");

        client.install().unwrap();
        assert_eq!(client.check_existing(), InstallStatus::Installed);
    }

    #[test]
    fn check_existing_needs_update_when_different_binary() {
        let dir = TempDir::new().unwrap();
        let client = test_client(&dir, "mcpServers");

        let config = json!({
            "mcpServers": {
                "memoryco": {
                    "command": "/old/path/memoryco",
                    "args": ["serve"]
                }
            }
        });
        std::fs::write(client.config_path(), serde_json::to_string_pretty(&config).unwrap()).unwrap();

        match client.check_existing() {
            InstallStatus::NeedsUpdate { current_command } => {
                assert_eq!(current_command, "/old/path/memoryco");
            }
            other => panic!("Expected NeedsUpdate, got {:?}", other),
        }
    }

    #[test]
    fn uninstall_removes_memoryco_only() {
        let dir = TempDir::new().unwrap();
        let client = test_client(&dir, "mcpServers");

        let config = json!({
            "mcpServers": {
                "memoryco": { "command": "memoryco", "args": ["serve"] },
                "other": { "command": "other", "args": ["run"] }
            }
        });
        std::fs::write(client.config_path(), serde_json::to_string_pretty(&config).unwrap()).unwrap();

        client.uninstall().unwrap();

        let config = client.read_config().unwrap();
        assert!(config["mcpServers"]["memoryco"].is_null());
        assert_eq!(config["mcpServers"]["other"]["command"], "other");
    }

    #[test]
    fn uninstall_noop_when_no_file() {
        let dir = TempDir::new().unwrap();
        let client = test_client(&dir, "mcpServers");
        // Should not error
        client.uninstall().unwrap();
    }

    #[test]
    fn works_with_servers_key() {
        // VS Code uses "servers" instead of "mcpServers"
        let dir = TempDir::new().unwrap();
        let client = test_client(&dir, "servers");

        client.install().unwrap();

        let config = client.read_config().unwrap();
        assert!(config["servers"]["memoryco"]["command"].is_string());
        assert!(config["mcpServers"].is_null()); // Should NOT create mcpServers
    }

    #[test]
    fn handles_empty_file() {
        let dir = TempDir::new().unwrap();
        let client = test_client(&dir, "mcpServers");

        std::fs::write(client.config_path(), "").unwrap();
        client.install().unwrap();

        let config = client.read_config().unwrap();
        assert!(config["mcpServers"]["memoryco"]["command"].is_string());
    }
}
