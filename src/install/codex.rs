//! OpenAI Codex MCP client configuration (TOML-based).
//!
//! Codex stores MCP config in `~/.codex/config.toml` with the structure:
//! ```toml
//! [mcp_servers.memoryco]
//! command = "/path/to/memoryco"
//! args = ["serve"]
//!
//! [mcp_servers.memoryco.env]
//! MEMORY_HOME = "~/.memoryco"
//! ```

use super::{McpClient, InstallStatus, InstallError, memoryco_server_entry};
use std::path::PathBuf;
use toml_edit::{DocumentMut, Item, Table, Array, value};

/// OpenAI Codex MCP client.
pub struct CodexClient {
    config_path: PathBuf,
}

impl CodexClient {
    pub fn new(config_path: PathBuf) -> Self {
        Self { config_path }
    }

    /// Read and parse the TOML config. Returns empty document if file doesn't exist.
    fn read_config(&self) -> Result<DocumentMut, InstallError> {
        if !self.config_path.exists() {
            return Ok(DocumentMut::new());
        }
        let contents = std::fs::read_to_string(&self.config_path)?;
        if contents.trim().is_empty() {
            return Ok(DocumentMut::new());
        }
        contents.parse::<DocumentMut>()
            .map_err(|e| InstallError::Parse(format!("{}: {}", self.config_path.display(), e)))
    }

    /// Write config back to file.
    fn write_config(&self, doc: &DocumentMut) -> Result<(), InstallError> {
        if let Some(parent) = self.config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&self.config_path, doc.to_string())?;
        Ok(())
    }
}

impl McpClient for CodexClient {
    fn name(&self) -> &str {
        "Codex (OpenAI)"
    }

    fn config_path(&self) -> PathBuf {
        self.config_path.clone()
    }

    fn check_existing(&self) -> InstallStatus {
        if !self.detect() {
            return InstallStatus::ClientNotFound;
        }

        let doc = match self.read_config() {
            Ok(d) => d,
            Err(_) => return InstallStatus::NotInstalled,
        };

        // Check for mcp_servers.memoryco
        let Some(servers) = doc.get("mcp_servers").and_then(|v| v.as_table()) else {
            return InstallStatus::NotInstalled;
        };

        let Some(entry) = servers.get("memoryco").and_then(|v| v.as_table()) else {
            return InstallStatus::NotInstalled;
        };

        let (our_command, _, _) = memoryco_server_entry();
        let their_command = entry
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
        let mut doc = self.read_config()?;
        let (command, args, env_pairs) = memoryco_server_entry();

        // Ensure [mcp_servers] exists
        if !doc.contains_key("mcp_servers") {
            doc["mcp_servers"] = Item::Table(Table::new());
        }

        // Build the memoryco table
        let mut entry = Table::new();
        entry.insert("command", value(&command));

        let mut args_array = Array::new();
        for arg in &args {
            args_array.push(arg.as_str());
        }
        entry.insert("args", value(args_array));

        // Build env sub-table
        let mut env_table = Table::new();
        for (k, v) in &env_pairs {
            env_table.insert(k.as_str(), value(v.as_str()));
        }
        entry.insert("env", Item::Table(env_table));

        // Insert under mcp_servers
        doc["mcp_servers"]["memoryco"] = Item::Table(entry);

        self.write_config(&doc)?;
        Ok(())
    }

    fn uninstall(&self) -> Result<(), InstallError> {
        if !self.config_path.exists() {
            return Ok(());
        }

        let mut doc = self.read_config()?;

        if let Some(servers) = doc.get_mut("mcp_servers").and_then(|v| v.as_table_mut()) {
            servers.remove("memoryco");
        }

        self.write_config(&doc)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_client(dir: &TempDir) -> CodexClient {
        CodexClient::new(dir.path().join("config.toml"))
    }

    #[test]
    fn install_creates_toml_from_scratch() {
        let dir = TempDir::new().unwrap();
        let client = test_client(&dir);

        client.install().unwrap();

        let contents = std::fs::read_to_string(client.config_path()).unwrap();
        let doc: DocumentMut = contents.parse().unwrap();

        let entry = doc["mcp_servers"]["memoryco"].as_table().unwrap();
        assert!(entry.get("command").unwrap().as_str().is_some());
        assert_eq!(entry["args"].as_array().unwrap().get(0).unwrap().as_str().unwrap(), "serve");
    }

    #[test]
    fn install_preserves_existing_toml() {
        let dir = TempDir::new().unwrap();
        let client = test_client(&dir);

        let initial = r#"
model = "o3"

[mcp_servers.filesystem]
command = "/usr/bin/fs-server"
args = ["--root", "/home"]
"#;
        std::fs::write(client.config_path(), initial).unwrap();

        client.install().unwrap();

        let contents = std::fs::read_to_string(client.config_path()).unwrap();
        let doc: DocumentMut = contents.parse().unwrap();

        // Existing config preserved
        assert_eq!(doc["model"].as_str().unwrap(), "o3");
        assert!(doc["mcp_servers"]["filesystem"].as_table().is_some());
        // Our entry added
        assert!(doc["mcp_servers"]["memoryco"].as_table().is_some());
    }

    #[test]
    fn check_existing_installed_after_install() {
        let dir = TempDir::new().unwrap();
        let client = test_client(&dir);

        client.install().unwrap();
        assert_eq!(client.check_existing(), InstallStatus::Installed);
    }

    #[test]
    fn uninstall_removes_memoryco_only() {
        let dir = TempDir::new().unwrap();
        let client = test_client(&dir);

        let initial = r#"
[mcp_servers.memoryco]
command = "memoryco"
args = ["serve"]

[mcp_servers.other]
command = "other"
"#;
        std::fs::write(client.config_path(), initial).unwrap();

        client.uninstall().unwrap();

        let contents = std::fs::read_to_string(client.config_path()).unwrap();
        let doc: DocumentMut = contents.parse().unwrap();

        assert!(doc["mcp_servers"].as_table().unwrap().get("memoryco").is_none());
        assert!(doc["mcp_servers"]["other"].as_table().is_some());
    }
}
