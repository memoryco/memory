//! Tool: open_dashboard — opens the memoryco dashboard in the user's browser.

use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, Tool, ToolEnv};

use crate::Context;
use crate::dashboard::{BIND_HOST, resolve_dashboard_port};
use crate::tools::text_response;

/// Resolve the dashboard URL, honoring `MEMORYCO_DASHBOARD_PORT` if set.
fn dashboard_url() -> String {
    format!("http://{}:{}", BIND_HOST, resolve_dashboard_port())
}

pub struct OpenDashboardTool;

impl Tool<Context> for OpenDashboardTool {
    fn name(&self) -> &str {
        "open_dashboard"
    }

    fn description(&self) -> &str {
        "Open the memoryco dashboard in your browser. View and manage your identity, references, memories, and association graph."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    fn execute(
        &self,
        _args: JsonValue,
        _context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let url = dashboard_url();
        let cmd = if cfg!(target_os = "macos") {
            "open"
        } else if cfg!(target_os = "windows") {
            "start"
        } else {
            "xdg-open"
        };

        match std::process::Command::new(cmd).arg(&url).spawn() {
            Ok(_) => Ok(text_response(format!(
                "Opened dashboard in your browser: {}",
                &url
            ))),
            Err(_) => Ok(text_response(format!(
                "Could not open browser automatically. Open this URL manually: {}",
                &url
            ))),
        }
    }
}
