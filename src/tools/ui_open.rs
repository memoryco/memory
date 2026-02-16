//! ui_open - Open the memoryco dashboard in the user's browser

use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult};

use crate::Context;
use crate::tools::text_response;

const DASHBOARD_URL: &str = "http://127.0.0.1:4243";

pub struct UiOpenTool;

impl Tool<Context> for UiOpenTool {
    fn name(&self) -> &str {
        "ui_open"
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
        let cmd = if cfg!(target_os = "macos") {
            "open"
        } else {
            "xdg-open"
        };

        match std::process::Command::new(cmd).arg(DASHBOARD_URL).spawn() {
            Ok(_) => Ok(text_response(format!(
                "Opened dashboard in your browser: {}",
                DASHBOARD_URL
            ))),
            Err(_) => Ok(text_response(format!(
                "Could not open browser automatically. Open this URL manually: {}",
                DASHBOARD_URL
            ))),
        }
    }
}
