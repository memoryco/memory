//! engram_get - Get a specific memory by ID (no side effects)

use crate::engram::EngramId;
use serde::Deserialize;
use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, McpError, Tool, ToolEnv};

use crate::Context;
use crate::tools::{format_engram, text_response};

pub struct EngramGetTool;

#[derive(Deserialize)]
struct Args {
    id: String,
}

impl Tool<Context> for EngramGetTool {
    fn name(&self) -> &str {
        "engram_get"
    }

    fn description(&self) -> &str {
        "Get a memory by ID without stimulating it. Use engram_recall if you \
         want to actively use the memory."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The UUID of the memory."
                }
            },
            "required": ["id"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let args: Args =
            serde_json::from_value(args).map_err(|e| McpError::InvalidParams(e.to_string()))?;

        let id: EngramId = args
            .id
            .parse()
            .map_err(|e| McpError::InvalidParams(format!("Invalid UUID: {}", e)))?;

        let mut brain = context.brain.lock().unwrap();
        let _ = brain.sync_from_storage();

        match brain.get_or_load(&id) {
            Some(engram) => Ok(text_response(format_engram(engram))),
            None => Ok(text_response(format!("Memory {} not found.", id))),
        }
    }
}
