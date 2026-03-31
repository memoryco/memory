//! memory_get - Get a specific memory by ID (no side effects)

use crate::memory_core::MemoryId;
use serde::Deserialize;
use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, McpError, Tool, ToolEnv};

use crate::Context;
use crate::tools::{format_memory, text_response};

pub struct MemoryGetTool;

#[derive(Deserialize)]
struct Args {
    id: String,
}

impl Tool<Context> for MemoryGetTool {
    fn name(&self) -> &str {
        "memory_get"
    }

    fn description(&self) -> &str {
        "Get a memory by ID without stimulating it."
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

        let id: MemoryId = args
            .id
            .parse()
            .map_err(|e| McpError::InvalidParams(format!("Invalid UUID: {}", e)))?;

        let mut brain = context.brain.write().unwrap();
        let _ = brain.sync_from_storage();

        match brain.get_or_load(&id) {
            Some(mem) => Ok(text_response(format_memory(mem))),
            None => Ok(text_response(format!("Memory {} not found.", id))),
        }
    }
}
