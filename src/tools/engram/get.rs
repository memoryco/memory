//! engram_get - Get a specific memory by ID (no side effects)

use engram::EngramId;
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};
use sovran_mcp::server::server::{McpTool, McpToolEnvironment};
use sovran_mcp::types::{CallToolResponse, McpError};

use crate::Context;
use crate::tools::{text_response, format_engram};

pub struct EngramGetTool;

#[derive(Deserialize)]
struct Args {
    id: String,
}

impl McpTool<Context> for EngramGetTool {
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
        _env: &McpToolEnvironment,
    ) -> Result<CallToolResponse, McpError> {
        let args: Args = serde_json::from_value(args)
            .map_err(|e| McpError::InvalidArguments(e.to_string()))?;

        let id: EngramId = args.id.parse()
            .map_err(|e| McpError::InvalidArguments(format!("Invalid UUID: {}", e)))?;

        let brain = context.brain.lock().unwrap();

        match brain.get(&id) {
            Some(engram) => Ok(text_response(format_engram(engram))),
            None => Ok(text_response(format!("Memory {} not found.", id))),
        }
    }
}
