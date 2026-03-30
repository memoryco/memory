//! memory_associate - Create an explicit association between memories

use crate::memory_core::MemoryId;
use serde::Deserialize;
use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, McpError, Tool, ToolEnv};

use crate::Context;
use crate::tools::text_response;

pub struct MemoryAssociateTool;

#[derive(Deserialize)]
struct Args {
    from: String,
    to: String,
    #[serde(default)]
    weight: Option<f64>,
    #[serde(default)]
    ordinal: Option<u32>,
}

impl Tool<Context> for MemoryAssociateTool {
    fn name(&self) -> &str {
        "memory_associate"
    }

    fn description(&self) -> &str {
        "Create an explicit association between two memories. Associations also \
         form automatically through Hebbian learning when memories are recalled together."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "from": {
                    "type": "string",
                    "description": "UUID of the source memory."
                },
                "to": {
                    "type": "string",
                    "description": "UUID of the target memory."
                },
                "weight": {
                    "type": "number",
                    "description": "Association strength (0.0-1.0). Default: 0.5"
                },
                "ordinal": {
                    "type": "integer",
                    "description": "Position in an ordered chain (e.g., procedure steps). \
                     Use ordinal to create ordered chains — step 1, step 2, etc. \
                     Omit for unordered associations."
                }
            },
            "required": ["from", "to"]
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

        let from: MemoryId = args
            .from
            .parse()
            .map_err(|e| McpError::InvalidParams(format!("Invalid 'from' UUID: {}", e)))?;
        let to: MemoryId = args
            .to
            .parse()
            .map_err(|e| McpError::InvalidParams(format!("Invalid 'to' UUID: {}", e)))?;

        let weight = args.weight.unwrap_or(0.5);
        let ordinal = args.ordinal;

        let mut brain = context.brain.write().unwrap();

        brain
            .associate_with_ordinal(from, to, weight, ordinal)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        let ordinal_str = match ordinal {
            Some(n) => format!(", ordinal: {}", n),
            None => String::new(),
        };

        Ok(text_response(format!(
            "Association created: {} → {} (weight: {:.2}{})",
            from, to, weight, ordinal_str
        )))
    }
}
