//! engram_associate - Create an explicit association between memories

use crate::engram::EngramId;
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult, McpError};

use crate::Context;
use crate::tools::text_response;

pub struct EngramAssociateTool;

#[derive(Deserialize)]
struct Args {
    from: String,
    to: String,
    #[serde(default)]
    weight: Option<f64>,
}

impl Tool<Context> for EngramAssociateTool {
    fn name(&self) -> &str {
        "engram_associate"
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
        let args: Args = serde_json::from_value(args)
            .map_err(|e| McpError::InvalidParams(e.to_string()))?;

        let from: EngramId = args.from.parse()
            .map_err(|e| McpError::InvalidParams(format!("Invalid 'from' UUID: {}", e)))?;
        let to: EngramId = args.to.parse()
            .map_err(|e| McpError::InvalidParams(format!("Invalid 'to' UUID: {}", e)))?;

        let weight = args.weight.unwrap_or(0.5);

        let mut brain = context.brain.lock().unwrap();

        brain.associate(from, to, weight)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        Ok(text_response(format!(
            "Association created: {} → {} (weight: {:.2})",
            from, to, weight
        )))
    }
}
