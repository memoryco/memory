//! engram_delete - Delete memories permanently

use crate::engram::EngramId;
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult, McpError};

use crate::Context;
use crate::tools::text_response;

pub struct EngramDeleteTool;

#[derive(Deserialize)]
struct Args {
    ids: Vec<String>,
}

impl Tool<Context> for EngramDeleteTool {
    fn name(&self) -> &str {
        "engram_delete"
    }

    fn description(&self) -> &str {
        "Delete memories permanently. Accepts an array of IDs to delete. \
         Also removes all associations from/to deleted memories. Use for \
         housekeeping: removing duplicates, correcting mistakes, cleaning up. \
         IMPORTANT: Never delete memories unless the user explicitly asks you to."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Array of UUIDs to delete permanently."
                }
            },
            "required": ["ids"]
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

        let mut brain = context.brain.lock().unwrap();
        let mut output = String::new();
        let mut deleted_count = 0;
        let mut not_found_count = 0;

        for id_str in &args.ids {
            let id: EngramId = id_str.parse()
                .map_err(|e| McpError::InvalidParams(format!("Invalid UUID '{}': {}", id_str, e)))?;

            let existed = brain.delete(id)
                .map_err(|e| McpError::ToolError(e.to_string()))?;

            if existed {
                deleted_count += 1;
                output.push_str(&format!("Deleted: {}\n", id));
            } else {
                not_found_count += 1;
                output.push_str(&format!("Not found: {}\n", id));
            }
        }

        let header = format!(
            "Deleted {} memories ({} not found).\n\n",
            deleted_count, not_found_count
        );

        Ok(text_response(format!("{}{}", header, output.trim())))
    }
}
