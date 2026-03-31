//! memory_delete - Delete memories permanently

use crate::memory_core::MemoryId;
use serde::Deserialize;
use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, McpError, Tool, ToolEnv};

use crate::Context;
use crate::tools::text_response;

pub struct MemoryDeleteTool;

#[derive(Deserialize)]
struct Args {
    ids: Vec<String>,
}

impl Tool<Context> for MemoryDeleteTool {
    fn name(&self) -> &str {
        "memory_delete"
    }

    fn description(&self) -> &str {
        "Permanently delete memories and their associations. Never delete \
         unless the user explicitly asks."
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
        let args: Args =
            serde_json::from_value(args).map_err(|e| McpError::InvalidParams(e.to_string()))?;

        let mut brain = context.brain.write().unwrap();
        let mut output = String::new();
        let mut deleted_count = 0;
        let mut not_found_count = 0;

        for id_str in &args.ids {
            let id: MemoryId = id_str.parse().map_err(|e| {
                McpError::InvalidParams(format!("Invalid UUID '{}': {}", id_str, e))
            })?;

            let existed = brain
                .delete(id)
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
