//! identity_remove_instruction - Remove an instruction by index

use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult, McpError};

use crate::Context;
use crate::tools::text_response;

pub struct IdentityRemoveInstructionTool;

impl Tool<Context> for IdentityRemoveInstructionTool {
    fn name(&self) -> &str {
        "identity_remove_instruction"
    }

    fn description(&self) -> &str {
        "Remove an instruction from identity by index (1-based). Use identity_get to see \
         current instructions and their indices."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "1-based index of the instruction to remove"
                }
            },
            "required": ["index"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let index = args.get("index")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| McpError::InvalidParams("index is required".into()))?;

        let mut brain = context.brain.write().unwrap();
        let mut identity = brain.identity().clone();
        
        let count = identity.instructions.len();
        
        if index < 1 || index as usize > count {
            return Ok(text_response(format!(
                "Invalid index {}. Valid range: 1-{}",
                index, count
            )));
        }
        
        let removed = identity.instructions.remove((index - 1) as usize);
        let preview: String = removed.chars().take(60).collect();
        
        brain.set_identity(identity)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        Ok(text_response(format!(
            "Removed instruction {}: \"{}...\"\n\nRemaining instructions: {}",
            index,
            preview.replace('\n', " "),
            brain.identity().instructions.len()
        )))
    }
}
