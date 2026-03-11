//! identity_add_instruction - Append an instruction to identity

use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult, McpError};

use crate::Context;
use crate::tools::text_response;

pub struct IdentityAddInstructionTool;

impl Tool<Context> for IdentityAddInstructionTool {
    fn name(&self) -> &str {
        "identity_add_instruction"
    }

    fn description(&self) -> &str {
        "Add an instruction to identity without replacing existing ones. Instructions are \
         permanent operational directives that don't decay. Use for workflow rules, tool \
         usage guides, or behavioral directives."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "The instruction text to add. Markdown supported."
                },
                "marker": {
                    "type": "string",
                    "description": "Optional: unique marker to check for duplicates (e.g., '## References'). If present in existing instructions, the add is skipped."
                }
            },
            "required": ["instruction"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let instruction = args.get("instruction")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::InvalidParams("instruction is required".into()))?;
        
        let marker = args.get("marker").and_then(|v| v.as_str());

        let mut brain = context.brain.write().unwrap();
        
        // Check for duplicate if marker provided
        if let Some(m) = marker {
            let already_present = brain.identity().instructions.iter()
                .any(|i| i.contains(m));
            
            if already_present {
                return Ok(text_response(format!(
                    "Instruction with marker '{}' already exists. Skipped.",
                    m
                )));
            }
        }

        // Add the instruction
        let mut identity = brain.identity().clone();
        identity = identity.with_instruction(instruction);
        
        brain.set_identity(identity)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        Ok(text_response(format!(
            "Instruction added. Total instructions: {}",
            brain.identity().instructions.len()
        )))
    }
}
