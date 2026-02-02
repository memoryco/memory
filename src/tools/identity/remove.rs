//! identity_remove - Remove an identity item by ID

use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult, McpError};

use crate::Context;
use crate::tools::text_response;

pub struct IdentityRemoveTool;

impl Tool<Context> for IdentityRemoveTool {
    fn name(&self) -> &str {
        "identity_remove"
    }

    fn description(&self) -> &str {
        "Remove an identity item by its ID. Use identity_list first to see items \
         and their IDs. This is permanent."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The ID of the identity item to remove (from identity_list)"
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
        let id = args.get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::InvalidParams("id is required".into()))?;

        let mut store = context.identity.lock().unwrap();
        
        store.remove(id)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        Ok(text_response(format!("Removed identity item: {}", id)))
    }
}
