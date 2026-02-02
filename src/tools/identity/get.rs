//! identity_get - Get the current identity

use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult, McpError};

use crate::Context;
use crate::tools::text_response;

pub struct IdentityGetTool;

impl Tool<Context> for IdentityGetTool {
    fn name(&self) -> &str {
        "identity_get"
    }

    fn description(&self) -> &str {
        "Get the current identity (persona, values, preferences, relationships). \
         Identity never decays - it's who you ARE, not what you remember."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    fn execute(
        &self,
        _args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let mut store = context.identity.lock().unwrap();
        let identity = store.get()
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        if identity.persona.name.is_empty() && identity.values.is_empty() && identity.instructions.is_empty() {
            return Ok(text_response("No identity configured yet.".to_string()));
        }

        Ok(text_response(identity.render()))
    }
}
