//! identity_get - Get the current identity

use serde_json::{json, Value as JsonValue};
use sovran_mcp::server::server::{McpTool, McpToolEnvironment};
use sovran_mcp::types::{CallToolResponse, McpError};

use crate::Context;
use crate::tools::text_response;

pub struct IdentityGetTool;

impl McpTool<Context> for IdentityGetTool {
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
        _env: &McpToolEnvironment,
    ) -> Result<CallToolResponse, McpError> {
        let brain = context.brain.lock().unwrap();
        let identity = brain.identity();

        if identity.persona.name.is_empty() && identity.values.is_empty() && identity.instructions.is_empty() {
            return Ok(text_response("No identity configured yet.".to_string()));
        }

        Ok(text_response(identity.render()))
    }
}
