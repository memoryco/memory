//! plans - List all active plans

use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, Tool, ToolEnv};

use crate::Context;
use crate::tools::text_response;

pub struct PlansListTool;

impl Tool<Context> for PlansListTool {
    fn name(&self) -> &str {
        "plans"
    }

    fn description(&self) -> &str {
        "List all active plans with their IDs and descriptions."
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
        let mut plans = context.plans.lock().unwrap();

        let plan_list = plans
            .list()
            .map_err(|e| sml_mcps::McpError::ToolError(e.to_string()))?;

        if plan_list.is_empty() {
            return Ok(text_response("No active plans.".to_string()));
        }

        let output = plan_list
            .iter()
            .map(|(id, desc)| format!("• {} - {}", id, desc))
            .collect::<Vec<_>>()
            .join("\n");

        Ok(text_response(format!("Active plans:\n\n{}", output)))
    }
}
