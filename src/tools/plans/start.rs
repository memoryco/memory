//! plan_start - Start a new plan

use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult};

use crate::Context;
use crate::tools::text_response;

pub struct PlanStartTool;

impl Tool<Context> for PlanStartTool {
    fn name(&self) -> &str {
        "plan_start"
    }

    fn description(&self) -> &str {
        "Start a new plan. Returns the plan ID which you'll need for adding steps."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "What this plan is for - the goal or objective."
                }
            },
            "required": ["description"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let description = args["description"].as_str()
            .ok_or_else(|| sml_mcps::McpError::InvalidParams("description is required".to_string()))?;

        let mut plans = context.plans.lock().unwrap();
        
        let id = plans.start(description)
            .map_err(|e| sml_mcps::McpError::ToolError(e.to_string()))?;

        Ok(text_response(format!(
            "Plan started!\n\nID: {}\nDescription: {}\n\nUse step_add to add steps to this plan.",
            id, description
        )))
    }
}
