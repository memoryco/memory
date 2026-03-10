//! plan_stop - Stop (delete) a plan

use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, Tool, ToolEnv};

use crate::Context;
use crate::plans::PlanId;
use crate::tools::text_response;

pub struct PlanStopTool;

impl Tool<Context> for PlanStopTool {
    fn name(&self) -> &str {
        "plan_stop"
    }

    fn description(&self) -> &str {
        "Stop and delete a plan. Use when the plan is complete or no longer needed."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The UUID of the plan to stop/delete."
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
        let id_str = args["id"]
            .as_str()
            .ok_or_else(|| sml_mcps::McpError::InvalidParams("id is required".to_string()))?;

        let id = PlanId::parse_str(id_str)
            .map_err(|e| sml_mcps::McpError::InvalidParams(format!("Invalid plan ID: {}", e)))?;

        let mut plans = context.plans.lock().unwrap();

        let deleted = plans
            .stop(&id)
            .map_err(|e| sml_mcps::McpError::ToolError(e.to_string()))?;

        if deleted {
            Ok(text_response(format!("Plan stopped and deleted: {}", id)))
        } else {
            Ok(text_response(format!("Plan not found: {}", id)))
        }
    }
}
