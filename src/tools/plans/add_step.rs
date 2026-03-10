//! step_add - Add a step to a plan

use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, Tool, ToolEnv};

use crate::Context;
use crate::plans::PlanId;
use crate::tools::text_response;

pub struct StepAddTool;

impl Tool<Context> for StepAddTool {
    fn name(&self) -> &str {
        "step_add"
    }

    fn description(&self) -> &str {
        "Add a step to a plan. Steps are appended in order and assigned sequential indices."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The UUID of the plan to add a step to."
                },
                "description": {
                    "type": "string",
                    "description": "What needs to be done in this step."
                }
            },
            "required": ["id", "description"]
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
        let description = args["description"].as_str().ok_or_else(|| {
            sml_mcps::McpError::InvalidParams("description is required".to_string())
        })?;

        let id = PlanId::parse_str(id_str)
            .map_err(|e| sml_mcps::McpError::InvalidParams(format!("Invalid plan ID: {}", e)))?;

        let mut plans = context.plans.lock().unwrap();

        let step_index = plans
            .add_step(&id, description)
            .map_err(|e| sml_mcps::McpError::ToolError(e.to_string()))?;

        Ok(text_response(format!(
            "Step {} added: {}",
            step_index, description
        )))
    }
}
