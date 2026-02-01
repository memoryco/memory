//! step_complete - Mark a step as completed

use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult};

use crate::Context;
use crate::plans::PlanId;
use crate::tools::text_response;

pub struct StepCompleteTool;

impl Tool<Context> for StepCompleteTool {
    fn name(&self) -> &str {
        "step_complete"
    }

    fn description(&self) -> &str {
        "Mark a step as completed. If this completes all steps, the plan is automatically closed."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The UUID of the plan."
                },
                "step": {
                    "type": "integer",
                    "description": "The step index (1-based) to mark as complete."
                }
            },
            "required": ["id", "step"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let id_str = args["id"].as_str()
            .ok_or_else(|| sml_mcps::McpError::InvalidParams("id is required".to_string()))?;
        let step = args["step"].as_u64()
            .ok_or_else(|| sml_mcps::McpError::InvalidParams("step is required".to_string()))? as usize;
        
        let id = PlanId::parse_str(id_str)
            .map_err(|e| sml_mcps::McpError::InvalidParams(format!("Invalid plan ID: {}", e)))?;

        let mut plans = context.plans.lock().unwrap();
        
        let completed = plans.complete_step(&id, step)
            .map_err(|e| sml_mcps::McpError::ToolError(e.to_string()))?;

        if completed {
            // Get the plan to check progress
            if let Ok(Some(plan)) = plans.get(&id) {
                if plan.is_complete() {
                    // All steps done - auto-close the plan
                    let description = plan.description.clone();
                    plans.stop(&id)
                        .map_err(|e| sml_mcps::McpError::ToolError(e.to_string()))?;
                    
                    Ok(text_response(format!(
                        "Step {} completed.\n\n🎉 All steps complete! Plan \"{}\" auto-closed.",
                        step, description
                    )))
                } else {
                    let progress = format!("{}% ({}/{})", 
                        (plan.progress() * 100.0) as usize,
                        plan.completed_count(),
                        plan.steps.len()
                    );
                    
                    Ok(text_response(format!("Step {} completed. Progress: {}", step, progress)))
                }
            } else {
                Ok(text_response(format!("Step {} completed.", step)))
            }
        } else {
            Ok(text_response(format!("Step {} not found in plan {}", step, id)))
        }
    }
}
