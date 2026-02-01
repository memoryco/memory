//! plan_get - Get a single plan with all its steps

use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult};

use crate::Context;
use crate::plans::PlanId;
use crate::tools::text_response;

pub struct PlanGetTool;

impl Tool<Context> for PlanGetTool {
    fn name(&self) -> &str {
        "plan_get"
    }

    fn description(&self) -> &str {
        "Get a single plan with all its steps and their completion status."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The UUID of the plan to retrieve."
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
        let id_str = args["id"].as_str()
            .ok_or_else(|| sml_mcps::McpError::InvalidParams("id is required".to_string()))?;
        
        let id = PlanId::parse_str(id_str)
            .map_err(|e| sml_mcps::McpError::InvalidParams(format!("Invalid plan ID: {}", e)))?;

        let mut plans = context.plans.lock().unwrap();
        
        let plan = plans.get(&id)
            .map_err(|e| sml_mcps::McpError::ToolError(e.to_string()))?;

        match plan {
            Some(p) => {
                let steps_output = if p.steps.is_empty() {
                    "  (no steps yet)".to_string()
                } else {
                    p.steps.iter()
                        .map(|s| {
                            let check = if s.completed { "✓" } else { " " };
                            format!("  [{}] {}. {}", check, s.index, s.description)
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                };

                let progress = if p.steps.is_empty() {
                    "0%".to_string()
                } else {
                    format!("{}% ({}/{})", 
                        (p.progress() * 100.0) as usize,
                        p.completed_count(),
                        p.steps.len()
                    )
                };

                let output = format!(
                    "Plan: {}\nID: {}\nProgress: {}\n\nSteps:\n{}",
                    p.description,
                    p.id,
                    progress,
                    steps_output
                );

                Ok(text_response(output))
            }
            None => Ok(text_response(format!("Plan not found: {}", id)))
        }
    }
}
