//! config_set - Update a configuration value

use serde::Deserialize;
use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult, McpError};

use crate::Context;
use crate::tools::text_response;

pub struct ConfigSetTool;

#[derive(Deserialize)]
struct Args {
    key: String,
    value: f64,
}

impl Tool<Context> for ConfigSetTool {
    fn name(&self) -> &str {
        "config_set"
    }

    fn description(&self) -> &str {
        "Update a configuration value. Keys: decay_rate_per_day, decay_interval_hours, \
         propagation_damping, hebbian_learning_rate, recall_strength"
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "enum": [
                        "decay_rate_per_day",
                        "decay_interval_hours",
                        "propagation_damping",
                        "hebbian_learning_rate",
                        "recall_strength"
                    ],
                    "description": "Configuration key to update."
                },
                "value": {
                    "type": "number",
                    "description": "New value."
                }
            },
            "required": ["key", "value"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let args: Args = serde_json::from_value(args)
            .map_err(|e| McpError::InvalidParams(e.to_string()))?;

        let mut brain = context.brain.lock().unwrap();

        let updated = brain.configure(&args.key, args.value)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        if updated {
            Ok(text_response(format!(
                "Configuration updated: {} = {}",
                args.key, args.value
            )))
        } else {
            Ok(text_response(format!(
                "Unknown configuration key: {}",
                args.key
            )))
        }
    }
}
