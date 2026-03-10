//! config_set - Update a configuration value

use serde::Deserialize;
use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, McpError, Tool, ToolEnv};

use crate::Context;
use crate::tools::text_response;

pub struct ConfigSetTool;

#[derive(Deserialize)]
struct Args {
    key: String,
    value: JsonValue,
}

impl Tool<Context> for ConfigSetTool {
    fn name(&self) -> &str {
        "config_set"
    }

    fn description(&self) -> &str {
        "Update a configuration value. Keys: decay_rate_per_day, decay_interval_hours, \
         propagation_damping, hebbian_learning_rate, recall_strength, \
         search_follow_associations, search_association_depth, embedding_model, \
         rerank_enabled, rerank_candidates, hybrid_search_enabled, \
         query_expansion_enabled"
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
                        "recall_strength",
                        "search_follow_associations",
                        "search_association_depth",
                        "embedding_model",
                        "rerank_enabled",
                        "rerank_candidates",
                        "hybrid_search_enabled",
                        "query_expansion_enabled"
                    ],
                    "description": "Configuration key to update."
                },
                "value": {
                    "description": "New value. Number for most keys, string for embedding_model."
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
        let args: Args =
            serde_json::from_value(args).map_err(|e| McpError::InvalidParams(e.to_string()))?;

        // Handle embedding_model separately (string value)
        if args.key == "embedding_model" {
            let model_name = args.value.as_str().ok_or_else(|| {
                McpError::InvalidParams("embedding_model value must be a string".to_string())
            })?;

            if !crate::embedding::is_valid_model(model_name) {
                return Ok(text_response(format!(
                    "Unknown embedding model: {}. Use a valid model name like \
                     SnowflakeArcticEmbedL, AllMiniLML6V2, BGELargeENV15, etc.",
                    model_name
                )));
            }

            let mut brain = context.brain.lock().unwrap();
            let mut config = brain.config().clone();
            config.embedding_model = model_name.to_string();
            // Don't update embedding_model_active — mismatch triggers migration on next startup
            brain
                .set_config(config)
                .map_err(|e| McpError::ToolError(e.to_string()))?;

            return Ok(text_response(format!(
                "Configuration updated: embedding_model = {}. \
                 Migration will occur on next restart (current embeddings: {:?}).",
                model_name,
                brain
                    .config()
                    .embedding_model_active
                    .as_deref()
                    .unwrap_or("(none)")
            )));
        }

        // All other keys are numeric
        let value = args.value.as_f64().ok_or_else(|| {
            McpError::InvalidParams("value must be a number for this config key".to_string())
        })?;

        let mut brain = context.brain.lock().unwrap();

        let updated = brain
            .configure(&args.key, value)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        if updated {
            Ok(text_response(format!(
                "Configuration updated: {} = {}",
                args.key, value
            )))
        } else {
            Ok(text_response(format!(
                "Unknown configuration key: {}",
                args.key
            )))
        }
    }
}
