//! config_set - Update a configuration value

use serde::Deserialize;
use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, McpError, Tool, ToolEnv};

use crate::Context;
use crate::engram::config_toml::{ConfigValue, write_config_key};
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

            // Write to config.toml (durable)
            write_config_key(
                &context.memory_home,
                "embedding_model",
                &ConfigValue::Str(model_name.to_string()),
            )
            .map_err(|e| McpError::ToolError(e.to_string()))?;

            // Also update in-memory config for the current session
            let mut brain = context.brain.lock().unwrap();
            let mut config = brain.config().clone();
            config.embedding_model = model_name.to_string();
            // Don't update embedding_model_active — mismatch triggers migration on next startup
            brain
                .set_config(config)
                .map_err(|e| McpError::ToolError(e.to_string()))?;

            return Ok(text_response(format!(
                "Configuration updated: embedding_model = {}. \
                 Migration will occur on next restart (current embeddings: {:?}). \
                 Change saved to config.toml. Takes full effect on restart.",
                model_name,
                brain
                    .config()
                    .embedding_model_active
                    .as_deref()
                    .unwrap_or("(none)")
            )));
        }

        // Boolean keys
        let bool_keys = [
            "search_follow_associations",
            "rerank_enabled",
            "hybrid_search_enabled",
            "query_expansion_enabled",
        ];

        // Integer keys
        let int_keys = ["search_association_depth", "rerank_candidates"];

        // All other keys are numeric (booleans accepted as true/false/0/1 in any form)
        let value = args
            .value
            .as_f64()
            .or_else(|| args.value.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
            .or_else(|| {
                // Handle string representations: "true", "false", "0", "1", "3.14"
                args.value.as_str().and_then(|s| match s {
                    "true" => Some(1.0),
                    "false" => Some(0.0),
                    other => other.parse::<f64>().ok(),
                })
            })
            .ok_or_else(|| {
                McpError::InvalidParams(
                    "value must be a number (or true/false for boolean keys)".to_string(),
                )
            })?;

        // Determine the TOML value type for this key
        let toml_value = if bool_keys.contains(&args.key.as_str()) {
            ConfigValue::Bool(value != 0.0)
        } else if int_keys.contains(&args.key.as_str()) {
            ConfigValue::Int(value.clamp(0.0, usize::MAX as f64) as usize)
        } else {
            ConfigValue::Float(value)
        };

        // Write to config.toml (durable)
        write_config_key(&context.memory_home, &args.key, &toml_value)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        // Also update in-memory config for the current session
        let mut brain = context.brain.lock().unwrap();
        let updated = brain
            .configure(&args.key, value)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        if updated {
            Ok(text_response(format!(
                "Configuration updated: {} = {}. Change saved to config.toml. Takes full effect on restart.",
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
