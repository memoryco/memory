//! identity_set - Set the identity from JSON

use crate::memory_core::Identity;
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult, McpError};

use crate::Context;
use crate::tools::text_response;

pub struct IdentitySetTool;

#[derive(Deserialize)]
struct Args {
    identity: Identity,
}

impl Tool<Context> for IdentitySetTool {
    fn name(&self) -> &str {
        "identity_set"
    }

    fn description(&self) -> &str {
        "Set the entire identity from JSON. Replaces existing."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "identity": {
                    "type": "object",
                    "description": "The identity object with persona, values, preferences, etc.",
                    "properties": {
                        "persona": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string" },
                                "description": { "type": "string" },
                                "traits": { "type": "array", "items": { "type": "string" } }
                            }
                        },
                        "values": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "principle": { "type": "string" },
                                    "why": { "type": "string" },
                                    "category": { "type": "string" }
                                }
                            }
                        },
                        "preferences": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "prefer": { "type": "string" },
                                    "over": { "type": "string" },
                                    "category": { "type": "string" }
                                }
                            }
                        },
                        "relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entity": { "type": "string" },
                                    "relation": { "type": "string" },
                                    "context": { "type": "string" }
                                }
                            }
                        },
                        "antipatterns": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "avoid": { "type": "string" },
                                    "why": { "type": "string" },
                                    "instead": { "type": "string" }
                                }
                            }
                        },
                        "communication": {
                            "type": "object",
                            "properties": {
                                "tone": { "type": "array", "items": { "type": "string" } },
                                "directives": { "type": "array", "items": { "type": "string" } }
                            }
                        },
                        "expertise": {
                            "type": "array",
                            "items": { "type": "string" }
                        }
                    }
                }
            },
            "required": ["identity"]
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

        let mut brain = context.brain.write().unwrap();

        brain.set_identity(args.identity.clone())
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        Ok(text_response(format!(
            "Identity set.\n\n{}",
            args.identity.render()
        )))
    }
}
