//! identity_search - Search identity content

use serde::Deserialize;
use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult, McpError};

use crate::Context;
use crate::tools::text_response;

pub struct IdentitySearchTool;

#[derive(Deserialize)]
struct Args {
    query: String,
}

impl Tool<Context> for IdentitySearchTool {
    fn name(&self) -> &str {
        "identity_search"
    }

    fn description(&self) -> &str {
        "Search identity for matching values, preferences, relationships, etc."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text to search for in identity."
                }
            },
            "required": ["query"]
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

        let brain = context.brain.lock().unwrap();
        let results = brain.identity().search(&args.query);

        if results.is_empty() {
            return Ok(text_response(format!(
                "No identity content matching '{}'",
                args.query
            )));
        }

        let mut output = format!(
            "Found {} matches for '{}':\n\n",
            results.total_count(),
            args.query
        );

        if !results.values.is_empty() {
            output.push_str("Values:\n");
            for v in results.values {
                output.push_str(&format!("  • {}\n", v.principle));
            }
            output.push('\n');
        }

        if !results.preferences.is_empty() {
            output.push_str("Preferences:\n");
            for p in results.preferences {
                if let Some(over) = &p.over {
                    output.push_str(&format!("  • {} > {}\n", p.prefer, over));
                } else {
                    output.push_str(&format!("  • {}\n", p.prefer));
                }
            }
            output.push('\n');
        }

        if !results.relationships.is_empty() {
            output.push_str("Relationships:\n");
            for r in results.relationships {
                output.push_str(&format!("  • {}: {}\n", r.entity, r.relation));
            }
            output.push('\n');
        }

        if !results.antipatterns.is_empty() {
            output.push_str("Antipatterns:\n");
            for a in results.antipatterns {
                output.push_str(&format!("  ✗ {}\n", a.avoid));
            }
            output.push('\n');
        }

        if !results.expertise.is_empty() {
            let exp: Vec<&str> = results.expertise.iter().map(|s| s.as_str()).collect();
            output.push_str(&format!("Expertise: {}\n", exp.join(", ")));
        }

        if !results.traits.is_empty() {
            let traits: Vec<&str> = results.traits.iter().map(|s| s.as_str()).collect();
            output.push_str(&format!("Traits: {}\n", traits.join(", ")));
        }

        if !results.instructions.is_empty() {
            output.push_str("Instructions:\n");
            for i in results.instructions {
                output.push_str(&format!("  • {}\n", i));
            }
        }

        Ok(text_response(output))
    }
}
