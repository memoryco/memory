//! identity_search - Search identity content

use serde::Deserialize;
use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, McpError, Tool, ToolEnv};

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
        let args: Args =
            serde_json::from_value(args).map_err(|e| McpError::InvalidParams(e.to_string()))?;

        let mut store = context.identity.lock().unwrap();
        let identity = store
            .get()
            .map_err(|e| McpError::ToolError(e.to_string()))?;
        let results = identity.search(&args.query);

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

        if !results.rules.is_empty() {
            output.push_str("Rules:\n");
            for r in results.rules {
                output.push_str(&format!("  • {}\n", r.content));
            }
            output.push('\n');
        }

        Ok(text_response(output))
    }
}
