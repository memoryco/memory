//! lenses_get - Get a lens by name

use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, Content, PromptDef, Tool, ToolEnv};

use crate::Context;
use crate::lenses::load_lenses;
use crate::tools::text_response;

pub struct LensesGetTool;

impl Tool<Context> for LensesGetTool {
    fn name(&self) -> &str {
        "lenses_get"
    }

    fn description(&self) -> &str {
        "Load a lens by name. Returns the full lens content to hold in working memory \
         during a task. Use lenses_list first to see available lenses."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the lens to load (e.g., 'humanizer')"
                }
            },
            "required": ["name"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| sml_mcps::McpError::InvalidParams("Missing 'name' parameter".into()))?;

        let lenses = load_lenses(&context.lenses_dir);

        let lens = lenses.into_iter().find(|l| l.name() == name);

        match lens {
            Some(l) => {
                // Get the content via get_messages (returns PromptMessage with content)
                let messages = l.get_messages(&std::collections::HashMap::new())?;
                if let Some(msg) = messages.first() {
                    // Extract text from Content enum
                    let text = match &msg.content {
                        Content::Text { text } => text.clone(),
                        _ => "[Non-text content]".to_string(),
                    };
                    Ok(text_response(text))
                } else {
                    Ok(text_response("[Empty lens]".to_string()))
                }
            }
            None => {
                let available: Vec<_> = load_lenses(&context.lenses_dir)
                    .iter()
                    .map(|l| l.name().to_string())
                    .collect();
                Ok(text_response(format!(
                    "Lens '{}' not found. Available: {}",
                    name,
                    available.join(", ")
                )))
            }
        }
    }
}
