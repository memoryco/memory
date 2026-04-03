//! identity_list - List identity items by type with IDs

use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, McpError, Tool, ToolEnv};

use crate::Context;
use crate::identity::{IdentityItemType, ListedItem};
use crate::tools::text_response;

pub struct IdentityListTool;

impl Tool<Context> for IdentityListTool {
    fn name(&self) -> &str {
        "identity_list"
    }

    fn description(&self) -> &str {
        "List identity items of a specific type with their IDs."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "The type of identity item to list",
                    "enum": [
                        "persona_name", "persona_description",
                        "value", "preference", "relationship", "rule"
                    ]
                }
            },
            "required": ["type"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let type_str = args
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::InvalidParams("type is required".into()))?;

        let item_type = parse_item_type(type_str)
            .ok_or_else(|| McpError::InvalidParams(format!("Invalid type: {}", type_str)))?;

        let mut store = context.identity.lock().unwrap();
        let items = store
            .list(item_type)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        if items.is_empty() {
            return Ok(text_response(format!("No {} items found.", type_str)));
        }

        let mut output = format!("{} items ({}):\n\n", type_str, items.len());

        for item in items {
            let display = format_item(&item);
            output.push_str(&format!("• [{}] {}\n", item.id, display));
        }

        Ok(text_response(output))
    }
}

fn format_item(item: &ListedItem) -> String {
    let mut parts = vec![item.content.clone()];
    if let Some(ref s) = item.secondary {
        parts.push(format!("({})", s));
    }
    if let Some(ref t) = item.tertiary {
        parts.push(format!("[{}]", t));
    }
    if let Some(ref c) = item.category {
        parts.push(format!("{{{}}}", c));
    }
    parts.join(" ")
}

fn parse_item_type(s: &str) -> Option<IdentityItemType> {
    match s {
        "persona_name" => Some(IdentityItemType::PersonaName),
        "persona_description" => Some(IdentityItemType::PersonaDescription),
        "value" => Some(IdentityItemType::Value),
        "preference" => Some(IdentityItemType::Preference),
        "relationship" => Some(IdentityItemType::Relationship),
        "rule" => Some(IdentityItemType::Rule),
        _ => None,
    }
}
