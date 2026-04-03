//! Typed identity add tools - one per identity type
//!
//! Each tool has explicit parameters matching what that type needs.
//! No more ambiguous "content" + "secondary" overloading.

use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, McpError, Tool, ToolEnv};

use crate::Context;
use crate::identity::{IdentityField, classify};
use crate::tools::text_response;

/// Classify content and return a warning string if the content looks like
/// it belongs to a different identity field than the one being added to.
fn classification_warning(content: &str, expected: IdentityField) -> Option<String> {
    classify(content).and_then(|c| c.mismatch_warning(expected))
}

// ================================
// PERSONA (singular, replaces)
// ================================

pub struct IdentitySetPersonaNameTool;

impl Tool<Context> for IdentitySetPersonaNameTool {
    fn name(&self) -> &str {
        "identity_set_persona_name"
    }

    fn description(&self) -> &str {
        "Set the persona name."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The persona name"
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
            .ok_or_else(|| McpError::InvalidParams("name is required".into()))?;

        let mut store = context.identity.lock().unwrap();
        let id = store
            .set_persona_name(name)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        Ok(text_response(format!(
            "Set persona name to \"{}\" [{}]",
            name, id
        )))
    }
}

pub struct IdentitySetPersonaDescriptionTool;

impl Tool<Context> for IdentitySetPersonaDescriptionTool {
    fn name(&self) -> &str {
        "identity_set_persona_description"
    }

    fn description(&self) -> &str {
        "Set the persona description."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The persona description"
                }
            },
            "required": ["description"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::InvalidParams("description is required".into()))?;

        let mut store = context.identity.lock().unwrap();
        let id = store
            .set_persona_description(description)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        Ok(text_response(format!("Set persona description [{}]", id)))
    }
}

// ================================
// STRUCTURED TYPES
// ================================

pub struct IdentityAddValueTool;

impl Tool<Context> for IdentityAddValueTool {
    fn name(&self) -> &str {
        "identity_add_value"
    }

    fn description(&self) -> &str {
        "Add a core value or principle."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "principle": {
                    "type": "string",
                    "description": "The value or principle"
                },
                "why": {
                    "type": "string",
                    "description": "Why this value matters (optional)"
                },
                "category": {
                    "type": "string",
                    "description": "Category for grouping (optional)"
                }
            },
            "required": ["principle"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let principle = args
            .get("principle")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::InvalidParams("principle is required".into()))?;
        let why = args.get("why").and_then(|v| v.as_str());
        let category = args.get("category").and_then(|v| v.as_str());

        let warning = classification_warning(principle, IdentityField::Value);

        let mut store = context.identity.lock().unwrap();
        let id = store
            .add_value(principle, why, category)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        let mut msg = format!("Added value: \"{}\" [{}]", principle, id);
        if let Some(w) = warning {
            msg = format!("⚠️ {}\n\n{}", w, msg);
        }
        Ok(text_response(msg))
    }
}

pub struct IdentityAddPreferenceTool;

impl Tool<Context> for IdentityAddPreferenceTool {
    fn name(&self) -> &str {
        "identity_add_preference"
    }

    fn description(&self) -> &str {
        "Add a preference."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "prefer": {
                    "type": "string",
                    "description": "What is preferred"
                },
                "over": {
                    "type": "string",
                    "description": "What it's preferred over (optional)"
                },
                "category": {
                    "type": "string",
                    "description": "Category for grouping (optional)"
                }
            },
            "required": ["prefer"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let prefer = args
            .get("prefer")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::InvalidParams("prefer is required".into()))?;
        let over = args.get("over").and_then(|v| v.as_str());
        let category = args.get("category").and_then(|v| v.as_str());

        let warning = classification_warning(prefer, IdentityField::Preference);

        let mut store = context.identity.lock().unwrap();
        let id = store
            .add_preference(prefer, over, category)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        let msg = if let Some(o) = over {
            format!("Added preference: \"{}\" over \"{}\" [{}]", prefer, o, id)
        } else {
            format!("Added preference: \"{}\" [{}]", prefer, id)
        };

        let mut msg = msg;
        if let Some(w) = warning {
            msg = format!("⚠️ {}\n\n{}", w, msg);
        }
        Ok(text_response(msg))
    }
}

pub struct IdentityAddRelationshipTool;

impl Tool<Context> for IdentityAddRelationshipTool {
    fn name(&self) -> &str {
        "identity_add_relationship"
    }

    fn description(&self) -> &str {
        "Add a relationship with an entity."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "The entity name (person, project, etc.)"
                },
                "relation": {
                    "type": "string",
                    "description": "The relationship description"
                },
                "context": {
                    "type": "string",
                    "description": "Additional context (optional)"
                }
            },
            "required": ["entity", "relation"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let entity = args
            .get("entity")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::InvalidParams("entity is required".into()))?;
        let relation = args
            .get("relation")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::InvalidParams("relation is required".into()))?;
        let context_opt = args.get("context").and_then(|v| v.as_str());

        let classify_text = format!("{} {}", entity, relation);
        let warning = classification_warning(&classify_text, IdentityField::Relationship);

        let mut store = context.identity.lock().unwrap();
        let id = store
            .add_relationship(entity, relation, context_opt)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        let mut msg = format!("Added relationship: {} - {} [{}]", entity, relation, id);
        if let Some(w) = warning {
            msg = format!("⚠️ {}\n\n{}", w, msg);
        }
        Ok(text_response(msg))
    }
}

pub struct IdentityAddRuleTool;

impl Tool<Context> for IdentityAddRuleTool {
    fn name(&self) -> &str {
        "identity_add_rule"
    }

    fn description(&self) -> &str {
        "Add a behavioral rule. Set `negative: true` for 'don't' rules (rendered under Don't:), \
         or omit / set `negative: false` for 'do' rules (rendered under Do:)."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The rule — either 'do X' or 'don't do X'"
                },
                "instead": {
                    "type": "string",
                    "description": "What to do instead (for negative rules) or additional context (optional)"
                },
                "why": {
                    "type": "string",
                    "description": "Why this rule matters — enables judgment in edge cases (optional)"
                },
                "negative": {
                    "type": "boolean",
                    "description": "True for 'don't' rules, false for 'do' rules (default: false)"
                }
            },
            "required": ["content"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::InvalidParams("content is required".into()))?;
        let instead = args.get("instead").and_then(|v| v.as_str());
        let why = args.get("why").and_then(|v| v.as_str());
        let negative = args
            .get("negative")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let warning = classification_warning(content, IdentityField::Rule);

        let mut store = context.identity.lock().unwrap();
        let id = store
            .add_rule(content, instead, why, negative)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        let mut msg = format!("Added rule: \"{}\" [{}]", content, id);
        if let Some(w) = warning {
            msg = format!("⚠️ {}\n\n{}", w, msg);
        }
        Ok(text_response(msg))
    }
}
