//! Typed identity add tools - one per identity type
//!
//! Each tool has explicit parameters matching what that type needs.
//! No more ambiguous "content" + "secondary" overloading.

use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult, McpError};

use crate::Context;
use crate::identity::{classify, IdentityField};
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
        "Set the persona name. Replaces any existing name."
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
        let name = args.get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::InvalidParams("name is required".into()))?;

        let mut store = context.identity.lock().unwrap();
        let id = store.set_persona_name(name)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        Ok(text_response(format!("Set persona name to \"{}\" [{}]", name, id)))
    }
}

pub struct IdentitySetPersonaDescriptionTool;

impl Tool<Context> for IdentitySetPersonaDescriptionTool {
    fn name(&self) -> &str {
        "identity_set_persona_description"
    }

    fn description(&self) -> &str {
        "Set the persona description. Replaces any existing description."
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
        let description = args.get("description")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::InvalidParams("description is required".into()))?;

        let mut store = context.identity.lock().unwrap();
        let id = store.set_persona_description(description)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        Ok(text_response(format!("Set persona description [{}]", id)))
    }
}

// ================================
// SIMPLE STRING TYPES
// ================================

pub struct IdentityAddTraitTool;

impl Tool<Context> for IdentityAddTraitTool {
    fn name(&self) -> &str {
        "identity_add_trait"
    }

    fn description(&self) -> &str {
        "Add a personality trait."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "trait": {
                    "type": "string",
                    "description": "The trait to add (e.g., 'direct', 'curious', 'pragmatic')"
                }
            },
            "required": ["trait"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let trait_name = args.get("trait")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::InvalidParams("trait is required".into()))?;

        let warning = classification_warning(trait_name, IdentityField::Trait);

        let mut store = context.identity.lock().unwrap();
        let id = store.add_trait(trait_name)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        let mut msg = format!("Added trait: \"{}\" [{}]", trait_name, id);
        if let Some(w) = warning { msg = format!("⚠️ {}\n\n{}", w, msg); }
        Ok(text_response(msg))
    }
}

pub struct IdentityAddExpertiseTool;

impl Tool<Context> for IdentityAddExpertiseTool {
    fn name(&self) -> &str {
        "identity_add_expertise"
    }

    fn description(&self) -> &str {
        "Add an area of expertise."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "area": {
                    "type": "string",
                    "description": "The area of expertise (e.g., 'Rust', 'distributed systems')"
                }
            },
            "required": ["area"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let area = args.get("area")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::InvalidParams("area is required".into()))?;

        let warning = classification_warning(area, IdentityField::Expertise);

        let mut store = context.identity.lock().unwrap();
        let id = store.add_expertise(area)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        let mut msg = format!("Added expertise: \"{}\" [{}]", area, id);
        if let Some(w) = warning { msg = format!("⚠️ {}\n\n{}", w, msg); }
        Ok(text_response(msg))
    }
}

pub struct IdentityAddInstructionTool;

impl Tool<Context> for IdentityAddInstructionTool {
    fn name(&self) -> &str {
        "identity_add_instruction_v2"
    }

    fn description(&self) -> &str {
        "Add an operational instruction. Instructions are permanent directives that don't decay."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "The instruction text"
                }
            },
            "required": ["instruction"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let instruction = args.get("instruction")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::InvalidParams("instruction is required".into()))?;

        let warning = classification_warning(instruction, IdentityField::Instruction);

        let mut store = context.identity.lock().unwrap();
        let id = store.add_instruction(instruction)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        let mut msg = format!("Added instruction [{}]", id);
        if let Some(w) = warning { msg = format!("⚠️ {}\n\n{}", w, msg); }
        Ok(text_response(msg))
    }
}

pub struct IdentityAddToneTool;

impl Tool<Context> for IdentityAddToneTool {
    fn name(&self) -> &str {
        "identity_add_tone"
    }

    fn description(&self) -> &str {
        "Add a communication tone."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "tone": {
                    "type": "string",
                    "description": "The tone to add (e.g., 'direct', 'friendly', 'technical')"
                }
            },
            "required": ["tone"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let tone = args.get("tone")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::InvalidParams("tone is required".into()))?;

        let mut store = context.identity.lock().unwrap();
        let id = store.add_tone(tone)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        Ok(text_response(format!("Added tone: \"{}\" [{}]", tone, id)))
    }
}

pub struct IdentityAddDirectiveTool;

impl Tool<Context> for IdentityAddDirectiveTool {
    fn name(&self) -> &str {
        "identity_add_directive"
    }

    fn description(&self) -> &str {
        "Add a communication directive."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "directive": {
                    "type": "string",
                    "description": "The directive to add"
                }
            },
            "required": ["directive"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let directive = args.get("directive")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::InvalidParams("directive is required".into()))?;

        let mut store = context.identity.lock().unwrap();
        let id = store.add_directive(directive)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        Ok(text_response(format!("Added directive: \"{}\" [{}]", directive, id)))
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
        let principle = args.get("principle")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::InvalidParams("principle is required".into()))?;
        let why = args.get("why").and_then(|v| v.as_str());
        let category = args.get("category").and_then(|v| v.as_str());

        let warning = classification_warning(principle, IdentityField::Value);

        let mut store = context.identity.lock().unwrap();
        let id = store.add_value(principle, why, category)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        let mut msg = format!("Added value: \"{}\" [{}]", principle, id);
        if let Some(w) = warning { msg = format!("⚠️ {}\n\n{}", w, msg); }
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
        let prefer = args.get("prefer")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::InvalidParams("prefer is required".into()))?;
        let over = args.get("over").and_then(|v| v.as_str());
        let category = args.get("category").and_then(|v| v.as_str());

        let warning = classification_warning(prefer, IdentityField::Preference);

        let mut store = context.identity.lock().unwrap();
        let id = store.add_preference(prefer, over, category)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        let msg = if let Some(o) = over {
            format!("Added preference: \"{}\" over \"{}\" [{}]", prefer, o, id)
        } else {
            format!("Added preference: \"{}\" [{}]", prefer, id)
        };

        let mut msg = msg;
        if let Some(w) = warning { msg = format!("⚠️ {}\n\n{}", w, msg); }
        Ok(text_response(msg))
    }
}

pub struct IdentityAddRelationshipTool;

impl Tool<Context> for IdentityAddRelationshipTool {
    fn name(&self) -> &str {
        "identity_add_relationship"
    }

    fn description(&self) -> &str {
        "Add a relationship with an entity (person, project, organization)."
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
        let entity = args.get("entity")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::InvalidParams("entity is required".into()))?;
        let relation = args.get("relation")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::InvalidParams("relation is required".into()))?;
        let context_opt = args.get("context").and_then(|v| v.as_str());

        let classify_text = format!("{} {}", entity, relation);
        let warning = classification_warning(&classify_text, IdentityField::Relationship);

        let mut store = context.identity.lock().unwrap();
        let id = store.add_relationship(entity, relation, context_opt)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        let mut msg = format!("Added relationship: {} - {} [{}]", entity, relation, id);
        if let Some(w) = warning { msg = format!("⚠️ {}\n\n{}", w, msg); }
        Ok(text_response(msg))
    }
}

pub struct IdentityAddAntipatternTool;

impl Tool<Context> for IdentityAddAntipatternTool {
    fn name(&self) -> &str {
        "identity_add_antipattern"
    }

    fn description(&self) -> &str {
        "Add an antipattern - something to avoid."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "avoid": {
                    "type": "string",
                    "description": "What to avoid"
                },
                "instead": {
                    "type": "string",
                    "description": "What to do instead (optional)"
                },
                "why": {
                    "type": "string",
                    "description": "Why this should be avoided (optional)"
                }
            },
            "required": ["avoid"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let avoid = args.get("avoid")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::InvalidParams("avoid is required".into()))?;
        let instead = args.get("instead").and_then(|v| v.as_str());
        let why = args.get("why").and_then(|v| v.as_str());

        let warning = classification_warning(avoid, IdentityField::Antipattern);

        let mut store = context.identity.lock().unwrap();
        let id = store.add_antipattern(avoid, instead, why)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        let mut msg = format!("Added antipattern: \"{}\" [{}]", avoid, id);
        if let Some(w) = warning { msg = format!("⚠️ {}\n\n{}", w, msg); }
        Ok(text_response(msg))
    }
}
