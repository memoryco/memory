//! identity_add - Add to identity with semantic field classification

use serde::Deserialize;
use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult, McpError};

use crate::Context;
use crate::identity::{classify, IdentityField};
use crate::memory_core::{Value, Preference, Relationship, Antipattern};
use crate::tools::text_response;

pub struct IdentityAddTool;

#[derive(Deserialize)]
struct Args {
    /// Which identity field to add to
    field: String,
    /// The content to add (interpretation depends on field)
    content: String,
    /// Optional: "why" for values/antipatterns, "over" for preferences, "relation" for relationships
    #[serde(default)]
    secondary: Option<String>,
    /// Optional: category for values/preferences, context for relationships
    #[serde(default)]
    category: Option<String>,
}

impl Tool<Context> for IdentityAddTool {
    fn name(&self) -> &str {
        "identity_add"
    }

    fn description(&self) -> &str {
        "Add to identity (values, preferences, relationships, expertise, traits, instructions, antipatterns). \
         Uses semantic classification to verify the field type matches the content."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "required": ["field", "content"],
            "properties": {
                "field": {
                    "type": "string",
                    "enum": ["value", "preference", "relationship", "expertise", "trait", "instruction", "antipattern"],
                    "description": "Which identity field to add to"
                },
                "content": {
                    "type": "string",
                    "description": "The content to add. For relationships, this is the entity name."
                },
                "secondary": {
                    "type": "string", 
                    "description": "Optional secondary content: 'why' for values/antipatterns, 'over' for preferences (what it's preferred over), 'relation' for relationships (the relationship description), 'instead' for antipatterns"
                },
                "category": {
                    "type": "string",
                    "description": "Optional category for values/preferences, or context for relationships"
                }
            }
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

        let field = IdentityField::from_str(&args.field)
            .ok_or_else(|| McpError::InvalidParams(format!("Unknown field: {}", args.field)))?;

        // Classify the content to check if it matches the requested field
        let warning = if let Some(classification) = classify(&args.content) {
            classification.mismatch_warning(field)
        } else {
            None
        };

        let mut brain = context.brain.write().unwrap();
        let mut identity = brain.identity().clone();

        // Add to the appropriate field
        let added_description = match field {
            IdentityField::Value => {
                let mut value = Value::new(&args.content);
                if let Some(why) = &args.secondary {
                    value = value.with_why(why);
                }
                if let Some(cat) = &args.category {
                    value = value.with_category(cat);
                }
                identity.values.push(value);
                format!("value: \"{}\"", args.content)
            }
            IdentityField::Preference => {
                let mut pref = Preference::new(&args.content);
                if let Some(over) = &args.secondary {
                    pref = pref.over(over);
                }
                if let Some(cat) = &args.category {
                    pref = pref.with_category(cat);
                }
                identity.preferences.push(pref);
                if let Some(over) = &args.secondary {
                    format!("preference: \"{}\" over \"{}\"", args.content, over)
                } else {
                    format!("preference: \"{}\"", args.content)
                }
            }
            IdentityField::Relationship => {
                let relation = args.secondary.as_deref().unwrap_or("(no relation specified)");
                let mut rel = Relationship::new(&args.content, relation);
                if let Some(ctx) = &args.category {
                    rel = rel.with_context(ctx);
                }
                identity.relationships.push(rel);
                format!("relationship: {} - {}", args.content, relation)
            }
            IdentityField::Expertise => {
                identity.expertise.push(args.content.clone());
                format!("expertise: \"{}\"", args.content)
            }
            IdentityField::Trait => {
                identity.persona.traits.push(args.content.clone());
                format!("trait: \"{}\"", args.content)
            }
            IdentityField::Instruction => {
                identity.instructions.push(args.content.clone());
                format!("instruction: \"{}\"", args.content)
            }
            IdentityField::Antipattern => {
                let mut ap = Antipattern::new(&args.content);
                if let Some(why_or_instead) = &args.secondary {
                    // If it starts with "instead", treat as alternative, otherwise as "why"
                    if why_or_instead.to_lowercase().starts_with("instead") {
                        ap = ap.instead(why_or_instead.trim_start_matches("instead").trim_start_matches(':').trim());
                    } else {
                        ap = ap.because(why_or_instead);
                    }
                }
                identity.antipatterns.push(ap);
                format!("antipattern: \"{}\"", args.content)
            }
        };

        // Save the updated identity
        brain.set_identity(identity)
            .map_err(|e| McpError::ToolError(e.to_string()))?;

        let mut output = format!("Added to identity.{}\n", added_description);
        
        if let Some(warn) = warning {
            output = format!("⚠️ {}\n\n{}", warn, output);
        }

        Ok(text_response(output))
    }
}
