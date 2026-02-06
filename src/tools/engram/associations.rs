//! engram_associations - Inspect associations for a single memory

use crate::engram::EngramId;
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult, McpError};

use crate::Context;
use crate::tools::{text_response, truncate_content};

pub struct EngramAssociationsTool;

#[derive(Deserialize)]
struct Args {
    id: String,
    #[serde(default)]
    direction: Option<String>,
}

impl Tool<Context> for EngramAssociationsTool {
    fn name(&self) -> &str {
        "engram_associations"
    }

    fn description(&self) -> &str {
        "Inspect associations for a specific memory. Shows detailed info including \
         weights, co-activation counts (Hebbian learning indicator), and timestamps. \
         Use direction to see outbound, inbound, or both."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The UUID of the memory to inspect."
                },
                "direction": {
                    "type": "string",
                    "enum": ["outbound", "inbound", "both"],
                    "description": "Which associations to show. Default: outbound"
                }
            },
            "required": ["id"]
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

        let id: EngramId = args.id.parse()
            .map_err(|e| McpError::InvalidParams(format!("Invalid UUID: {}", e)))?;

        let direction = args.direction.as_deref().unwrap_or("outbound");
        let mut brain = context.brain.lock().unwrap();
        let _ = brain.sync_from_storage();

        let memory_info = match brain.get(&id) {
            Some(e) => format!("Memory: {} (energy: {:.2})\n", truncate_content(&e.content, 60), e.energy),
            None => return Ok(text_response(format!("Memory {} not found.", id))),
        };

        let mut output = memory_info;
        output.push_str(&format!("Direction: {}\n\n", direction));

        // Outbound associations
        if direction == "outbound" || direction == "both" {
            output.push_str("=== OUTBOUND (this memory → others) ===\n");
            match brain.associations_from(&id) {
                Some(assocs) if !assocs.is_empty() => {
                    let mut sorted: Vec<_> = assocs.iter().collect();
                    sorted.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal));
                    
                    for assoc in sorted {
                        let target_preview = brain.get(&assoc.to)
                            .map(|e| truncate_content(&e.content, 40))
                            .unwrap_or_else(|| "[deleted]".to_string());
                        
                        output.push_str(&format!(
                            "  → {} (weight: {:.3}, co-activations: {})\n    {}\n",
                            assoc.to,
                            assoc.weight,
                            assoc.co_activation_count,
                            target_preview
                        ));
                    }
                }
                _ => output.push_str("  (none)\n"),
            }
            output.push('\n');
        }

        // Inbound associations
        if direction == "inbound" || direction == "both" {
            output.push_str("=== INBOUND (others → this memory) ===\n");
            match brain.associations_to(&id) {
                Some(source_ids) if !source_ids.is_empty() => {
                    for source_id in source_ids {
                        let assoc_info = brain.associations_from(source_id)
                            .and_then(|assocs| assocs.iter().find(|a| a.to == id));
                        
                        let source_preview = brain.get(source_id)
                            .map(|e| truncate_content(&e.content, 40))
                            .unwrap_or_else(|| "[deleted]".to_string());
                        
                        if let Some(assoc) = assoc_info {
                            output.push_str(&format!(
                                "  ← {} (weight: {:.3}, co-activations: {})\n    {}\n",
                                source_id,
                                assoc.weight,
                                assoc.co_activation_count,
                                source_preview
                            ));
                        } else {
                            output.push_str(&format!(
                                "  ← {}\n    {}\n",
                                source_id,
                                source_preview
                            ));
                        }
                    }
                }
                _ => output.push_str("  (none)\n"),
            }
        }

        Ok(text_response(output))
    }
}
