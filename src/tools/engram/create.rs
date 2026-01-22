//! engram_create - Create new memories (batch)

use engram::EngramId;
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult, McpError};

use crate::Context;
use crate::tools::text_response;

pub struct EngramCreateTool;

#[derive(Deserialize, Clone)]
struct MemoryInput {
    content: String,
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Deserialize)]
struct Args {
    memories: Vec<MemoryInput>,
}

/// Result of filtering tags through the blocklist and cardinality checks
struct FilteredTags {
    /// Tags that passed filtering
    tags: Vec<String>,
    /// Tags that were blocked
    blocked: Vec<String>,
    /// Tags that exceed cardinality threshold (tag, count)
    high_cardinality: Vec<(String, usize)>,
}

impl Tool<Context> for EngramCreateTool {
    fn name(&self) -> &str {
        "engram_create"
    }

    fn description(&self) -> &str {
        "Create new memories. Accepts an array of memories to create in one call. \
         Memories start with full energy and decay over time without use."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "memories": {
                    "type": "array",
                    "description": "Array of memories to create. Each item has 'content' and optional 'tags'.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": { "type": "string" },
                            "tags": { "type": "array", "items": { "type": "string" } }
                        },
                        "required": ["content"]
                    }
                }
            },
            "required": ["memories"]
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

        let mut brain = context.brain.lock().unwrap();
        
        // Get config for tag filtering
        let blocked_tags: Vec<String> = brain.config().blocked_tags
            .iter()
            .map(|t| t.to_lowercase())
            .collect();
        let max_cardinality = brain.config().max_tag_cardinality;
        
        let mut output = String::new();
        let mut warnings = String::new();
        let mut created_count = 0;

        for memory in &args.memories {
            // Filter tags
            let filtered = filter_tags(
                &memory.tags,
                &blocked_tags,
                max_cardinality,
                |tag| brain.tag_count(tag),
            );
            
            // Collect warnings
            if !filtered.blocked.is_empty() {
                warnings.push_str(&format!(
                    "⚠️ Blocked tags stripped: {:?}\n",
                    filtered.blocked
                ));
            }
            if !filtered.high_cardinality.is_empty() {
                for (tag, count) in &filtered.high_cardinality {
                    warnings.push_str(&format!(
                        "⚠️ Tag '{}' is on {} memories (threshold: {}) - consider more specific tag\n",
                        tag, count, max_cardinality
                    ));
                }
            }
            
            let id: EngramId = if filtered.tags.is_empty() {
                brain.create(&memory.content)
            } else {
                brain.create_with_tags(&memory.content, filtered.tags.clone())
            }.map_err(|e| McpError::ToolError(e.to_string()))?;

            created_count += 1;
            output.push_str(&format!(
                "ID: {}\nContent: {}\nTags: {:?}\n\n",
                id, memory.content, filtered.tags
            ));
        }

        let header = format!("{} memories created.\n\n", created_count);
        let result = if warnings.is_empty() {
            format!("{}{}", header, output.trim())
        } else {
            format!("{}{}\n{}", warnings, header, output.trim())
        };
        
        Ok(text_response(result))
    }
}

/// Filter tags through blocklist and cardinality checks
fn filter_tags<F>(
    tags: &[String],
    blocked_tags: &[String],
    max_cardinality: usize,
    tag_count_fn: F,
) -> FilteredTags 
where
    F: Fn(&str) -> usize,
{
    let mut result = FilteredTags {
        tags: Vec::new(),
        blocked: Vec::new(),
        high_cardinality: Vec::new(),
    };
    
    for tag in tags {
        let tag_lower = tag.to_lowercase();
        
        // Check blocklist
        if blocked_tags.contains(&tag_lower) {
            result.blocked.push(tag.clone());
            continue;
        }
        
        // Check cardinality
        let count = tag_count_fn(tag);
        if count >= max_cardinality {
            result.high_cardinality.push((tag.clone(), count));
            // Still allow it, just warn
        }
        
        result.tags.push(tag.clone());
    }
    
    result
}
