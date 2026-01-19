//! engram_create - Create new memories (batch)

use engram::EngramId;
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};
use sovran_mcp::server::server::{McpTool, McpToolEnvironment};
use sovran_mcp::types::{CallToolResponse, McpError};

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

impl McpTool<Context> for EngramCreateTool {
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
        _env: &McpToolEnvironment,
    ) -> Result<CallToolResponse, McpError> {
        let args: Args = serde_json::from_value(args)
            .map_err(|e| McpError::InvalidArguments(e.to_string()))?;

        let mut brain = context.brain.lock().unwrap();
        let mut output = String::new();
        let mut created_count = 0;

        for memory in &args.memories {
            let id: EngramId = if memory.tags.is_empty() {
                brain.create(&memory.content)
            } else {
                brain.create_with_tags(&memory.content, memory.tags.clone())
            }.map_err(|e| McpError::Other(e.to_string()))?;

            created_count += 1;
            output.push_str(&format!(
                "ID: {}\nContent: {}\nTags: {:?}\n\n",
                id, memory.content, memory.tags
            ));
        }

        let header = format!("{} memories created.\n\n", created_count);
        Ok(text_response(format!("{}{}", header, output.trim())))
    }
}
