//! engram_create - Create new memories (batch)

use crate::embedding::EmbeddingGenerator;
use crate::engram::EngramId;
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult, McpError};

use crate::Context;
use crate::tools::text_response;

pub struct EngramCreateTool;

#[derive(Deserialize, Clone)]
struct MemoryInput {
    content: String,
}

#[derive(Deserialize)]
struct Args {
    memories: Vec<MemoryInput>,
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
                    "description": "Array of memories to create. Each item has 'content'.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": { "type": "string" }
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
        
        // Create embedding generator once for the batch
        let generator = EmbeddingGenerator::new();
        
        let mut output = String::new();
        let mut created_count = 0;

        for memory in &args.memories {
            let id: EngramId = brain.create(&memory.content)
                .map_err(|e| McpError::ToolError(e.to_string()))?;

            // Generate and save embedding
            if let Ok(embedding) = generator.generate(&memory.content) {
                let _ = brain.set_embedding(&id, &embedding);
            }

            created_count += 1;
            output.push_str(&format!(
                "ID: {}\nContent: {}\n\n",
                id, memory.content
            ));
        }

        let header = format!("{} memories created.\n\n", created_count);
        Ok(text_response(format!("{}{}", header, output.trim())))
    }
}
