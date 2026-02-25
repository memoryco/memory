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

        let mut created: Vec<(EngramId, String)> = Vec::new();
        let mut output = String::new();

        // Phase 1: Create memories (holding lock)
        {
            let mut brain = context.brain.lock().unwrap();
            for memory in &args.memories {
                let id: EngramId = brain.create(&memory.content)
                    .map_err(|e| McpError::ToolError(e.to_string()))?;
                
                output.push_str(&format!(
                    "ID: {}\nContent: {}\n\n",
                    id, memory.content
                ));
                
                created.push((id, memory.content.clone()));
            }
        } // Lock released here

        // Phase 2: Spawn background embedding tasks (no lock held)
        for (id, content) in created.iter() {
            let brain_clone = context.brain.clone();
            let id_clone = *id;
            let content_clone = content.clone();

            std::thread::spawn(move || {
                let generator = EmbeddingGenerator::new();
                if let Ok(embedding) = generator.generate(&content_clone)
                    && let Ok(mut brain) = brain_clone.lock()
                {
                    let _ = brain.set_embedding(&id_clone, &embedding);
                }
            });
        }

        // Phase 3: Return immediately
        let header = format!("{} memories created (embeddings generating in background).\n\n", created.len());
        Ok(text_response(format!("{}{}", header, output.trim())))
    }
}
