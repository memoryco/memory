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
    /// Optional creation timestamp (ISO 8601 or unix epoch seconds).
    /// If omitted, defaults to now. Useful for importing historical data.
    #[serde(default)]
    created_at: Option<String>,
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
                            "content": { "type": "string" },
                            "created_at": {
                                "type": "string",
                                "description": "Optional creation timestamp (ISO 8601 datetime or unix epoch seconds). Defaults to now. Use for importing historical data."
                            }
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
                let id: EngramId = if let Some(ref ts) = memory.created_at {
                    let epoch = parse_timestamp(ts)
                        .map_err(|e| McpError::InvalidParams(format!("Invalid created_at '{}': {}", ts, e)))?;
                    brain.create_with_timestamp(&memory.content, epoch)
                } else {
                    brain.create(&memory.content)
                }.map_err(|e| McpError::ToolError(e.to_string()))?;
                
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

/// Parse a timestamp string into unix epoch seconds.
/// Accepts:
///   - Unix epoch seconds (integer or float): "1684937600"
///   - ISO 8601 date: "2023-05-24"
///   - ISO 8601 datetime: "2023-05-24T14:30:00Z" or "2023-05-24T14:30:00+00:00"
fn parse_timestamp(s: &str) -> Result<i64, String> {
    let trimmed = s.trim();

    // Try as unix epoch (integer)
    if let Ok(epoch) = trimmed.parse::<i64>() {
        return Ok(epoch);
    }

    // Try as unix epoch (float)
    if let Ok(epoch_f) = trimmed.parse::<f64>() {
        return Ok(epoch_f as i64);
    }

    // Try ISO 8601 datetime with timezone
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(trimmed) {
        return Ok(dt.timestamp());
    }

    // Try ISO 8601 datetime without timezone (assume UTC)
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(trimmed, "%Y-%m-%dT%H:%M:%S") {
        return Ok(dt.and_utc().timestamp());
    }

    // Try ISO 8601 date only (midnight UTC)
    if let Ok(d) = chrono::NaiveDate::parse_from_str(trimmed, "%Y-%m-%d") {
        return Ok(d.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp());
    }

    // Try common date formats: "24 May 2023", "May 24, 2023"
    if let Ok(d) = chrono::NaiveDate::parse_from_str(trimmed, "%d %B %Y") {
        return Ok(d.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp());
    }
    if let Ok(d) = chrono::NaiveDate::parse_from_str(trimmed, "%B %d, %Y") {
        return Ok(d.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp());
    }

    Err(format!("Unrecognized timestamp format. Expected unix epoch, ISO 8601 date (YYYY-MM-DD), or ISO 8601 datetime."))
}
