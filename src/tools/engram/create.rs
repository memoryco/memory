//! engram_create - Create new memories (batch)

use crate::embedding::EmbeddingGenerator;
use crate::engram::EngramId;
use crate::llm::SharedLlmService;
use serde::Deserialize;
use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, McpError, Tool, ToolEnv};

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
    /// Session ID for context-aware retrieval
    session_id: String,
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
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID for context-aware retrieval."
                }
            },
            "required": ["memories", "session_id"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let args: Args =
            serde_json::from_value(args).map_err(|e| McpError::InvalidParams(e.to_string()))?;

        let mut created: Vec<(EngramId, String)> = Vec::new();
        let mut output = String::new();

        // Phase 1: Create memories (holding write lock briefly)
        {
            let mut brain = context.brain.write().unwrap();
            for memory in &args.memories {
                let id: EngramId = if let Some(ref ts) = memory.created_at {
                    let epoch = parse_timestamp(ts).map_err(|e| {
                        McpError::InvalidParams(format!("Invalid created_at '{}': {}", ts, e))
                    })?;
                    brain.create_with_timestamp(&memory.content, epoch)
                } else {
                    brain.create(&memory.content)
                }
                .map_err(|e| McpError::ToolError(e.to_string()))?;

                output.push_str(&format!("ID: {}\nContent: {}\n\n", id, memory.content));

                created.push((id, memory.content.clone()));
            }
        } // Lock released here

        // Session accumulation: feed each memory's content into the session context.
        // Collect any embeddings generated here so we can reuse them for storage
        // instead of regenerating them in the background thread.
        let mut precomputed_embeddings: std::collections::HashMap<EngramId, Vec<f32>> =
            std::collections::HashMap::new();

        for (id, memory) in created.iter().zip(args.memories.iter()) {
            if let Some(embedding) =
                super::accumulate_session_signal(context, &args.session_id, &memory.content)
            {
                precomputed_embeddings.insert(id.0, embedding);
            }
        }

        // Phase 2a: Spawn per-memory threads for primary embeddings (fast, parallel).
        // If session accumulation already generated the embedding, store it directly
        // instead of regenerating — same model, same text, deterministic output.
        for (id, content) in created.iter() {
            if let Some(embedding) = precomputed_embeddings.remove(id) {
                // Reuse the embedding from session accumulation
                let brain_clone = context.brain.clone();
                let id_clone = *id;
                std::thread::spawn(move || {
                    if let Ok(brain) = brain_clone.read() {
                        let _ = brain.set_embedding(&id_clone, &embedding);
                    }
                });
            } else {
                // No session or embedding generation failed — generate fresh
                let brain_clone = context.brain.clone();
                let id_clone = *id;
                let content_clone = content.clone();
                std::thread::spawn(move || {
                    let generator = EmbeddingGenerator::new();
                    if let Ok(embedding) = generator.generate(&content_clone)
                        && let Ok(brain) = brain_clone.read()
                    {
                        let _ = brain.set_embedding(&id_clone, &embedding);
                    }
                });
            }
        }

        // Phase 2b: Spawn ONE thread for LLM enrichment (sequential, no contention)
        // LLM inference serializes internally, so 438 threads just thrash.
        // One worker processes the queue and avoids the thread storm.
        {
            let brain_clone = context.brain.clone();
            let llm_clone: SharedLlmService = context.llm.clone();
            let batch: Vec<(EngramId, String)> = created.clone();

            std::thread::spawn(move || {
                if !llm_clone.available() {
                    return;
                }

                let generator = EmbeddingGenerator::new();

                for (id, content) in &batch {
                    if let Ok(queries) = llm_clone.generate_training_queries(content, 5) {
                        let enrichment_embeddings: Vec<Vec<f32>> = queries
                            .iter()
                            .filter_map(|q| generator.generate(q).ok())
                            .collect();
                        if !enrichment_embeddings.is_empty() {
                            // Use read lock: set_enrichment_embeddings is &self.
                            if let Ok(brain) = brain_clone.read() {
                                let n = enrichment_embeddings.len();
                                let preview: String = content.chars().take(50).collect();
                                let _ = brain.set_enrichment_embeddings(
                                    id,
                                    &enrichment_embeddings,
                                    "llm",
                                );
                                eprintln!(
                                    "[create] enrichment: {} vectors for \"{}\"",
                                    n, preview
                                );
                            }
                        }
                    }
                }
            });
        }

        // Phase 3: Return immediately
        let header = format!(
            "session_id: {}\n\n{} memories created (embeddings generating in background).\n\n",
            args.session_id,
            created.len()
        );
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

    Err(format!(
        "Unrecognized timestamp format. Expected unix epoch, ISO 8601 date (YYYY-MM-DD), or ISO 8601 datetime."
    ))
}
