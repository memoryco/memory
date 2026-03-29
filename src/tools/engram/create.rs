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
        "Create new memories. MANDATORY: call at least once per turn if you learned \
         anything new. Skipping creation means permanent information loss.\n\
         \n\
         ATOMICITY: Each memory generates an embedding vector. Multiple unrelated concepts \
         in one memory dilute the embedding, making retrieval unreliable. One fact per memory.\n\
         \n\
         Splitting rules:\n\
         - Numbered lists (1), (2), (3) \u{2014} each item = separate memory\n\
         - More than 2 sentences \u{2014} likely multiple facts, split them\n\
         - If removing half would leave a complete useful memory, split it\n\
         - Repeat shared context as prefix: 'Project X: decision one', 'Project X: decision two'\n\
         - Remove date prefixes \u{2014} created_at captures timestamps automatically\n\
         \n\
         What to store: project facts, architectural decisions, gotchas/workarounds, corrections, \
         personal context, workflow discoveries, user preferences, repeatable processes.\n\
         Skip: exact duplicates, ephemeral task state, info already in Identity, meta-observations.\n\
         \n\
         On long turns, create incrementally in batches of ~3\u{2013}5 facts as you discover them. \
         Use the memories array parameter to batch multiple atomic memories per call."
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

        // Session accumulation: feed each memory's content into the session context,
        // track created IDs, and collect precomputed embeddings — all in one load/save.
        let mut precomputed_embeddings: std::collections::HashMap<EngramId, Vec<f32>> =
            std::collections::HashMap::new();
        {
            let brain = context.brain.read().unwrap();
            let config = brain.config();
            let max_queries = config.session_max_queries;
            let smoothing = config.session_centroid_smoothing;

            let mut session = brain
                .load_session(&args.session_id)
                .unwrap_or(None)
                .unwrap_or_else(|| crate::engram::SessionContext::new(&args.session_id));

            session.touch();

            let generator = crate::embedding::EmbeddingGenerator::new();

            for (id, memory) in created.iter().zip(args.memories.iter()) {
                session.add_query(&memory.content, max_queries);
                if let Ok(embedding) = generator.generate(&memory.content) {
                    session.update_centroid(&embedding, smoothing);
                    precomputed_embeddings.insert(id.0, embedding);
                }
            }

            // Track created IDs in the same session save
            let created_uuids: Vec<uuid::Uuid> = created.iter().map(|(id, _)| *id).collect();
            session.add_created(&created_uuids);

            if let Err(e) = brain.save_session(&session) {
                eprintln!("[session] Failed to save session {}: {}", &args.session_id, e);
            }
        }

        // Phase 1b: Wire associations between all created/recalled IDs in this session.
        // Requires write lock (separate phase from read-lock session accumulation above).
        super::wire_session_associations(context, &args.session_id);

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
