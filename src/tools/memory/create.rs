//! memory_create - Create new memories (batch)

use crate::embedding::EmbeddingGenerator;
use crate::memory_core::MemoryId;
use crate::llm::SharedLlmService;
use serde::Deserialize;
use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, McpError, Tool, ToolEnv};

use crate::Context;
use crate::tools::text_response;

/// Strip redundant date anchors from memory content.
/// The AI habitually inserts " (YYYY-MM-DD)" as temporal markers,
/// but created_at already captures this. Stripping avoids embedding pollution.
fn strip_date_anchors(content: &str) -> String {
    // Match " (YYYY-MM-DD)" — just the parenthesized date, preserving any trailing colon
    static RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r" \(\d{4}-\d{2}-\d{2}\)").unwrap()
    });
    let result = RE.replace_all(content, "");
    // Clean up any resulting double spaces
    let cleaned = result.replace("  ", " ");
    cleaned.trim().to_string()
}

pub struct MemoryCreateTool;

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

impl Tool<Context> for MemoryCreateTool {
    fn name(&self) -> &str {
        "memory_create"
    }

    fn description(&self) -> &str {
        "Store new memories. One fact per memory \u{2014} compound memories dilute \
         embeddings. Mandatory at least once per turn if you learned anything."
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

        let mut created: Vec<(MemoryId, String)> = Vec::new();
        let mut output = String::new();

        // Phase 1: Create memories (holding write lock briefly)
        {
            let mut brain = context.brain.write().unwrap();
            for memory in &args.memories {
                let content = strip_date_anchors(&memory.content);
                let id: MemoryId = if let Some(ref ts) = memory.created_at {
                    let epoch = parse_timestamp(ts).map_err(|e| {
                        McpError::InvalidParams(format!("Invalid created_at '{}': {}", ts, e))
                    })?;
                    brain.create_with_timestamp(&content, epoch)
                } else {
                    brain.create(&content)
                }
                .map_err(|e| McpError::ToolError(e.to_string()))?;

                output.push_str(&format!("{}, ", id));

                created.push((id, content));
            }
        } // Lock released here

        // Session accumulation: feed each memory's content into the session context,
        // track created IDs, and collect precomputed embeddings — all in one load/save.
        let mut precomputed_embeddings: std::collections::HashMap<MemoryId, Vec<f32>> =
            std::collections::HashMap::new();
        {
            let brain = context.brain.read().unwrap();
            let config = brain.config();
            let max_queries = config.session_max_queries;
            let smoothing = config.session_centroid_smoothing;

            let mut session = brain
                .load_session(&args.session_id)
                .unwrap_or(None)
                .unwrap_or_else(|| crate::memory_core::SessionContext::new(&args.session_id));

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
            let batch: Vec<(MemoryId, String)> = created.clone();

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
            "session_id: {}\n\n{} memories created.\n\nIDs: {}",
            args.session_id,
            created.len(),
            output.trim_end_matches(", ")
        );
        Ok(text_response(header))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_date_anchor_with_colon() {
        assert_eq!(
            strip_date_anchors("SEARCH PIPELINE CHANGE (2026-04-08): Removed energy blending"),
            "SEARCH PIPELINE CHANGE: Removed energy blending"
        );
    }

    #[test]
    fn strip_date_anchor_without_colon() {
        assert_eq!(
            strip_date_anchors("Tool output slimming (2026-04-08) summary"),
            "Tool output slimming summary"
        );
    }

    #[test]
    fn strip_date_anchor_mid_sentence() {
        assert_eq!(
            strip_date_anchors("cosmos.gl confirmed working (2026-04-03): renders 3362 nodes"),
            "cosmos.gl confirmed working: renders 3362 nodes"
        );
    }

    #[test]
    fn preserve_date_as_fact() {
        // Date IS the fact — no parens, no anchor pattern
        assert_eq!(
            strip_date_anchors("Brandon's birthday is 2026-03-15"),
            "Brandon's birthday is 2026-03-15"
        );
    }

    #[test]
    fn preserve_date_reference_in_prose() {
        assert_eq!(
            strip_date_anchors("The API changed on 2026-01-15 to use v2 endpoints"),
            "The API changed on 2026-01-15 to use v2 endpoints"
        );
    }

    #[test]
    fn strip_multiple_date_anchors() {
        assert_eq!(
            strip_date_anchors("First (2026-01-01): then second (2026-02-02): done"),
            "First: then second: done"
        );
    }

    #[test]
    fn no_op_on_clean_content() {
        assert_eq!(
            strip_date_anchors("Memory with no date anchors at all"),
            "Memory with no date anchors at all"
        );
    }
}
