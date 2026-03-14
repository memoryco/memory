//! engram_recall - Actively recall memories and optionally create new ones

use crate::engram::EngramId;
use serde::Deserialize;
use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, McpError, Tool, ToolEnv};

use crate::Context;
use crate::tools::engram::EngramCreateTool;
use crate::tools::{extract_text, format_engram, text_response};

pub struct EngramRecallTool;

#[derive(Deserialize)]
struct MemoryInput {
    content: String,
    #[serde(default)]
    created_at: Option<String>,
}

#[derive(Deserialize)]
struct Args {
    ids: Vec<String>,
    #[serde(default)]
    strength: Option<f64>,
    #[serde(default)]
    create_memories: Option<Vec<MemoryInput>>,
    /// Session ID for context-aware retrieval
    session_id: String,
}

impl Tool<Context> for EngramRecallTool {
    fn name(&self) -> &str {
        "engram_recall"
    }

    fn description(&self) -> &str {
        "Actively recall memories and optionally create new ones in a single call. \
         Stimulates recalled memories (increases energy), triggers Hebbian learning \
         between them, and can resurrect archived memories. Pass create_memories to also \
         store new facts in the same call \u{2014} use this at end of turn to handle \
         both recall and creation in one round trip."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Array of UUIDs to recall. All memories will be linked via Hebbian learning."
                },
                "strength": {
                    "type": "number",
                    "description": "Stimulation strength (0.0-1.0). Default uses config value."
                },
                "create_memories": {
                    "type": "array",
                    "description": "Optional array of memories to create alongside recall. \
                        Same format as engram_create. Use this to combine end-of-turn recall \
                        and storage into a single call.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": { "type": "string" },
                            "created_at": {
                                "type": "string",
                                "description": "Optional creation timestamp (ISO 8601 or unix epoch seconds). Defaults to now."
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
            "required": ["ids", "session_id"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let args: Args =
            serde_json::from_value(args).map_err(|e| McpError::InvalidParams(e.to_string()))?;

        // Phase 1: Recall memories (brain write lock scoped to drop before create)
        let (header, output) = {
            let mut brain = context.brain.write().unwrap();

            // Lazy maintenance: decay + cross-process sync
            let _ = brain.apply_time_decay();
            let _ = brain.sync_from_storage();

            let mut output = String::new();
            let mut recalled_count = 0;
            let mut not_found_count = 0;
            let mut total_affected = 0;

            for id_str in &args.ids {
                let id: EngramId = id_str.parse().map_err(|e| {
                    McpError::InvalidParams(format!("Invalid UUID '{}': {}", id_str, e))
                })?;

                let result = if let Some(s) = args.strength {
                    brain.recall_with_strength(id, s)
                } else {
                    brain.recall(id)
                }
                .map_err(|e| McpError::ToolError(e.to_string()))?;

                if !result.found() {
                    not_found_count += 1;
                    output.push_str(&format!("Memory {} not found.\n\n", id));
                    continue;
                }

                recalled_count += 1;
                total_affected += result.affected_count();

                let engram = result.engram.as_ref().unwrap();
                output.push_str(&format_engram(engram));

                if result.resurrected {
                    output.push_str(&format!(
                        "\n🔄 RESURRECTED from {:?}!",
                        result.previous_state.unwrap()
                    ));
                }
                output.push_str("\n\n");
            }

            // Log the search→recall cycle for membed training data extraction
            {
                let query_text = context
                    .last_search_query
                    .lock()
                    .ok()
                    .and_then(|mut q| q.take());
                let result_ids = context
                    .last_search_result_ids
                    .lock()
                    .ok()
                    .map(|mut r| std::mem::take(&mut *r))
                    .unwrap_or_default();

                if let Some(query) = query_text {
                    let recalled: Vec<EngramId> = args
                        .ids
                        .iter()
                        .filter_map(|id_str| id_str.parse().ok())
                        .collect();
                    if let Err(e) = brain.log_access(&query, &result_ids, &recalled) {
                        eprintln!("[recall] access log write failed: {}", e);
                    }
                }
            }

            let header = format!(
                "Recalled {} memories ({} not found). Affected {} total via Hebbian learning.\n\n",
                recalled_count, not_found_count, total_affected
            );

            (header, output)
        }; // brain write lock dropped here

        // Session accumulation: blend recalled memory embeddings into session centroid.
        // Uses existing embeddings from storage — no new embedding generation needed.
        {
            let brain = context.brain.read().unwrap();
            let config = brain.config();
            let smoothing = config.session_centroid_smoothing;

            let mut session = brain
                .load_session(&args.session_id)
                .unwrap_or(None)
                .unwrap_or_else(|| crate::engram::SessionContext::new(&args.session_id));

            session.touch();

            for id_str in &args.ids {
                if let Ok(id) = id_str.parse::<crate::engram::EngramId>()
                    && let Ok(Some(embedding)) = brain.get_embedding(&id)
                {
                    session.update_centroid(&embedding, smoothing);
                }
            }

            if let Err(e) = brain.save_session(&session) {
                eprintln!("[session] Failed to save session {}: {}", &args.session_id, e);
            }
        }

        let mut final_output = format!("session_id: {}\n\n{}\n{}", args.session_id, header, output.trim());

        // Phase 2: Create new memories if provided (bounce to EngramCreateTool)
        if let Some(create_memories) = &args.create_memories {
            if !create_memories.is_empty() {
                let create_tool = EngramCreateTool;
                let memories_json: Vec<JsonValue> = create_memories
                    .iter()
                    .map(|m| {
                        let mut obj = json!({ "content": m.content });
                        if let Some(ref ts) = m.created_at {
                            obj["created_at"] = json!(ts);
                        }
                        obj
                    })
                    .collect();

                let create_args = json!({ "memories": memories_json, "session_id": args.session_id });

                final_output.push_str("\n\n---\n\n");
                match create_tool.execute(create_args, context, env) {
                    Ok(result) => {
                        final_output.push_str(&extract_text(&result));
                    }
                    Err(e) => {
                        final_output.push_str(&format!("Memory creation failed: {}\n", e));
                    }
                }
            }
        }

        Ok(text_response(final_output))
    }
}
