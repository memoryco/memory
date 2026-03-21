//! engram_search - Semantic memory search with energy weighting

use serde::Deserialize;
use serde_json::{Value as JsonValue, json};
use sml_mcps::{CallToolResult, McpError, Tool, ToolEnv};

use crate::Context;
use crate::tools::text_response;

use std::path::{Path, PathBuf};

/// Write a verbose pipeline trace to MEMORY_HOME/logs/<session_id>.log.
///
/// Each search query appends to the same session log file, separated by
/// a timestamp header. Returns the file path on success, None on failure.
fn write_trace_file(
    memory_home: &Path,
    session_id: &str,
    query: &str,
    info: &crate::engram::search::DebugInfo,
) -> Option<PathBuf> {
    use std::fmt::Write as FmtWrite;
    use std::io::Write;

    let logs_dir = memory_home.join("logs");
    if let Err(e) = std::fs::create_dir_all(&logs_dir) {
        eprintln!("[search] failed to create logs dir: {}", e);
        return None;
    }

    let log_path = logs_dir.join(format!("{}.log", session_id));

    let mut buf = String::new();
    let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
    let _ = writeln!(buf, "\n{}", "=".repeat(80));
    let _ = writeln!(buf, "SEARCH TRACE  |  {}  |  query: {:?}", timestamp, query);
    let _ = writeln!(buf, "{}", "=".repeat(80));

    // Summary lines
    for line in &info.lines {
        let _ = writeln!(buf, "  {}", line);
    }

    // Stage snapshots
    for stage in &info.stages {
        let _ = writeln!(buf, "\n--- stage: {} ({} candidates) ---", stage.name, stage.entries.len());
        let _ = writeln!(buf, "{:<4} {:<38} {:>8} {:>8} {:>8}  {}",
            "#", "ID", "sim", "energy", "blended", "content");
        let _ = writeln!(buf, "{}", "-".repeat(120));
        for (i, entry) in stage.entries.iter().enumerate() {
            let _ = writeln!(buf, "{:<4} {:<38} {:>8.4} {:>8.4} {:>8.4}  {}",
                i + 1,
                entry.id,
                entry.similarity,
                entry.energy,
                entry.blended,
                entry.content_preview,
            );
        }
    }

    let _ = writeln!(buf);

    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
    {
        Ok(mut file) => {
            if let Err(e) = file.write_all(buf.as_bytes()) {
                eprintln!("[search] failed to write trace: {}", e);
                return None;
            }
            Some(log_path)
        }
        Err(e) => {
            eprintln!("[search] failed to open trace file: {}", e);
            None
        }
    }
}

pub struct EngramSearchTool;

#[derive(Deserialize)]
struct Args {
    /// Text query to search for
    query: String,
    /// Maximum results to return (default: 10)
    #[serde(default)]
    limit: Option<usize>,
    /// Minimum similarity score (0.0-1.0, default: 0.4)
    #[serde(default)]
    min_score: Option<f32>,
    /// Include deep storage memories (default: false)
    #[serde(default)]
    include_deep: Option<bool>,
    /// Include archived memories (default: false)
    #[serde(default)]
    include_archived: Option<bool>,
    /// Only include memories created after this time (ISO 8601 or unix epoch seconds)
    #[serde(default)]
    created_after: Option<String>,
    /// Only include memories created before this time (ISO 8601 or unix epoch seconds)
    #[serde(default)]
    created_before: Option<String>,
    /// Session ID for context-aware retrieval
    session_id: String,
}

#[derive(Deserialize)]
struct BatchArgs {
    /// Array of search queries to run
    queries: Vec<String>,
    /// Maximum results per query (default: 10)
    #[serde(default)]
    limit: Option<usize>,
    /// Minimum similarity score (0.0-1.0, default: 0.4)
    #[serde(default)]
    min_score: Option<f32>,
    /// Include deep storage memories (default: false)
    #[serde(default)]
    include_deep: Option<bool>,
    /// Include archived memories (default: false)
    #[serde(default)]
    include_archived: Option<bool>,
    /// Only include memories created after this time (ISO 8601 or unix epoch seconds)
    #[serde(default)]
    created_after: Option<String>,
    /// Only include memories created before this time (ISO 8601 or unix epoch seconds)
    #[serde(default)]
    created_before: Option<String>,
    /// Session ID for context-aware retrieval
    session_id: String,
}

impl EngramSearchTool {
    /// Run a single search query through the full pipeline.
    /// Called by the batch `execute` wrapper and by `identity_get`.
    pub fn search_for_query(
        &self,
        args: JsonValue,
        context: &mut Context,
        _env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let args: Args =
            serde_json::from_value(args).map_err(|e| McpError::InvalidParams(e.to_string()))?;

        let limit = args.limit.unwrap_or(10);
        let include_deep = args.include_deep.unwrap_or(false);
        let include_archived = args.include_archived.unwrap_or(false);
        let composite_query = crate::engram::search::is_composite_query(&args.query);
        let inferential_query = crate::engram::search::is_inferential_query(&args.query);

        // Parse optional date-range filters
        let created_after: Option<i64> = args
            .created_after
            .as_deref()
            .map(crate::engram::search::parse_timestamp)
            .transpose()
            .map_err(|e| McpError::InvalidParams(e))?;
        let created_before: Option<i64> = args
            .created_before
            .as_deref()
            .map(crate::engram::search::parse_timestamp)
            .transpose()
            .map_err(|e| McpError::InvalidParams(e))?;

        let (
            follow_associations,
            association_depth,
            rerank_mode,
            rerank_candidates,
            hybrid_search_enabled,
            query_expansion_enabled,
            search_min_score_config,
            composite_limit_min,
            composite_limit_max,
            association_cap_min,
            association_cap_max,
            debug,
        ) = {
            // Phase 0: brief write lock for maintenance + config snapshot.
            // Drop the write lock before expensive embedding generation so enrichment
            // threads can acquire write locks for set_embedding / set_enrichment_embeddings.
            let mut brain = context.brain.write().unwrap();

            // Lazy maintenance: decay + cross-process sync
            let _ = brain.apply_time_decay();
            let _ = brain.sync_from_storage();

            // Read config for association-following, reranking, hybrid search, and query expansion
            (
                brain.config().search_follow_associations,
                brain.config().search_association_depth,
                brain.config().rerank_mode.clone(),
                brain.config().rerank_candidates,
                brain.config().hybrid_search_enabled,
                brain.config().query_expansion_enabled,
                brain.config().search_min_score,
                brain.config().composite_limit_min,
                brain.config().composite_limit_max,
                brain.config().association_cap_min,
                brain.config().association_cap_max,
                brain.config().debug,
            )
        }; // write lock released here

        let min_score = args.min_score.unwrap_or(search_min_score_config as f32);

        // List/composite questions often need more context than the nominal limit.
        let effective_limit = if composite_query {
            limit.max(composite_limit_min).min(composite_limit_max)
        } else {
            limit
        };

        // Query expansion: generate variant queries
        let variants = if query_expansion_enabled {
            crate::engram::search::llm_query_variants(&context.llm, &args.query)
                .unwrap_or_else(|| super::query_expansion::expand_query(&args.query))
        } else {
            vec![args.query.clone()]
        };

        if variants.len() > 1 {
            eprintln!(
                "[search] query expansion: {} variants from {:?}",
                variants.len(),
                &args.query
            );
        }

        // Pre-compute fallback terms (used by pipeline phases 2, 3, and diversity shaping)
        let fallback_terms = super::query_expansion::fallback_terms(&args.query);

        // Phase 1+: read lock for the actual search pipeline.
        // Multiple concurrent searches can hold read locks simultaneously.
        // Enrichment threads use read locks (set_embedding is &self via storage Mutex),
        // so they can interleave with search rather than being blocked for 10-15 seconds.
        let brain = context.brain.read().unwrap();

        // When reranking is enabled, fetch more candidates for the reranker to work with
        let fetch_count = if rerank_mode != "off" {
            rerank_candidates.max(effective_limit * 3)
        } else {
            effective_limit * 3
        };

        // Load session centroid for affinity biasing
        let (session_centroid, session_context_weight) = {
            let config_weight = brain.config().session_context_weight;
            if config_weight > 0.0 {
                match brain.load_session(&args.session_id) {
                    Ok(Some(session)) => (session.centroid, config_weight),
                    _ => (None, 0.0),
                }
            } else {
                (None, 0.0)
            }
        };

        let params = crate::engram::search::SearchPipelineParams {
            query: args.query.clone(),
            variants,
            fallback_terms,
            effective_limit,
            composite_query,
            inferential_query,
            min_score,
            fetch_count,
            follow_associations,
            association_depth,
            rerank_mode,
            hybrid_search_enabled,
            association_cap_min,
            association_cap_max,
            include_deep,
            include_archived,
            created_after,
            created_before,
            query_expansion_enabled,
            session_centroid,
            session_context_weight,
            debug,
        };

        let pipeline_result =
            crate::engram::search::run_search_pipeline(&brain, &context.llm, &params)
                .map_err(|e| McpError::ToolError(e))?;

        let scored = pipeline_result.results;
        let chain_hints = pipeline_result.chain_hints;
        let association_merged_count = pipeline_result.association_merged_count;
        let debug_info = pipeline_result.debug_info;
        let association_discovery_count = pipeline_result.association_discovery_count;

        // Save search state for access log (correlated with next recall)
        {
            let result_ids: Vec<uuid::Uuid> = scored.iter().map(|r| r.id).collect();
            if let Ok(mut q) = context.last_search_query.lock() {
                *q = Some(args.query.clone());
            }
            if let Ok(mut r) = context.last_search_result_ids.lock() {
                *r = result_ids;
            }
        }

        // Drop read lock before session accumulation (accumulate_session_signal acquires its own read lock)
        drop(brain);

        // Session accumulation: feed the query into the session context
        super::accumulate_session_signal(context, &args.session_id, &args.query);

        if scored.is_empty() {
            let mut output = format!(
                "session_id: {}\n\n\
                 ⚡ **REQUIRED:** Call engram_recall on IDs you use. \n\
                 💾 **REQUIRED:** Call engram_create if you learn ANY new facts this turn.\n\
                 ---\n\n",
                args.session_id
            );
            output.push_str("No memories found.\n");
            output.push_str("\n💡 Tip: Try a different query or lower min_score.");
            return Ok(text_response(output));
        }

        let mut output = format!(
            "session_id: {}\n\n\
             ⚡ **REQUIRED:** Call engram_recall on IDs you use. \n\
             💾 **REQUIRED:** Call engram_create if you learn ANY new facts this turn.\n\
             ---\n\n",
            args.session_id
        );

        // Show chain hints before memory listings
        for hint in &chain_hints {
            let truncated_content =
                crate::engram::search::truncate_chain_hint_content(&hint.anchor_content, 40);
            output.push_str(&format!(
                "🔗 Procedure chain detected: {} [{}] has {} ordered steps. Use engram_associations to walk the chain.\n\n",
                truncated_content, hint.anchor_id, hint.ordered_step_count
            ));
        }

        if association_merged_count > 0 {
            output.push_str(&format!(
                "Found {} memories ({} via associations, {} total discovered):\n\n",
                scored.len(),
                association_merged_count,
                association_discovery_count
            ));
        } else {
            output.push_str(&format!("Found {} memories:\n\n", scored.len()));
        }

        for r in &scored {
            // Format created_at as readable date
            let created_date = chrono::DateTime::from_timestamp(r.created_at, 0)
                .map(|dt| dt.format("%Y-%m-%d %H:%M UTC").to_string())
                .unwrap_or_else(|| "unknown".to_string());

            output.push_str(&format!(
                "ID: {}\nContent: {}\nCreated: {}\nScore: {:.4} (sim: {:.4}, energy: {:.4}) | State: {}\n\n",
                r.id,
                r.content,
                created_date,
                r.blended,
                r.similarity,
                r.energy,
                r.state_emoji,
            ));
        }

        // Sparse/weak results hint: nudge the caller to try related searches
        let should_hint =
            scored.len() < 3 || scored.first().map(|r| r.blended < 0.45).unwrap_or(false);
        if should_hint {
            output.push_str(
                "⚠️ Results may not be relevant. Try searching with related concepts:\n\
                 - Break abstract questions into concrete topics (e.g. \"relationship status\" → \
                   \"breakup\", \"dating\", \"partner\", \"married\", \"single\")\n\
                 - Search for specific events or actions rather than status or state\n\
                 - Try [person's name] + a related action, event, or feeling\n",
            );
        }

        // Append debug diagnostics when enabled
        if let Some(info) = &debug_info {
            output.push_str("\n--- memoryco debug info ---\n");
            for line in &info.lines {
                output.push_str(line);
                output.push('\n');
            }

            // Write verbose trace file when stages were collected
            if !info.stages.is_empty() {
                let trace_path = write_trace_file(
                    &context.memory_home,
                    &args.session_id,
                    &args.query,
                    info,
                );
                if let Some(path) = trace_path {
                    output.push_str(&format!("trace: {}\n", path.display()));
                }
            }
        }

        Ok(text_response(output))
    }
}

impl Tool<Context> for EngramSearchTool {
    fn name(&self) -> &str {
        "engram_search"
    }

    fn description(&self) -> &str {
        "Search memories by semantic similarity using vector embeddings. \
         Accepts an array of queries to batch multiple searches in a single call. \
         Finds memories with similar meaning even if they don't share exact keywords. \
         Supports date-range filtering via created_after/created_before to narrow \
         results by when memories were stored. \
         If results seem weak or irrelevant, try decomposing abstract queries into \
         concrete related terms. For example, instead of 'relationship status', \
         try 'breakup', 'dating', 'partner', or 'married'. Search for actions \
         and events rather than abstract states."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "required": ["queries"],
            "properties": {
                "queries": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Array of search queries. Each query runs through the \
                        full search pipeline independently."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results per query. Default: 10"
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum similarity score (0.0-1.0). Default: 0.4"
                },
                "include_deep": {
                    "type": "boolean",
                    "description": "Include deep storage memories. Default: false"
                },
                "include_archived": {
                    "type": "boolean",
                    "description": "Include archived memories. Default: false"
                },
                "created_after": {
                    "type": "string",
                    "description": "Only include memories created after this time. \
                        Accepts ISO 8601 (e.g. 2026-03-14, 2026-03-14T19:00:00Z) \
                        or unix epoch seconds."
                },
                "created_before": {
                    "type": "string",
                    "description": "Only include memories created before this time. \
                        Accepts ISO 8601 (e.g. 2026-03-14, 2026-03-14T19:00:00Z) \
                        or unix epoch seconds."
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID for context-aware retrieval."
                }
            },
            "required": ["queries", "session_id"]
        })
    }

    fn execute(
        &self,
        args: JsonValue,
        context: &mut Context,
        env: &ToolEnv,
    ) -> sml_mcps::Result<CallToolResult> {
        let batch: BatchArgs = serde_json::from_value(args)
            .map_err(|e| McpError::InvalidParams(e.to_string()))?;

        let mut output = String::new();

        for (i, query) in batch.queries.iter().enumerate() {
            if batch.queries.len() > 1 {
                output.push_str(&format!("\n## Query {}: {:?}\n\n", i + 1, query));
            }

            let single_args = json!({
                "query": query,
                "limit": batch.limit,
                "min_score": batch.min_score,
                "include_deep": batch.include_deep,
                "include_archived": batch.include_archived,
                "created_after": batch.created_after,
                "created_before": batch.created_before,
                "session_id": batch.session_id,
            });

            match self.search_for_query(single_args, context, env) {
                Ok(result) => {
                    output.push_str(&crate::tools::extract_text(&result));
                }
                Err(e) => {
                    output.push_str(&format!("Search failed for {:?}: {}\n", query, e));
                }
            }
        }

        Ok(text_response(output))
    }
}
