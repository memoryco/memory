//! engram_search - Semantic memory search with energy weighting

use crate::embedding::EmbeddingGenerator;
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};
use sml_mcps::{Tool, ToolEnv, CallToolResult, McpError};

use crate::Context;
use crate::tools::text_response;

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
}

impl Tool<Context> for EngramSearchTool {
    fn name(&self) -> &str {
        "engram_search"
    }

    fn description(&self) -> &str {
        "Search memories by semantic similarity using vector embeddings. \
         Finds memories with similar meaning even if they don't share exact keywords."
    }

    fn schema(&self) -> JsonValue {
        json!({
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text to search for semantically. Finds memories with similar meaning."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return. Default: 10"
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
                }
            }
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

        let limit = args.limit.unwrap_or(10);
        let min_score = args.min_score.unwrap_or(0.4);
        let include_deep = args.include_deep.unwrap_or(false);
        let include_archived = args.include_archived.unwrap_or(false);

        let mut brain = context.brain.lock().unwrap();

        // Lazy maintenance: decay + cross-process sync
        let _ = brain.apply_time_decay();
        let _ = brain.sync_from_storage();

        // Read config for association-following
        let follow_associations = brain.config().search_follow_associations;
        let association_depth = brain.config().search_association_depth;

        // Generate query embedding
        let generator = EmbeddingGenerator::new();
        let query_embedding = generator.generate(&args.query)
            .map_err(|e| McpError::ToolError(format!("Failed to generate embedding: {}", e)))?;

        // Find similar memories (fetch extra to allow for filtering)
        let vector_results = brain.find_similar_by_embedding(&query_embedding, limit * 3, min_score)
            .map_err(|e| McpError::ToolError(format!("Vector search failed: {}", e)))?;

        // Score blending: 70% semantic similarity, 30% energy (recency/relevance)
        // This boosts recently-used memories even if semantic match is slightly lower
        struct ScoredResult {
            id: uuid::Uuid,
            content: String,
            similarity: f32,
            energy: f64,
            blended: f32,
            state_emoji: &'static str,
        }

        let mut scored: Vec<ScoredResult> = vector_results.iter().filter_map(|r| {
            let engram = brain.get_or_load(&r.id)?;

            // State filter
            let state_ok = if include_archived {
                true
            } else if include_deep {
                !engram.is_archived()
            } else {
                engram.is_searchable()
            };
            if !state_ok { return None; }

            let blended = r.score * 0.7 + (engram.energy as f32) * 0.3;

            Some(ScoredResult {
                id: r.id,
                content: r.content.clone(),
                similarity: r.score,
                energy: engram.energy,
                blended,
                state_emoji: engram.state.emoji(),
            })
        }).collect();

        // Sort by blended score descending
        scored.sort_by(|a, b| b.blended.partial_cmp(&a.blended).unwrap_or(std::cmp::Ordering::Equal));

        // Association-following: discover related memories via associations
        let mut association_discovery_count: usize = 0;
        let mut association_merged_count: usize = 0;
        if follow_associations && association_depth > 0 && !scored.is_empty() {
            use std::collections::HashSet;

            let seed_ids: Vec<uuid::Uuid> = scored.iter().map(|r| r.id).collect();
            let mut seen_ids: HashSet<uuid::Uuid> = seed_ids.iter().copied().collect();

            let discoveries = brain.discover_associated_memories(
                &query_embedding,
                &seed_ids,
                &mut seen_ids,
                association_depth,
            ).map_err(|e| McpError::ToolError(format!("Association discovery failed: {}", e)))?;

            // Cap discoveries to avoid swamping vector results
            association_discovery_count = discoveries.len();
            let capped: Vec<_> = {
                let mut sorted = discoveries;
                sorted.sort_by(|a, b| b.blended_score.partial_cmp(&a.blended_score).unwrap_or(std::cmp::Ordering::Equal));
                sorted.into_iter().take(5).collect()
            };

            // Merge discovered memories into scored results
            for d in capped {
                let engram = match brain.get_or_load(&d.id) {
                    Some(e) => e,
                    None => continue,
                };

                // State filter (same as above)
                let state_ok = if include_archived {
                    true
                } else if include_deep {
                    !engram.is_archived()
                } else {
                    engram.is_searchable()
                };
                if !state_ok { continue; }

                scored.push(ScoredResult {
                    id: d.id,
                    content: engram.content.clone(),
                    similarity: d.similarity,
                    energy: d.energy,
                    blended: d.blended_score,
                    state_emoji: engram.state.emoji(),
                });
            }

            // Re-sort after merging
            scored.sort_by(|a, b| b.blended.partial_cmp(&a.blended).unwrap_or(std::cmp::Ordering::Equal));

            association_merged_count = association_discovery_count.min(5);

            // Log discovery stats for visibility
            eprintln!("[search] association-following: {} discovered, {} merged (cap: 5, depth: {})",
                association_discovery_count, association_merged_count, association_depth);
        }

        // Limit results
        scored.truncate(limit);

        if scored.is_empty() {
            let mut output = String::from(
                "⚡ **REQUIRED:** Call engram_recall on IDs you use. \n\
                 💾 **REQUIRED:** Call engram_create if you learn ANY new facts this turn.\n\
                 ---\n\n"
            );
            output.push_str("No memories found.\n");
            output.push_str("\n💡 Tip: Try a different query or lower min_score.");
            return Ok(text_response(output));
        }

        let mut output = String::from(
            "⚡ **REQUIRED:** Call engram_recall on IDs you use. \n\
             💾 **REQUIRED:** Call engram_create if you learn ANY new facts this turn.\n\
             ---\n\n"
        );
        if association_merged_count > 0 {
            output.push_str(&format!("Found {} memories ({} via associations, {} total discovered):\n\n",
                scored.len(), association_merged_count, association_discovery_count));
        } else {
            output.push_str(&format!("Found {} memories:\n\n", scored.len()));
        }

        for r in &scored {
            output.push_str(&format!(
                "ID: {}\nContent: {}\nScore: {:.2} (sim: {:.2}, energy: {:.2}) | State: {}\n\n",
                r.id,
                r.content,
                r.blended,
                r.similarity,
                r.energy,
                r.state_emoji,
            ));
        }

        Ok(text_response(output))
    }
}
