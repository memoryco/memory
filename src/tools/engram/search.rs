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

        // Capture seed IDs (vector search results) before association merge
        // for chain detection later
        let seed_ids: Vec<uuid::Uuid> = scored.iter().map(|r| r.id).collect();

        if follow_associations && association_depth > 0 && !scored.is_empty() {
            use std::collections::HashSet;

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

        // Chain detection: check if any seed results (vector search hits) are procedure anchors
        let chain_hints = detect_procedure_chains(&brain, &seed_ids);

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

        // Show chain hints before memory listings
        for hint in &chain_hints {
            let truncated_content = truncate_chain_hint_content(&hint.anchor_content, 40);
            output.push_str(&format!(
                "🔗 Procedure chain detected: {} [{}] has {} ordered steps. Use engram_associations to walk the chain.\n\n",
                truncated_content, hint.anchor_id, hint.ordered_step_count
            ));
        }

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

/// A detected procedure chain anchor
#[derive(Debug, Clone)]
pub struct ChainHint {
    pub anchor_id: uuid::Uuid,
    pub anchor_content: String,
    pub ordered_step_count: usize,
}

/// Truncate chain hint content for display while preserving UTF-8 boundaries.
fn truncate_chain_hint_content(content: &str, max_bytes: usize) -> String {
    if content.len() <= max_bytes {
        return content.to_string();
    }

    let mut end = max_bytes.min(content.len());
    while end > 0 && !content.is_char_boundary(end) {
        end -= 1;
    }

    let prefix = content.get(..end).unwrap_or_default();
    format!("{}...", prefix)
}

/// Check seed results for procedure chain anchors.
///
/// A procedure anchor is an engram that has outbound associations where
/// at least one has an ordinal set. Returns hints for each anchor found.
pub fn detect_procedure_chains(
    brain: &crate::engram::Brain,
    seed_ids: &[uuid::Uuid],
) -> Vec<ChainHint> {
    let mut hints = Vec::new();

    for id in seed_ids {
        let assocs = match brain.associations_from(id) {
            Some(a) => a,
            None => continue,
        };

        let ordered_count = assocs.iter().filter(|a| a.ordinal.is_some()).count();
        if ordered_count == 0 {
            continue;
        }

        let content = match brain.get(id) {
            Some(e) => e.content.clone(),
            None => continue,
        };

        hints.push(ChainHint {
            anchor_id: *id,
            anchor_content: content,
            ordered_step_count: ordered_count,
        });
    }

    hints
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::{Brain, storage::EngramStorage};

    /// Helper: create a Brain backed by in-memory SQLite
    fn brain_with_sqlite() -> Brain {
        let storage = EngramStorage::in_memory().unwrap();
        Brain::new(storage).unwrap()
    }

    #[test]
    fn detect_chains_finds_procedure_anchor() {
        let mut brain = brain_with_sqlite();

        let anchor = brain.create("Deploy procedure").unwrap();
        let step1 = brain.create("Pull code").unwrap();
        let step2 = brain.create("Run tests").unwrap();
        let step3 = brain.create("Deploy").unwrap();

        brain.associate_with_ordinal(anchor, step1, 0.9, Some(1)).unwrap();
        brain.associate_with_ordinal(anchor, step2, 0.9, Some(2)).unwrap();
        brain.associate_with_ordinal(anchor, step3, 0.9, Some(3)).unwrap();

        let hints = detect_procedure_chains(&brain, &[anchor]);
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].anchor_id, anchor);
        assert_eq!(hints[0].ordered_step_count, 3);
        assert!(hints[0].anchor_content.contains("Deploy procedure"));
    }

    #[test]
    fn detect_chains_ignores_unordered_associations() {
        let mut brain = brain_with_sqlite();

        let memory = brain.create("Just a memory").unwrap();
        let related = brain.create("Related memory").unwrap();

        // No ordinal
        brain.associate(memory, related, 0.8).unwrap();

        let hints = detect_procedure_chains(&brain, &[memory]);
        assert!(hints.is_empty(), "Unordered associations should not trigger chain detection");
    }

    #[test]
    fn detect_chains_mixed_ordered_and_unordered() {
        let mut brain = brain_with_sqlite();

        let anchor = brain.create("Mixed anchor").unwrap();
        let step1 = brain.create("Step 1").unwrap();
        let related = brain.create("Related but unordered").unwrap();

        brain.associate_with_ordinal(anchor, step1, 0.9, Some(1)).unwrap();
        brain.associate(anchor, related, 0.5).unwrap();

        let hints = detect_procedure_chains(&brain, &[anchor]);
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].ordered_step_count, 1, "Only count ordered associations");
    }

    #[test]
    fn detect_chains_multiple_anchors() {
        let mut brain = brain_with_sqlite();

        let anchor1 = brain.create("Procedure A").unwrap();
        let a_step = brain.create("A step 1").unwrap();
        brain.associate_with_ordinal(anchor1, a_step, 0.9, Some(1)).unwrap();

        let anchor2 = brain.create("Procedure B").unwrap();
        let b_step1 = brain.create("B step 1").unwrap();
        let b_step2 = brain.create("B step 2").unwrap();
        brain.associate_with_ordinal(anchor2, b_step1, 0.9, Some(1)).unwrap();
        brain.associate_with_ordinal(anchor2, b_step2, 0.9, Some(2)).unwrap();

        let non_anchor = brain.create("Not a procedure").unwrap();

        let hints = detect_procedure_chains(&brain, &[anchor1, non_anchor, anchor2]);
        assert_eq!(hints.len(), 2);

        // Verify both anchors detected with correct step counts
        let hint_map: std::collections::HashMap<_, _> = hints.iter()
            .map(|h| (h.anchor_id, h.ordered_step_count))
            .collect();
        assert_eq!(hint_map[&anchor1], 1);
        assert_eq!(hint_map[&anchor2], 2);
    }

    #[test]
    fn detect_chains_no_associations_returns_empty() {
        let mut brain = brain_with_sqlite();

        let isolated = brain.create("Isolated memory").unwrap();

        let hints = detect_procedure_chains(&brain, &[isolated]);
        assert!(hints.is_empty());
    }

    #[test]
    fn detect_chains_empty_seed_ids_returns_empty() {
        let brain = brain_with_sqlite();

        let hints = detect_procedure_chains(&brain, &[]);
        assert!(hints.is_empty());
    }

    #[test]
    fn truncate_chain_hint_content_handles_unicode_boundary() {
        let input = format!("{}—XYZ", "a".repeat(39));
        let result = truncate_chain_hint_content(&input, 40);
        assert_eq!(result, format!("{}...", "a".repeat(39)));
    }

    #[test]
    fn truncate_chain_hint_content_short_string_passthrough() {
        assert_eq!(truncate_chain_hint_content("short", 40), "short");
    }
}
