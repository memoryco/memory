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

/// Heuristic: list/multi-hop style queries often need broader recall.
fn is_composite_query(query: &str) -> bool {
    let q = query.to_lowercase();
    let conjunction = q.contains(" and ") || q.contains(" both ");
    let list_cues = [
        "activities", "events", "books", "states", "cities", "names", "causes",
        "traits", "attributes", "both have", "in common", "all ",
    ];
    conjunction || list_cues.iter().any(|cue| q.contains(cue))
}

/// Heuristic: inferential/open-domain questions benefit from weaker-signal recall.
fn is_inferential_query(query: &str) -> bool {
    let q = query.to_lowercase();
    let cues = [
        "might", "likely", "considered", "would", "could", "personality",
        "attributes", "traits", "describe", "what job",
    ];
    cues.iter().any(|cue| q.contains(cue))
}

/// Very small stop-word list for diversity/cue tokenization.
const SHAPING_STOP_WORDS: &[&str] = &[
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "from", "is", "are", "was", "were", "be", "been", "this", "that", "it",
    "as", "at", "by", "what", "when", "where", "who", "how", "does", "did",
];

/// Normalize a token for result-shaping analysis.
fn normalize_shaping_token(raw: &str) -> Option<String> {
    let mut t = raw
        .trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '’' && c != '`')
        .replace('’', "'")
        .replace('`', "")
        .to_lowercase();

    if t.ends_with("'s") {
        t.truncate(t.len().saturating_sub(2));
    }
    t = t.trim_matches(|c: char| !c.is_alphanumeric()).to_string();
    if t.is_empty() || t.len() < 3 {
        return None;
    }
    if SHAPING_STOP_WORDS.contains(&t.as_str()) {
        return None;
    }
    Some(t)
}

/// Tokenize content to support overlap/diversity calculations.
fn tokenize_for_shaping(text: &str) -> std::collections::HashSet<String> {
    text.split_whitespace()
        .filter_map(normalize_shaping_token)
        .collect()
}

/// Jaccard overlap between two token sets.
fn token_jaccard_overlap(
    a: &std::collections::HashSet<String>,
    b: &std::collections::HashSet<String>,
) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    if union == 0.0 { 0.0 } else { inter / union }
}

/// Derive a source bucket key from memory content + created_at.
/// Used to avoid over-concentrating top results from one session snippet.
fn source_bucket_key(content: &str, created_at: i64) -> String {
    if content.starts_with('[') {
        if let Some(end) = content.find(']') {
            if end > 1 && end <= 48 {
                return content[..=end].to_lowercase();
            }
        }
    }
    chrono::DateTime::from_timestamp(created_at, 0)
        .map(|dt| dt.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| created_at.to_string())
}

impl Tool<Context> for EngramSearchTool {
    fn name(&self) -> &str {
        "engram_search"
    }

    fn description(&self) -> &str {
        "Search memories by semantic similarity using vector embeddings. \
         Finds memories with similar meaning even if they don't share exact keywords. \
         If results seem weak or irrelevant, try decomposing abstract queries into \
         concrete related terms. For example, instead of 'relationship status', \
         try 'breakup', 'dating', 'partner', or 'married'. Search for actions \
         and events rather than abstract states."
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
        let composite_query = is_composite_query(&args.query);
        let inferential_query = is_inferential_query(&args.query);

        // List/composite questions often need more context than the nominal limit.
        let effective_limit = if composite_query {
            limit.max(15).min(30)
        } else {
            limit
        };

        let mut brain = context.brain.lock().unwrap();

        // Lazy maintenance: decay + cross-process sync
        let _ = brain.apply_time_decay();
        let _ = brain.sync_from_storage();

        // Read config for association-following, reranking, hybrid search, and query expansion
        let follow_associations = brain.config().search_follow_associations;
        let association_depth = brain.config().search_association_depth;
        let rerank_enabled = brain.config().rerank_enabled;
        let rerank_candidates = brain.config().rerank_candidates;
        let hybrid_search_enabled = brain.config().hybrid_search_enabled;
        let query_expansion_enabled = brain.config().query_expansion_enabled;

        // Query expansion: generate variant queries
        let variants = if query_expansion_enabled {
            super::query_expansion::expand_query(&args.query)
        } else {
            vec![args.query.clone()]
        };

        if variants.len() > 1 {
            eprintln!("[search] query expansion: {} variants from {:?}",
                variants.len(), &args.query);
        }

        let generator = EmbeddingGenerator::new();

        // Find similar memories (fetch extra to allow for filtering)
        // When reranking is enabled, fetch more candidates for the reranker to work with
        let fetch_count = if rerank_enabled {
            rerank_candidates.max(effective_limit * 3)
        } else {
            effective_limit * 3
        };

        // Run retrieval for each variant, merging results (keep highest score per engram).
        // Cache the original query embedding (variants[0]) so we don't compute it twice —
        // it's also needed for association discovery later.
        let mut all_results: std::collections::HashMap<uuid::Uuid, crate::engram::SimilarityResult> =
            std::collections::HashMap::new();
        let mut original_query_embedding: Option<Vec<f32>> = None;

        // Helper: run vector + BM25 for a single query variant and merge into all_results
        fn run_variant(
            variant_query: &str,
            brain: &mut crate::engram::Brain,
            generator: &EmbeddingGenerator,
            fetch_count: usize,
            min_score: f32,
            hybrid_search_enabled: bool,
            all_results: &mut std::collections::HashMap<uuid::Uuid, crate::engram::SimilarityResult>,
            original_embedding: &mut Option<Vec<f32>>,
            log_variant: bool,
        ) -> sml_mcps::Result<()> {
            let variant_embedding = generator.generate(variant_query)
                .map_err(|e| McpError::ToolError(format!("Failed to generate embedding: {}", e)))?;

            if original_embedding.is_none() {
                *original_embedding = Some(variant_embedding.clone());
            }

            let vector_results = brain.find_similar_by_embedding(&variant_embedding, fetch_count, min_score)
                .map_err(|e| McpError::ToolError(format!("Vector search failed: {}", e)))?;

            let variant_merged = if hybrid_search_enabled {
                let bm25_results = brain.keyword_search(variant_query, fetch_count)
                    .unwrap_or_else(|e| {
                        eprintln!("[search] BM25 keyword search failed, using vector only: {}", e);
                        Vec::new()
                    });

                if bm25_results.is_empty() {
                    vector_results
                } else {
                    use crate::engram::storage::rrf;
                    let merged = rrf::reciprocal_rank_fusion(
                        &[&vector_results, &bm25_results],
                        rrf::DEFAULT_K,
                    );
                    if log_variant {
                        eprintln!("[search] hybrid variant {:?}: {} vector + {} BM25 → {} merged",
                            variant_query, vector_results.len(), bm25_results.len(), merged.len());
                    } else {
                        eprintln!("[search] hybrid: {} vector + {} BM25 → {} merged via RRF",
                            vector_results.len(), bm25_results.len(), merged.len());
                    }
                    merged
                }
            } else {
                vector_results
            };

            for result in variant_merged {
                all_results.entry(result.id)
                    .and_modify(|existing| {
                        if result.score > existing.score {
                            *existing = result.clone();
                        }
                    })
                    .or_insert(result);
            }
            Ok(())
        }

        // Phase 1: Run primary variants (original + stop-word-stripped)
        for variant_query in &variants {
            run_variant(variant_query, &mut brain, &generator, fetch_count, min_score,
                       hybrid_search_enabled, &mut all_results,
                       &mut original_query_embedding, variants.len() > 1)?;
        }

        // Phase 2 (fallback): run individual term queries when sparse OR
        // when the question looks list/multi-hop style.
        const SPARSE_THRESHOLD: usize = 3;
        let sparse_results = all_results.len() < SPARSE_THRESHOLD;

        if query_expansion_enabled && (sparse_results || composite_query) {
            let fallback = super::query_expansion::fallback_terms(&args.query);
            if !fallback.is_empty() {
                let max_fallback_terms = if sparse_results {
                    fallback.len()
                } else {
                    // Composite queries already have some signal; limit expansion fan-out.
                    fallback.len().min(3)
                };
                eprintln!(
                    "[search] {} results (composite={}), running {} fallback term queries",
                    all_results.len(),
                    composite_query,
                    max_fallback_terms
                );
                for term in fallback.iter().take(max_fallback_terms) {
                    run_variant(term, &mut brain, &generator, fetch_count, min_score,
                               hybrid_search_enabled, &mut all_results,
                               &mut original_query_embedding, true)?;
                }
            }
        }

        // Phase 3 (inferential questions): if still under-filled, run a relaxed
        // similarity pass so weak but relevant clues can surface.
        if query_expansion_enabled && inferential_query && all_results.len() < effective_limit {
            let relaxed_min_score = (min_score * 0.6).max(0.15);
            if relaxed_min_score < min_score {
                let mut relaxed_queries = vec![args.query.clone()];
                for term in super::query_expansion::fallback_terms(&args.query).into_iter().take(2) {
                    if !relaxed_queries.iter().any(|q| q == &term) {
                        relaxed_queries.push(term);
                    }
                }
                eprintln!(
                    "[search] inferential query under-filled ({}<{}), relaxed min_score {:.2}->{:.2} across {} queries",
                    all_results.len(),
                    effective_limit,
                    min_score,
                    relaxed_min_score,
                    relaxed_queries.len()
                );
                for rq in &relaxed_queries {
                    run_variant(rq, &mut brain, &generator, fetch_count, relaxed_min_score,
                               hybrid_search_enabled, &mut all_results,
                               &mut original_query_embedding, true)?;
                }
            }
        }

        // Convert to sorted vec
        let mut merged_results: Vec<crate::engram::SimilarityResult> = all_results.into_values().collect();
        merged_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Use the ORIGINAL query embedding for association discovery and reranking
        let query_embedding = original_query_embedding.expect("at least one variant always exists");

        // Score blending: 70% semantic similarity, 30% energy (recency/relevance)
        // This boosts recently-used memories even if semantic match is slightly lower
        #[derive(Clone)]
        struct ScoredResult {
            id: uuid::Uuid,
            content: String,
            created_at: i64,
            similarity: f32,
            energy: f64,
            blended: f32,
            state_emoji: &'static str,
        }

        let mut scored: Vec<ScoredResult> = merged_results.iter().filter_map(|r| {
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
                created_at: engram.created_at,
                similarity: r.score,
                energy: engram.energy,
                blended,
                state_emoji: engram.state.emoji(),
            })
        }).collect();

        // Sort by blended score descending
        scored.sort_by(|a, b| b.blended.partial_cmp(&a.blended).unwrap_or(std::cmp::Ordering::Equal));

        // Cross-encoder re-ranking: feed (query, content) pairs through a cross-encoder
        // for significantly better relevance ordering than cosine similarity alone.
        // Falls back silently to cosine results if reranking fails.
        if rerank_enabled && !scored.is_empty() {
            let contents: Vec<String> = scored.iter().map(|r| r.content.clone()).collect();
            let content_refs: Vec<&str> = contents.iter().map(|s| s.as_str()).collect();
            let candidate_count = contents.len();

            match crate::embedding::reranker::rerank(&args.query, &content_refs) {
                Ok(reranked) => {
                    let mut reranked_results = Vec::with_capacity(reranked.len());
                    for rs in &reranked {
                        if rs.index < scored.len() {
                            let orig = &scored[rs.index];
                            reranked_results.push(ScoredResult {
                                id: orig.id,
                                content: orig.content.clone(),
                                created_at: orig.created_at,
                                similarity: orig.similarity,
                                energy: orig.energy,
                                blended: rs.score, // Replace blended with reranker score
                                state_emoji: orig.state_emoji,
                            });
                        }
                    }
                    scored = reranked_results;
                    eprintln!("[search] re-ranked {} candidates with cross-encoder", candidate_count);
                }
                Err(e) => {
                    eprintln!("[search] reranking failed, falling back to cosine: {}", e);
                }
            }
        }

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

            // Cap discoveries to avoid swamping vector results.
            // Scale the cap with requested limit so list/multi-hop questions
            // can merge more associated clues.
            association_discovery_count = discoveries.len();
            let association_cap = effective_limit.clamp(5, 12);
            let capped: Vec<_> = {
                let mut sorted = discoveries;
                sorted.sort_by(|a, b| b.blended_score.partial_cmp(&a.blended_score).unwrap_or(std::cmp::Ordering::Equal));
                sorted.into_iter().take(association_cap).collect()
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
                    created_at: engram.created_at,
                    similarity: d.similarity,
                    energy: d.energy,
                    blended: d.blended_score,
                    state_emoji: engram.state.emoji(),
                });
            }

            // Re-sort after merging
            scored.sort_by(|a, b| b.blended.partial_cmp(&a.blended).unwrap_or(std::cmp::Ordering::Equal));

            association_merged_count = association_discovery_count.min(association_cap);

            // Log discovery stats for visibility
            eprintln!("[search] association-following: {} discovered, {} merged (cap: {}, depth: {})",
                association_discovery_count, association_merged_count, association_cap, association_depth);
        }

        // Aggregation-friendly shaping for list/composite queries:
        // pick a diverse, coverage-rich, source-balanced subset for final context.
        if composite_query && scored.len() > 1 {
            let original_count = scored.len();
            let target = effective_limit.min(scored.len());

            let cue_terms: Vec<String> = {
                let mut cues: Vec<String> = super::query_expansion::fallback_terms(&args.query)
                    .into_iter()
                    .filter_map(|t| normalize_shaping_token(&t))
                    .collect();
                if cues.is_empty() {
                    cues = tokenize_for_shaping(&args.query).into_iter().collect();
                }
                cues.sort();
                cues.dedup();
                cues
            };

            let token_sets: Vec<std::collections::HashSet<String>> =
                scored.iter().map(|r| tokenize_for_shaping(&r.content)).collect();
            let lower_contents: Vec<String> = scored.iter().map(|r| r.content.to_lowercase()).collect();
            let source_keys: Vec<String> = scored
                .iter()
                .map(|r| source_bucket_key(&r.content, r.created_at))
                .collect();

            let mut selected_indices: Vec<usize> = Vec::with_capacity(target);
            let mut used = vec![false; scored.len()];
            let mut covered_cues: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut source_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

            while selected_indices.len() < target {
                let mut best_idx: Option<usize> = None;
                let mut best_score = f32::NEG_INFINITY;
                let mut best_new_cues: Vec<String> = Vec::new();

                for idx in 0..scored.len() {
                    if used[idx] {
                        continue;
                    }

                    // Penalize near-duplicate results versus what's already selected.
                    let max_overlap = selected_indices
                        .iter()
                        .map(|sel| token_jaccard_overlap(&token_sets[idx], &token_sets[*sel]))
                        .fold(0.0_f32, f32::max);
                    let diversity_penalty = max_overlap * 0.20;

                    // Reward candidates that introduce uncovered cue terms.
                    let mut new_cues = Vec::new();
                    for cue in &cue_terms {
                        if !covered_cues.contains(cue) && lower_contents[idx].contains(cue.as_str()) {
                            new_cues.push(cue.clone());
                        }
                    }
                    let cue_bonus = (new_cues.len() as f32) * 0.05;

                    // Soft source balancing: down-rank overrepresented source buckets.
                    let src_count = *source_counts.get(&source_keys[idx]).unwrap_or(&0);
                    let source_penalty = if src_count >= 2 {
                        (src_count as f32) * 0.08
                    } else {
                        0.0
                    };

                    let adjusted = scored[idx].blended + cue_bonus - diversity_penalty - source_penalty;

                    if adjusted > best_score {
                        best_score = adjusted;
                        best_idx = Some(idx);
                        best_new_cues = new_cues;
                    }
                }

                let Some(chosen) = best_idx else {
                    break;
                };

                used[chosen] = true;
                selected_indices.push(chosen);
                for cue in best_new_cues {
                    covered_cues.insert(cue);
                }
                *source_counts.entry(source_keys[chosen].clone()).or_insert(0) += 1;
            }

            if !selected_indices.is_empty() {
                let mut reshaped = Vec::with_capacity(selected_indices.len());
                for idx in selected_indices {
                    reshaped.push(scored[idx].clone());
                }
                scored = reshaped;
                eprintln!(
                    "[search] aggregation shaping: {} -> {} results (covered cues: {}/{})",
                    original_count,
                    scored.len(),
                    covered_cues.len(),
                    cue_terms.len()
                );
            }
        }

        // Chain detection: check if any seed results (vector search hits) are procedure anchors
        let chain_hints = detect_procedure_chains(&brain, &seed_ids);

        // Limit results
        scored.truncate(effective_limit);

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
        let should_hint = scored.len() < 3
            || scored.first().map(|r| r.blended < 0.45).unwrap_or(false);
        if should_hint {
            output.push_str(
                "⚠️ Results may not be relevant. Try searching with related concepts:\n\
                 - Break abstract questions into concrete topics (e.g. \"relationship status\" → \
                   \"breakup\", \"dating\", \"partner\", \"married\", \"single\")\n\
                 - Search for specific events or actions rather than status or state\n\
                 - Try [person's name] + a related action, event, or feeling\n"
            );
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

    #[test]
    fn composite_query_heuristic_detects_list_style_questions() {
        assert!(is_composite_query("What activities does Melanie partake in?"));
        assert!(is_composite_query("What do Jon and Gina both have in common?"));
        assert!(!is_composite_query("Where did Caroline move from?"));
    }

    #[test]
    fn inferential_query_heuristic_detects_open_domain_style_questions() {
        assert!(is_inferential_query("What might John's degree be in?"));
        assert!(is_inferential_query("Would Caroline be considered religious?"));
        assert!(!is_inferential_query("When did Caroline attend the support group?"));
    }

    #[test]
    fn shaping_tokenizer_cleans_possessives_and_punctuation() {
        let tokens = tokenize_for_shaping("Caroline's relationship status?");
        assert!(tokens.contains("caroline"));
        assert!(tokens.contains("relationship"));
        assert!(tokens.contains("status"));
    }

    #[test]
    fn token_overlap_detects_similarity() {
        let a = tokenize_for_shaping("sunset painting pottery class");
        let b = tokenize_for_shaping("pottery class with sunset");
        let c = tokenize_for_shaping("veterans hospital fundraiser");
        assert!(token_jaccard_overlap(&a, &b) > 0.2);
        assert!(token_jaccard_overlap(&a, &c) < 0.2);
    }

    #[test]
    fn source_bucket_uses_bracket_prefix_when_present() {
        let key = source_bucket_key("[1:14 pm on 25 May, 2023] Melanie said: hello", 0);
        assert!(key.starts_with("[1:14 pm on 25 may, 2023]"));
    }
}
