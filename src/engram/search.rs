//! Search pipeline — domain logic for semantic memory search.
//!
//! This module contains the core search/recall pipeline as composable, testable functions.
//! No MCP dependencies. Operates on domain types only.

use crate::embedding::{EmbeddingGenerator, cosine_similarity};
use crate::engram::{Brain, SimilarityResult};
use crate::llm::{LlmTier, SharedLlmService};
use std::collections::{HashMap, HashSet};

/// Parse a timestamp string into unix epoch seconds.
/// Accepts ISO 8601 datetime (e.g. "2026-03-14T19:00:00Z", "2026-03-14") or
/// unix epoch seconds as a plain integer string (e.g. "1710432000").
pub fn parse_timestamp(s: &str) -> Result<i64, String> {
    let trimmed = s.trim();

    // Try unix epoch (plain integer)
    if let Ok(epoch) = trimmed.parse::<i64>() {
        return Ok(epoch);
    }

    // Try full ISO 8601 with timezone
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(trimmed) {
        return Ok(dt.timestamp());
    }

    // Try ISO 8601 without timezone (assume UTC)
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(trimmed, "%Y-%m-%dT%H:%M:%S") {
        return Ok(dt.and_utc().timestamp());
    }

    // Try date-only ("2026-03-14") — treat as start of day UTC
    if let Ok(d) = chrono::NaiveDate::parse_from_str(trimmed, "%Y-%m-%d") {
        return Ok(d.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp());
    }

    Err(format!("Cannot parse timestamp: '{}'. Expected ISO 8601 (e.g. 2026-03-14, 2026-03-14T19:00:00Z) or unix epoch seconds.", trimmed))
}

/// Cosine similarity threshold for semantic deduplication in the diversity shaping pass.
/// Embeddings with similarity >= this value are considered near-duplicates.
/// This catches semantic overlap that Jaccard token overlap misses — e.g. two memories
/// phrased differently but about the same fact will have high cosine similarity even
/// with moderate token overlap.
pub const SEMANTIC_DEDUP_THRESHOLD: f32 = 0.85;

/// Very small stop-word list for diversity/cue tokenization.
const SHAPING_STOP_WORDS: &[&str] = &[
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "from", "is", "are",
    "was", "were", "be", "been", "this", "that", "it", "as", "at", "by", "what", "when", "where",
    "who", "how", "does", "did",
];

/// Normalize a token for result-shaping analysis.
pub fn normalize_shaping_token(raw: &str) -> Option<String> {
    let mut t = raw
        .trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '\u{2019}' && c != '`')
        .replace('\u{2019}', "'")
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
pub fn tokenize_for_shaping(text: &str) -> HashSet<String> {
    text.split_whitespace()
        .filter_map(normalize_shaping_token)
        .collect()
}

/// Jaccard overlap between two token sets.
pub fn token_jaccard_overlap(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    if union == 0.0 { 0.0 } else { inter / union }
}

/// Derive a source bucket key from memory content + created_at.
/// Used to avoid over-concentrating top results from one session snippet.
pub fn source_bucket_key(content: &str, created_at: i64) -> String {
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

/// Heuristic: list/multi-hop style queries often need broader recall.
// TODO: English-specific keyword heuristic — should be revisited for i18n support.
pub fn is_composite_query(query: &str) -> bool {
    let q = query.to_lowercase();
    let conjunction = q.contains(" and ") || q.contains(" both ");
    let list_cues = [
        "activities",
        "events",
        "books",
        "states",
        "cities",
        "names",
        "causes",
        "traits",
        "attributes",
        "both have",
        "in common",
        "all ",
    ];
    conjunction || list_cues.iter().any(|cue| q.contains(cue))
}

/// Heuristic: inferential/open-domain questions benefit from weaker-signal recall.
// TODO: English-specific keyword heuristic — should be revisited for i18n support.
pub fn is_inferential_query(query: &str) -> bool {
    let q = query.to_lowercase();
    let cues = [
        "might",
        "likely",
        "considered",
        "would",
        "could",
        "personality",
        "attributes",
        "traits",
        "describe",
        "what job",
    ];
    cues.iter().any(|cue| q.contains(cue))
}

fn normalize_variants(original: &str, generated: Vec<String>) -> Vec<String> {
    let mut variants = Vec::new();
    let mut seen = HashSet::new();

    for candidate in std::iter::once(original.to_string()).chain(generated.into_iter()) {
        let trimmed = candidate.trim();
        if trimmed.is_empty() {
            continue;
        }

        let key = trimmed.to_lowercase();
        if seen.insert(key) {
            variants.push(trimmed.to_string());
        }
    }

    if variants.is_empty() {
        vec![original.to_string()]
    } else {
        variants
    }
}

pub fn llm_query_variants(llm: &SharedLlmService, query: &str) -> Option<Vec<String>> {
    if !llm.available() || llm.tier() < LlmTier::Minimal {
        return None;
    }

    match llm.expand_query(query, 3) {
        Ok(variants) => Some(normalize_variants(query, variants)),
        Err(e) => {
            eprintln!("[search] llm query expansion failed: {:?}", e);
            None
        }
    }
}

/// A scored search result after energy blending.
#[derive(Clone)]
pub struct ScoredResult {
    pub id: uuid::Uuid,
    pub content: String,
    pub created_at: i64,
    pub similarity: f32,
    pub energy: f64,
    pub blended: f32,
    pub state_emoji: &'static str,
}

/// A detected procedure chain anchor.
#[derive(Debug, Clone)]
pub struct ChainHint {
    pub anchor_id: uuid::Uuid,
    pub anchor_content: String,
    pub ordered_step_count: usize,
}

/// A single candidate snapshot at a given pipeline stage.
#[derive(Clone)]
pub struct TraceEntry {
    pub id: uuid::Uuid,
    /// First ~100 chars of memory content.
    pub content_preview: String,
    pub similarity: f32,
    pub energy: f64,
    pub blended: f32,
}

impl TraceEntry {
    fn from_scored(s: &ScoredResult) -> Self {
        let preview = if s.content.len() > 100 {
            let mut end = 100;
            while end > 0 && !s.content.is_char_boundary(end) {
                end -= 1;
            }
            format!("{}...", &s.content[..end])
        } else {
            s.content.clone()
        };
        Self {
            id: s.id,
            content_preview: preview,
            similarity: s.similarity,
            energy: s.energy,
            blended: s.blended,
        }
    }
}

/// A snapshot of all candidates at a given pipeline stage.
pub struct TraceStage {
    pub name: String,
    pub entries: Vec<TraceEntry>,
}

/// Debug diagnostics collected during the search pipeline.
#[derive(Default)]
pub struct DebugInfo {
    /// Ordered log of pipeline stages and their outcomes.
    pub lines: Vec<String>,
    /// Full candidate snapshots at each pipeline stage (for trace file output).
    pub stages: Vec<TraceStage>,
}

/// Result of running the search pipeline.
pub struct SearchPipelineResult {
    pub results: Vec<ScoredResult>,
    pub association_merged_count: usize,
    pub association_discovery_count: usize,
    pub chain_hints: Vec<ChainHint>,
    /// Pipeline diagnostics (only populated when config.debug = true).
    pub debug_info: Option<DebugInfo>,
}

/// Parameters for the search pipeline, pre-computed by the caller.
pub struct SearchPipelineParams {
    /// The original query string.
    pub query: String,
    /// Pre-computed query variants (original + expanded).
    pub variants: Vec<String>,
    /// Pre-computed fallback terms for sparse/composite fallback.
    pub fallback_terms: Vec<String>,
    /// Effective limit (may be larger than user's requested limit for composite queries).
    pub effective_limit: usize,
    /// Whether this is a composite/list-style query.
    pub composite_query: bool,
    /// Whether this is an inferential/open-domain query.
    pub inferential_query: bool,
    /// Minimum similarity score threshold.
    pub min_score: f32,
    /// How many candidates to fetch for reranking.
    pub fetch_count: usize,
    /// Config: follow associations after vector search?
    pub follow_associations: bool,
    /// Config: how many hops to follow for associations.
    pub association_depth: u8,
    /// Config: reranking mode ("off", "cross-encoder", "llm", "hybrid").
    pub rerank_mode: String,
    /// Config: use hybrid BM25+vector search?
    pub hybrid_search_enabled: bool,
    /// Config: how many candidates to send to LLM reranker.
    pub llm_rerank_candidates: usize,
    /// Config: minimum association cap.
    pub association_cap_min: usize,
    /// Config: maximum association cap.
    pub association_cap_max: usize,
    /// Include deep storage memories?
    pub include_deep: bool,
    /// Include archived memories?
    pub include_archived: bool,
    /// Only include memories created after this epoch.
    pub created_after: Option<i64>,
    /// Only include memories created before this epoch.
    pub created_before: Option<i64>,
    /// Config: query expansion enabled?
    pub query_expansion_enabled: bool,
    /// Pre-loaded session centroid for affinity biasing (None = no session context).
    pub session_centroid: Option<Vec<f32>>,
    /// How much session context biases retrieval (β from config). 0.0 = off.
    pub session_context_weight: f32,
    /// When true, collect pipeline diagnostics into the result.
    pub debug: bool,
}

/// Run vector + BM25 for a single query variant and merge into all_results.
/// Takes &Brain (not &mut Brain) because find_similar_by_embedding and keyword_search
/// are now &self methods that lock storage internally.
#[allow(clippy::too_many_arguments)]
fn run_variant(
    variant_query: &str,
    brain: &Brain,
    generator: &EmbeddingGenerator,
    fetch_count: usize,
    min_score: f32,
    hybrid_search_enabled: bool,
    all_results: &mut HashMap<uuid::Uuid, SimilarityResult>,
    original_embedding: &mut Option<Vec<f32>>,
    log_variant: bool,
) -> Result<(), String> {
    let variant_embedding = generator
        .generate(variant_query)
        .map_err(|e| format!("Failed to generate embedding: {}", e))?;

    if original_embedding.is_none() {
        *original_embedding = Some(variant_embedding.clone());
    }

    let vector_results = brain
        .find_similar_by_embedding(&variant_embedding, fetch_count, min_score)
        .map_err(|e| format!("Vector search failed: {}", e))?;

    let variant_merged = if hybrid_search_enabled {
        let bm25_results = brain
            .keyword_search(variant_query, fetch_count)
            .unwrap_or_else(|e| {
                eprintln!(
                    "[search] BM25 keyword search failed, using vector only: {}",
                    e
                );
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
                eprintln!(
                    "[search] hybrid variant {:?}: {} vector + {} BM25 → {} merged",
                    variant_query,
                    vector_results.len(),
                    bm25_results.len(),
                    merged.len()
                );
            } else {
                eprintln!(
                    "[search] hybrid: {} vector + {} BM25 → {} merged via RRF",
                    vector_results.len(),
                    bm25_results.len(),
                    merged.len()
                );
            }
            merged
        }
    } else {
        vector_results
    };

    for result in variant_merged {
        all_results
            .entry(result.id)
            .and_modify(|existing| {
                if result.score > existing.score {
                    *existing = result.clone();
                }
            })
            .or_insert(result);
    }
    Ok(())
}

/// Truncate chain hint content for display while preserving UTF-8 boundaries.
pub fn truncate_chain_hint_content(content: &str, max_bytes: usize) -> String {
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
pub fn detect_procedure_chains(brain: &Brain, seed_ids: &[uuid::Uuid]) -> Vec<ChainHint> {
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

/// Run the complete search pipeline.
///
/// # Arguments
/// - `brain`: read lock already held by the caller
/// - `llm`: LLM service for query expansion and reranking
/// - `params`: all pre-computed search parameters
pub fn run_search_pipeline(
    brain: &Brain,
    llm: &SharedLlmService,
    params: &SearchPipelineParams,
) -> Result<SearchPipelineResult, String> {
    let generator = EmbeddingGenerator::new();

    // Debug diagnostics (only collected when config.debug = true)
    let mut dbg = if params.debug { Some(DebugInfo::default()) } else { None };
    macro_rules! dbg_push {
        ($($arg:tt)*) => { if let Some(d) = &mut dbg { d.lines.push(format!($($arg)*)); } };
    }
    macro_rules! dbg_snapshot {
        ($name:expr, $scored:expr) => {
            if let Some(d) = &mut dbg {
                d.stages.push(TraceStage {
                    name: $name.to_string(),
                    entries: $scored.iter().map(TraceEntry::from_scored).collect(),
                });
            }
        };
    }
    dbg_push!("query: {:?}", params.query);
    dbg_push!("variants: {} (expansion={})", params.variants.len(), params.query_expansion_enabled);
    dbg_push!("rerank_mode: {}", params.rerank_mode);
    dbg_push!("fetch_count: {}, effective_limit: {}", params.fetch_count, params.effective_limit);

    // Find similar memories (fetch extra to allow for filtering)
    // When reranking is enabled, fetch more candidates for the reranker to work with
    // Run retrieval for each variant, merging results (keep highest score per engram).
    // Cache the original query embedding (variants[0]) so we don't compute it twice —
    // it's also needed for association discovery later.
    let mut all_results: HashMap<uuid::Uuid, SimilarityResult> = HashMap::new();
    let mut original_query_embedding: Option<Vec<f32>> = None;

    // Phase 1: Run primary variants (original + stop-word-stripped)
    for variant_query in &params.variants {
        run_variant(
            variant_query,
            brain,
            &generator,
            params.fetch_count,
            params.min_score,
            params.hybrid_search_enabled,
            &mut all_results,
            &mut original_query_embedding,
            params.variants.len() > 1,
        )?;
    }

    // Phase 2 (fallback): run individual term queries when sparse OR
    // when the question looks list/multi-hop style.
    const SPARSE_THRESHOLD: usize = 3;
    let sparse_results = all_results.len() < SPARSE_THRESHOLD;

    if params.query_expansion_enabled && (sparse_results || params.composite_query) {
        let fallback = &params.fallback_terms;
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
                params.composite_query,
                max_fallback_terms
            );
            for term in fallback.iter().take(max_fallback_terms) {
                run_variant(
                    term,
                    brain,
                    &generator,
                    params.fetch_count,
                    params.min_score,
                    params.hybrid_search_enabled,
                    &mut all_results,
                    &mut original_query_embedding,
                    true,
                )?;
            }
        }
    }

    // Phase 3 (inferential questions): if still under-filled, run a relaxed
    // similarity pass so weak but relevant clues can surface.
    if params.query_expansion_enabled
        && params.inferential_query
        && all_results.len() < params.effective_limit
    {
        let relaxed_min_score = (params.min_score * 0.6).max(0.15);
        if relaxed_min_score < params.min_score {
            let mut relaxed_queries = vec![params.query.clone()];
            for term in params.fallback_terms.iter().take(2) {
                if !relaxed_queries.iter().any(|q| q == term) {
                    relaxed_queries.push(term.clone());
                }
            }
            eprintln!(
                "[search] inferential query under-filled ({}<{}), relaxed min_score {:.2}->{:.2} across {} queries",
                all_results.len(),
                params.effective_limit,
                params.min_score,
                relaxed_min_score,
                relaxed_queries.len()
            );
            for rq in &relaxed_queries {
                run_variant(
                    rq,
                    brain,
                    &generator,
                    params.fetch_count,
                    relaxed_min_score,
                    params.hybrid_search_enabled,
                    &mut all_results,
                    &mut original_query_embedding,
                    true,
                )?;
            }
        }
    }

    // Convert to sorted vec
    let mut merged_results: Vec<SimilarityResult> = all_results.into_values().collect();
    merged_results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Use the ORIGINAL query embedding for association discovery and reranking
    let query_embedding = original_query_embedding.expect("at least one variant always exists");

    // Score blending: multiplicative — energy boosts relevance but can't replace it.
    // sim * (0.5 + energy * 0.5) means: zero similarity = zero score regardless of energy.
    // A fully-energized memory gets up to 1.0x boost; a decayed memory gets 0.5x floor.
    let mut scored: Vec<ScoredResult> = merged_results
        .iter()
        .filter_map(|r| {
            let engram = brain.get(&r.id)?;

            // State filter
            let state_ok = if params.include_archived {
                true
            } else if params.include_deep {
                !engram.is_archived()
            } else {
                engram.is_searchable()
            };
            if !state_ok {
                return None;
            }

            // Date-range filter
            if let Some(after) = params.created_after {
                if engram.created_at <= after {
                    return None;
                }
            }
            if let Some(before) = params.created_before {
                if engram.created_at >= before {
                    return None;
                }
            }

            let blended = r.score * (0.5 + (engram.energy as f32) * 0.5);

            Some(ScoredResult {
                id: r.id,
                content: r.content.clone(),
                created_at: engram.created_at,
                similarity: r.score,
                energy: engram.energy,
                blended,
                state_emoji: engram.state.emoji(),
            })
        })
        .collect();

    // Sort by blended score descending
    scored.sort_by(|a, b| {
        b.blended
            .partial_cmp(&a.blended)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    dbg_push!("retrieval: {} candidates after cosine scoring + state/date filtering", scored.len());
    dbg_snapshot!("retrieval", scored);

    // Re-ranking: cross-encoder, LLM, hybrid, or off.
    // Falls back silently to cosine results if reranking fails.
    match params.rerank_mode.as_str() {
        "llm" => {
            if !scored.is_empty() && llm.available() && llm.tier() >= LlmTier::Minimal {
                let cap = params.llm_rerank_candidates.min(scored.len());
                let contents: Vec<String> = scored[..cap].iter().map(|r| r.content.clone()).collect();
                let content_refs: Vec<&str> = contents.iter().map(|s| s.as_str()).collect();

                let rerank_start = std::time::Instant::now();
                match llm.rerank(&params.query, &content_refs, params.effective_limit) {
                    Ok(indices) => {
                        let rerank_ms = rerank_start.elapsed().as_millis();
                        let mut reranked_results = Vec::with_capacity(indices.len());
                        for (rank, &idx) in indices.iter().enumerate() {
                            if idx < cap {
                                let orig = &scored[idx];
                                reranked_results.push(ScoredResult {
                                    id: orig.id,
                                    content: orig.content.clone(),
                                    created_at: orig.created_at,
                                    similarity: orig.similarity,
                                    energy: orig.energy,
                                    blended: 1.0 - (rank as f32 * 0.05),
                                    state_emoji: orig.state_emoji,
                                });
                            }
                        }
                        if !reranked_results.is_empty() {
                            scored = reranked_results;
                            eprintln!(
                                "[search] LLM re-ranked {} candidates, selected {} in {}ms",
                                cap, scored.len(), rerank_ms
                            );
                            dbg_push!(
                                "llm-rerank: re-ranked {} candidates, selected {} in {}ms",
                                cap, scored.len(), rerank_ms
                            );
                        }
                    }
                    Err(e) => {
                        eprintln!("[search] LLM reranking failed, falling back to cosine: {}", e);
                        dbg_push!("llm-rerank: FAILED ({}), fell back to cosine", e);
                    }
                }
            } else if !scored.is_empty() {
                eprintln!("[search] LLM reranking requested but LLM not available, using cosine order");
                dbg_push!("llm-rerank: skipped (LLM not available)");
            }
        }
        "hybrid" => {
            // Stage 1: cross-encoder narrows candidates
            if !scored.is_empty() {
                let contents: Vec<String> = scored.iter().map(|r| r.content.clone()).collect();
                let content_refs: Vec<&str> = contents.iter().map(|s| s.as_str()).collect();
                let candidate_count = contents.len();

                let rerank_start = std::time::Instant::now();
                match crate::embedding::reranker::rerank(&params.query, &content_refs) {
                    Ok(reranked) => {
                        let rerank_ms = rerank_start.elapsed().as_millis();
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
                                    blended: rs.score,
                                    state_emoji: orig.state_emoji,
                                });
                            }
                        }
                        scored = reranked_results;
                        eprintln!(
                            "[search] hybrid stage 1: cross-encoder re-ranked {} candidates in {}ms",
                            candidate_count, rerank_ms
                        );
                        dbg_push!(
                            "hybrid/cross-encoder: re-ranked {} candidates in {}ms",
                            candidate_count, rerank_ms
                        );
                    }
                    Err(e) => {
                        eprintln!("[search] hybrid stage 1 failed, skipping to LLM: {}", e);
                        dbg_push!("hybrid/cross-encoder: FAILED ({})", e);
                    }
                }
            }
            // Stage 2: LLM reorders the top N
            if !scored.is_empty() && llm.available() && llm.tier() >= LlmTier::Minimal {
                let cap = params.llm_rerank_candidates.min(scored.len());
                let contents: Vec<String> = scored[..cap].iter().map(|r| r.content.clone()).collect();
                let content_refs: Vec<&str> = contents.iter().map(|s| s.as_str()).collect();

                let rerank_start = std::time::Instant::now();
                match llm.rerank(&params.query, &content_refs, params.effective_limit) {
                    Ok(indices) => {
                        let rerank_ms = rerank_start.elapsed().as_millis();
                        let mut reranked_results = Vec::with_capacity(indices.len());
                        for (rank, &idx) in indices.iter().enumerate() {
                            if idx < cap {
                                let orig = &scored[idx];
                                reranked_results.push(ScoredResult {
                                    id: orig.id,
                                    content: orig.content.clone(),
                                    created_at: orig.created_at,
                                    similarity: orig.similarity,
                                    energy: orig.energy,
                                    blended: 1.0 - (rank as f32 * 0.05),
                                    state_emoji: orig.state_emoji,
                                });
                            }
                        }
                        if !reranked_results.is_empty() {
                            scored = reranked_results;
                            eprintln!(
                                "[search] hybrid stage 2: LLM re-ranked {} candidates, selected {} in {}ms",
                                cap, scored.len(), rerank_ms
                            );
                            dbg_push!(
                                "hybrid/llm: re-ranked {} candidates, selected {} in {}ms",
                                cap, scored.len(), rerank_ms
                            );
                        }
                    }
                    Err(e) => {
                        eprintln!("[search] hybrid stage 2 LLM reranking failed: {}", e);
                        dbg_push!("hybrid/llm: FAILED ({}), kept cross-encoder order", e);
                    }
                }
            } else if !scored.is_empty() && !(llm.available() && llm.tier() >= LlmTier::Minimal) {
                eprintln!("[search] hybrid stage 2: LLM not available, keeping cross-encoder order");
                dbg_push!("hybrid/llm: skipped (LLM not available)");
            }
        }
        "cross-encoder" => {
            if !scored.is_empty() {
                let contents: Vec<String> = scored.iter().map(|r| r.content.clone()).collect();
                let content_refs: Vec<&str> = contents.iter().map(|s| s.as_str()).collect();
                let candidate_count = contents.len();
                let pre_order: Vec<uuid::Uuid> = scored.iter().map(|r| r.id).collect();

                let rerank_start = std::time::Instant::now();
                match crate::embedding::reranker::rerank(&params.query, &content_refs) {
                    Ok(reranked) => {
                        let rerank_ms = rerank_start.elapsed().as_millis();
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
                                    blended: rs.score,
                                    state_emoji: orig.state_emoji,
                                });
                            }
                        }
                        let post_order: Vec<uuid::Uuid> = reranked_results.iter().map(|r| r.id).collect();
                        let order_changed = pre_order != post_order;
                        let score_min = reranked.last().map(|r| r.score).unwrap_or(0.0);
                        let score_max = reranked.first().map(|r| r.score).unwrap_or(0.0);
                        scored = reranked_results;
                        eprintln!(
                            "[search] re-ranked {} candidates with cross-encoder",
                            candidate_count
                        );
                        dbg_push!(
                            "cross-encoder: re-ranked {} candidates in {}ms (scores: {:.4}..{:.4}, order_changed: {})",
                            candidate_count, rerank_ms, score_min, score_max, order_changed
                        );
                    }
                    Err(e) => {
                        eprintln!("[search] reranking failed, falling back to cosine: {}", e);
                        dbg_push!("cross-encoder: FAILED ({}), fell back to cosine", e);
                    }
                }
            }
        }
        _ => {
            dbg_push!("rerank: skipped (mode={})", params.rerank_mode);
        }
    }
    dbg_snapshot!("rerank", scored);

    // Session affinity gating: bias scores toward the conversation topic.
    // Applied after reranking (so we nudge reranked scores, not interfere with reranker)
    // and before association-following (so association merge uses adjusted scores).
    if let Some(ref centroid) = params.session_centroid
        && params.session_context_weight > 0.0
        && !scored.is_empty()
    {
        let mut session_adjusted = 0usize;
        for result in &mut scored {
            if let Ok(Some(emb)) = brain.get_embedding(&result.id) {
                let affinity = cosine_similarity(&emb, centroid);
                result.blended *= 1.0 + params.session_context_weight * affinity;
                session_adjusted += 1;
            }
        }
        // Re-sort after adjustment
        scored.sort_by(|a, b| {
            b.blended.partial_cmp(&a.blended).unwrap_or(std::cmp::Ordering::Equal)
        });
        if session_adjusted > 0 {
            eprintln!(
                "[search] session affinity: adjusted {} results (weight={})",
                session_adjusted, params.session_context_weight
            );
        }
        dbg_push!("session_affinity: adjusted {} results (weight={})", session_adjusted, params.session_context_weight);
    } else {
        dbg_push!("session_affinity: skipped (no centroid or weight=0)");
    }
    dbg_snapshot!("session_affinity", scored);

    // Association-following: discover related memories via associations
    let mut association_discovery_count: usize = 0;
    let mut association_merged_count: usize = 0;

    // Capture seed IDs (vector search results) before association merge
    // for chain detection later
    let seed_ids: Vec<uuid::Uuid> = scored.iter().map(|r| r.id).collect();

    if params.follow_associations && params.association_depth > 0 && !scored.is_empty() {
        let mut seen_ids: HashSet<uuid::Uuid> = seed_ids.iter().copied().collect();

        let discoveries = brain
            .discover_associated_memories(
                &query_embedding,
                &seed_ids,
                &mut seen_ids,
                params.association_depth,
            )
            .map_err(|e| format!("Association discovery failed: {}", e))?;

        // Cap discoveries to avoid swamping vector results.
        // Scale the cap with requested limit so list/multi-hop questions
        // can merge more associated clues.
        association_discovery_count = discoveries.len();
        let association_cap =
            params.effective_limit.clamp(params.association_cap_min, params.association_cap_max);
        let capped: Vec<_> = {
            let mut sorted = discoveries;
            sorted.sort_by(|a, b| {
                b.blended_score
                    .partial_cmp(&a.blended_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            sorted.into_iter().take(association_cap).collect()
        };

        // Merge discovered memories into scored results
        for d in capped {
            // After sync_from_storage, all engrams are in substrate cache.
            let engram = match brain.get(&d.id) {
                Some(e) => e,
                None => continue,
            };

            // State filter (same as above)
            let state_ok = if params.include_archived {
                true
            } else if params.include_deep {
                !engram.is_archived()
            } else {
                engram.is_searchable()
            };
            if !state_ok {
                continue;
            }

            // Date-range filter (same as above)
            if let Some(after) = params.created_after {
                if engram.created_at <= after {
                    continue;
                }
            }
            if let Some(before) = params.created_before {
                if engram.created_at >= before {
                    continue;
                }
            }

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
        scored.sort_by(|a, b| {
            b.blended
                .partial_cmp(&a.blended)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        association_merged_count = association_discovery_count.min(association_cap);

        // Log discovery stats for visibility
        eprintln!(
            "[search] association-following: {} discovered, {} merged (cap: {}, depth: {})",
            association_discovery_count,
            association_merged_count,
            association_cap,
            params.association_depth
        );
        dbg_push!("associations: {} discovered, {} merged (cap: {}, depth: {})",
            association_discovery_count, association_merged_count, association_cap, params.association_depth);
    } else {
        dbg_push!("associations: skipped (follow={}, depth={})", params.follow_associations, params.association_depth);
    }
    dbg_snapshot!("associations", scored);

    // Diversity shaping: pick a diverse, coverage-rich, source-balanced subset.
    // Redundancy is a property of results, not the query — always filter near-duplicates.
    if scored.len() > 1 {
        let original_count = scored.len();
        let target = params.effective_limit.min(scored.len());

        let cue_terms: Vec<String> = {
            let mut cues: Vec<String> = params
                .fallback_terms
                .iter()
                .filter_map(|t| normalize_shaping_token(t))
                .collect();
            if cues.is_empty() {
                cues = tokenize_for_shaping(&params.query).into_iter().collect();
            }
            cues.sort();
            cues.dedup();
            cues
        };

        let token_sets: Vec<HashSet<String>> = scored
            .iter()
            .map(|r| tokenize_for_shaping(&r.content))
            .collect();
        let lower_contents: Vec<String> =
            scored.iter().map(|r| r.content.to_lowercase()).collect();
        let source_keys: Vec<String> = scored
            .iter()
            .map(|r| source_bucket_key(&r.content, r.created_at))
            .collect();

        // Embedding cache: fetch once per candidate, reuse for subsequent comparisons.
        // Keyed by index into `scored` to avoid re-fetching from storage each iteration.
        let mut embedding_cache: HashMap<usize, Vec<f32>> = HashMap::new();
        let mut semantic_dedup_count: usize = 0;

        let mut selected_indices: Vec<usize> = Vec::with_capacity(target);
        let mut used = vec![false; scored.len()];
        let mut covered_cues: HashSet<String> = HashSet::new();
        let mut source_counts: HashMap<String, usize> = HashMap::new();

        while selected_indices.len() < target {
            let mut best_idx: Option<usize> = None;
            let mut best_score = f32::NEG_INFINITY;
            let mut best_new_cues: Vec<String> = Vec::new();

            for idx in 0..scored.len() {
                if used[idx] {
                    continue;
                }

                // --- Semantic dedup (stronger signal) ---
                // Try embedding-based cosine similarity first. If the candidate's
                // embedding is too similar to any already-selected result, skip it
                // entirely — these are effectively duplicates that Jaccard would miss
                // when the same fact is phrased differently.
                let candidate_embedding =
                    embedding_cache.get(&idx).cloned().unwrap_or_else(|| {
                        match brain.get_embedding(&scored[idx].id) {
                            Ok(Some(emb)) => {
                                embedding_cache.insert(idx, emb.clone());
                                emb
                            }
                            _ => Vec::new(), // empty = no embedding available
                        }
                    });

                if !candidate_embedding.is_empty() && !selected_indices.is_empty() {
                    let max_semantic_sim = selected_indices
                        .iter()
                        .filter_map(|&sel_idx| {
                            let sel_emb =
                                embedding_cache.get(&sel_idx).cloned().unwrap_or_else(|| {
                                    match brain.get_embedding(&scored[sel_idx].id) {
                                        Ok(Some(emb)) => {
                                            // Can't insert into cache here (borrowing issues),
                                            // but selected embeddings are already cached from
                                            // when they were candidates.
                                            emb
                                        }
                                        _ => Vec::new(),
                                    }
                                });
                            if sel_emb.is_empty() {
                                None
                            } else {
                                Some(cosine_similarity(&candidate_embedding, &sel_emb))
                            }
                        })
                        .fold(0.0_f32, f32::max);

                    if max_semantic_sim >= SEMANTIC_DEDUP_THRESHOLD {
                        // This result is a semantic duplicate — skip it entirely.
                        // Skipping (rather than penalizing) is appropriate because at
                        // >= 0.85 cosine similarity the content is near-identical.
                        used[idx] = true;
                        semantic_dedup_count += 1;
                        continue;
                    }
                }

                // --- Jaccard dedup (fallback for results without embeddings) ---
                let max_overlap = selected_indices
                    .iter()
                    .map(|sel| token_jaccard_overlap(&token_sets[idx], &token_sets[*sel]))
                    .fold(0.0_f32, f32::max);
                let diversity_penalty = max_overlap * 0.20;

                // Reward candidates that introduce uncovered cue terms.
                let mut new_cues = Vec::new();
                for cue in &cue_terms {
                    if !covered_cues.contains(cue) && lower_contents[idx].contains(cue.as_str())
                    {
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

                let adjusted =
                    scored[idx].blended + cue_bonus - diversity_penalty - source_penalty;

                if adjusted > best_score {
                    best_score = adjusted;
                    best_idx = Some(idx);
                    best_new_cues = new_cues;
                }
            }

            let Some(chosen) = best_idx else {
                break;
            };

            // Ensure the chosen result's embedding is cached for future comparisons
            if !embedding_cache.contains_key(&chosen) {
                if let Ok(Some(emb)) = brain.get_embedding(&scored[chosen].id) {
                    embedding_cache.insert(chosen, emb);
                }
            }

            used[chosen] = true;
            selected_indices.push(chosen);
            for cue in best_new_cues {
                covered_cues.insert(cue);
            }
            *source_counts
                .entry(source_keys[chosen].clone())
                .or_insert(0) += 1;
        }

        if !selected_indices.is_empty() {
            let mut reshaped = Vec::with_capacity(selected_indices.len());
            for idx in selected_indices {
                reshaped.push(scored[idx].clone());
            }
            scored = reshaped;
            eprintln!(
                "[search] diversity shaping: {} -> {} results (covered cues: {}/{}, semantic dedup: {})",
                original_count,
                scored.len(),
                covered_cues.len(),
                cue_terms.len(),
                semantic_dedup_count,
            );
            dbg_push!("diversity: {} -> {} results (covered cues: {}/{}, semantic dedup: {})",
                original_count, scored.len(), covered_cues.len(), cue_terms.len(), semantic_dedup_count);
        }
    }
    dbg_snapshot!("diversity", scored);

    // Chain detection: check if any seed results (vector search hits) are procedure anchors
    let chain_hints = detect_procedure_chains(brain, &seed_ids);

    // Limit results
    scored.truncate(params.effective_limit);

    dbg_push!("final: {} results returned", scored.len());

    Ok(SearchPipelineResult {
        results: scored,
        association_merged_count,
        association_discovery_count,
        chain_hints,
        debug_info: dbg,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::{Brain, storage::EngramStorage};

    /// Helper: create a Brain backed by in-memory SQLite
    fn brain_with_sqlite() -> Brain {
        let storage = EngramStorage::in_memory().unwrap();
        Brain::new(storage, crate::engram::Config::default()).unwrap()
    }

    #[test]
    fn detect_chains_finds_procedure_anchor() {
        let mut brain = brain_with_sqlite();

        let anchor = brain.create("Deploy procedure").unwrap();
        let step1 = brain.create("Pull code").unwrap();
        let step2 = brain.create("Run tests").unwrap();
        let step3 = brain.create("Deploy").unwrap();

        brain
            .associate_with_ordinal(anchor, step1, 0.9, Some(1))
            .unwrap();
        brain
            .associate_with_ordinal(anchor, step2, 0.9, Some(2))
            .unwrap();
        brain
            .associate_with_ordinal(anchor, step3, 0.9, Some(3))
            .unwrap();

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
        assert!(
            hints.is_empty(),
            "Unordered associations should not trigger chain detection"
        );
    }

    #[test]
    fn detect_chains_mixed_ordered_and_unordered() {
        let mut brain = brain_with_sqlite();

        let anchor = brain.create("Mixed anchor").unwrap();
        let step1 = brain.create("Step 1").unwrap();
        let related = brain.create("Related but unordered").unwrap();

        brain
            .associate_with_ordinal(anchor, step1, 0.9, Some(1))
            .unwrap();
        brain.associate(anchor, related, 0.5).unwrap();

        let hints = detect_procedure_chains(&brain, &[anchor]);
        assert_eq!(hints.len(), 1);
        assert_eq!(
            hints[0].ordered_step_count, 1,
            "Only count ordered associations"
        );
    }

    #[test]
    fn detect_chains_multiple_anchors() {
        let mut brain = brain_with_sqlite();

        let anchor1 = brain.create("Procedure A").unwrap();
        let a_step = brain.create("A step 1").unwrap();
        brain
            .associate_with_ordinal(anchor1, a_step, 0.9, Some(1))
            .unwrap();

        let anchor2 = brain.create("Procedure B").unwrap();
        let b_step1 = brain.create("B step 1").unwrap();
        let b_step2 = brain.create("B step 2").unwrap();
        brain
            .associate_with_ordinal(anchor2, b_step1, 0.9, Some(1))
            .unwrap();
        brain
            .associate_with_ordinal(anchor2, b_step2, 0.9, Some(2))
            .unwrap();

        let non_anchor = brain.create("Not a procedure").unwrap();

        let hints = detect_procedure_chains(&brain, &[anchor1, non_anchor, anchor2]);
        assert_eq!(hints.len(), 2);

        // Verify both anchors detected with correct step counts
        let hint_map: std::collections::HashMap<_, _> = hints
            .iter()
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
        assert!(is_composite_query(
            "What activities does Melanie partake in?"
        ));
        assert!(is_composite_query(
            "What do Jon and Gina both have in common?"
        ));
        assert!(!is_composite_query("Where did Caroline move from?"));
    }

    #[test]
    fn inferential_query_heuristic_detects_open_domain_style_questions() {
        assert!(is_inferential_query("What might John's degree be in?"));
        assert!(is_inferential_query(
            "Would Caroline be considered religious?"
        ));
        assert!(!is_inferential_query(
            "When did Caroline attend the support group?"
        ));
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

    #[test]
    fn source_bucket_falls_back_to_date_without_bracket_prefix() {
        // created_at = 1685059200 is 2023-05-26 00:00:00 UTC
        let key = source_bucket_key("Just a regular memory", 1685059200);
        assert_eq!(key, "2023-05-26");
    }

    /// Verify diversity shaping gate is unconditional — the code path no longer
    /// checks `is_composite_query()`. This test confirms the shaping helpers work
    /// correctly for a simple, non-composite query that would NOT have triggered
    /// the old gate.
    #[test]
    fn shaping_helpers_work_for_non_composite_queries() {
        let query = "Where did Caroline move from?";
        assert!(
            !is_composite_query(query),
            "sanity: query should NOT be composite"
        );

        // Tokenization and overlap still function for non-composite queries
        let tokens = tokenize_for_shaping(query);
        assert!(tokens.contains("caroline"));
        assert!(tokens.contains("move"));

        // Two near-duplicate contents should have high overlap
        let a = tokenize_for_shaping("Caroline moved from Portland to Seattle");
        let b = tokenize_for_shaping("Caroline moved from Portland to a new city");
        let overlap = token_jaccard_overlap(&a, &b);
        assert!(
            overlap > 0.3,
            "near-duplicate content should have significant overlap: {overlap}"
        );

        // Dissimilar content should have low overlap
        let c = tokenize_for_shaping("John enjoys hiking in the mountains every weekend");
        let overlap_ac = token_jaccard_overlap(&a, &c);
        assert!(
            overlap_ac < 0.15,
            "unrelated content should have low overlap: {overlap_ac}"
        );
    }

    /// The diversity penalty should down-rank near-duplicate results regardless
    /// of whether the query is composite. This simulates the shaping selection
    /// loop to verify that duplicates get penalized.
    #[test]
    fn diversity_penalty_demotes_near_duplicates() {
        // Simulate 3 results: two near-duplicates and one unique
        let contents = [
            "Caroline moved from Portland to Seattle last year",
            "Caroline moved from Portland to Seattle in March",
            "John works as a software engineer in Austin",
        ];
        let scores: [f32; 3] = [0.90, 0.88, 0.85];

        let token_sets: Vec<std::collections::HashSet<String>> =
            contents.iter().map(|c| tokenize_for_shaping(c)).collect();

        // Simulate greedy selection with diversity penalty (same as search.rs shaping loop)
        let mut selected: Vec<usize> = Vec::new();
        let mut used = [false; 3];
        let target = 3;

        while selected.len() < target {
            let mut best_idx: Option<usize> = None;
            let mut best_adj = f32::NEG_INFINITY;

            for idx in 0..3 {
                if used[idx] {
                    continue;
                }

                let max_overlap = selected
                    .iter()
                    .map(|sel| token_jaccard_overlap(&token_sets[idx], &token_sets[*sel]))
                    .fold(0.0_f32, f32::max);
                let diversity_penalty = max_overlap * 0.20;
                let adjusted = scores[idx] - diversity_penalty;

                if adjusted > best_adj {
                    best_adj = adjusted;
                    best_idx = Some(idx);
                }
            }

            let chosen = best_idx.unwrap();
            used[chosen] = true;
            selected.push(chosen);
        }

        // First pick should be highest-scored (index 0)
        assert_eq!(selected[0], 0, "highest score should be picked first");

        // Second pick should be the UNIQUE result (index 2), not the near-duplicate (index 1),
        // because the diversity penalty on index 1 should push it below index 2.
        assert_eq!(
            selected[1], 2,
            "unique result should be preferred over near-duplicate due to diversity penalty"
        );
        assert_eq!(selected[2], 1, "near-duplicate should be selected last");
    }

    /// Source bucket balancing should penalize results that are over-concentrated
    /// from the same source.
    #[test]
    fn source_balancing_penalizes_overrepresented_sources() {
        let contents = [
            "[chat-1] Message about cooking",
            "[chat-1] Message about baking",
            "[chat-1] Message about grilling",
            "[chat-2] Message about hiking",
        ];
        let scores: [f32; 4] = [0.90, 0.85, 0.80, 0.78];

        let source_keys: Vec<String> = contents.iter().map(|c| source_bucket_key(c, 0)).collect();

        // All chat-1 items should have the same bucket key
        assert_eq!(source_keys[0], source_keys[1]);
        assert_eq!(source_keys[1], source_keys[2]);
        assert_ne!(source_keys[0], source_keys[3]);

        // Simulate selection with source balancing (no diversity penalty for simplicity)
        let mut selected: Vec<usize> = Vec::new();
        let mut used = [false; 4];
        let mut source_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        while selected.len() < 4 {
            let mut best_idx: Option<usize> = None;
            let mut best_adj = f32::NEG_INFINITY;

            for idx in 0..4 {
                if used[idx] {
                    continue;
                }

                let src_count = *source_counts.get(&source_keys[idx]).unwrap_or(&0);
                let source_penalty = if src_count >= 2 {
                    (src_count as f32) * 0.08
                } else {
                    0.0
                };

                let adjusted = scores[idx] - source_penalty;
                if adjusted > best_adj {
                    best_adj = adjusted;
                    best_idx = Some(idx);
                }
            }

            let chosen = best_idx.unwrap();
            used[chosen] = true;
            selected.push(chosen);
            *source_counts
                .entry(source_keys[chosen].clone())
                .or_insert(0) += 1;
        }

        // First two from chat-1 are fine (under threshold). The third from chat-1
        // should get penalized, potentially letting chat-2 jump ahead.
        // With scores [0.90, 0.85, 0.80, 0.78] and penalty kicking in at src_count>=2:
        // Pick 1: idx 0 (0.90, no penalty) -> src_count chat-1 = 1
        // Pick 2: idx 1 (0.85, no penalty) -> src_count chat-1 = 2
        // Pick 3: idx 2 (0.80 - 2*0.08 = 0.64) vs idx 3 (0.78, no penalty) -> idx 3 wins
        // Pick 4: idx 2 (remaining)
        assert_eq!(selected[0], 0);
        assert_eq!(selected[1], 1);
        assert_eq!(
            selected[2], 3,
            "chat-2 result should be promoted over 3rd chat-1 result due to source penalty"
        );
        assert_eq!(selected[3], 2);
    }

    #[test]
    fn normalize_shaping_token_filters_stop_words() {
        assert_eq!(normalize_shaping_token("the"), None);
        assert_eq!(normalize_shaping_token("and"), None);
        assert_eq!(normalize_shaping_token("with"), None);
    }

    #[test]
    fn normalize_shaping_token_filters_short_tokens() {
        assert_eq!(normalize_shaping_token("ab"), None);
        assert_eq!(normalize_shaping_token("x"), None);
    }

    #[test]
    fn normalize_shaping_token_strips_possessive() {
        assert_eq!(normalize_shaping_token("John's"), Some("john".to_string()));
        // Smart quote possessive
        assert_eq!(
            normalize_shaping_token("John\u{2019}s"),
            Some("john".to_string())
        );
    }

    #[test]
    fn token_jaccard_overlap_empty_sets() {
        let empty: std::collections::HashSet<String> = std::collections::HashSet::new();
        let nonempty = tokenize_for_shaping("some content here");
        assert_eq!(token_jaccard_overlap(&empty, &nonempty), 0.0);
        assert_eq!(token_jaccard_overlap(&nonempty, &empty), 0.0);
        assert_eq!(token_jaccard_overlap(&empty, &empty), 0.0);
    }

    #[test]
    fn token_jaccard_overlap_identical_sets() {
        let a = tokenize_for_shaping("pottery class sunset");
        let b = tokenize_for_shaping("pottery class sunset");
        assert!((token_jaccard_overlap(&a, &b) - 1.0).abs() < f32::EPSILON);
    }

    /// Semantic dedup: results with cosine similarity >= SEMANTIC_DEDUP_THRESHOLD
    /// against an already-selected result should be skipped entirely.
    #[test]
    fn semantic_dedup_skips_high_cosine_similarity() {
        use crate::embedding::cosine_similarity;

        let mut brain = brain_with_sqlite();

        let id1 = brain
            .create("Memory MCP server location: /work/memoryco/memory")
            .unwrap();
        let id2 = brain
            .create("MemoryCo source code lives at /work/memoryco/memory")
            .unwrap();
        let id3 = brain
            .create("Brandon enjoys hiking in the Cascades every weekend")
            .unwrap();

        // Create embeddings: id1 and id2 are near-identical (cosine sim ~1.0),
        // id3 is orthogonal.
        let emb_similar: Vec<f32> = vec![0.8, 0.2, 0.1, 0.0];
        // Slight perturbation — still very high cosine similarity
        let emb_similar2: Vec<f32> = vec![0.79, 0.21, 0.11, 0.01];
        let emb_different: Vec<f32> = vec![0.0, 0.1, 0.2, 0.9];

        // Verify our test embeddings actually have the properties we expect
        let sim_12 = cosine_similarity(&emb_similar, &emb_similar2);
        let sim_13 = cosine_similarity(&emb_similar, &emb_different);
        assert!(
            sim_12 >= SEMANTIC_DEDUP_THRESHOLD,
            "test embeddings 1&2 should be above threshold: {sim_12}"
        );
        assert!(
            sim_13 < SEMANTIC_DEDUP_THRESHOLD,
            "test embeddings 1&3 should be below threshold: {sim_13}"
        );

        brain.set_embedding(&id1, &emb_similar).unwrap();
        brain.set_embedding(&id2, &emb_similar2).unwrap();
        brain.set_embedding(&id3, &emb_different).unwrap();

        // Simulate the shaping loop's semantic dedup logic
        let scores: [f32; 3] = [0.90, 0.88, 0.80];
        let mut embedding_cache: std::collections::HashMap<usize, Vec<f32>> =
            std::collections::HashMap::new();

        // Pre-fill cache
        embedding_cache.insert(0, emb_similar.clone());
        embedding_cache.insert(1, emb_similar2.clone());
        embedding_cache.insert(2, emb_different.clone());

        let mut selected: Vec<usize> = Vec::new();
        let mut used = [false; 3];
        let mut semantic_dedup_count = 0;

        // Greedy selection with semantic dedup
        for _ in 0..3 {
            let mut best_idx: Option<usize> = None;
            let mut best_score = f32::NEG_INFINITY;

            for idx in 0..3 {
                if used[idx] {
                    continue;
                }

                // Semantic dedup check
                let candidate_emb = embedding_cache.get(&idx).unwrap();
                let max_sim = selected
                    .iter()
                    .filter_map(|&sel_idx| {
                        embedding_cache
                            .get(&sel_idx)
                            .map(|sel_emb| cosine_similarity(candidate_emb, sel_emb))
                    })
                    .fold(0.0_f32, f32::max);

                if max_sim >= SEMANTIC_DEDUP_THRESHOLD {
                    used[idx] = true;
                    semantic_dedup_count += 1;
                    continue;
                }

                if scores[idx] > best_score {
                    best_score = scores[idx];
                    best_idx = Some(idx);
                }
            }

            if let Some(chosen) = best_idx {
                used[chosen] = true;
                selected.push(chosen);
            } else {
                break;
            }
        }

        // id1 should be selected (highest score), id2 should be deduped (similar to id1),
        // id3 should survive (different embedding)
        assert_eq!(selected, vec![0, 2], "id2 (index 1) should be deduped");
        assert_eq!(
            semantic_dedup_count, 1,
            "exactly one result should be semantic-deduped"
        );
    }

    /// Results below the semantic dedup threshold should survive even if they have
    /// moderate cosine similarity.
    #[test]
    fn semantic_dedup_preserves_results_below_threshold() {
        use crate::embedding::cosine_similarity;

        // Two embeddings with moderate similarity (below threshold)
        let emb_a: Vec<f32> = vec![0.8, 0.2, 0.1, 0.0];
        let emb_b: Vec<f32> = vec![0.5, 0.5, 0.2, 0.1];

        let sim = cosine_similarity(&emb_a, &emb_b);
        assert!(
            sim < SEMANTIC_DEDUP_THRESHOLD,
            "test embeddings should be below threshold: {sim}"
        );
        assert!(
            sim > 0.3,
            "test embeddings should have some similarity: {sim}"
        );

        // Simulate: both should survive the dedup
        let mut embedding_cache: std::collections::HashMap<usize, Vec<f32>> =
            std::collections::HashMap::new();
        embedding_cache.insert(0, emb_a);
        embedding_cache.insert(1, emb_b);

        let scores: [f32; 2] = [0.90, 0.85];
        let mut selected: Vec<usize> = Vec::new();
        let mut used = [false; 2];
        let mut semantic_dedup_count = 0;

        for _ in 0..2 {
            let mut best_idx: Option<usize> = None;
            let mut best_score = f32::NEG_INFINITY;

            for idx in 0..2 {
                if used[idx] {
                    continue;
                }

                let candidate_emb = embedding_cache.get(&idx).unwrap();
                let max_sim = selected
                    .iter()
                    .filter_map(|&sel_idx| {
                        embedding_cache
                            .get(&sel_idx)
                            .map(|sel_emb| cosine_similarity(candidate_emb, sel_emb))
                    })
                    .fold(0.0_f32, f32::max);

                if max_sim >= SEMANTIC_DEDUP_THRESHOLD {
                    used[idx] = true;
                    semantic_dedup_count += 1;
                    continue;
                }

                if scores[idx] > best_score {
                    best_score = scores[idx];
                    best_idx = Some(idx);
                }
            }

            if let Some(chosen) = best_idx {
                used[chosen] = true;
                selected.push(chosen);
            } else {
                break;
            }
        }

        assert_eq!(
            selected,
            vec![0, 1],
            "both results should survive below-threshold similarity"
        );
        assert_eq!(semantic_dedup_count, 0, "no results should be deduped");
    }

    /// When embedding is unavailable (None from storage), the Jaccard fallback
    /// should still penalize near-duplicate content.
    #[test]
    fn jaccard_fallback_when_embedding_unavailable() {
        let mut brain = brain_with_sqlite();

        // Create engrams but do NOT set embeddings — simulates embedding unavailability
        let id1 = brain
            .create("Caroline moved from Portland to Seattle last year")
            .unwrap();
        let id2 = brain
            .create("Caroline moved from Portland to Seattle in March")
            .unwrap();
        let id3 = brain
            .create("John works as a software engineer in Austin")
            .unwrap();

        // Verify no embeddings exist
        assert!(brain.get_embedding(&id1).unwrap().is_none());
        assert!(brain.get_embedding(&id2).unwrap().is_none());
        assert!(brain.get_embedding(&id3).unwrap().is_none());

        let contents = [
            "Caroline moved from Portland to Seattle last year",
            "Caroline moved from Portland to Seattle in March",
            "John works as a software engineer in Austin",
        ];
        let scores: [f32; 3] = [0.90, 0.88, 0.85];

        let token_sets: Vec<std::collections::HashSet<String>> =
            contents.iter().map(|c| tokenize_for_shaping(c)).collect();

        // Simulate greedy selection with Jaccard fallback (no embeddings = empty cache)
        let embedding_cache: std::collections::HashMap<usize, Vec<f32>> =
            std::collections::HashMap::new();

        let mut selected: Vec<usize> = Vec::new();
        let mut used = [false; 3];

        for _ in 0..3 {
            let mut best_idx: Option<usize> = None;
            let mut best_adj = f32::NEG_INFINITY;

            for idx in 0..3 {
                if used[idx] {
                    continue;
                }

                // Semantic dedup: skip if embedding available and above threshold
                let candidate_emb = embedding_cache.get(&idx);
                if let Some(c_emb) = candidate_emb {
                    if !c_emb.is_empty() {
                        let max_sim = selected
                            .iter()
                            .filter_map(|&sel| {
                                embedding_cache
                                    .get(&sel)
                                    .filter(|e| !e.is_empty())
                                    .map(|sel_emb| cosine_similarity(c_emb, sel_emb))
                            })
                            .fold(0.0_f32, f32::max);
                        if max_sim >= SEMANTIC_DEDUP_THRESHOLD {
                            used[idx] = true;
                            continue;
                        }
                    }
                }

                // Jaccard fallback
                let max_overlap = selected
                    .iter()
                    .map(|sel| token_jaccard_overlap(&token_sets[idx], &token_sets[*sel]))
                    .fold(0.0_f32, f32::max);
                let diversity_penalty = max_overlap * 0.20;
                let adjusted = scores[idx] - diversity_penalty;

                if adjusted > best_adj {
                    best_adj = adjusted;
                    best_idx = Some(idx);
                }
            }

            if let Some(chosen) = best_idx {
                used[chosen] = true;
                selected.push(chosen);
            } else {
                break;
            }
        }

        // Without embeddings, Jaccard kicks in. The near-duplicate (idx 1) should be
        // penalized, making idx 2 (unique content) preferred in position 2.
        assert_eq!(selected[0], 0, "highest score should be first");
        assert_eq!(
            selected[1], 2,
            "Jaccard fallback should promote unique result over near-duplicate"
        );
        assert_eq!(selected[2], 1, "near-duplicate should be selected last");
    }

    #[test]
    fn semantic_dedup_threshold_constant_is_reasonable() {
        assert!(
            SEMANTIC_DEDUP_THRESHOLD > 0.5,
            "threshold should be well above random similarity"
        );
        assert!(
            SEMANTIC_DEDUP_THRESHOLD < 1.0,
            "threshold should allow some variation"
        );
        assert!(
            (SEMANTIC_DEDUP_THRESHOLD - 0.85).abs() < f32::EPSILON,
            "threshold should be 0.85"
        );
    }

    // ==================
    // parse_timestamp tests
    // ==================

    #[test]
    fn parse_timestamp_unix_epoch() {
        assert_eq!(parse_timestamp("1710432000"), Ok(1710432000));
    }

    #[test]
    fn parse_timestamp_unix_epoch_with_whitespace() {
        assert_eq!(parse_timestamp("  1710432000  "), Ok(1710432000));
    }

    #[test]
    fn parse_timestamp_iso8601_full_utc() {
        // 2026-03-14T19:00:00Z
        let result = parse_timestamp("2026-03-14T19:00:00Z").unwrap();
        let dt = chrono::DateTime::from_timestamp(result, 0).unwrap();
        assert_eq!(dt.format("%Y-%m-%d %H:%M:%S").to_string(), "2026-03-14 19:00:00");
    }

    #[test]
    fn parse_timestamp_iso8601_with_offset() {
        // 2026-03-14T12:00:00-07:00 = 2026-03-14T19:00:00Z
        let result = parse_timestamp("2026-03-14T12:00:00-07:00").unwrap();
        let dt = chrono::DateTime::from_timestamp(result, 0).unwrap();
        assert_eq!(dt.format("%Y-%m-%d %H:%M:%S").to_string(), "2026-03-14 19:00:00");
    }

    #[test]
    fn parse_timestamp_iso8601_no_timezone() {
        // Treated as UTC
        let result = parse_timestamp("2026-03-14T19:00:00").unwrap();
        let dt = chrono::DateTime::from_timestamp(result, 0).unwrap();
        assert_eq!(dt.format("%Y-%m-%d %H:%M:%S").to_string(), "2026-03-14 19:00:00");
    }

    #[test]
    fn parse_timestamp_date_only() {
        // "2026-03-14" → start of day UTC
        let result = parse_timestamp("2026-03-14").unwrap();
        let dt = chrono::DateTime::from_timestamp(result, 0).unwrap();
        assert_eq!(dt.format("%Y-%m-%d %H:%M:%S").to_string(), "2026-03-14 00:00:00");
    }

    #[test]
    fn parse_timestamp_invalid_string() {
        let result = parse_timestamp("not-a-date");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Cannot parse timestamp"));
    }

    #[test]
    fn parse_timestamp_empty_string() {
        let result = parse_timestamp("");
        assert!(result.is_err());
    }

    #[test]
    fn parse_timestamp_negative_epoch() {
        // Before unix epoch (e.g. 1969)
        assert_eq!(parse_timestamp("-86400"), Ok(-86400));
    }

    #[test]
    fn parse_timestamp_zero_epoch() {
        assert_eq!(parse_timestamp("0"), Ok(0));
    }

    // ==================
    // Session affinity gate tests
    // ==================

    /// When a session centroid is provided and weight > 0, results whose embeddings
    /// align with the centroid should receive a score boost.
    #[test]
    fn test_session_affinity_gate_boosts_matching_results() {
        use crate::embedding::cosine_similarity;

        let mut brain = brain_with_sqlite();

        // Two engrams: one whose embedding aligns with the session centroid, one orthogonal.
        let id_matching = brain.create("Rust memory management and ownership").unwrap();
        let id_unrelated = brain.create("Banana bread recipe with walnuts").unwrap();

        // Session centroid: points toward Rust/tech
        let centroid: Vec<f32> = vec![0.9, 0.1, 0.0, 0.0];
        // Matching: closely aligned with centroid
        let emb_matching: Vec<f32> = vec![0.85, 0.15, 0.0, 0.0];
        // Unrelated: orthogonal to centroid
        let emb_unrelated: Vec<f32> = vec![0.0, 0.0, 0.85, 0.15];

        brain.set_embedding(&id_matching, &emb_matching).unwrap();
        brain.set_embedding(&id_unrelated, &emb_unrelated).unwrap();

        let base_score = 0.7_f32;
        let weight = 0.3_f32;

        // Compute expected affinity scores
        let affinity_matching = cosine_similarity(&emb_matching, &centroid);
        let affinity_unrelated = cosine_similarity(&emb_unrelated, &centroid);
        assert!(
            affinity_matching > affinity_unrelated,
            "matching embedding should have higher centroid affinity"
        );

        // Simulate the affinity gate logic
        let expected_matching = base_score * (1.0 + weight * affinity_matching);
        let expected_unrelated = base_score * (1.0 + weight * affinity_unrelated);

        assert!(
            expected_matching > expected_unrelated,
            "matching result should score higher after session affinity gate: {} vs {}",
            expected_matching,
            expected_unrelated
        );
        assert!(
            expected_matching > base_score,
            "session affinity should boost the matching result above baseline"
        );
    }

    /// When session_centroid is None, the gate is a no-op: scored results are unchanged.
    #[test]
    fn test_session_affinity_gate_noop_when_no_centroid() {
        let brain = brain_with_sqlite();

        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();
        let original_scores = [0.9_f32, 0.7_f32];

        // Simulate the gate with no centroid: nothing should change.
        let centroid: Option<Vec<f32>> = None;
        let weight = 0.3_f32;

        let mut scores = original_scores;

        if let Some(ref c) = centroid {
            if weight > 0.0 {
                for (score, id) in scores.iter_mut().zip([id1, id2].iter()) {
                    if let Ok(Some(emb)) = brain.get_embedding(id) {
                        let affinity = cosine_similarity(&emb, c);
                        *score *= 1.0 + weight * affinity;
                    }
                }
            }
        }

        assert_eq!(
            scores, original_scores,
            "scores should be unchanged when centroid is None"
        );
    }

    /// When session_context_weight is 0.0, the gate is a no-op even if a centroid is present.
    #[test]
    fn test_session_affinity_gate_noop_when_weight_zero() {
        let mut brain = brain_with_sqlite();

        let id1 = brain.create("Rust memory management").unwrap();
        let emb: Vec<f32> = vec![0.9, 0.1, 0.0, 0.0];
        brain.set_embedding(&id1, &emb).unwrap();

        let centroid: Option<Vec<f32>> = Some(vec![0.9, 0.1, 0.0, 0.0]);
        let weight = 0.0_f32;

        let original_score = 0.8_f32;
        let mut score = original_score;

        if let Some(ref c) = centroid {
            if weight > 0.0 {
                if let Ok(Some(emb_loaded)) = brain.get_embedding(&id1) {
                    let affinity = cosine_similarity(&emb_loaded, c);
                    score *= 1.0 + weight * affinity;
                }
            }
        }

        assert_eq!(
            score, original_score,
            "score should be unchanged when weight is 0.0"
        );
    }
}
