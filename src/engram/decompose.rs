//! Compound memory decomposition
//!
//! Scans engrams for compound memories (multiple facts crammed into one)
//! and splits them into atomic memories. This improves embedding quality
//! since each child memory represents a single concept.

use super::brain::Brain;
use super::association::Association;
use super::engram::MemoryState;
use super::EngramId;
use crate::storage::StorageResult;
use crate::embedding::EmbeddingGenerator;

/// Report returned after decomposition
#[derive(Debug, Clone)]
pub struct DecomposeReport {
    /// Total engrams scanned
    pub total_scanned: usize,
    /// Number of engrams that were decomposed
    pub total_decomposed: usize,
    /// Total children created across all decompositions
    pub total_children_created: usize,
    /// Associations skipped because they had ordinals (procedure chains)
    pub skipped_procedural: usize,
    /// Errors encountered (non-fatal)
    pub errors: Vec<String>,
}

/// Scoring breakdown for a single engram
#[derive(Debug, Clone)]
pub struct CompoundScore {
    pub total: f64,
    pub sentence_count: usize,
    pub numbered_items: usize,
    pub semicolons: usize,
    pub connectors: usize,
    pub length: usize,
    pub line_breaks: usize,
}

/// Properties collected from a parent engram before decomposition.
/// Children inherit these so they reflect the parent's actual state
/// rather than starting fresh.
struct DecomposePlan {
    parent_id: EngramId,
    children_content: Vec<String>,
    created_at: i64,
    #[allow(dead_code)]
    tags: Vec<String>,
    energy: f64,
    state: MemoryState,
    access_count: u64,
    last_accessed: i64,
    compound_score: f64,
}

/// Count sentences in text using punctuation boundaries
fn count_sentences(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    // Split on sentence-ending punctuation followed by space or end
    let mut count = 0;
    let chars: Vec<char> = text.chars().collect();
    for i in 0..chars.len() {
        if chars[i] == '.' || chars[i] == '!' || chars[i] == '?' {
            // Check it's not part of a number like "1.5" or abbreviation
            let prev_is_digit = i > 0 && chars[i - 1].is_ascii_digit();
            let next_is_digit = i + 1 < chars.len() && chars[i + 1].is_ascii_digit();
            if prev_is_digit && next_is_digit {
                continue; // Skip decimals like "1.5"
            }
            // Must be followed by space, end of string, or quote
            let at_end = i + 1 >= chars.len();
            let next_is_space = !at_end && (chars[i + 1] == ' ' || chars[i + 1] == '\n');
            let next_is_quote = !at_end && (chars[i + 1] == '"' || chars[i + 1] == '\'');
            if at_end || next_is_space || next_is_quote {
                count += 1;
            }
        }
    }
    // If we found no sentence-enders but there's content, count as 1
    if count == 0 && !text.trim().is_empty() {
        count = 1;
    }
    count
}

/// Count numbered items like (1), 1), 1. patterns
fn count_numbered_items(text: &str) -> usize {
    let mut count = 0;
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    for i in 0..len {
        // Pattern: (N) where N is one or more digits
        if chars[i] == '(' {
            let mut j = i + 1;
            while j < len && chars[j].is_ascii_digit() {
                j += 1;
            }
            if j > i + 1 && j < len && chars[j] == ')' {
                count += 1;
            }
        }
        // Pattern: N) at start of line or after space
        // Pattern: N. at start of line or after space (but not decimals)
        if chars[i].is_ascii_digit() {
            let at_start = i == 0 || chars[i - 1] == ' ' || chars[i - 1] == '\n';
            if at_start {
                // Consume all digits
                let mut j = i;
                while j < len && chars[j].is_ascii_digit() {
                    j += 1;
                }
                if j < len {
                    if chars[j] == ')' {
                        count += 1;
                    } else if chars[j] == '.' {
                        // Make sure it's not a decimal (next char should be space or letter)
                        let next_is_digit = j + 1 < len && chars[j + 1].is_ascii_digit();
                        if !next_is_digit {
                            count += 1;
                        }
                    }
                }
            }
        }
    }
    count
}

/// Count connectors that indicate compound thoughts
fn count_connectors(text: &str) -> usize {
    let lower = text.to_lowercase();
    let words = [" also ", " additionally ", " furthermore ", " moreover "];
    words.iter().map(|w| lower.matches(w).count()).sum()
}

/// Count semicolons separating facts
fn count_semicolons(text: &str) -> usize {
    text.matches(';').count()
}

/// Count line breaks with content (list-like structure)
fn count_content_line_breaks(text: &str) -> usize {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= 1 {
        return 0;
    }
    // Count non-empty lines after the first
    lines[1..].iter().filter(|l| !l.trim().is_empty()).count()
}

/// Score how compound a memory is
pub fn score_compound(content: &str) -> CompoundScore {
    let sentence_count = count_sentences(content);
    let numbered_items = count_numbered_items(content);
    let semicolons = count_semicolons(content);
    let connectors = count_connectors(content);
    let length = content.len();
    let line_breaks = count_content_line_breaks(content);

    let mut total = 0.0;

    // Each sentence above 2 adds 1.0
    if sentence_count > 2 {
        total += (sentence_count - 2) as f64 * 1.0;
    }

    // Each numbered item adds 1.5
    total += numbered_items as f64 * 1.5;

    // Each semicolon adds 0.5
    total += semicolons as f64 * 0.5;

    // Each connector adds 0.5
    total += connectors as f64 * 0.5;

    // Length penalty: +1.0 per 200 chars above 400
    if length > 400 {
        total += ((length - 400) as f64 / 200.0).floor() * 1.0;
    }

    // Line breaks with content: each adds 0.3
    total += line_breaks as f64 * 0.3;

    CompoundScore {
        total,
        sentence_count,
        numbered_items,
        semicolons,
        connectors,
        length,
        line_breaks,
    }
}

/// Minimum compound score to trigger decomposition
const DECOMPOSE_THRESHOLD: f64 = 3.0;

/// Split a compound memory into atomic pieces.
/// Returns a vec of child content strings, or empty if not decomposable.
pub fn split_content(content: &str, score: &CompoundScore) -> Vec<String> {
    // Tier 1: numbered items
    if score.numbered_items >= 2 {
        let result = split_numbered(content);
        if result.len() >= 2 {
            return result;
        }
    }

    // Tier 2: sentence boundary split (4+ sentences)
    if score.sentence_count >= 4 {
        let result = split_sentences(content);
        if result.len() >= 2 {
            return result;
        }
    }

    Vec::new()
}

/// Tier 1: Split on numbered patterns like (1)...(2)... or 1)...2)... or 1. ...2. ...
fn split_numbered(content: &str) -> Vec<String> {
    // Try each pattern family and pick the one that yields the most splits
    let patterns: &[&[&str]] = &[
        // (N) pattern
        &["(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)", "(8)", "(9)", "(10)",
          "(11)", "(12)", "(13)", "(14)", "(15)"],
        // N) pattern — only match at word boundary
        &["1)", "2)", "3)", "4)", "5)", "6)", "7)", "8)", "9)", "10)",
          "11)", "12)", "13)", "14)", "15)"],
        // N. pattern
        &["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.",
          "11.", "12.", "13.", "14.", "15."],
    ];

    let mut best_result: Vec<String> = Vec::new();

    for pattern_family in patterns {
        let result = try_split_numbered_with_patterns(content, pattern_family);
        if result.len() > best_result.len() {
            best_result = result;
        }
    }

    best_result
}

/// Try splitting with a specific numbered pattern family
fn try_split_numbered_with_patterns(content: &str, patterns: &[&str]) -> Vec<String> {
    // Find the first pattern that appears
    let mut first_idx = None;
    let mut first_pattern_num = 0;

    for (i, pat) in patterns.iter().enumerate() {
        if let Some(pos) = find_pattern_position(content, pat) {
            if first_idx.is_none() || pos < first_idx.unwrap() {
                first_idx = Some(pos);
                first_pattern_num = i;
                break; // patterns are in order, so the first match is pattern #1
            }
        }
    }

    let first_idx = match first_idx {
        Some(idx) => idx,
        None => return Vec::new(),
    };

    // The prefix is everything before the first numbered item
    let prefix = content[..first_idx].trim();

    // Now find all consecutive numbered items starting from the first
    let mut items: Vec<String> = Vec::new();
    let mut remaining = &content[first_idx..];

    for i in first_pattern_num..patterns.len() {
        let current_pat = patterns[i];
        let next_pat = if i + 1 < patterns.len() {
            Some(patterns[i + 1])
        } else {
            None
        };

        if let Some(start) = find_pattern_position(remaining, current_pat) {
            let content_start = start + current_pat.len();
            let content_after = &remaining[content_start..];

            // Find end: either next pattern or end of string
            let end = if let Some(next) = next_pat {
                find_pattern_position(content_after, next).unwrap_or(content_after.len())
            } else {
                content_after.len()
            };

            let item_content = content_after[..end].trim();
            if !item_content.is_empty() {
                // Build child: prefix + item content
                let child = if !prefix.is_empty() {
                    // Remove trailing colon from prefix if present for cleaner joining
                    let clean_prefix = prefix.trim_end_matches(':').trim();
                    format!("{}: {}", clean_prefix, item_content)
                } else {
                    item_content.to_string()
                };
                items.push(child);
            }

            // Advance remaining past this item
            remaining = &remaining[content_start + end..];
        } else {
            break; // Pattern not found, stop
        }
    }

    items
}

/// Find pattern position, respecting word boundaries for short patterns like "1)"
fn find_pattern_position(text: &str, pattern: &str) -> Option<usize> {
    let mut search_from = 0;
    while let Some(pos) = text[search_from..].find(pattern) {
        let absolute_pos = search_from + pos;
        // For patterns like "1)", "2." etc, check word boundary
        let needs_boundary = pattern.len() <= 3
            && pattern.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false);

        if needs_boundary {
            let at_start = absolute_pos == 0;
            let prev_is_boundary = absolute_pos > 0 && {
                let prev = text.as_bytes()[absolute_pos - 1];
                prev == b' ' || prev == b'\n' || prev == b',' || prev == b';'
            };
            if at_start || prev_is_boundary {
                return Some(absolute_pos);
            }
            search_from = absolute_pos + 1;
        } else {
            return Some(absolute_pos);
        }
    }
    None
}

/// Tier 2: Split on sentence boundaries, grouping into chunks of 1-2 sentences
fn split_sentences(content: &str) -> Vec<String> {
    let sentences = extract_sentences(content);
    if sentences.len() < 4 {
        return Vec::new();
    }

    // Check if first sentence looks like a topic header
    let first = &sentences[0];
    let is_header = first.len() < 80 && (first.ends_with(':') || first.ends_with('.'));

    let prefix = if is_header && sentences.len() > 2 {
        Some(first.trim_end_matches('.').trim_end_matches(':').trim())
    } else {
        None
    };

    let start_idx = if prefix.is_some() { 1 } else { 0 };
    let remaining = &sentences[start_idx..];

    // Group into chunks of 2 sentences
    let mut children = Vec::new();
    for chunk in remaining.chunks(2) {
        let combined = chunk.join(" ");
        let child = if let Some(pfx) = prefix {
            format!("{}: {}", pfx, combined.trim())
        } else {
            combined.trim().to_string()
        };
        if !child.is_empty() {
            children.push(child);
        }
    }

    children
}

/// Extract sentences from text
fn extract_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();

    for i in 0..chars.len() {
        current.push(chars[i]);

        if chars[i] == '.' || chars[i] == '!' || chars[i] == '?' {
            // Check for decimal numbers
            let prev_is_digit = i > 0 && chars[i - 1].is_ascii_digit();
            let next_is_digit = i + 1 < chars.len() && chars[i + 1].is_ascii_digit();
            if prev_is_digit && next_is_digit {
                continue;
            }

            let at_end = i + 1 >= chars.len();
            let next_is_space = !at_end && (chars[i + 1] == ' ' || chars[i + 1] == '\n');
            let next_is_quote = !at_end && (chars[i + 1] == '"' || chars[i + 1] == '\'');

            if at_end || next_is_space || next_is_quote {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current.clear();
            }
        }
    }

    // Don't forget trailing content without sentence-ender
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

/// Run compound memory decomposition across all engrams in the brain.
///
/// Scans every engram, scores it for compound-ness, and splits those
/// above the threshold into atomic children. Transfers associations
/// from parent to children (skipping ordinal/procedural ones).
pub fn decompose_compound_memories(brain: &mut Brain) -> StorageResult<DecomposeReport> {
    let mut report = DecomposeReport {
        total_scanned: 0,
        total_decomposed: 0,
        total_children_created: 0,
        skipped_procedural: 0,
        errors: Vec::new(),
    };

    // Collect all engrams first to avoid borrow issues
    let engrams: Vec<_> = brain
        .all_engrams()
        .map(|e| (e.id, e.content.clone(), e.created_at, e.tags.clone(), e.energy, e.state, e.access_count, e.last_accessed))
        .collect();

    report.total_scanned = engrams.len();

    // Build decomposition plans
    let mut plans: Vec<DecomposePlan> = Vec::new();

    for (id, content, created_at, tags, energy, state, access_count, last_accessed) in &engrams {
        let score = score_compound(content);
        if score.total < DECOMPOSE_THRESHOLD {
            continue;
        }

        let children = split_content(content, &score);
        if children.len() < 2 {
            continue; // Splitting didn't produce useful results
        }

        plans.push(DecomposePlan {
            parent_id: *id,
            children_content: children,
            created_at: *created_at,
            tags: tags.clone(),
            energy: *energy,
            state: *state,
            access_count: *access_count,
            last_accessed: *last_accessed,
            compound_score: score.total,
        });
    }

    // Execute each decomposition plan
    for plan in plans {
        let num_children = plan.children_content.len();
        let compound_score = plan.compound_score;
        let parent_id = plan.parent_id;
        match decompose_one(brain, plan, &mut report) {
            Ok(()) => {
                eprintln!(
                    "[decompose] {} → {} children (score: {:.1})",
                    parent_id, num_children, compound_score
                );
                report.total_decomposed += 1;
            }
            Err(e) => {
                report.errors.push(format!(
                    "Failed to decompose {}: {}",
                    parent_id, e
                ));
            }
        }
    }

    eprintln!(
        "[decompose] Done: scanned={}, decomposed={}, children={}, skipped_procedural={}, errors={}",
        report.total_scanned,
        report.total_decomposed,
        report.total_children_created,
        report.skipped_procedural,
        report.errors.len(),
    );

    Ok(report)
}

/// Decompose a single parent engram into children
fn decompose_one(
    brain: &mut Brain,
    plan: DecomposePlan,
    report: &mut DecomposeReport,
) -> StorageResult<()> {
    let parent_id = plan.parent_id;

    // Collect association info before creating children
    let outbound: Vec<(EngramId, f64, Option<u32>)> = brain
        .associations_from(&parent_id)
        .map(|assocs| {
            assocs
                .iter()
                .map(|a| (a.to, a.weight, a.ordinal))
                .collect()
        })
        .unwrap_or_default();

    let inbound: Vec<(EngramId, f64, Option<u32>)> = brain
        .associations_to(&parent_id)
        .map(|sources| {
            sources
                .iter()
                .filter_map(|from_id| {
                    brain
                        .associations_from(from_id)
                        .and_then(|assocs| assocs.iter().find(|a| a.to == parent_id))
                        .map(|a| (a.from, a.weight, a.ordinal))
                })
                .collect()
        })
        .unwrap_or_default();

    // Create children with parent's timestamp
    let mut child_ids: Vec<EngramId> = Vec::new();
    for content in &plan.children_content {
        let child_id = brain.create_with_timestamp(content, plan.created_at)?;

        // Inherit parent's properties so children reflect the parent's actual state
        if let Some(child) = brain.substrate.get_mut(&child_id) {
            child.energy = plan.energy;
            child.state = plan.state;
            child.access_count = plan.access_count;
            child.last_accessed = plan.last_accessed;
        }
        // Persist inherited properties
        if let Some(child) = brain.substrate.get(&child_id).cloned() {
            brain.storage_save_engram(&child)?;
        }

        child_ids.push(child_id);
    }

    // Generate embeddings for children
    let generator = EmbeddingGenerator::new();
    let content_refs: Vec<&str> = plan.children_content.iter().map(|s| s.as_str()).collect();
    match generator.generate_batch(&content_refs) {
        Ok(embeddings) => {
            for (id, embedding) in child_ids.iter().zip(embeddings.iter()) {
                if let Err(e) = brain.set_embedding(id, embedding) {
                    report.errors.push(format!(
                        "Failed to set embedding for child {}: {}",
                        id, e
                    ));
                }
            }
        }
        Err(e) => {
            report.errors.push(format!(
                "Batch embedding failed for children of {}: {}",
                parent_id, e
            ));
        }
    }

    // Transfer associations from parent to children
    // Outbound: P→X becomes C1→X, C2→X, etc.
    for (target_id, weight, ordinal) in &outbound {
        if ordinal.is_some() {
            report.skipped_procedural += 1;
            eprintln!(
                "[decompose] WARN: Skipping ordinal association {}→{} (ordinal={:?})",
                parent_id, target_id, ordinal
            );
            continue;
        }
        for child_id in &child_ids {
            // Create association with reset co_activation_count
            let assoc = Association::with_weight(*child_id, *target_id, *weight);
            brain.substrate.insert_association(assoc.clone());
            // Persist
            if let Err(e) = brain.storage_save_association(&assoc) {
                report.errors.push(format!(
                    "Failed to save association {}→{}: {}",
                    child_id, target_id, e
                ));
            }
        }
    }

    // Inbound: X→P becomes X→C1, X→C2, etc.
    for (source_id, weight, ordinal) in &inbound {
        if ordinal.is_some() {
            report.skipped_procedural += 1;
            eprintln!(
                "[decompose] WARN: Skipping ordinal association {}→{} (ordinal={:?})",
                source_id, parent_id, ordinal
            );
            continue;
        }
        for child_id in &child_ids {
            let assoc = Association::with_weight(*source_id, *child_id, *weight);
            brain.substrate.insert_association(assoc.clone());
            if let Err(e) = brain.storage_save_association(&assoc) {
                report.errors.push(format!(
                    "Failed to save association {}→{}: {}",
                    source_id, child_id, e
                ));
            }
        }
    }

    report.total_children_created += child_ids.len();

    // Delete the parent
    brain.delete(parent_id)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::storage::EngramStorage;

    /// Helper: create a Brain backed by in-memory SQLite
    fn brain_with_sqlite() -> Brain {
        let storage = EngramStorage::in_memory().unwrap();
        Brain::new(storage).unwrap()
    }

    // ==================
    // SCORING TESTS
    // ==================

    #[test]
    fn score_atomic_memory() {
        let score = score_compound("Brandon prefers beer over coffee.");
        assert!(score.total < DECOMPOSE_THRESHOLD);
        assert_eq!(score.sentence_count, 1);
        assert_eq!(score.numbered_items, 0);
    }

    #[test]
    fn score_two_sentence_memory() {
        let score = score_compound("Rust is fast. It has a borrow checker.");
        assert!(score.total < DECOMPOSE_THRESHOLD);
        assert_eq!(score.sentence_count, 2);
    }

    #[test]
    fn score_numbered_items() {
        let content = "Code review: (1) null pointer in auth (2) missing error handling (3) unused import";
        let score = score_compound(content);
        assert_eq!(score.numbered_items, 3);
        assert!(score.total >= DECOMPOSE_THRESHOLD, "score={}", score.total);
    }

    #[test]
    fn score_many_sentences() {
        let content = "First fact. Second fact. Third fact. Fourth fact. Fifth fact.";
        let score = score_compound(content);
        assert_eq!(score.sentence_count, 5);
        // sentences above 2 = 3 * 1.0 = 3.0
        assert!(score.total >= DECOMPOSE_THRESHOLD);
    }

    #[test]
    fn score_long_with_semicolons() {
        let content = "A".repeat(500) + "; fact two; fact three";
        let score = score_compound(&content);
        assert_eq!(score.semicolons, 2);
        assert!(score.length > 400);
    }

    #[test]
    fn score_connectors() {
        let content = "Rust is fast. It also has great tooling. Additionally it has a strong type system. Furthermore the community is welcoming.";
        let score = score_compound(content);
        assert!(score.connectors >= 3, "connectors={}", score.connectors);
    }

    #[test]
    fn score_line_breaks() {
        let content = "List of things:\nFirst item\nSecond item\nThird item";
        let score = score_compound(content);
        assert_eq!(score.line_breaks, 3);
    }

    #[test]
    fn score_real_compound_memory() {
        // From the analysis report - worst offender pattern
        let content = "Code review findings: (1) null pointer in auth.rs (2) missing error handling in api.rs (3) unused import in lib.rs (4) dead code in utils.rs (5) naming inconsistency";
        let score = score_compound(content);
        assert!(score.total >= DECOMPOSE_THRESHOLD);
        assert_eq!(score.numbered_items, 5);
    }

    // ==================
    // SPLITTING TESTS
    // ==================

    #[test]
    fn split_numbered_parenthesized() {
        let content = "Code review findings: (1) null pointer in auth.rs (2) missing error handling in api.rs (3) unused import in lib.rs";
        let score = score_compound(content);
        let children = split_content(content, &score);

        assert_eq!(children.len(), 3);
        assert!(children[0].contains("null pointer"));
        assert!(children[1].contains("missing error handling"));
        assert!(children[2].contains("unused import"));
        // Each should have prefix
        for child in &children {
            assert!(child.contains("Code review findings"), "child='{}' missing prefix", child);
        }
    }

    #[test]
    fn split_numbered_bare_paren() {
        let content = "Issues found: 1) memory leak 2) race condition 3) deadlock";
        let score = score_compound(content);
        let children = split_content(content, &score);

        assert_eq!(children.len(), 3, "children={:?}", children);
        assert!(children[0].contains("memory leak"));
        assert!(children[1].contains("race condition"));
        assert!(children[2].contains("deadlock"));
    }

    #[test]
    fn split_numbered_dot_pattern() {
        let content = "Steps: 1. install rust 2. create project 3. run tests";
        let score = score_compound(content);
        let children = split_content(content, &score);

        assert_eq!(children.len(), 3, "children={:?}", children);
        assert!(children[0].contains("install rust"));
    }

    #[test]
    fn split_sentences_four_plus() {
        let content = "First fact here. Second fact here. Third fact here. Fourth fact here.";
        let score = score_compound(content);
        let children = split_content(content, &score);

        assert!(children.len() >= 2, "children={:?}", children);
    }

    #[test]
    fn split_sentences_with_topic_header() {
        let content = "Project setup: First you clone the repo. Then you install deps. Next run the build. Finally run tests.";
        let score = score_compound(content);
        let children = split_content(content, &score);

        assert!(children.len() >= 2, "children={:?}", children);
        // Children should include the prefix
        for child in &children {
            assert!(child.starts_with("Project setup:"), "child='{}' missing prefix", child);
        }
    }

    #[test]
    fn split_returns_empty_for_atomic() {
        let content = "Simple atomic memory.";
        let score = score_compound(content);
        let children = split_content(content, &score);
        assert!(children.is_empty());
    }

    // ==================
    // INTEGRATION TESTS
    // ==================

    #[test]
    fn decompose_skips_atomic_memories() {
        let mut brain = brain_with_sqlite();

        brain.create("Simple atomic memory.").unwrap();
        brain.create("Another simple one.").unwrap();

        let report = decompose_compound_memories(&mut brain).unwrap();

        assert_eq!(report.total_scanned, 2);
        assert_eq!(report.total_decomposed, 0);
        assert_eq!(report.total_children_created, 0);
    }

    #[test]
    fn decompose_splits_compound_numbered() {
        let mut brain = brain_with_sqlite();

        let parent_id = brain
            .create("Findings: (1) null pointer in auth (2) missing error handling (3) unused import")
            .unwrap();

        // Give it an embedding so we can verify it's replaced
        brain.set_embedding(&parent_id, &[1.0, 0.0, 0.0]).unwrap();

        let report = decompose_compound_memories(&mut brain).unwrap();

        assert_eq!(report.total_decomposed, 1);
        assert_eq!(report.total_children_created, 3);

        // Parent should be deleted
        assert!(brain.get(&parent_id).is_none(), "Parent should be deleted");

        // Should now have 3 memories
        let all: Vec<_> = brain.all_engrams().collect();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn decompose_transfers_outbound_associations() {
        let mut brain = brain_with_sqlite();

        let parent_id = brain
            .create("Issues: (1) null pointer (2) race condition (3) deadlock")
            .unwrap();
        let target = brain.create("Related project note").unwrap();

        // Create outbound association from parent
        brain.associate(parent_id, target, 0.7).unwrap();

        let report = decompose_compound_memories(&mut brain).unwrap();

        assert_eq!(report.total_decomposed, 1);

        // Each child should now have an association to the target
        let children: Vec<_> = brain.all_engrams().filter(|e| e.id != target).collect();
        assert_eq!(children.len(), 3);

        for child in &children {
            let assocs = brain.associations_from(&child.id);
            assert!(
                assocs.is_some() && assocs.unwrap().iter().any(|a| a.to == target),
                "Child {} should have association to target",
                child.id
            );
        }
    }

    #[test]
    fn decompose_transfers_inbound_associations() {
        let mut brain = brain_with_sqlite();

        let parent_id = brain
            .create("Issues: (1) null pointer (2) race condition (3) deadlock")
            .unwrap();
        let source = brain.create("Source memory").unwrap();

        // Create inbound association to parent
        brain.associate(source, parent_id, 0.6).unwrap();

        let report = decompose_compound_memories(&mut brain).unwrap();

        assert_eq!(report.total_decomposed, 1);

        // Source should now point to all children
        let assocs = brain.associations_from(&source).unwrap();
        // Original association to parent is gone (parent deleted), but new ones to children exist
        let child_ids: Vec<EngramId> = brain
            .all_engrams()
            .filter(|e| e.id != source)
            .map(|e| e.id)
            .collect();

        for child_id in &child_ids {
            assert!(
                assocs.iter().any(|a| a.to == *child_id),
                "Source should have association to child {}",
                child_id
            );
        }
    }

    #[test]
    fn decompose_skips_procedural_associations() {
        let mut brain = brain_with_sqlite();

        let parent_id = brain
            .create("Steps: (1) clone repo (2) install deps (3) run tests")
            .unwrap();
        let step_target = brain.create("Deployment guide").unwrap();

        // Create a procedural (ordinal) association
        brain
            .associate_with_ordinal(parent_id, step_target, 0.9, Some(1))
            .unwrap();

        let report = decompose_compound_memories(&mut brain).unwrap();

        assert_eq!(report.total_decomposed, 1);
        assert!(
            report.skipped_procedural >= 1,
            "Should skip procedural: {}",
            report.skipped_procedural
        );

        // Children should NOT have the ordinal association
        let children: Vec<_> = brain
            .all_engrams()
            .filter(|e| e.id != step_target)
            .collect();
        for child in &children {
            if let Some(assocs) = brain.associations_from(&child.id) {
                for assoc in assocs {
                    assert!(
                        assoc.ordinal.is_none(),
                        "Child should not have ordinal associations"
                    );
                }
            }
        }
    }

    #[test]
    fn decompose_children_have_reset_co_activation() {
        let mut brain = brain_with_sqlite();

        let parent_id = brain
            .create("Findings: (1) issue A (2) issue B (3) issue C")
            .unwrap();
        let target = brain.create("Target memory").unwrap();
        brain.associate(parent_id, target, 0.5).unwrap();

        decompose_compound_memories(&mut brain).unwrap();

        // Check co_activation_count on new associations
        let children: Vec<_> = brain.all_engrams().filter(|e| e.id != target).collect();
        for child in &children {
            if let Some(assocs) = brain.associations_from(&child.id) {
                for assoc in assocs {
                    assert_eq!(
                        assoc.co_activation_count, 0,
                        "New associations should have co_activation_count=0"
                    );
                }
            }
        }
    }

    #[test]
    fn decompose_report_counts() {
        let mut brain = brain_with_sqlite();

        // 2 atomic, 2 compound
        brain.create("Atomic memory one.").unwrap();
        brain.create("Atomic memory two.").unwrap();
        brain
            .create("Compound: (1) fact A (2) fact B (3) fact C")
            .unwrap();
        brain
            .create("Another: (1) item X (2) item Y (3) item Z")
            .unwrap();

        let report = decompose_compound_memories(&mut brain).unwrap();

        assert_eq!(report.total_scanned, 4);
        assert_eq!(report.total_decomposed, 2);
        assert_eq!(report.total_children_created, 6);
    }

    #[test]
    fn decompose_empty_brain() {
        let mut brain = brain_with_sqlite();

        let report = decompose_compound_memories(&mut brain).unwrap();

        assert_eq!(report.total_scanned, 0);
        assert_eq!(report.total_decomposed, 0);
    }

    // ==================
    // SENTENCE COUNTING TESTS
    // ==================

    #[test]
    fn count_sentences_basic() {
        assert_eq!(count_sentences("Hello world."), 1);
        assert_eq!(count_sentences("One. Two. Three."), 3);
        assert_eq!(count_sentences("Question? Answer!"), 2);
    }

    #[test]
    fn count_sentences_with_decimals() {
        // "1.5" should not be counted as a sentence boundary
        assert_eq!(count_sentences("The value is 1.5 and that is fine."), 1);
    }

    #[test]
    fn count_sentences_empty() {
        assert_eq!(count_sentences(""), 0);
        assert_eq!(count_sentences("   "), 0); // whitespace-only has no real content
    }

    #[test]
    fn count_numbered_paren_pattern() {
        assert_eq!(
            count_numbered_items("(1) first (2) second (3) third"),
            3
        );
    }

    #[test]
    fn count_numbered_bare_paren() {
        assert_eq!(count_numbered_items("1) first 2) second"), 2);
    }

    #[test]
    fn count_numbered_dot_pattern() {
        assert_eq!(count_numbered_items("1. first 2. second 3. third"), 3);
    }

    #[test]
    fn count_numbered_avoids_decimals() {
        // "3.14" should not match
        assert_eq!(count_numbered_items("Pi is 3.14 approximately"), 0);
    }

    // ==================
    // FIND PATTERN TESTS
    // ==================

    #[test]
    fn decompose_children_inherit_parent_properties() {
        use super::super::engram::MemoryState;

        let mut brain = brain_with_sqlite();

        let parent_id = brain
            .create("Issues: (1) null pointer (2) race condition (3) deadlock")
            .unwrap();

        // Simulate a decayed, well-accessed parent
        if let Some(parent) = brain.substrate.get_mut(&parent_id) {
            parent.energy = 0.25;
            parent.state = MemoryState::Dormant;
            parent.access_count = 42;
            parent.last_accessed = 1_700_000_000;
        }
        // Persist parent changes so the collection picks them up
        if let Some(parent) = brain.substrate.get(&parent_id).cloned() {
            brain.storage_save_engram(&parent).unwrap();
        }

        let report = decompose_compound_memories(&mut brain).unwrap();

        assert_eq!(report.total_decomposed, 1);
        assert_eq!(report.total_children_created, 3);

        // Verify all children inherited parent properties
        let children: Vec<_> = brain.all_engrams().collect();
        assert_eq!(children.len(), 3);

        for child in &children {
            assert!(
                (child.energy - 0.25).abs() < 0.001,
                "Child energy should be 0.25, got {}",
                child.energy
            );
            assert_eq!(
                child.state,
                MemoryState::Dormant,
                "Child state should be Dormant, got {:?}",
                child.state
            );
            assert_eq!(
                child.access_count, 42,
                "Child access_count should be 42, got {}",
                child.access_count
            );
            assert_eq!(
                child.last_accessed, 1_700_000_000,
                "Child last_accessed should be 1700000000, got {}",
                child.last_accessed
            );
        }
    }

    #[test]
    fn find_pattern_respects_boundary() {
        // "1)" should only match at word boundary, not inside "21)"
        let text = "item 21) and then 1) real match";
        let pos = find_pattern_position(text, "1)");
        assert!(pos.is_some());
        // Should find the "1)" after "then ", not "21)"
        let found = &text[pos.unwrap()..pos.unwrap() + 2];
        assert_eq!(found, "1)");
        // The position should be at the standalone "1)"
        assert!(pos.unwrap() > 15, "Should find the standalone 1), not inside 21)");
    }
}
