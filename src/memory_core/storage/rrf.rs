//! Reciprocal Rank Fusion (RRF) for merging multiple retrieval result lists.
//!
//! RRF score for a document = sum over all lists of: 1 / (k + rank_in_list)
//! where k is a constant (default 60, from Cormack et al.).
//!
//! Documents appearing in multiple lists get higher scores, but documents
//! appearing in only one list are still included.
//!
//! Output scores are normalized to 0..1 for downstream blending with other
//! signals (for example semantic similarity and energy). Without this,
//! raw RRF magnitudes (~0.01-0.03 with k=60) get swamped by other terms.

use super::SimilarityResult;
use crate::memory_core::MemoryId;
use std::collections::HashMap;

/// Default k constant from the original RRF paper (Cormack, Clarke, Buettcher 2009).
pub const DEFAULT_K: f64 = 60.0;

/// Merge results from multiple retrieval paths using Reciprocal Rank Fusion.
///
/// For each document, the RRF score = sum over all lists of: 1 / (k + rank + 1)
/// where rank is 0-based position in each list.
///
/// Documents appearing in only one list still receive a score.
/// Results are returned sorted by RRF score descending.
///
/// Returned `SimilarityResult.score` values are normalized to 0..1, where:
/// - 1.0 = theoretical best case (document ranked #1 in every list)
/// - Lower values preserve relative RRF ordering.
pub fn reciprocal_rank_fusion(
    result_lists: &[&[SimilarityResult]],
    k: f64,
) -> Vec<SimilarityResult> {
    // Map from memory ID to (accumulated RRF score, best SimilarityResult)
    let mut scores: HashMap<MemoryId, (f64, SimilarityResult)> = HashMap::new();

    for list in result_lists {
        for (rank, result) in list.iter().enumerate() {
            let rrf_score = 1.0 / (k + rank as f64 + 1.0);

            scores
                .entry(result.id)
                .and_modify(|(acc_score, _)| {
                    *acc_score += rrf_score;
                })
                .or_insert_with(|| (rrf_score, result.clone()));
        }
    }

    // Collect and sort by RRF score descending
    let mut merged: Vec<(f64, SimilarityResult)> = scores.into_values().collect();
    merged.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Normalize to 0..1 using the theoretical max:
    // max_rrf = num_lists * (1 / (k + 1)) when doc is rank 0 in every list.
    let max_rrf = if result_lists.is_empty() {
        1.0
    } else {
        (result_lists.len() as f64) / (k + 1.0)
    };

    // Set the score field to normalized RRF score for downstream use
    merged
        .into_iter()
        .map(|(rrf_score, mut result)| {
            let normalized = if max_rrf > 0.0 {
                (rrf_score / max_rrf).clamp(0.0, 1.0)
            } else {
                0.0
            };
            result.score = normalized as f32;
            result
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(id_byte: u8, score: f32, content: &str) -> SimilarityResult {
        SimilarityResult {
            id: uuid::Uuid::from_bytes([id_byte, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            score,
            content: content.to_string(),
        }
    }

    #[test]
    fn rrf_merges_disjoint_lists() {
        let list_a = vec![make_result(1, 0.9, "A1"), make_result(2, 0.8, "A2")];
        let list_b = vec![make_result(3, 0.95, "B1"), make_result(4, 0.85, "B2")];

        let merged = reciprocal_rank_fusion(&[&list_a, &list_b], DEFAULT_K);

        // All 4 items should appear
        assert_eq!(merged.len(), 4);

        // Items at rank 0 in their lists should have the same RRF score
        // since they each appear in exactly one list at rank 0
        let ids: Vec<u8> = merged.iter().map(|r| r.id.as_bytes()[0]).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
        assert!(ids.contains(&4));
    }

    #[test]
    fn rrf_merges_overlapping_lists() {
        // Item 1 appears in both lists — should get highest score
        let list_a = vec![make_result(1, 0.9, "Shared"), make_result(2, 0.8, "A only")];
        let list_b = vec![
            make_result(1, 0.95, "Shared"),
            make_result(3, 0.85, "B only"),
        ];

        let merged = reciprocal_rank_fusion(&[&list_a, &list_b], DEFAULT_K);

        assert_eq!(merged.len(), 3);

        // The shared item (id=1) should be first because it appears in both lists
        assert_eq!(merged[0].id.as_bytes()[0], 1);

        // Its RRF score should be higher than any single-list item
        assert!(merged[0].score > merged[1].score);
    }

    #[test]
    fn rrf_handles_empty_lists() {
        let empty: Vec<SimilarityResult> = Vec::new();
        let merged = reciprocal_rank_fusion(&[&empty], DEFAULT_K);
        assert!(merged.is_empty());

        let merged2 = reciprocal_rank_fusion(&[], DEFAULT_K);
        assert!(merged2.is_empty());
    }

    #[test]
    fn rrf_single_list() {
        let list = vec![
            make_result(1, 0.9, "First"),
            make_result(2, 0.8, "Second"),
            make_result(3, 0.7, "Third"),
        ];

        let merged = reciprocal_rank_fusion(&[&list], DEFAULT_K);

        // Same order preserved
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].id.as_bytes()[0], 1);
        assert_eq!(merged[1].id.as_bytes()[0], 2);
        assert_eq!(merged[2].id.as_bytes()[0], 3);

        // Scores should be decreasing
        assert!(merged[0].score > merged[1].score);
        assert!(merged[1].score > merged[2].score);
    }

    #[test]
    fn rrf_overlapping_item_gets_double_score() {
        // In normalized mode, item at rank 0 in both lists is the theoretical max: 1.0
        let list_a = vec![make_result(1, 0.9, "Shared")];
        let list_b = vec![make_result(1, 0.95, "Shared")];

        let merged = reciprocal_rank_fusion(&[&list_a, &list_b], DEFAULT_K);

        assert_eq!(merged.len(), 1);
        assert!(
            (merged[0].score - 1.0).abs() < 1e-6,
            "Expected normalized score 1.0, got {}",
            merged[0].score
        );
    }

    #[test]
    fn rrf_scores_are_normalized_to_unit_interval() {
        let list_a = vec![
            make_result(1, 0.9, "A1"),
            make_result(2, 0.8, "A2"),
            make_result(3, 0.7, "A3"),
        ];
        let list_b = vec![
            make_result(1, 0.95, "Shared top"),
            make_result(4, 0.85, "B2"),
        ];

        let merged = reciprocal_rank_fusion(&[&list_a, &list_b], DEFAULT_K);
        assert!(!merged.is_empty());

        for result in &merged {
            assert!(
                (0.0..=1.0).contains(&result.score),
                "score out of range: {}",
                result.score
            );
        }
    }
}
