//! Reciprocal Rank Fusion (RRF) for merging multiple retrieval result lists.
//!
//! RRF score for a document = sum over all lists of: 1 / (k + rank_in_list)
//! where k is a constant (default 60, from Cormack et al.).
//!
//! Documents appearing in multiple lists get higher scores, but documents
//! appearing in only one list are still included.

use std::collections::HashMap;
use super::SimilarityResult;
use crate::engram::EngramId;

/// Default k constant from the original RRF paper (Cormack, Clarke, Buettcher 2009).
pub const DEFAULT_K: f64 = 60.0;

/// Merge results from multiple retrieval paths using Reciprocal Rank Fusion.
///
/// For each document, the RRF score = sum over all lists of: 1 / (k + rank + 1)
/// where rank is 0-based position in each list.
///
/// Documents appearing in only one list still receive a score.
/// Results are returned sorted by RRF score descending.
pub fn reciprocal_rank_fusion(
    result_lists: &[&[SimilarityResult]],
    k: f64,
) -> Vec<SimilarityResult> {
    // Map from engram ID to (accumulated RRF score, best SimilarityResult)
    let mut scores: HashMap<EngramId, (f64, SimilarityResult)> = HashMap::new();

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

    // Set the score field to the RRF score for downstream use
    merged
        .into_iter()
        .map(|(rrf_score, mut result)| {
            result.score = rrf_score as f32;
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
        let list_a = vec![
            make_result(1, 0.9, "A1"),
            make_result(2, 0.8, "A2"),
        ];
        let list_b = vec![
            make_result(3, 0.95, "B1"),
            make_result(4, 0.85, "B2"),
        ];

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
        let list_a = vec![
            make_result(1, 0.9, "Shared"),
            make_result(2, 0.8, "A only"),
        ];
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
        // Verify the math: item at rank 0 in both lists should get 2/(k+1)
        let list_a = vec![make_result(1, 0.9, "Shared")];
        let list_b = vec![make_result(1, 0.95, "Shared")];

        let merged = reciprocal_rank_fusion(&[&list_a, &list_b], DEFAULT_K);

        assert_eq!(merged.len(), 1);
        let expected = 2.0 / (DEFAULT_K + 1.0);
        assert!((merged[0].score as f64 - expected).abs() < 1e-6,
            "Expected score ~{}, got {}", expected, merged[0].score);
    }
}
