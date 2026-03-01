//! Hybrid search: combines vector similarity with BM25 full-text scores.
//! Supports Reciprocal Rank Fusion (RRF) and weighted linear combination.

use std::collections::HashMap;

// ── Scoring strategies ──────────────────────────────────────────────────────

/// Result from a single ranking source (vector search or FTS).
#[derive(Debug, Clone)]
pub struct RankedResult {
    pub pk: String,
    pub score: f32,
}

/// Merged result with hybrid score.
#[derive(Debug, Clone)]
pub struct HybridResult {
    pub pk: String,
    pub score: f32,
    pub vector_score: Option<f32>,
    pub text_score: Option<f32>,
}

/// Reciprocal Rank Fusion: combines two ranked lists.
///
/// RRF score = 1/(k + rank_vector) + 1/(k + rank_text)
/// where k is a constant (default 60).
pub fn reciprocal_rank_fusion(
    vector_results: &[RankedResult],
    text_results: &[RankedResult],
    k: f32,
    limit: usize,
) -> Vec<HybridResult> {
    let mut scores: HashMap<String, HybridResult> = HashMap::new();

    // Score from vector results (rank 1-based)
    for (rank, result) in vector_results.iter().enumerate() {
        let rrf_score = 1.0 / (k + (rank + 1) as f32);
        let entry = scores.entry(result.pk.clone()).or_insert(HybridResult {
            pk: result.pk.clone(),
            score: 0.0,
            vector_score: None,
            text_score: None,
        });
        entry.score += rrf_score;
        entry.vector_score = Some(result.score);
    }

    // Score from text results
    for (rank, result) in text_results.iter().enumerate() {
        let rrf_score = 1.0 / (k + (rank + 1) as f32);
        let entry = scores.entry(result.pk.clone()).or_insert(HybridResult {
            pk: result.pk.clone(),
            score: 0.0,
            vector_score: None,
            text_score: None,
        });
        entry.score += rrf_score;
        entry.text_score = Some(result.score);
    }

    let mut merged: Vec<HybridResult> = scores.into_values().collect();
    merged.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    merged.truncate(limit);
    merged
}

/// Weighted linear combination of vector and text scores.
///
/// hybrid_score = vector_weight * normalized_vector_score + text_weight * normalized_text_score
pub fn weighted_combination(
    vector_results: &[RankedResult],
    text_results: &[RankedResult],
    vector_weight: f32,
    text_weight: f32,
    limit: usize,
) -> Vec<HybridResult> {
    // Normalize scores to [0, 1]
    let v_max = vector_results
        .iter()
        .map(|r| r.score)
        .fold(f32::NEG_INFINITY, f32::max);
    let v_min = vector_results
        .iter()
        .map(|r| r.score)
        .fold(f32::INFINITY, f32::min);
    let t_max = text_results
        .iter()
        .map(|r| r.score)
        .fold(f32::NEG_INFINITY, f32::max);
    let t_min = text_results
        .iter()
        .map(|r| r.score)
        .fold(f32::INFINITY, f32::min);

    let v_range = (v_max - v_min).max(1e-10);
    let t_range = (t_max - t_min).max(1e-10);

    let mut scores: HashMap<String, HybridResult> = HashMap::new();

    for result in vector_results {
        let normalized = (result.score - v_min) / v_range;
        // For distance metrics, invert: lower distance = higher score
        let inverted = 1.0 - normalized;
        let entry = scores.entry(result.pk.clone()).or_insert(HybridResult {
            pk: result.pk.clone(),
            score: 0.0,
            vector_score: None,
            text_score: None,
        });
        entry.score += vector_weight * inverted;
        entry.vector_score = Some(result.score);
    }

    for result in text_results {
        let normalized = (result.score - t_min) / t_range;
        let entry = scores.entry(result.pk.clone()).or_insert(HybridResult {
            pk: result.pk.clone(),
            score: 0.0,
            vector_score: None,
            text_score: None,
        });
        entry.score += text_weight * normalized;
        entry.text_score = Some(result.score);
    }

    let mut merged: Vec<HybridResult> = scores.into_values().collect();
    merged.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    merged.truncate(limit);
    merged
}

/// Compute a hybrid score for a single record given vector distance and text score.
/// Used by HYBRID_SCORE() SQL function.
///
/// score = vector_weight * (1 / (1 + distance)) + text_weight * text_score
pub fn compute_hybrid_score(
    vector_distance: f64,
    text_score: f64,
    vector_weight: f64,
    text_weight: f64,
) -> f64 {
    let vector_similarity = 1.0 / (1.0 + vector_distance);
    vector_weight * vector_similarity + text_weight * text_score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_basic() {
        let vector = vec![
            RankedResult {
                pk: "a".into(),
                score: 0.1,
            },
            RankedResult {
                pk: "b".into(),
                score: 0.5,
            },
            RankedResult {
                pk: "c".into(),
                score: 0.9,
            },
        ];
        let text = vec![
            RankedResult {
                pk: "b".into(),
                score: 5.0,
            },
            RankedResult {
                pk: "d".into(),
                score: 3.0,
            },
            RankedResult {
                pk: "a".into(),
                score: 1.0,
            },
        ];

        let results = reciprocal_rank_fusion(&vector, &text, 60.0, 10);
        assert!(!results.is_empty());

        // 'a' and 'b' appear in both lists, should score higher
        let pks: Vec<&str> = results.iter().map(|r| r.pk.as_str()).collect();
        assert!(pks.contains(&"a"));
        assert!(pks.contains(&"b"));

        // 'b' is rank 1 in text, rank 2 in vector — should have high score
        let b = results.iter().find(|r| r.pk == "b").unwrap();
        assert!(b.vector_score.is_some());
        assert!(b.text_score.is_some());
    }

    #[test]
    fn test_weighted_combination() {
        let vector = vec![
            RankedResult {
                pk: "a".into(),
                score: 0.1,
            },
            RankedResult {
                pk: "b".into(),
                score: 0.9,
            },
        ];
        let text = vec![
            RankedResult {
                pk: "a".into(),
                score: 5.0,
            },
            RankedResult {
                pk: "c".into(),
                score: 2.0,
            },
        ];

        let results = weighted_combination(&vector, &text, 0.7, 0.3, 10);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_compute_hybrid_score() {
        // Close vector (low distance) + high text score
        let score = compute_hybrid_score(0.1, 5.0, 0.7, 0.3);
        assert!(score > 0.0);

        // Far vector + no text
        let score2 = compute_hybrid_score(10.0, 0.0, 0.7, 0.3);
        assert!(score2 < score);
    }
}
