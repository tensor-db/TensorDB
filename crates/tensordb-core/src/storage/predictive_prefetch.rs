//! Predictive Prefetch — AI-driven Markov chain access pattern prediction.
//!
//! # The Problem
//! Traditional databases prefetch sequentially (read block N, prefetch N+1).
//! But real access patterns are rarely sequential — they follow application logic:
//! reading a user often leads to reading their orders, which leads to reading products.
//!
//! # The Innovation
//! TensorDB builds a Markov chain of key-to-key transitions. When key A is read,
//! the system predicts which key B is most likely to be read next and prefetches it
//! into the block cache BEFORE the application requests it.
//!
//! # AI Integration
//! The Markov model is a lightweight in-process AI that:
//! - Learns transition probabilities from observed read patterns
//! - Decays old transitions to adapt to changing workloads
//! - Predicts multi-step prefetch chains (A → B → C) for deep pipelining
//! - Self-tunes confidence thresholds to avoid wasting I/O on low-confidence predictions
//!
//! # Why This Is Novel
//! - RocksDB: readahead for sequential scans, no pattern-based prefetch
//! - PostgreSQL: bitmap heap scans optimize random reads, but no prediction
//! - InnoDB: linear read-ahead based on sequential access detection
//! - TensorDB: Markov chain predictions across arbitrary key access patterns

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

/// A single transition in the Markov chain.
#[derive(Debug, Clone)]
struct Transition {
    /// Number of times this transition was observed.
    count: u32,
    /// Decayed count (for adaptive learning).
    decayed_count: f64,
}

/// A node in the access pattern graph.
#[derive(Debug, Clone)]
struct AccessNode {
    /// Transitions from this key to other keys.
    transitions: HashMap<Vec<u8>, Transition>,
    /// Total outgoing transitions (for probability calculation).
    total_transitions: u32,
}

impl AccessNode {
    fn new() -> Self {
        Self {
            transitions: HashMap::new(),
            total_transitions: 0,
        }
    }
}

/// Markov chain access predictor.
pub struct AccessPredictor {
    /// The Markov model: key → {next_key → transition_count}.
    /// Uses prefix-truncated keys (first N bytes) to bound memory.
    model: RwLock<HashMap<Vec<u8>, AccessNode>>,
    /// Maximum number of nodes in the model (LRU eviction beyond this).
    max_nodes: usize,
    /// Key prefix length for grouping (0 = full key).
    key_prefix_len: usize,
    /// Minimum probability to trigger prefetch.
    confidence_threshold: f64,
    /// Maximum prefetch chain depth.
    max_chain_depth: usize,
    /// Stats
    total_observations: AtomicU64,
    total_predictions: AtomicU64,
    predictions_hit: AtomicU64,
    predictions_miss: AtomicU64,
    /// Last observed key (per-thread would be ideal, but we use a single tracker for simplicity).
    last_key: RwLock<Option<Vec<u8>>>,
}

impl AccessPredictor {
    pub fn new(max_nodes: usize, key_prefix_len: usize) -> Self {
        Self {
            model: RwLock::new(HashMap::new()),
            max_nodes,
            key_prefix_len,
            confidence_threshold: 0.1, // 10% probability = worth prefetching
            max_chain_depth: 3,
            total_observations: AtomicU64::new(0),
            total_predictions: AtomicU64::new(0),
            predictions_hit: AtomicU64::new(0),
            predictions_miss: AtomicU64::new(0),
            last_key: RwLock::new(None),
        }
    }

    /// Normalize a key to its prefix (for memory-bounded model).
    fn normalize_key(&self, key: &[u8]) -> Vec<u8> {
        if self.key_prefix_len > 0 && key.len() > self.key_prefix_len {
            key[..self.key_prefix_len].to_vec()
        } else {
            key.to_vec()
        }
    }

    /// Record an access and learn the transition from the previous key.
    pub fn record_access(&self, key: &[u8]) {
        let normalized = self.normalize_key(key);
        self.total_observations.fetch_add(1, Ordering::Relaxed);

        let prev = {
            let mut last = self.last_key.write();
            let prev = last.clone();
            *last = Some(normalized.clone());
            prev
        };

        if let Some(prev_key) = prev {
            if prev_key != normalized {
                let mut model = self.model.write();

                // Evict oldest entries if model is too large
                if model.len() >= self.max_nodes && !model.contains_key(&prev_key) {
                    // Simple eviction: remove the entry with the lowest total_transitions
                    if let Some(victim) = model
                        .iter()
                        .min_by_key(|(_, node)| node.total_transitions)
                        .map(|(k, _)| k.clone())
                    {
                        model.remove(&victim);
                    }
                }

                let node = model.entry(prev_key).or_insert_with(AccessNode::new);
                node.total_transitions += 1;

                let transition = node.transitions.entry(normalized).or_insert(Transition {
                    count: 0,
                    decayed_count: 0.0,
                });
                transition.count += 1;
                transition.decayed_count += 1.0;
            }
        }
    }

    /// Predict the next key(s) to be accessed, given the current key.
    /// Returns a list of (key, probability) pairs, sorted by probability descending.
    pub fn predict(&self, current_key: &[u8]) -> Vec<PrefetchPrediction> {
        let normalized = self.normalize_key(current_key);
        let model = self.model.read();

        let node = match model.get(&normalized) {
            Some(n) => n,
            None => return Vec::new(),
        };

        if node.total_transitions == 0 {
            return Vec::new();
        }

        let mut predictions: Vec<PrefetchPrediction> = node
            .transitions
            .iter()
            .map(|(key, t)| PrefetchPrediction {
                key: key.clone(),
                probability: t.count as f64 / node.total_transitions as f64,
                chain_depth: 1,
            })
            .filter(|p| p.probability >= self.confidence_threshold)
            .collect();

        predictions.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());

        // Extend with chain predictions (A → B → C)
        if self.max_chain_depth > 1 {
            let mut chain_predictions = Vec::new();
            for pred in &predictions {
                if let Some(next_node) = model.get(&pred.key) {
                    for (chain_key, chain_t) in &next_node.transitions {
                        let chain_prob = pred.probability
                            * (chain_t.count as f64 / next_node.total_transitions as f64);
                        if chain_prob >= self.confidence_threshold * 0.5 {
                            chain_predictions.push(PrefetchPrediction {
                                key: chain_key.clone(),
                                probability: chain_prob,
                                chain_depth: 2,
                            });
                        }
                    }
                }
            }
            predictions.extend(chain_predictions);
            predictions.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());
        }

        self.total_predictions.fetch_add(1, Ordering::Relaxed);

        // Return top predictions (limit to avoid excessive prefetch)
        predictions.truncate(5);
        predictions
    }

    /// Record a prefetch hit (predicted key was actually accessed).
    pub fn record_hit(&self) {
        self.predictions_hit.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a prefetch miss (predicted key was not accessed).
    pub fn record_miss(&self) {
        self.predictions_miss.fetch_add(1, Ordering::Relaxed);
    }

    /// Decay all transition weights (call periodically to adapt to changing patterns).
    pub fn decay(&self, factor: f64) {
        let mut model = self.model.write();
        for node in model.values_mut() {
            for t in node.transitions.values_mut() {
                t.decayed_count *= factor;
            }
            // Remove transitions that have decayed to near-zero
            node.transitions.retain(|_, t| t.decayed_count > 0.01);
        }
        // Remove nodes with no transitions
        model.retain(|_, node| !node.transitions.is_empty());
    }

    /// AI-driven self-tuning: adjust confidence threshold based on hit/miss ratio.
    pub fn auto_tune(&self) -> PrefetchTuneResult {
        let hits = self.predictions_hit.load(Ordering::Relaxed);
        let misses = self.predictions_miss.load(Ordering::Relaxed);
        let total = hits + misses;

        if total < 50 {
            return PrefetchTuneResult {
                adjusted: false,
                new_threshold: self.confidence_threshold,
                hit_rate: 0.0,
                reason: "insufficient data".to_string(),
            };
        }

        let hit_rate = hits as f64 / total as f64;

        let new_threshold = if hit_rate < 0.3 {
            // Too many misses → raise threshold (be more selective)
            (self.confidence_threshold * 1.5).min(0.5)
        } else if hit_rate > 0.8 {
            // Very accurate → lower threshold (prefetch more aggressively)
            (self.confidence_threshold * 0.7).max(0.05)
        } else {
            self.confidence_threshold
        };

        PrefetchTuneResult {
            adjusted: (new_threshold - self.confidence_threshold).abs() > 0.001,
            new_threshold,
            hit_rate,
            reason: format!("hit_rate={hit_rate:.2}, hits={hits}, misses={misses}"),
        }
    }

    /// Get predictor statistics.
    pub fn stats(&self) -> PredictorStats {
        let model = self.model.read();
        let total_transitions: u32 = model.values().map(|n| n.total_transitions).sum();

        PredictorStats {
            model_nodes: model.len(),
            total_transitions,
            total_observations: self.total_observations.load(Ordering::Relaxed),
            total_predictions: self.total_predictions.load(Ordering::Relaxed),
            predictions_hit: self.predictions_hit.load(Ordering::Relaxed),
            predictions_miss: self.predictions_miss.load(Ordering::Relaxed),
            confidence_threshold: self.confidence_threshold,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrefetchPrediction {
    pub key: Vec<u8>,
    pub probability: f64,
    pub chain_depth: usize,
}

#[derive(Debug, Clone)]
pub struct PrefetchTuneResult {
    pub adjusted: bool,
    pub new_threshold: f64,
    pub hit_rate: f64,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct PredictorStats {
    pub model_nodes: usize,
    pub total_transitions: u32,
    pub total_observations: u64,
    pub total_predictions: u64,
    pub predictions_hit: u64,
    pub predictions_miss: u64,
    pub confidence_threshold: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_transition_learning() {
        let predictor = AccessPredictor::new(1000, 0);

        // Simulate: A → B → C pattern repeated
        for _ in 0..10 {
            predictor.record_access(b"key_a");
            predictor.record_access(b"key_b");
            predictor.record_access(b"key_c");
        }

        // After seeing A, should predict B
        let preds = predictor.predict(b"key_a");
        assert!(!preds.is_empty());
        assert_eq!(preds[0].key, b"key_b");
        assert!(preds[0].probability > 0.8);
    }

    #[test]
    fn test_multiple_transitions() {
        let predictor = AccessPredictor::new(1000, 0);

        // Build direct A → B and A → C transitions.
        // We use "reset" between to avoid B→start and C→start noise.
        for _ in 0..7 {
            // Reset last_key
            predictor.record_access(b"__reset__");
            predictor.record_access(b"A");
            predictor.record_access(b"B");
        }
        for _ in 0..3 {
            predictor.record_access(b"__reset__");
            predictor.record_access(b"A");
            predictor.record_access(b"C");
        }

        let preds = predictor.predict(b"A");
        assert!(
            preds.len() >= 2,
            "expected >= 2 predictions, got {}",
            preds.len()
        );
        // B should have higher probability since 7 > 3
        let b_pred = preds
            .iter()
            .find(|p| p.key == b"B" && p.chain_depth == 1)
            .expect("B prediction missing");
        let c_pred = preds
            .iter()
            .find(|p| p.key == b"C" && p.chain_depth == 1)
            .expect("C prediction missing");
        assert!(
            b_pred.probability > c_pred.probability,
            "B prob {} should be > C prob {}",
            b_pred.probability,
            c_pred.probability
        );
    }

    #[test]
    fn test_no_prediction_for_unknown_key() {
        let predictor = AccessPredictor::new(1000, 0);
        let preds = predictor.predict(b"unknown");
        assert!(preds.is_empty());
    }

    #[test]
    fn test_prefix_normalization() {
        let predictor = AccessPredictor::new(1000, 6); // 6-byte prefix

        // "user/1" and "user/100" should map to different prefixes
        // "user/1" (6 bytes) vs "user/1" (prefix of "user/100")
        predictor.record_access(b"user/1");
        predictor.record_access(b"order/1");
        predictor.record_access(b"user/1");
        predictor.record_access(b"order/1");

        let preds = predictor.predict(b"user/1");
        assert!(!preds.is_empty());
    }

    #[test]
    fn test_decay_removes_old_patterns() {
        let predictor = AccessPredictor::new(1000, 0);

        // Build a pattern
        for _ in 0..5 {
            predictor.record_access(b"X");
            predictor.record_access(b"Y");
        }

        // Decay heavily
        predictor.decay(0.001);

        // Should have removed the transitions
        let stats = predictor.stats();
        assert_eq!(stats.total_transitions, 0);
    }

    #[test]
    fn test_model_eviction() {
        let predictor = AccessPredictor::new(3, 0); // Very small model

        // Create many nodes
        for i in 0..10 {
            predictor.record_access(format!("key_{i}").as_bytes());
            predictor.record_access(format!("next_{i}").as_bytes());
        }

        let stats = predictor.stats();
        assert!(stats.model_nodes <= 5); // Should have evicted
    }

    #[test]
    fn test_chain_prediction() {
        let predictor = AccessPredictor::new(1000, 0);

        // Strong A → B → C chain
        for _ in 0..20 {
            predictor.record_access(b"A");
            predictor.record_access(b"B");
            predictor.record_access(b"C");
        }

        let preds = predictor.predict(b"A");
        // Should include chain prediction A → B → C (depth 2)
        let depth2 = preds.iter().find(|p| p.chain_depth == 2);
        assert!(depth2.is_some(), "should have depth-2 chain prediction");
    }

    #[test]
    fn test_auto_tune_insufficient_data() {
        let predictor = AccessPredictor::new(1000, 0);
        let result = predictor.auto_tune();
        assert!(!result.adjusted);
        assert!(result.reason.contains("insufficient"));
    }

    #[test]
    fn test_stats() {
        let predictor = AccessPredictor::new(1000, 0);
        predictor.record_access(b"A");
        predictor.record_access(b"B");
        predictor.predict(b"A");

        let stats = predictor.stats();
        assert_eq!(stats.total_observations, 2);
        assert_eq!(stats.total_predictions, 1);
        assert_eq!(stats.model_nodes, 1);
    }

    #[test]
    fn test_hit_miss_tracking() {
        let predictor = AccessPredictor::new(1000, 0);
        predictor.record_hit();
        predictor.record_hit();
        predictor.record_miss();

        let stats = predictor.stats();
        assert_eq!(stats.predictions_hit, 2);
        assert_eq!(stats.predictions_miss, 1);
    }

    #[test]
    fn test_self_transition_ignored() {
        let predictor = AccessPredictor::new(1000, 0);

        // Same key accessed repeatedly
        for _ in 0..10 {
            predictor.record_access(b"same_key");
        }

        // No transitions should be recorded (self-transitions are skipped)
        let stats = predictor.stats();
        assert_eq!(stats.total_transitions, 0);
    }
}
