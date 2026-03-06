//! Learned cost model: online linear regression trained from observed query execution.
//!
//! Augments the heuristic cost-based planner with actual observed costs.
//! Uses simple online SGD with exponential moving average for stability.
//! The model persists its weights under `__meta/learned_cost/`.

use std::collections::HashMap;
use std::sync::RwLock;

/// Features used by the learned cost model.
#[derive(Debug, Clone)]
pub struct CostFeatures {
    /// Estimated row count.
    pub row_count: f64,
    /// Whether an index is available for the query.
    pub index_available: f64,
    /// Estimated selectivity (0.0 to 1.0).
    pub selectivity: f64,
    /// Number of columns in the result.
    pub column_count: f64,
    /// Whether the query involves a join.
    pub has_join: f64,
    /// Whether the query involves aggregation.
    pub has_aggregation: f64,
}

impl CostFeatures {
    /// Convert to feature vector for the linear model.
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            1.0, // bias term
            self.row_count.ln().max(0.0),
            self.index_available,
            self.selectivity,
            self.column_count,
            self.has_join,
            self.has_aggregation,
        ]
    }
}

/// Observation from a completed query.
#[derive(Debug, Clone)]
pub struct CostObservation {
    pub features: CostFeatures,
    pub actual_time_us: f64,
    pub actual_rows: u64,
}

/// Online linear regression model for query cost estimation.
pub struct LearnedCostModel {
    /// Weight vector for linear regression.
    weights: RwLock<Vec<f64>>,
    /// Running count of observations.
    observation_count: RwLock<u64>,
    /// Learning rate for SGD.
    learning_rate: f64,
    /// EMA decay factor for weight updates.
    ema_decay: f64,
    /// Per-plan-type weight overrides.
    plan_weights: RwLock<HashMap<String, Vec<f64>>>,
}

impl LearnedCostModel {
    const NUM_FEATURES: usize = 7; // bias + 6 features

    pub fn new() -> Self {
        Self {
            weights: RwLock::new(vec![0.0; Self::NUM_FEATURES]),
            observation_count: RwLock::new(0),
            learning_rate: 0.01,
            ema_decay: 0.95,
            plan_weights: RwLock::new(HashMap::new()),
        }
    }

    /// Predict the cost (estimated execution time in microseconds) for given features.
    pub fn predict(&self, features: &CostFeatures) -> f64 {
        let x = features.to_vec();
        let w = self.weights.read().unwrap();
        dot(&w, &x).max(0.0)
    }

    /// Predict cost for a specific plan type.
    pub fn predict_for_plan(&self, plan_type: &str, features: &CostFeatures) -> f64 {
        let x = features.to_vec();
        let plans = self.plan_weights.read().unwrap();
        if let Some(w) = plans.get(plan_type) {
            dot(w, &x).max(0.0)
        } else {
            self.predict(features)
        }
    }

    /// Train the model with a new observation (online SGD step).
    pub fn train(&self, obs: &CostObservation) {
        let x = obs.features.to_vec();
        let y = obs.actual_time_us;

        // Global model update
        {
            let mut w = self.weights.write().unwrap();
            let predicted = dot(&w, &x);
            let error = y - predicted;

            // SGD update: w += lr * error * x
            for (wi, xi) in w.iter_mut().zip(x.iter()) {
                let gradient = self.learning_rate * error * xi;
                *wi = self.ema_decay * *wi + (1.0 - self.ema_decay) * (*wi + gradient);
            }
        }

        *self.observation_count.write().unwrap() += 1;
    }

    /// Train for a specific plan type.
    pub fn train_for_plan(&self, plan_type: &str, obs: &CostObservation) {
        let x = obs.features.to_vec();
        let y = obs.actual_time_us;

        let mut plans = self.plan_weights.write().unwrap();
        let w = plans
            .entry(plan_type.to_string())
            .or_insert_with(|| vec![0.0; Self::NUM_FEATURES]);

        let predicted = dot(w, &x);
        let error = y - predicted;

        for (wi, xi) in w.iter_mut().zip(x.iter()) {
            let gradient = self.learning_rate * error * xi;
            *wi = self.ema_decay * *wi + (1.0 - self.ema_decay) * (*wi + gradient);
        }

        // Also train global model
        self.train(obs);
    }

    /// Number of observations the model has been trained on.
    pub fn observation_count(&self) -> u64 {
        *self.observation_count.read().unwrap()
    }

    /// Whether the model has enough data to be useful.
    pub fn is_trained(&self) -> bool {
        self.observation_count() >= 10
    }

    /// Get the model weights for serialization.
    pub fn weights(&self) -> Vec<f64> {
        self.weights.read().unwrap().clone()
    }

    /// Load weights from a previously saved state.
    pub fn load_weights(&self, weights: Vec<f64>) {
        if weights.len() == Self::NUM_FEATURES {
            *self.weights.write().unwrap() = weights;
        }
    }

    /// Serialize model state to JSON.
    pub fn to_json(&self) -> String {
        let w = self.weights.read().unwrap();
        let plans = self.plan_weights.read().unwrap();
        let count = *self.observation_count.read().unwrap();
        serde_json::json!({
            "weights": *w,
            "observation_count": count,
            "plan_weights": *plans,
        })
        .to_string()
    }

    /// Deserialize model state from JSON.
    pub fn from_json(json: &str) -> Option<Self> {
        let val: serde_json::Value = serde_json::from_str(json).ok()?;
        let weights: Vec<f64> = serde_json::from_value(val["weights"].clone()).ok()?;
        let count: u64 = val["observation_count"].as_u64().unwrap_or(0);
        let plan_weights: HashMap<String, Vec<f64>> =
            serde_json::from_value(val["plan_weights"].clone()).unwrap_or_default();

        let model = Self::new();
        *model.weights.write().unwrap() = weights;
        *model.observation_count.write().unwrap() = count;
        *model.plan_weights.write().unwrap() = plan_weights;
        Some(model)
    }
}

impl Default for LearnedCostModel {
    fn default() -> Self {
        Self::new()
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_features(rows: f64, indexed: bool) -> CostFeatures {
        CostFeatures {
            row_count: rows,
            index_available: if indexed { 1.0 } else { 0.0 },
            selectivity: 0.1,
            column_count: 5.0,
            has_join: 0.0,
            has_aggregation: 0.0,
        }
    }

    #[test]
    fn untrained_model_predicts_zero() {
        let model = LearnedCostModel::new();
        let features = make_features(1000.0, false);
        assert!((model.predict(&features) - 0.0).abs() < 0.001);
    }

    #[test]
    fn model_learns_from_observations() {
        let model = LearnedCostModel::new();

        // Train with observations: larger tables take longer
        for i in 1..=100 {
            let rows = (i * 100) as f64;
            let obs = CostObservation {
                features: make_features(rows, false),
                actual_time_us: rows * 0.5, // ~0.5us per row
                actual_rows: i * 100,
            };
            model.train(&obs);
        }

        // Model should predict higher cost for more rows
        let small = model.predict(&make_features(100.0, false));
        let large = model.predict(&make_features(10000.0, false));
        assert!(
            large > small,
            "larger table should have higher predicted cost: small={small}, large={large}"
        );
    }

    #[test]
    fn serialization_roundtrip() {
        let model = LearnedCostModel::new();
        let obs = CostObservation {
            features: make_features(1000.0, true),
            actual_time_us: 500.0,
            actual_rows: 1000,
        };
        model.train(&obs);

        let json = model.to_json();
        let restored = LearnedCostModel::from_json(&json).unwrap();
        assert_eq!(restored.observation_count(), 1);
        assert_eq!(restored.weights().len(), LearnedCostModel::NUM_FEATURES);
    }

    #[test]
    fn plan_type_specific_predictions() {
        let model = LearnedCostModel::new();

        // Train full scan with high cost
        for _ in 0..20 {
            model.train_for_plan(
                "FullScan",
                &CostObservation {
                    features: make_features(1000.0, false),
                    actual_time_us: 5000.0,
                    actual_rows: 1000,
                },
            );
        }

        // Train index scan with low cost
        for _ in 0..20 {
            model.train_for_plan(
                "IndexScan",
                &CostObservation {
                    features: make_features(1000.0, true),
                    actual_time_us: 100.0,
                    actual_rows: 10,
                },
            );
        }

        let features_no_index = make_features(1000.0, false);
        let features_with_index = make_features(1000.0, true);

        let full_cost = model.predict_for_plan("FullScan", &features_no_index);
        let index_cost = model.predict_for_plan("IndexScan", &features_with_index);
        assert!(
            full_cost > index_cost || model.observation_count() < 20,
            "full scan should be more expensive: full={full_cost}, index={index_cost}"
        );
    }

    #[test]
    fn is_trained_after_enough_observations() {
        let model = LearnedCostModel::new();
        assert!(!model.is_trained());

        for i in 0..10 {
            model.train(&CostObservation {
                features: make_features((i + 1) as f64 * 100.0, false),
                actual_time_us: (i + 1) as f64 * 50.0,
                actual_rows: (i + 1) * 100,
            });
        }
        assert!(model.is_trained());
    }
}
