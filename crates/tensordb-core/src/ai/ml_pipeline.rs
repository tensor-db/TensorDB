use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use crate::util::time::unix_millis;

// ---------------------------------------------------------------------------
// Feature Store
// ---------------------------------------------------------------------------

/// Data type for a feature column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureDType {
    Float32,
    Float64,
    Int64,
    Bool,
    /// Embedding with a fixed dimension.
    Embedding(usize),
}

/// Schema definition for a single feature column.
#[derive(Debug, Clone)]
pub struct FeatureDefinition {
    pub name: String,
    pub dtype: FeatureDType,
    pub default_value: Option<f64>,
    pub description: String,
}

/// Schema for a feature set (ordered list of feature columns).
#[derive(Debug, Clone)]
pub struct FeatureSchema {
    pub features: Vec<FeatureDefinition>,
}

/// A single entity's feature vector.
#[derive(Debug, Clone)]
pub struct FeatureVector {
    pub entity_id: String,
    pub values: Vec<f64>,
    pub updated_at: u64,
}

/// A named feature set containing schema and feature vectors keyed by entity id.
#[derive(Debug, Clone)]
pub struct FeatureSet {
    pub name: String,
    pub schema: FeatureSchema,
    pub vectors: HashMap<String, FeatureVector>,
    pub created_at: u64,
    pub updated_at: u64,
    pub version: u64,
}

/// Summary info for a feature set (returned by `list_feature_sets`).
#[derive(Debug, Clone)]
pub struct FeatureSetInfo {
    pub name: String,
    pub feature_count: usize,
    pub entity_count: usize,
    pub version: u64,
    pub created_at: u64,
    pub updated_at: u64,
}

/// Aggregate statistics for the feature store.
#[derive(Debug, Clone)]
pub struct FeatureStoreStats {
    pub total_feature_sets: usize,
    pub total_entities: usize,
    pub total_reads: u64,
    pub total_writes: u64,
}

/// In-process feature store for ML features computed from database facts.
pub struct FeatureStore {
    features: RwLock<HashMap<String, FeatureSet>>,
    total_reads: AtomicU64,
    total_writes: AtomicU64,
}

impl FeatureStore {
    pub fn new() -> Self {
        Self {
            features: RwLock::new(HashMap::new()),
            total_reads: AtomicU64::new(0),
            total_writes: AtomicU64::new(0),
        }
    }

    /// Create a new feature set with the given name and schema.
    /// Returns an error if a feature set with that name already exists.
    pub fn create_feature_set(&self, name: &str, schema: FeatureSchema) -> Result<(), String> {
        let mut map = self.features.write();
        if map.contains_key(name) {
            return Err(format!("feature set '{}' already exists", name));
        }
        let now = unix_millis();
        map.insert(
            name.to_string(),
            FeatureSet {
                name: name.to_string(),
                schema,
                vectors: HashMap::new(),
                created_at: now,
                updated_at: now,
                version: 1,
            },
        );
        Ok(())
    }

    /// Drop an existing feature set. Returns an error if it does not exist.
    pub fn drop_feature_set(&self, name: &str) -> Result<(), String> {
        let mut map = self.features.write();
        if map.remove(name).is_none() {
            return Err(format!("feature set '{}' not found", name));
        }
        Ok(())
    }

    /// Ingest a feature vector for an entity into a feature set.
    pub fn ingest(
        &self,
        feature_set: &str,
        entity_id: &str,
        values: Vec<f64>,
    ) -> Result<(), String> {
        let mut map = self.features.write();
        let fs = map
            .get_mut(feature_set)
            .ok_or_else(|| format!("feature set '{}' not found", feature_set))?;
        let now = unix_millis();
        fs.vectors.insert(
            entity_id.to_string(),
            FeatureVector {
                entity_id: entity_id.to_string(),
                values,
                updated_at: now,
            },
        );
        fs.updated_at = now;
        fs.version += 1;
        self.total_writes.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Get features for a single entity.
    pub fn get_features(&self, feature_set: &str, entity_id: &str) -> Option<FeatureVector> {
        let map = self.features.read();
        self.total_reads.fetch_add(1, Ordering::Relaxed);
        map.get(feature_set)
            .and_then(|fs| fs.vectors.get(entity_id).cloned())
    }

    /// Batch feature retrieval for multiple entity ids.
    pub fn get_batch(
        &self,
        feature_set: &str,
        entity_ids: &[String],
    ) -> Vec<(String, FeatureVector)> {
        let map = self.features.read();
        self.total_reads.fetch_add(1, Ordering::Relaxed);
        let Some(fs) = map.get(feature_set) else {
            return Vec::new();
        };
        entity_ids
            .iter()
            .filter_map(|id| fs.vectors.get(id).map(|v| (id.clone(), v.clone())))
            .collect()
    }

    /// List all feature sets with summary info.
    pub fn list_feature_sets(&self) -> Vec<FeatureSetInfo> {
        let map = self.features.read();
        map.values()
            .map(|fs| FeatureSetInfo {
                name: fs.name.clone(),
                feature_count: fs.schema.features.len(),
                entity_count: fs.vectors.len(),
                version: fs.version,
                created_at: fs.created_at,
                updated_at: fs.updated_at,
            })
            .collect()
    }

    /// Temporal feature retrieval: only return features updated before `as_of`.
    pub fn point_in_time_join(
        &self,
        feature_set: &str,
        entity_ids: &[String],
        as_of: u64,
    ) -> Vec<(String, Option<FeatureVector>)> {
        let map = self.features.read();
        self.total_reads.fetch_add(1, Ordering::Relaxed);
        let Some(fs) = map.get(feature_set) else {
            return entity_ids.iter().map(|id| (id.clone(), None)).collect();
        };
        entity_ids
            .iter()
            .map(|id| {
                let vec = fs.vectors.get(id).and_then(|v| {
                    if v.updated_at <= as_of {
                        Some(v.clone())
                    } else {
                        None
                    }
                });
                (id.clone(), vec)
            })
            .collect()
    }

    /// Aggregate statistics for the feature store.
    pub fn stats(&self) -> FeatureStoreStats {
        let map = self.features.read();
        let total_entities: usize = map.values().map(|fs| fs.vectors.len()).sum();
        FeatureStoreStats {
            total_feature_sets: map.len(),
            total_entities,
            total_reads: self.total_reads.load(Ordering::Relaxed),
            total_writes: self.total_writes.load(Ordering::Relaxed),
        }
    }
}

impl Default for FeatureStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Model Registry
// ---------------------------------------------------------------------------

/// Lifecycle status of a registered model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelStatus {
    Registered,
    Validated,
    Deployed,
    Deprecated,
    Archived,
}

/// Inference metrics for a model.
#[derive(Debug, Clone)]
pub struct ModelMetrics {
    pub total_inferences: u64,
    pub avg_latency_us: f64,
    pub error_count: u64,
    pub last_inference_at: u64,
}

/// A registered model entry (versioned).
#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub name: String,
    pub version: u64,
    pub framework: String,
    pub input_schema: Vec<String>,
    pub output_schema: Vec<String>,
    pub metadata: serde_json::Value,
    pub status: ModelStatus,
    pub registered_at: u64,
    pub updated_at: u64,
    pub metrics: ModelMetrics,
}

/// Summary info for a model (returned by `list`).
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub latest_version: u64,
    pub framework: String,
    pub status: ModelStatus,
    pub registered_at: u64,
}

/// Aggregate statistics for the model registry.
#[derive(Debug, Clone)]
pub struct ModelRegistryStats {
    pub total_models: usize,
    pub total_versions: usize,
    pub total_inferences: u64,
    pub total_registered: u64,
}

/// In-process model registry for tracking ML model versions and metadata.
///
/// Each model name maps to a list of versioned entries (append-only).
/// The outer `HashMap` key is the model name; the value stores all versions.
pub struct ModelRegistry {
    /// model_name -> Vec<ModelEntry> (one per version, ordered by version ascending)
    models: RwLock<HashMap<String, Vec<ModelEntry>>>,
    total_registered: AtomicU64,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            total_registered: AtomicU64::new(0),
        }
    }

    /// Register a new model (or a new version of an existing model).
    /// Returns the version number that was assigned.
    pub fn register(
        &self,
        name: &str,
        framework: &str,
        input_schema: Vec<String>,
        output_schema: Vec<String>,
        metadata: serde_json::Value,
    ) -> Result<u64, String> {
        let mut map = self.models.write();
        let now = unix_millis();
        let versions = map.entry(name.to_string()).or_default();
        let version = versions.len() as u64 + 1;
        versions.push(ModelEntry {
            name: name.to_string(),
            version,
            framework: framework.to_string(),
            input_schema,
            output_schema,
            metadata,
            status: ModelStatus::Registered,
            registered_at: now,
            updated_at: now,
            metrics: ModelMetrics {
                total_inferences: 0,
                avg_latency_us: 0.0,
                error_count: 0,
                last_inference_at: 0,
            },
        });
        self.total_registered.fetch_add(1, Ordering::Relaxed);
        Ok(version)
    }

    /// Get the latest version of a model by name.
    pub fn get(&self, name: &str) -> Option<ModelEntry> {
        let map = self.models.read();
        map.get(name).and_then(|v| v.last().cloned())
    }

    /// Get a specific version of a model.
    pub fn get_version(&self, name: &str, version: u64) -> Option<ModelEntry> {
        let map = self.models.read();
        map.get(name)
            .and_then(|versions| versions.iter().find(|e| e.version == version).cloned())
    }

    /// List all models (latest version info only).
    pub fn list(&self) -> Vec<ModelInfo> {
        let map = self.models.read();
        map.values()
            .filter_map(|versions| {
                versions.last().map(|e| ModelInfo {
                    name: e.name.clone(),
                    latest_version: e.version,
                    framework: e.framework.clone(),
                    status: e.status.clone(),
                    registered_at: e.registered_at,
                })
            })
            .collect()
    }

    /// Update the status of the latest version of a model.
    pub fn update_status(&self, name: &str, status: ModelStatus) -> Result<(), String> {
        let mut map = self.models.write();
        let versions = map
            .get_mut(name)
            .ok_or_else(|| format!("model '{}' not found", name))?;
        let entry = versions
            .last_mut()
            .ok_or_else(|| format!("model '{}' has no versions", name))?;
        entry.status = status;
        entry.updated_at = unix_millis();
        Ok(())
    }

    /// Record an inference for the latest version of a model.
    /// Updates latency statistics and error counts.
    pub fn record_inference(
        &self,
        name: &str,
        latency_us: f64,
        success: bool,
    ) -> Result<(), String> {
        let mut map = self.models.write();
        let versions = map
            .get_mut(name)
            .ok_or_else(|| format!("model '{}' not found", name))?;
        let entry = versions
            .last_mut()
            .ok_or_else(|| format!("model '{}' has no versions", name))?;

        let m = &mut entry.metrics;
        let prev_total = m.total_inferences as f64;
        m.total_inferences += 1;
        // Running average: new_avg = (old_avg * old_count + new_value) / new_count
        m.avg_latency_us = (m.avg_latency_us * prev_total + latency_us) / m.total_inferences as f64;
        if !success {
            m.error_count += 1;
        }
        m.last_inference_at = unix_millis();
        Ok(())
    }

    /// Get the latest version of a model that has `Deployed` status.
    pub fn latest_deployed(&self, name: &str) -> Option<ModelEntry> {
        let map = self.models.read();
        map.get(name).and_then(|versions| {
            versions
                .iter()
                .rev()
                .find(|e| e.status == ModelStatus::Deployed)
                .cloned()
        })
    }

    /// Aggregate statistics for the model registry.
    pub fn stats(&self) -> ModelRegistryStats {
        let map = self.models.read();
        let total_versions: usize = map.values().map(|v| v.len()).sum();
        let total_inferences: u64 = map
            .values()
            .flat_map(|versions| versions.iter())
            .map(|e| e.metrics.total_inferences)
            .sum();
        ModelRegistryStats {
            total_models: map.len(),
            total_versions,
            total_inferences,
            total_registered: self.total_registered.load(Ordering::Relaxed),
        }
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Pipeline Advisor
// ---------------------------------------------------------------------------

/// Aggregate statistics for the pipeline advisor.
#[derive(Debug, Clone)]
pub struct PipelineAdvisorStats {
    pub staleness_samples: usize,
    pub accuracy_samples: usize,
    pub latest_staleness: Option<f64>,
    pub latest_accuracy: Option<f64>,
    pub health_score: f64,
}

/// AI advisor that monitors the ML pipeline and recommends actions.
pub struct PipelineAdvisor {
    feature_staleness_window: RwLock<Vec<(u64, f64)>>,
    model_accuracy_window: RwLock<Vec<(u64, f64)>>,
}

/// Threshold below which accuracy triggers a retrain recommendation.
const ACCURACY_RETRAIN_THRESHOLD: f64 = 0.7;
/// Threshold above which staleness triggers a refresh recommendation.
const STALENESS_REFRESH_THRESHOLD: f64 = 0.5;
/// Maximum number of samples to keep in the sliding windows.
const MAX_WINDOW_SIZE: usize = 1000;

impl PipelineAdvisor {
    pub fn new() -> Self {
        Self {
            feature_staleness_window: RwLock::new(Vec::new()),
            model_accuracy_window: RwLock::new(Vec::new()),
        }
    }

    /// Record what fraction of features are stale (0.0 = all fresh, 1.0 = all stale).
    pub fn record_staleness(&self, ratio: f64) {
        let mut window = self.feature_staleness_window.write();
        let now = unix_millis();
        window.push((now, ratio.clamp(0.0, 1.0)));
        if window.len() > MAX_WINDOW_SIZE {
            let excess = window.len() - MAX_WINDOW_SIZE;
            window.drain(..excess);
        }
    }

    /// Record model accuracy (0.0 to 1.0).
    pub fn record_accuracy(&self, accuracy: f64) {
        let mut window = self.model_accuracy_window.write();
        let now = unix_millis();
        window.push((now, accuracy.clamp(0.0, 1.0)));
        if window.len() > MAX_WINDOW_SIZE {
            let excess = window.len() - MAX_WINDOW_SIZE;
            window.drain(..excess);
        }
    }

    /// AI decision: recommend retraining if recent accuracy drops below threshold.
    pub fn should_retrain(&self) -> bool {
        let window = self.model_accuracy_window.read();
        if window.is_empty() {
            return false;
        }
        // Use the average of the last 5 samples (or all if fewer).
        let tail = &window[window.len().saturating_sub(5)..];
        let avg: f64 = tail.iter().map(|(_, v)| v).sum::<f64>() / tail.len() as f64;
        avg < ACCURACY_RETRAIN_THRESHOLD
    }

    /// AI decision: recommend feature refresh if staleness is high.
    pub fn should_refresh_features(&self) -> bool {
        let window = self.feature_staleness_window.read();
        if window.is_empty() {
            return false;
        }
        let tail = &window[window.len().saturating_sub(5)..];
        let avg: f64 = tail.iter().map(|(_, v)| v).sum::<f64>() / tail.len() as f64;
        avg > STALENESS_REFRESH_THRESHOLD
    }

    /// Overall pipeline health score (0.0 = unhealthy, 1.0 = healthy).
    ///
    /// Combines freshness (1 - staleness) and accuracy into a weighted score.
    /// If no data is available, returns 1.0 (assume healthy).
    pub fn health_score(&self) -> f64 {
        let staleness_window = self.feature_staleness_window.read();
        let accuracy_window = self.model_accuracy_window.read();

        let freshness = if staleness_window.is_empty() {
            1.0
        } else {
            let tail = &staleness_window[staleness_window.len().saturating_sub(5)..];
            let avg_staleness: f64 = tail.iter().map(|(_, v)| v).sum::<f64>() / tail.len() as f64;
            1.0 - avg_staleness
        };

        let accuracy = if accuracy_window.is_empty() {
            1.0
        } else {
            let tail = &accuracy_window[accuracy_window.len().saturating_sub(5)..];
            tail.iter().map(|(_, v)| v).sum::<f64>() / tail.len() as f64
        };

        // Weighted: 40% freshness, 60% accuracy.
        let score = 0.4 * freshness + 0.6 * accuracy;
        score.clamp(0.0, 1.0)
    }

    /// Aggregate statistics for the pipeline advisor.
    pub fn stats(&self) -> PipelineAdvisorStats {
        let staleness_window = self.feature_staleness_window.read();
        let accuracy_window = self.model_accuracy_window.read();
        PipelineAdvisorStats {
            staleness_samples: staleness_window.len(),
            accuracy_samples: accuracy_window.len(),
            latest_staleness: staleness_window.last().map(|(_, v)| *v),
            latest_accuracy: accuracy_window.last().map(|(_, v)| *v),
            health_score: self.health_score_inner(&staleness_window, &accuracy_window),
        }
    }

    /// Inner health score calculation (avoids double-locking in `stats`).
    fn health_score_inner(
        &self,
        staleness_window: &[(u64, f64)],
        accuracy_window: &[(u64, f64)],
    ) -> f64 {
        let freshness = if staleness_window.is_empty() {
            1.0
        } else {
            let tail = &staleness_window[staleness_window.len().saturating_sub(5)..];
            let avg_staleness: f64 = tail.iter().map(|(_, v)| v).sum::<f64>() / tail.len() as f64;
            1.0 - avg_staleness
        };

        let accuracy = if accuracy_window.is_empty() {
            1.0
        } else {
            let tail = &accuracy_window[accuracy_window.len().saturating_sub(5)..];
            tail.iter().map(|(_, v)| v).sum::<f64>() / tail.len() as f64
        };

        (0.4 * freshness + 0.6 * accuracy).clamp(0.0, 1.0)
    }
}

impl Default for PipelineAdvisor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Feature Store tests ----

    #[test]
    fn feature_store_create_and_list() {
        let store = FeatureStore::new();
        let schema = FeatureSchema {
            features: vec![FeatureDefinition {
                name: "age".to_string(),
                dtype: FeatureDType::Float64,
                default_value: Some(0.0),
                description: "user age".to_string(),
            }],
        };
        store.create_feature_set("users", schema).unwrap();
        let list = store.list_feature_sets();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "users");
        assert_eq!(list[0].feature_count, 1);
    }

    #[test]
    fn feature_store_ingest_and_get() {
        let store = FeatureStore::new();
        let schema = FeatureSchema {
            features: vec![FeatureDefinition {
                name: "score".to_string(),
                dtype: FeatureDType::Float64,
                default_value: None,
                description: "user score".to_string(),
            }],
        };
        store.create_feature_set("scores", schema).unwrap();
        store.ingest("scores", "entity-1", vec![42.0]).unwrap();
        let fv = store.get_features("scores", "entity-1").unwrap();
        assert_eq!(fv.entity_id, "entity-1");
        assert_eq!(fv.values, vec![42.0]);
    }

    #[test]
    fn feature_store_batch_get() {
        let store = FeatureStore::new();
        let schema = FeatureSchema {
            features: vec![FeatureDefinition {
                name: "x".to_string(),
                dtype: FeatureDType::Float32,
                default_value: None,
                description: "feature x".to_string(),
            }],
        };
        store.create_feature_set("fs", schema).unwrap();
        store.ingest("fs", "a", vec![1.0]).unwrap();
        store.ingest("fs", "b", vec![2.0]).unwrap();
        store.ingest("fs", "c", vec![3.0]).unwrap();

        let ids = vec!["a".to_string(), "c".to_string(), "missing".to_string()];
        let batch = store.get_batch("fs", &ids);
        assert_eq!(batch.len(), 2);
        assert!(batch.iter().any(|(id, _)| id == "a"));
        assert!(batch.iter().any(|(id, _)| id == "c"));
    }

    #[test]
    fn feature_store_point_in_time_join() {
        let store = FeatureStore::new();
        let schema = FeatureSchema {
            features: vec![FeatureDefinition {
                name: "v".to_string(),
                dtype: FeatureDType::Float64,
                default_value: None,
                description: "value".to_string(),
            }],
        };
        store.create_feature_set("fs", schema).unwrap();
        store.ingest("fs", "e1", vec![10.0]).unwrap();

        // Get the updated_at timestamp so we can query around it.
        let fv = store.get_features("fs", "e1").unwrap();
        let ts = fv.updated_at;

        // Query with as_of >= updated_at should return the vector.
        let result = store.point_in_time_join("fs", &["e1".to_string()], ts + 1000);
        assert_eq!(result.len(), 1);
        assert!(result[0].1.is_some());

        // Query with as_of < updated_at should NOT return the vector.
        let result = store.point_in_time_join("fs", &["e1".to_string()], ts - 1);
        assert_eq!(result.len(), 1);
        assert!(result[0].1.is_none());
    }

    #[test]
    fn feature_store_drop() {
        let store = FeatureStore::new();
        let schema = FeatureSchema { features: vec![] };
        store.create_feature_set("temp", schema).unwrap();
        assert_eq!(store.list_feature_sets().len(), 1);
        store.drop_feature_set("temp").unwrap();
        assert_eq!(store.list_feature_sets().len(), 0);
    }

    #[test]
    fn feature_store_duplicate_create_fails() {
        let store = FeatureStore::new();
        let schema = FeatureSchema { features: vec![] };
        store.create_feature_set("dup", schema.clone()).unwrap();
        let result = store.create_feature_set("dup", schema);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already exists"));
    }

    #[test]
    fn feature_store_stats() {
        let store = FeatureStore::new();
        let schema = FeatureSchema {
            features: vec![FeatureDefinition {
                name: "f".to_string(),
                dtype: FeatureDType::Int64,
                default_value: None,
                description: "".to_string(),
            }],
        };
        store.create_feature_set("fs", schema).unwrap();
        store.ingest("fs", "e1", vec![1.0]).unwrap();
        store.ingest("fs", "e2", vec![2.0]).unwrap();
        let _ = store.get_features("fs", "e1");

        let s = store.stats();
        assert_eq!(s.total_feature_sets, 1);
        assert_eq!(s.total_entities, 2);
        assert_eq!(s.total_writes, 2);
        assert!(s.total_reads >= 1);
    }

    #[test]
    fn feature_store_ingest_missing_set_fails() {
        let store = FeatureStore::new();
        let result = store.ingest("nonexistent", "e1", vec![1.0]);
        assert!(result.is_err());
    }

    // ---- Model Registry tests ----

    #[test]
    fn model_registry_register_and_get() {
        let registry = ModelRegistry::new();
        let v = registry
            .register(
                "my-model",
                "onnx",
                vec!["feature_a".to_string()],
                vec!["prediction".to_string()],
                serde_json::json!({"author": "test"}),
            )
            .unwrap();
        assert_eq!(v, 1);

        let entry = registry.get("my-model").unwrap();
        assert_eq!(entry.name, "my-model");
        assert_eq!(entry.version, 1);
        assert_eq!(entry.framework, "onnx");
        assert_eq!(entry.status, ModelStatus::Registered);
    }

    #[test]
    fn model_registry_version_tracking() {
        let registry = ModelRegistry::new();
        let v1 = registry
            .register("m", "custom", vec![], vec![], serde_json::json!({}))
            .unwrap();
        let v2 = registry
            .register("m", "custom", vec![], vec![], serde_json::json!({}))
            .unwrap();
        assert_eq!(v1, 1);
        assert_eq!(v2, 2);

        // get() returns latest.
        let latest = registry.get("m").unwrap();
        assert_eq!(latest.version, 2);

        // get_version returns specific.
        let first = registry.get_version("m", 1).unwrap();
        assert_eq!(first.version, 1);
    }

    #[test]
    fn model_registry_update_status() {
        let registry = ModelRegistry::new();
        registry
            .register("m", "onnx", vec![], vec![], serde_json::json!({}))
            .unwrap();
        registry.update_status("m", ModelStatus::Deployed).unwrap();
        let entry = registry.get("m").unwrap();
        assert_eq!(entry.status, ModelStatus::Deployed);
    }

    #[test]
    fn model_registry_record_inference() {
        let registry = ModelRegistry::new();
        registry
            .register("m", "onnx", vec![], vec![], serde_json::json!({}))
            .unwrap();
        registry.record_inference("m", 100.0, true).unwrap();
        registry.record_inference("m", 200.0, true).unwrap();
        registry.record_inference("m", 300.0, false).unwrap();

        let entry = registry.get("m").unwrap();
        assert_eq!(entry.metrics.total_inferences, 3);
        assert_eq!(entry.metrics.error_count, 1);
        // avg_latency_us = (100 + 200 + 300) / 3 = 200
        assert!((entry.metrics.avg_latency_us - 200.0).abs() < 0.01);
    }

    #[test]
    fn model_registry_list() {
        let registry = ModelRegistry::new();
        registry
            .register("a", "onnx", vec![], vec![], serde_json::json!({}))
            .unwrap();
        registry
            .register("b", "custom", vec![], vec![], serde_json::json!({}))
            .unwrap();
        let list = registry.list();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn model_registry_latest_deployed() {
        let registry = ModelRegistry::new();
        registry
            .register("m", "onnx", vec![], vec![], serde_json::json!({}))
            .unwrap();
        registry
            .register("m", "onnx", vec![], vec![], serde_json::json!({}))
            .unwrap();
        // Deploy version 1 only.
        {
            let mut map = registry.models.write();
            map.get_mut("m").unwrap()[0].status = ModelStatus::Deployed;
        }
        let deployed = registry.latest_deployed("m").unwrap();
        assert_eq!(deployed.version, 1);

        // No deployed version for unknown model.
        assert!(registry.latest_deployed("unknown").is_none());
    }

    #[test]
    fn model_registry_stats() {
        let registry = ModelRegistry::new();
        registry
            .register("a", "onnx", vec![], vec![], serde_json::json!({}))
            .unwrap();
        registry
            .register("a", "onnx", vec![], vec![], serde_json::json!({}))
            .unwrap();
        registry
            .register("b", "custom", vec![], vec![], serde_json::json!({}))
            .unwrap();
        registry.record_inference("a", 50.0, true).unwrap();

        let s = registry.stats();
        assert_eq!(s.total_models, 2);
        assert_eq!(s.total_versions, 3);
        assert_eq!(s.total_registered, 3);
        assert_eq!(s.total_inferences, 1);
    }

    // ---- Pipeline Advisor tests ----

    #[test]
    fn pipeline_advisor_staleness_tracking() {
        let advisor = PipelineAdvisor::new();
        advisor.record_staleness(0.2);
        advisor.record_staleness(0.3);
        let stats = advisor.stats();
        assert_eq!(stats.staleness_samples, 2);
        assert!((stats.latest_staleness.unwrap() - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn pipeline_advisor_accuracy_tracking() {
        let advisor = PipelineAdvisor::new();
        advisor.record_accuracy(0.95);
        advisor.record_accuracy(0.90);
        let stats = advisor.stats();
        assert_eq!(stats.accuracy_samples, 2);
        assert!((stats.latest_accuracy.unwrap() - 0.90).abs() < f64::EPSILON);
    }

    #[test]
    fn pipeline_advisor_retrain_decision() {
        let advisor = PipelineAdvisor::new();
        // No data => no retrain.
        assert!(!advisor.should_retrain());

        // High accuracy => no retrain.
        advisor.record_accuracy(0.95);
        advisor.record_accuracy(0.92);
        assert!(!advisor.should_retrain());

        // Low accuracy => retrain.
        let advisor2 = PipelineAdvisor::new();
        for _ in 0..5 {
            advisor2.record_accuracy(0.55);
        }
        assert!(advisor2.should_retrain());
    }

    #[test]
    fn pipeline_advisor_refresh_decision() {
        let advisor = PipelineAdvisor::new();
        // No data => no refresh.
        assert!(!advisor.should_refresh_features());

        // Low staleness => no refresh.
        advisor.record_staleness(0.1);
        advisor.record_staleness(0.2);
        assert!(!advisor.should_refresh_features());

        // High staleness => refresh.
        let advisor2 = PipelineAdvisor::new();
        for _ in 0..5 {
            advisor2.record_staleness(0.8);
        }
        assert!(advisor2.should_refresh_features());
    }

    #[test]
    fn pipeline_advisor_health_score() {
        let advisor = PipelineAdvisor::new();
        // No data => healthy (1.0).
        assert!((advisor.health_score() - 1.0).abs() < f64::EPSILON);

        // Perfect accuracy, no staleness => 1.0.
        advisor.record_accuracy(1.0);
        advisor.record_staleness(0.0);
        assert!((advisor.health_score() - 1.0).abs() < f64::EPSILON);

        // All stale, zero accuracy => 0.0.
        let advisor2 = PipelineAdvisor::new();
        for _ in 0..5 {
            advisor2.record_staleness(1.0);
            advisor2.record_accuracy(0.0);
        }
        assert!(advisor2.health_score() < f64::EPSILON);
    }
}
