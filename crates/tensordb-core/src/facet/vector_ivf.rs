//! IVF (Inverted File) index: k-means centroids, cell assignment, posting lists.
//! Implements IVF-PQ for approximate nearest neighbor search on large datasets.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use serde::{Deserialize, Serialize};

use crate::error::{Result, TensorError};
use crate::facet::vector_quantization::PQCodebook;
use crate::facet::vector_search::DistanceMetric;

// ── IVF Index ───────────────────────────────────────────────────────────────

/// IVF index configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfConfig {
    /// Number of Voronoi cells (k-means clusters).
    pub nlist: usize,
    /// Number of cells to probe during search.
    pub nprobe: usize,
    /// PQ sub-quantizer count (0 = no PQ, store raw vectors).
    pub pq_m: usize,
    /// PQ bits per code.
    pub pq_bits: u8,
}

impl Default for IvfConfig {
    fn default() -> Self {
        Self {
            nlist: 256,
            nprobe: 16,
            pq_m: 0,
            pq_bits: 8,
        }
    }
}

/// An IVF index for approximate nearest-neighbor search.
#[derive(Debug)]
pub struct IvfIndex {
    pub config: IvfConfig,
    pub metric: DistanceMetric,
    pub dims: usize,
    /// Centroids for Voronoi cells, one per nlist.
    pub centroids: Vec<Vec<f32>>,
    /// Posting lists: for each cell, a list of (pk, vector_or_codes).
    pub posting_lists: Vec<Vec<IvfEntry>>,
    /// Optional PQ codebook (trained after centroids).
    pub codebook: Option<PQCodebook>,
    /// Total vector count.
    pub count: usize,
}

/// Entry in an IVF posting list.
#[derive(Debug, Clone)]
pub struct IvfEntry {
    pub pk: String,
    /// If PQ is enabled, this holds the PQ codes. Otherwise, full f32 vector.
    pub data: IvfEntryData,
}

#[derive(Debug, Clone)]
pub enum IvfEntryData {
    Raw(Vec<f32>),
    PQCodes(Vec<u8>),
}

/// Search result from IVF.
#[derive(Debug, Clone)]
pub struct IvfSearchResult {
    pub pk: String,
    pub distance: f32,
}

impl IvfIndex {
    /// Create a new empty IVF index.
    pub fn new(dims: usize, metric: DistanceMetric, config: IvfConfig) -> Self {
        Self {
            centroids: Vec::new(),
            posting_lists: Vec::new(),
            codebook: None,
            count: 0,
            config,
            metric,
            dims,
        }
    }

    /// Train the IVF index on a set of vectors (builds centroids and optional PQ codebook).
    pub fn train(&mut self, vectors: &[(String, Vec<f32>)]) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        let nlist = self.config.nlist.min(vectors.len());

        // K-means for centroids
        let vecs: Vec<Vec<f32>> = vectors.iter().map(|(_, v)| v.clone()).collect();
        self.centroids = simple_kmeans(&vecs, nlist, 25);

        // Initialize posting lists
        self.posting_lists = vec![Vec::new(); nlist];

        // Train PQ if configured
        if self.config.pq_m > 0 {
            let codebook = PQCodebook::train(&vecs, self.config.pq_m, self.config.pq_bits, 25)?;
            self.codebook = Some(codebook);
        }

        // Assign vectors to cells
        for (pk, vec) in vectors {
            self.insert_internal(pk.clone(), vec)?;
        }

        Ok(())
    }

    /// Insert a vector into the index.
    pub fn insert(&mut self, pk: String, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dims {
            return Err(TensorError::VectorError(format!(
                "expected {} dimensions, got {}",
                self.dims,
                vector.len()
            )));
        }
        if self.centroids.is_empty() {
            // Not yet trained — add as a single-cell index
            if self.posting_lists.is_empty() {
                self.centroids.push(vector.to_vec());
                self.posting_lists.push(Vec::new());
            }
        }
        self.insert_internal(pk, vector)
    }

    fn insert_internal(&mut self, pk: String, vector: &[f32]) -> Result<()> {
        let cell = self.nearest_centroid(vector);
        let data = if let Some(ref codebook) = self.codebook {
            // Compute residual (vector - centroid) and PQ encode
            let residual: Vec<f32> = vector
                .iter()
                .zip(self.centroids[cell].iter())
                .map(|(v, c)| v - c)
                .collect();
            IvfEntryData::PQCodes(codebook.encode(&residual))
        } else {
            IvfEntryData::Raw(vector.to_vec())
        };

        self.posting_lists[cell].push(IvfEntry { pk, data });
        self.count += 1;
        Ok(())
    }

    /// Search for k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<IvfSearchResult> {
        if self.centroids.is_empty() {
            return Vec::new();
        }

        // Find nearest nprobe centroids
        let nprobe = self.config.nprobe.min(self.centroids.len());
        let mut centroid_dists: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, self.metric.compute(query, c)))
            .collect();
        centroid_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        let mut heap: BinaryHeap<OrdF32Entry> = BinaryHeap::new();

        // Precompute PQ distance table if using PQ
        let pq_table = self.codebook.as_ref().map(|cb| {
            // Query residual for each probed cell will differ,
            // but for simplicity we use the raw query table (ADC approximation)
            cb.precompute_distance_table(query)
        });

        for &(cell, _) in centroid_dists.iter().take(nprobe) {
            for entry in &self.posting_lists[cell] {
                let distance = match &entry.data {
                    IvfEntryData::Raw(vec) => self.metric.compute(query, vec),
                    IvfEntryData::PQCodes(codes) => {
                        if let Some(ref table) = pq_table {
                            PQCodebook::distance_from_table(table, codes)
                        } else {
                            f32::MAX
                        }
                    }
                };

                if heap.len() < k {
                    heap.push(OrdF32Entry {
                        pk: entry.pk.clone(),
                        distance,
                    });
                } else if let Some(top) = heap.peek() {
                    if distance < top.distance {
                        heap.pop();
                        heap.push(OrdF32Entry {
                            pk: entry.pk.clone(),
                            distance,
                        });
                    }
                }
            }
        }

        let mut results: Vec<IvfSearchResult> = heap
            .into_iter()
            .map(|e| IvfSearchResult {
                pk: e.pk,
                distance: e.distance,
            })
            .collect();
        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        results
    }

    /// Delete a vector by primary key.
    pub fn delete(&mut self, pk: &str) -> bool {
        for list in &mut self.posting_lists {
            if let Some(pos) = list.iter().position(|e| e.pk == pk) {
                list.swap_remove(pos);
                self.count -= 1;
                return true;
            }
        }
        false
    }

    fn nearest_centroid(&self, vector: &[f32]) -> usize {
        let mut best = 0;
        let mut best_dist = f32::MAX;
        for (i, c) in self.centroids.iter().enumerate() {
            let dist = l2_sq(vector, c);
            if dist < best_dist {
                best_dist = dist;
                best = i;
            }
        }
        best
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct OrdF32Entry {
    pk: String,
    distance: f32,
}

impl PartialEq for OrdF32Entry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for OrdF32Entry {}

impl PartialOrd for OrdF32Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: larger distance = higher priority (so we can pop the worst)
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Simple k-means for IVF centroid training.
fn simple_kmeans(data: &[Vec<f32>], k: usize, max_iters: usize) -> Vec<Vec<f32>> {
    if data.is_empty() || k == 0 {
        return Vec::new();
    }

    let n = data.len();
    let d = data[0].len();
    let actual_k = k.min(n);

    let mut centroids: Vec<Vec<f32>> = (0..actual_k)
        .map(|i| data[i * n / actual_k].clone())
        .collect();

    let mut assignments = vec![0usize; n];

    for _ in 0..max_iters {
        let mut changed = false;

        for (i, point) in data.iter().enumerate() {
            let mut best = 0;
            let mut best_dist = l2_sq(point, &centroids[0]);
            for (j, centroid) in centroids.iter().enumerate().skip(1) {
                let dist = l2_sq(point, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best = j;
                }
            }
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        let mut sums = vec![vec![0.0_f32; d]; actual_k];
        let mut counts = vec![0usize; actual_k];
        for (i, point) in data.iter().enumerate() {
            let cluster = assignments[i];
            counts[cluster] += 1;
            for (j, &v) in point.iter().enumerate() {
                sums[cluster][j] += v;
            }
        }
        for c in 0..actual_k {
            if counts[c] > 0 {
                for j in 0..d {
                    centroids[c][j] = sums[c][j] / counts[c] as f32;
                }
            }
        }
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ivf_basic_search() {
        let mut idx = IvfIndex::new(
            4,
            DistanceMetric::Euclidean,
            IvfConfig {
                nlist: 4,
                nprobe: 4,
                pq_m: 0,
                pq_bits: 8,
            },
        );

        let vectors: Vec<(String, Vec<f32>)> = (0..20)
            .map(|i| (format!("pk{i}"), vec![i as f32; 4]))
            .collect();

        idx.train(&vectors).unwrap();
        assert_eq!(idx.count, 20);

        let results = idx.search(&[5.0; 4], 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].pk, "pk5");
    }

    #[test]
    fn test_ivf_delete() {
        let mut idx = IvfIndex::new(
            2,
            DistanceMetric::Euclidean,
            IvfConfig {
                nlist: 2,
                nprobe: 2,
                pq_m: 0,
                pq_bits: 8,
            },
        );

        let vectors: Vec<(String, Vec<f32>)> = (0..10)
            .map(|i| (format!("pk{i}"), vec![i as f32; 2]))
            .collect();

        idx.train(&vectors).unwrap();
        assert_eq!(idx.count, 10);

        assert!(idx.delete("pk5"));
        assert_eq!(idx.count, 9);
        assert!(!idx.delete("pk_nonexistent"));
    }
}
