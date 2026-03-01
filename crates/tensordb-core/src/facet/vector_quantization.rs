//! Vector quantization: PQ codebook, INT8/FP16 encoding, k-means.
//! Implemented in Phase V3.

use half::f16;
use serde::{Deserialize, Serialize};

use crate::error::{Result, TensorError};

// ── Scalar quantization ─────────────────────────────────────────────────────

/// Quantize f32 values to FP16 (half-precision).
pub fn quantize_f16(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        out.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    out
}

/// Dequantize FP16 bytes back to f32.
pub fn dequantize_f16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

/// Quantize f32 values to INT8 (signed, normalized to [-1, 1]).
pub fn quantize_int8(values: &[f32]) -> Vec<u8> {
    let max_abs = values
        .iter()
        .map(|v| v.abs())
        .fold(f32::NEG_INFINITY, f32::max);
    let scale = if max_abs > 0.0 { 127.0 / max_abs } else { 1.0 };
    values
        .iter()
        .map(|&v| (v * scale).round().clamp(-127.0, 127.0) as i8 as u8)
        .collect()
}

/// Dequantize INT8 bytes back to f32 (approximate).
pub fn dequantize_int8(bytes: &[u8]) -> Vec<f32> {
    bytes.iter().map(|&b| (b as i8) as f32 / 127.0).collect()
}

// ── Product Quantization ────────────────────────────────────────────────────

/// PQ codebook: divides vectors into `m` sub-vectors, each quantized to `k_bits` bits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQCodebook {
    /// Number of sub-quantizers.
    pub m: usize,
    /// Bits per code (typically 8).
    pub k_bits: u8,
    /// Number of centroids per sub-quantizer (2^k_bits).
    pub k: usize,
    /// Dimension of the full vector.
    pub dims: usize,
    /// Dimension of each sub-vector (dims / m).
    pub sub_dims: usize,
    /// Centroids: m * k sub-vectors, each of length sub_dims.
    /// Indexed as centroids[sub_q * k + centroid_id][component].
    pub centroids: Vec<Vec<f32>>,
}

impl PQCodebook {
    /// Train PQ codebook using k-means on the given vectors.
    pub fn train(vectors: &[Vec<f32>], m: usize, k_bits: u8, max_iters: usize) -> Result<Self> {
        if vectors.is_empty() {
            return Err(TensorError::VectorError(
                "cannot train PQ on empty dataset".to_string(),
            ));
        }

        let dims = vectors[0].len();
        if !dims.is_multiple_of(m) {
            return Err(TensorError::VectorError(format!(
                "vector dimension {dims} not divisible by m={m}"
            )));
        }

        let sub_dims = dims / m;
        let k = 1usize << k_bits;
        let mut centroids = Vec::with_capacity(m * k);

        for sub_q in 0..m {
            let start = sub_q * sub_dims;
            let end = start + sub_dims;

            // Extract sub-vectors for this sub-quantizer
            let sub_vectors: Vec<Vec<f32>> =
                vectors.iter().map(|v| v[start..end].to_vec()).collect();

            // Run k-means
            let sub_centroids = kmeans(&sub_vectors, k, max_iters);
            centroids.extend(sub_centroids);
        }

        Ok(Self {
            m,
            k_bits,
            k,
            dims,
            sub_dims,
            centroids,
        })
    }

    /// Encode a vector to PQ codes.
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.m);
        for sub_q in 0..self.m {
            let start = sub_q * self.sub_dims;
            let end = start + self.sub_dims;
            let sub_vec = &vector[start..end];

            // Find nearest centroid
            let base = sub_q * self.k;
            let mut best_idx = 0u8;
            let mut best_dist = f32::MAX;
            for c in 0..self.k {
                let dist = l2_distance_sq(sub_vec, &self.centroids[base + c]);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = c as u8;
                }
            }
            codes.push(best_idx);
        }
        codes
    }

    /// Decode PQ codes back to an approximate vector.
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let mut vector = Vec::with_capacity(self.dims);
        for (sub_q, &code) in codes.iter().enumerate() {
            let base = sub_q * self.k;
            vector.extend_from_slice(&self.centroids[base + code as usize]);
        }
        vector
    }

    /// Compute asymmetric distance: exact query sub-vectors vs PQ codes.
    pub fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        let mut dist = 0.0_f32;
        for (sub_q, &code) in codes.iter().enumerate() {
            let start = sub_q * self.sub_dims;
            let end = start + self.sub_dims;
            let base = sub_q * self.k;
            dist += l2_distance_sq(&query[start..end], &self.centroids[base + code as usize]);
        }
        dist
    }

    /// Precompute distance table for a query (for faster batch search).
    pub fn precompute_distance_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let mut table = Vec::with_capacity(self.m);
        for sub_q in 0..self.m {
            let start = sub_q * self.sub_dims;
            let end = start + self.sub_dims;
            let sub_query = &query[start..end];
            let base = sub_q * self.k;
            let mut distances = Vec::with_capacity(self.k);
            for c in 0..self.k {
                distances.push(l2_distance_sq(sub_query, &self.centroids[base + c]));
            }
            table.push(distances);
        }
        table
    }

    /// Compute distance using precomputed distance table.
    pub fn distance_from_table(table: &[Vec<f32>], codes: &[u8]) -> f32 {
        let mut dist = 0.0_f32;
        for (sub_q, &code) in codes.iter().enumerate() {
            dist += table[sub_q][code as usize];
        }
        dist
    }

    /// Serialize codebook to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| TensorError::VectorError(e.to_string()))
    }

    /// Deserialize codebook from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).map_err(|e| TensorError::VectorError(e.to_string()))
    }
}

// ── K-means clustering ──────────────────────────────────────────────────────

/// Simple k-means clustering for PQ training.
fn kmeans(data: &[Vec<f32>], k: usize, max_iters: usize) -> Vec<Vec<f32>> {
    if data.is_empty() || k == 0 {
        return Vec::new();
    }

    let n = data.len();
    let d = data[0].len();
    let actual_k = k.min(n);

    // Initialize centroids by evenly spacing through data
    let mut centroids: Vec<Vec<f32>> = (0..actual_k)
        .map(|i| data[i * n / actual_k].clone())
        .collect();

    let mut assignments = vec![0usize; n];

    for _ in 0..max_iters {
        let mut changed = false;

        // Assignment step
        for (i, point) in data.iter().enumerate() {
            let mut best = 0;
            let mut best_dist = l2_distance_sq(point, &centroids[0]);
            for (j, centroid) in centroids.iter().enumerate().skip(1) {
                let dist = l2_distance_sq(point, centroid);
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

        // Update step
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

    // Pad to k centroids if we had fewer data points
    while centroids.len() < k {
        centroids.push(vec![0.0; d]);
    }

    centroids
}

/// Squared L2 distance between two vectors.
fn l2_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_roundtrip() {
        let values = vec![1.0_f32, -0.5, 0.0, 3.14];
        let encoded = quantize_f16(&values);
        let decoded = dequantize_f16(&encoded);
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 0.01, "f16 roundtrip: {a} != {b}");
        }
    }

    #[test]
    fn test_int8_roundtrip() {
        let values = vec![1.0_f32, -0.5, 0.0, 0.75];
        let encoded = quantize_int8(&values);
        let decoded = dequantize_int8(&encoded);
        // INT8 has lower precision
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 0.1, "int8 roundtrip: {a} != {b}");
        }
    }

    #[test]
    fn test_pq_encode_decode() {
        let dims = 8;
        let m = 4;
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| (0..dims).map(|j| (i * dims + j) as f32 / 100.0).collect())
            .collect();

        let codebook = PQCodebook::train(&vectors, m, 4, 10).unwrap();
        assert_eq!(codebook.m, m);
        assert_eq!(codebook.dims, dims);
        assert_eq!(codebook.sub_dims, 2);

        let codes = codebook.encode(&vectors[0]);
        assert_eq!(codes.len(), m);

        let decoded = codebook.decode(&codes);
        assert_eq!(decoded.len(), dims);
    }

    #[test]
    fn test_pq_distance_table() {
        let vectors: Vec<Vec<f32>> = (0..50).map(|i| vec![i as f32 / 10.0; 4]).collect();

        let codebook = PQCodebook::train(&vectors, 2, 4, 5).unwrap();
        let query = vec![1.0; 4];
        let codes = codebook.encode(&vectors[5]);

        let direct = codebook.asymmetric_distance(&query, &codes);
        let table = codebook.precompute_distance_table(&query);
        let from_table = PQCodebook::distance_from_table(&table, &codes);

        assert!((direct - from_table).abs() < 1e-6);
    }
}
