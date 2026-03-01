//! Vector persistence: LSM key encoding and serialization for vector data and indexes.
//!
//! Storage layout:
//! ```text
//! __vec/{table}/{column}/{pk}                    -> VectorRecord (encoding + raw bytes)
//! __vec_idx/{table}/{column}/hnsw/graph          -> serialized HNSW graph
//! __vec_idx/{table}/{column}/ivf/centroids       -> IVF centroid vectors
//! __vec_idx/{table}/{column}/ivf/list/{list_id}  -> posting list
//! __vec_idx/{table}/{column}/pq/codebook         -> PQ sub-quantizers
//! __meta/vector_index/{table}/{column}           -> VectorIndexMetadata JSON
//! ```

use serde::{Deserialize, Serialize};

use crate::error::{Result, TensorError};
use crate::sql::parser::VectorIndexType;

// ── Key builders ────────────────────────────────────────────────────────────

/// Key for storing a single vector: `__vec/{table}/{column}/{pk}`
pub fn vector_data_key(table: &str, column: &str, pk: &str) -> String {
    format!("__vec/{table}/{column}/{pk}")
}

/// Prefix for scanning all vectors of a column: `__vec/{table}/{column}/`
pub fn vector_data_prefix(table: &str, column: &str) -> String {
    format!("__vec/{table}/{column}/")
}

/// Key for HNSW graph data: `__vec_idx/{table}/{column}/hnsw/graph`
pub fn vector_hnsw_graph_key(table: &str, column: &str) -> String {
    format!("__vec_idx/{table}/{column}/hnsw/graph")
}

/// Key for IVF centroids: `__vec_idx/{table}/{column}/ivf/centroids`
pub fn vector_ivf_centroids_key(table: &str, column: &str) -> String {
    format!("__vec_idx/{table}/{column}/ivf/centroids")
}

/// Key for IVF posting list: `__vec_idx/{table}/{column}/ivf/list/{list_id}`
pub fn vector_ivf_list_key(table: &str, column: &str, list_id: u32) -> String {
    format!("__vec_idx/{table}/{column}/ivf/list/{list_id}")
}

/// Key for PQ codebook: `__vec_idx/{table}/{column}/pq/codebook`
pub fn vector_pq_codebook_key(table: &str, column: &str) -> String {
    format!("__vec_idx/{table}/{column}/pq/codebook")
}

/// Metadata key: `__meta/vector_index/{table}/{column}`
pub fn vector_index_meta_key(table: &str, column: &str) -> String {
    format!("__meta/vector_index/{table}/{column}")
}

/// Prefix for scanning all vector index metadata: `__meta/vector_index/`
pub fn vector_index_meta_prefix() -> &'static str {
    "__meta/vector_index/"
}

// ── Vector encoding ─────────────────────────────────────────────────────────

/// How vector components are stored on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorEncoding {
    /// 32-bit IEEE 754 floats (4 bytes per dimension).
    Float32,
    /// 16-bit IEEE 754 half-precision floats (2 bytes per dimension).
    Float16,
    /// Signed 8-bit integers, quantized from original floats (1 byte per dimension).
    Int8,
}

impl VectorEncoding {
    pub fn from_str_name(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "f32" | "float32" => Some(Self::Float32),
            "f16" | "float16" => Some(Self::Float16),
            "int8" | "i8" => Some(Self::Int8),
            _ => None,
        }
    }

    /// Bytes per dimension for this encoding.
    pub fn bytes_per_dim(&self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float16 => 2,
            Self::Int8 => 1,
        }
    }
}

// ── On-disk record ──────────────────────────────────────────────────────────

/// Binary record stored at `__vec/{table}/{column}/{pk}`.
///
/// Layout: `[encoding: u8] [dims: u16 LE] [raw bytes: dims * bytes_per_dim]`
///
/// Total size = 3 + dims * bytes_per_dim.
#[derive(Debug, Clone)]
pub struct VectorRecord {
    pub encoding: VectorEncoding,
    pub dims: u16,
    pub data: Vec<u8>,
}

impl VectorRecord {
    /// Create a Float32 record from an f32 slice.
    pub fn from_f32(vec: &[f32]) -> Self {
        let mut data = Vec::with_capacity(vec.len() * 4);
        for &v in vec {
            data.extend_from_slice(&v.to_le_bytes());
        }
        Self {
            encoding: VectorEncoding::Float32,
            dims: vec.len() as u16,
            data,
        }
    }

    /// Decode to f32 values regardless of on-disk encoding.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self.encoding {
            VectorEncoding::Float32 => self
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            VectorEncoding::Float16 => {
                use half::f16;
                self.data
                    .chunks_exact(2)
                    .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect()
            }
            VectorEncoding::Int8 => {
                // INT8 stores signed values in [-127, 127] normalized to [-1.0, 1.0]
                self.data
                    .iter()
                    .map(|&b| (b as i8) as f32 / 127.0)
                    .collect()
            }
        }
    }

    /// Serialize to bytes for LSM storage.
    pub fn to_bytes(&self) -> Vec<u8> {
        let enc_byte: u8 = match self.encoding {
            VectorEncoding::Float32 => 0,
            VectorEncoding::Float16 => 1,
            VectorEncoding::Int8 => 2,
        };
        let mut out = Vec::with_capacity(3 + self.data.len());
        out.push(enc_byte);
        out.extend_from_slice(&self.dims.to_le_bytes());
        out.extend_from_slice(&self.data);
        out
    }

    /// Deserialize from LSM-stored bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 3 {
            return Err(TensorError::VectorError(
                "vector record too short".to_string(),
            ));
        }
        let encoding = match bytes[0] {
            0 => VectorEncoding::Float32,
            1 => VectorEncoding::Float16,
            2 => VectorEncoding::Int8,
            other => {
                return Err(TensorError::VectorError(format!(
                    "unknown vector encoding: {other}"
                )))
            }
        };
        let dims = u16::from_le_bytes([bytes[1], bytes[2]]);
        let expected_data_len = dims as usize * encoding.bytes_per_dim();
        if bytes.len() < 3 + expected_data_len {
            return Err(TensorError::VectorError(format!(
                "vector record data too short: expected {} bytes, got {}",
                3 + expected_data_len,
                bytes.len()
            )));
        }
        let data = bytes[3..3 + expected_data_len].to_vec();
        Ok(Self {
            encoding,
            dims,
            data,
        })
    }
}

// ── Index metadata ──────────────────────────────────────────────────────────

/// Persisted metadata for a vector index, stored as JSON under
/// `__meta/vector_index/{table}/{column}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndexMetadata {
    pub index_name: String,
    pub table: String,
    pub column: String,
    pub dims: u16,
    pub index_type: String, // "hnsw" or "ivf_pq"
    pub metric: String,     // "euclidean", "cosine", "dot_product"
    pub params: Vec<(String, String)>,
}

impl VectorIndexMetadata {
    pub fn new(
        index_name: String,
        table: String,
        column: String,
        dims: u16,
        index_type: VectorIndexType,
        metric: String,
        params: Vec<(String, String)>,
    ) -> Self {
        let type_str = match index_type {
            VectorIndexType::Hnsw => "hnsw",
            VectorIndexType::IvfPq => "ivf_pq",
        };
        Self {
            index_name,
            table,
            column,
            dims,
            index_type: type_str.to_string(),
            metric,
            params,
        }
    }
}

// ── Parse vector literal ────────────────────────────────────────────────────

/// Parse a vector literal string like "[0.1, 0.2, 0.3]" into f32 values.
pub fn parse_vector_literal(s: &str) -> Result<Vec<f32>> {
    let trimmed = s.trim();
    let inner = if trimmed.starts_with('[') && trimmed.ends_with(']') {
        &trimmed[1..trimmed.len() - 1]
    } else {
        trimmed
    };

    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }

    inner
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<f32>()
                .map_err(|e| TensorError::VectorError(format!("invalid vector component: {e}")))
        })
        .collect()
}

/// Format an f32 vector as a bracket-delimited string.
pub fn format_vector(vec: &[f32]) -> String {
    let parts: Vec<String> = vec.iter().map(|v| format!("{v}")).collect();
    format!("[{}]", parts.join(", "))
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_record_f32_roundtrip() {
        let vec = vec![1.0_f32, 2.0, 3.0, -0.5];
        let record = VectorRecord::from_f32(&vec);
        let bytes = record.to_bytes();
        let decoded = VectorRecord::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.dims, 4);
        assert_eq!(decoded.to_f32_vec(), vec);
    }

    #[test]
    fn test_vector_record_too_short() {
        assert!(VectorRecord::from_bytes(&[0, 1]).is_err());
    }

    #[test]
    fn test_parse_vector_literal() {
        let v = parse_vector_literal("[1.0, 2.0, 3.0]").unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0]);

        let v2 = parse_vector_literal("0.5, -0.5").unwrap();
        assert_eq!(v2, vec![0.5, -0.5]);

        let empty = parse_vector_literal("[]").unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_format_vector() {
        assert_eq!(format_vector(&[1.0, 2.5, 3.0]), "[1, 2.5, 3]");
    }

    #[test]
    fn test_key_builders() {
        assert_eq!(
            vector_data_key("docs", "embedding", "pk1"),
            "__vec/docs/embedding/pk1"
        );
        assert_eq!(
            vector_data_prefix("docs", "embedding"),
            "__vec/docs/embedding/"
        );
        assert_eq!(
            vector_hnsw_graph_key("docs", "embedding"),
            "__vec_idx/docs/embedding/hnsw/graph"
        );
        assert_eq!(
            vector_index_meta_key("docs", "embedding"),
            "__meta/vector_index/docs/embedding"
        );
    }

    #[test]
    fn test_vector_encoding() {
        assert_eq!(
            VectorEncoding::from_str_name("f32"),
            Some(VectorEncoding::Float32)
        );
        assert_eq!(
            VectorEncoding::from_str_name("f16"),
            Some(VectorEncoding::Float16)
        );
        assert_eq!(
            VectorEncoding::from_str_name("int8"),
            Some(VectorEncoding::Int8)
        );
        assert_eq!(VectorEncoding::from_str_name("unknown"), None);
    }
}
