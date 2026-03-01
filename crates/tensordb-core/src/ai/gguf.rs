//! GGUF v3 file parser — zero-copy tensor access via mmap.
//!
//! Parses the GGUF binary format (magic, version, tensor count, metadata KV pairs),
//! builds a tensor name→offset index, and provides dequantization helpers for Q8_0 and F16.

use std::collections::HashMap;
use std::path::Path;

use memmap2::Mmap;

use crate::error::{Result, TensorError};

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as little-endian u32 (bytes: G G U F)

/// Quantization / data type of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufDtype {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
}

impl GgufDtype {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(GgufDtype::F32),
            1 => Ok(GgufDtype::F16),
            2 => Ok(GgufDtype::Q4_0),
            3 => Ok(GgufDtype::Q4_1),
            6 => Ok(GgufDtype::Q5_0),
            7 => Ok(GgufDtype::Q5_1),
            8 => Ok(GgufDtype::Q8_0),
            9 => Ok(GgufDtype::Q8_1),
            10 => Ok(GgufDtype::Q2K),
            11 => Ok(GgufDtype::Q3K),
            12 => Ok(GgufDtype::Q4K),
            13 => Ok(GgufDtype::Q5K),
            14 => Ok(GgufDtype::Q6K),
            15 => Ok(GgufDtype::Q8K),
            16 => Ok(GgufDtype::IQ2XXS),
            17 => Ok(GgufDtype::IQ2XS),
            18 => Ok(GgufDtype::IQ3XXS),
            19 => Ok(GgufDtype::IQ1S),
            20 => Ok(GgufDtype::IQ4NL),
            21 => Ok(GgufDtype::IQ3S),
            22 => Ok(GgufDtype::IQ2S),
            23 => Ok(GgufDtype::IQ4XS),
            24 => Ok(GgufDtype::I8),
            25 => Ok(GgufDtype::I16),
            26 => Ok(GgufDtype::I32),
            27 => Ok(GgufDtype::I64),
            28 => Ok(GgufDtype::F64),
            29 => Ok(GgufDtype::IQ1M),
            _ => Err(TensorError::LlmError(format!("unknown GGUF dtype: {v}"))),
        }
    }

    /// Bytes per element for fixed-size types; for block-quantized types
    /// this is the block size in bytes.
    pub fn type_size(&self) -> usize {
        match self {
            GgufDtype::F32 => 4,
            GgufDtype::F16 => 2,
            GgufDtype::Q4_0 => 18, // 2 bytes scale + 16 bytes data (32 values)
            GgufDtype::Q4_1 => 20, // 2+2 bytes scale/min + 16 bytes data
            GgufDtype::Q5_0 => 22, // 2 bytes scale + 4 bytes high bits + 16 bytes data
            GgufDtype::Q5_1 => 24,
            GgufDtype::Q8_0 => 34, // 2 bytes scale + 32 bytes data
            GgufDtype::Q8_1 => 40,
            GgufDtype::I8 => 1,
            GgufDtype::I16 => 2,
            GgufDtype::I32 => 4,
            GgufDtype::I64 => 8,
            GgufDtype::F64 => 8,
            _ => 0, // K-quant and IQ types not yet supported
        }
    }

    /// Number of elements per quantization block.
    pub fn block_size(&self) -> usize {
        match self {
            GgufDtype::F32 | GgufDtype::F16 | GgufDtype::F64 => 1,
            GgufDtype::I8 | GgufDtype::I16 | GgufDtype::I32 | GgufDtype::I64 => 1,
            GgufDtype::Q4_0 | GgufDtype::Q4_1 => 32,
            GgufDtype::Q5_0 | GgufDtype::Q5_1 => 32,
            GgufDtype::Q8_0 | GgufDtype::Q8_1 => 32,
            _ => 256, // K-quant block sizes
        }
    }
}

/// Metadata value types in GGUF.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::U32(v) => Some(*v),
            GgufValue::I32(v) if *v >= 0 => Some(*v as u32),
            GgufValue::U64(v) if *v <= u32::MAX as u64 => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U64(v) => Some(*v),
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::I32(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::F32(v) => Some(*v),
            GgufValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            GgufValue::Array(a) => Some(a.as_slice()),
            _ => None,
        }
    }
}

/// Information about a single tensor stored in the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: GgufDtype,
    pub shape: Vec<usize>,
    pub offset: usize, // byte offset from tensor data start
}

impl TensorInfo {
    /// Total number of elements in this tensor.
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }

    /// Total bytes occupied by this tensor's data.
    pub fn n_bytes(&self) -> usize {
        let n = self.n_elements();
        let bs = self.dtype.block_size();
        let ts = self.dtype.type_size();
        (n / bs) * ts
    }
}

/// A parsed GGUF file with mmap-backed zero-copy tensor access.
pub struct GgufFile {
    mmap: Mmap,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<TensorInfo>,
    tensor_index: HashMap<String, usize>, // name → index into tensors
    tensor_data_offset: usize,            // byte offset where tensor data begins
}

impl GgufFile {
    /// Open and parse a GGUF file. The file is memory-mapped for zero-copy tensor access.
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(|e| {
            TensorError::LlmError(format!("failed to open GGUF file {}: {e}", path.display()))
        })?;

        // Safety: We treat the mmap as read-only. The file should not be modified
        // while we have it mapped.
        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| TensorError::LlmError(format!("failed to mmap GGUF file: {e}")))?
        };

        let mut cursor = Cursor::new(&mmap);

        // Parse header
        let magic = cursor.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(TensorError::LlmError(format!(
                "invalid GGUF magic: 0x{magic:08x} (expected 0x{GGUF_MAGIC:08x})"
            )));
        }

        let version = cursor.read_u32()?;
        if !(2..=3).contains(&version) {
            return Err(TensorError::LlmError(format!(
                "unsupported GGUF version: {version} (expected 2 or 3)"
            )));
        }

        let tensor_count = cursor.read_u64()? as usize;
        let metadata_kv_count = cursor.read_u64()? as usize;

        // Parse metadata
        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = cursor.read_gguf_string()?;
            let value = cursor.read_gguf_value()?;
            metadata.insert(key, value);
        }

        // Parse tensor info entries
        let mut tensors = Vec::with_capacity(tensor_count);
        let mut tensor_index = HashMap::with_capacity(tensor_count);
        for i in 0..tensor_count {
            let name = cursor.read_gguf_string()?;
            let n_dims = cursor.read_u32()? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(cursor.read_u64()? as usize);
            }
            let dtype = GgufDtype::from_u32(cursor.read_u32()?)?;
            let offset = cursor.read_u64()? as usize;

            tensor_index.insert(name.clone(), i);
            tensors.push(TensorInfo {
                name,
                dtype,
                shape,
                offset,
            });
        }

        // Tensor data starts at the next alignment boundary (32 bytes) after the header
        let header_end = cursor.pos;
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as usize;
        let tensor_data_offset = header_end.div_ceil(alignment) * alignment;

        Ok(Self {
            mmap,
            metadata,
            tensors,
            tensor_index,
            tensor_data_offset,
        })
    }

    /// Get raw tensor data bytes (zero-copy slice into the mmap).
    pub fn tensor_data(&self, name: &str) -> Result<&[u8]> {
        let idx = self
            .tensor_index
            .get(name)
            .ok_or_else(|| TensorError::LlmError(format!("tensor not found: {name}")))?;
        let info = &self.tensors[*idx];
        let start = self.tensor_data_offset + info.offset;
        let end = start + info.n_bytes();
        if end > self.mmap.len() {
            return Err(TensorError::LlmError(format!(
                "tensor {name} extends beyond file (offset {start}..{end}, file len {})",
                self.mmap.len()
            )));
        }
        Ok(&self.mmap[start..end])
    }

    /// Get tensor info by name.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index.get(name).map(|idx| &self.tensors[*idx])
    }

    /// Get a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }
}

/// Dequantize Q8_0 block data to f32.
///
/// Q8_0 format: each block is 34 bytes — 1 f16 scale factor (2 bytes) + 32 int8 values.
/// Each output f32 = scale * int8_value.
pub fn dequant_q8_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34; // 2 (f16 scale) + 32 (int8 values)

    let n_blocks = n_elements / BLOCK_SIZE;
    let mut out = Vec::with_capacity(n_elements);

    for i in 0..n_blocks {
        let block = &data[i * BLOCK_BYTES..];
        let scale_bits = u16::from_le_bytes([block[0], block[1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();

        for j in 0..BLOCK_SIZE {
            let val = block[2 + j] as i8;
            out.push(scale * val as f32);
        }
    }

    out
}

/// Dequantize Q4_0 block data to f32.
///
/// Q4_0 format: each block is 18 bytes — 1 f16 scale factor (2 bytes) + 16 bytes of packed 4-bit values (32 values).
/// Each nibble is a signed 4-bit integer (range -8..7), value = scale * (nibble - 8).
pub fn dequant_q4_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18; // 2 (f16 scale) + 16 (packed nibbles)

    let n_blocks = n_elements / BLOCK_SIZE;
    let mut out = Vec::with_capacity(n_elements);

    for i in 0..n_blocks {
        let block = &data[i * BLOCK_BYTES..];
        let scale_bits = u16::from_le_bytes([block[0], block[1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();

        for j in 0..16 {
            let byte = block[2 + j];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            out.push(scale * lo as f32);
            out.push(scale * hi as f32);
        }
    }

    out
}

/// Convert f16 data to f32.
pub fn dequant_f16(data: &[u8], n_elements: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_elements);
    for i in 0..n_elements {
        let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        out.push(half::f16::from_bits(bits).to_f32());
    }
    out
}

/// Read f32 data directly (no conversion needed, just reinterpret bytes).
pub fn read_f32(data: &[u8], n_elements: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_elements);
    for i in 0..n_elements {
        let start = i * 4;
        let bits = u32::from_le_bytes([
            data[start],
            data[start + 1],
            data[start + 2],
            data[start + 3],
        ]);
        out.push(f32::from_bits(bits));
    }
    out
}

/// Dequantize tensor data to f32 based on its dtype.
pub fn dequant_tensor(data: &[u8], dtype: GgufDtype, n_elements: usize) -> Result<Vec<f32>> {
    match dtype {
        GgufDtype::F32 => Ok(read_f32(data, n_elements)),
        GgufDtype::F16 => Ok(dequant_f16(data, n_elements)),
        GgufDtype::Q8_0 => Ok(dequant_q8_0(data, n_elements)),
        GgufDtype::Q4_0 => Ok(dequant_q4_0(data, n_elements)),
        _ => Err(TensorError::LlmError(format!(
            "unsupported quantization format for dequantization: {:?}",
            dtype
        ))),
    }
}

// ── Internal cursor for parsing binary data ──────────────────────────────

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(TensorError::LlmError(format!(
                "GGUF parse: unexpected EOF at offset {} (need {n} bytes, have {})",
                self.pos,
                self.remaining()
            )));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8> {
        let b = self.read_bytes(1)?;
        Ok(b[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_i16(&mut self) -> Result<i16> {
        let b = self.read_bytes(2)?;
        Ok(i16::from_le_bytes([b[0], b[1]]))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i32(&mut self) -> Result<i32> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64> {
        let b = self.read_bytes(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_f32(&mut self) -> Result<f32> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f64(&mut self) -> Result<f64> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    fn read_gguf_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| TensorError::LlmError(format!("GGUF string is not valid UTF-8: {e}")))
    }

    fn read_gguf_value(&mut self) -> Result<GgufValue> {
        let value_type = self.read_u32()?;
        self.read_gguf_typed_value(value_type)
    }

    fn read_gguf_typed_value(&mut self, value_type: u32) -> Result<GgufValue> {
        match value_type {
            0 => Ok(GgufValue::U8(self.read_u8()?)),
            1 => Ok(GgufValue::I8(self.read_i8()?)),
            2 => Ok(GgufValue::U16(self.read_u16()?)),
            3 => Ok(GgufValue::I16(self.read_i16()?)),
            4 => Ok(GgufValue::U32(self.read_u32()?)),
            5 => Ok(GgufValue::I32(self.read_i32()?)),
            6 => Ok(GgufValue::F32(self.read_f32()?)),
            7 => Ok(GgufValue::Bool(self.read_bool()?)),
            8 => Ok(GgufValue::String(self.read_gguf_string()?)),
            9 => {
                // Array: element_type (u32) + count (u64) + elements
                let elem_type = self.read_u32()?;
                let count = self.read_u64()? as usize;
                let mut arr = Vec::with_capacity(count.min(1_000_000));
                for _ in 0..count {
                    arr.push(self.read_gguf_typed_value(elem_type)?);
                }
                Ok(GgufValue::Array(arr))
            }
            10 => Ok(GgufValue::U64(self.read_u64()?)),
            11 => Ok(GgufValue::I64(self.read_i64()?)),
            12 => Ok(GgufValue::F64(self.read_f64()?)),
            _ => Err(TensorError::LlmError(format!(
                "unknown GGUF value type: {value_type}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequant_q8_0_basic() {
        // One block: scale=1.0 (f16), values 0..31
        let scale_bits = half::f16::from_f32(1.0).to_bits();
        let mut block = vec![scale_bits as u8, (scale_bits >> 8) as u8];
        for i in 0i8..32 {
            block.push(i as u8);
        }
        let result = dequant_q8_0(&block, 32);
        assert_eq!(result.len(), 32);
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - i as f32).abs() < 0.01,
                "element {i}: expected {i}, got {val}"
            );
        }
    }

    #[test]
    fn dequant_q8_0_scaled() {
        // One block: scale=2.0, all values = 1
        let scale_bits = half::f16::from_f32(2.0).to_bits();
        let mut block = vec![scale_bits as u8, (scale_bits >> 8) as u8];
        block.extend_from_slice(&[1u8; 32]);
        let result = dequant_q8_0(&block, 32);
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - 2.0).abs() < 0.01,
                "element {i}: expected 2.0, got {val}"
            );
        }
    }

    #[test]
    fn dequant_f16_basic() {
        let values: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5];
        let mut data = Vec::new();
        for &v in &values {
            let bits = half::f16::from_f32(v).to_bits();
            data.push(bits as u8);
            data.push((bits >> 8) as u8);
        }
        let result = dequant_f16(&data, 4);
        assert_eq!(result.len(), 4);
        for (i, (&expected, &got)) in values.iter().zip(result.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 0.01,
                "element {i}: expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn read_f32_basic() {
        let values: Vec<f32> = vec![1.0, -2.5, 0.0, 42.0];
        let mut data = Vec::new();
        for &v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let result = read_f32(&data, 4);
        assert_eq!(result, values);
    }

    #[test]
    fn dequant_q4_0_basic() {
        // One block: scale=1.0, all nibbles = 8 (meaning value = 0 after subtracting 8)
        let scale_bits = half::f16::from_f32(1.0).to_bits();
        let mut block = vec![scale_bits as u8, (scale_bits >> 8) as u8];
        // 16 bytes, each byte = 0x88 → lo nibble=8, hi nibble=8 → both map to 0
        block.extend_from_slice(&[0x88u8; 16]);
        let result = dequant_q4_0(&block, 32);
        assert_eq!(result.len(), 32);
        for (i, &val) in result.iter().enumerate() {
            assert!(val.abs() < 0.01, "element {i}: expected 0.0, got {val}");
        }
    }

    #[test]
    fn tensor_info_n_bytes() {
        let info = TensorInfo {
            name: "test".to_string(),
            dtype: GgufDtype::Q8_0,
            shape: vec![64],
            offset: 0,
        };
        // 64 elements / 32 per block = 2 blocks * 34 bytes = 68
        assert_eq!(info.n_bytes(), 68);
    }

    #[test]
    fn tensor_info_f32_n_bytes() {
        let info = TensorInfo {
            name: "test".to_string(),
            dtype: GgufDtype::F32,
            shape: vec![10, 20],
            offset: 0,
        };
        assert_eq!(info.n_elements(), 200);
        assert_eq!(info.n_bytes(), 800);
    }
}
