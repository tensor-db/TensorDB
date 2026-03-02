//! Qwen2/3 transformer runtime — pure-Rust forward pass with KV cache.
//!
//! Implements the full Qwen2 architecture: RMSNorm, Grouped-Query Attention (GQA)
//! with RoPE, SwiGLU FFN, and a KV cache for efficient autoregressive generation.

use crate::error::{Result, TensorError};

#[cfg(feature = "llm")]
use rayon::prelude::*;

use super::gguf::{dequant_tensor, GgufDtype, GgufFile};

/// Minimum number of output rows before switching to parallel execution.
/// Below this threshold the rayon scheduling overhead exceeds the benefit.
const MIN_PARALLEL_ROWS: usize = 128;

/// Model hyperparameters extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub intermediate_dim: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub head_dim: usize,
    pub context_length: usize,
    /// Linear RoPE scaling factor (1.0 = no scaling, >1.0 = extends context).
    pub rope_scale: f32,
    /// Precomputed RoPE frequencies: `1.0 / (theta^(2i/head_dim) * rope_scale)` for `i in 0..head_dim/2`.
    pub rope_freqs: Vec<f32>,
}

impl ModelConfig {
    /// Extract model config from GGUF metadata.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let get_u32 = |key: &str| -> Result<u32> {
            gguf.get_metadata(key)
                .and_then(|v| v.as_u32())
                .ok_or_else(|| TensorError::LlmError(format!("missing GGUF metadata: {key}")))
        };

        let hidden_dim = get_u32("llama.embedding_length")? as usize;
        let n_layers = get_u32("llama.block_count")? as usize;
        let n_heads = get_u32("llama.attention.head_count")? as usize;
        let n_kv_heads = get_u32("llama.attention.head_count_kv")? as usize;
        let intermediate_dim = get_u32("llama.feed_forward_length")? as usize;
        let vocab_size = gguf
            .get_metadata("llama.vocab_size")
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                // Fall back to counting tokens in vocabulary
                gguf.get_metadata("tokenizer.ggml.tokens")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len())
                    .unwrap_or(151936) // Qwen default
            });

        let rope_theta = gguf
            .get_metadata("llama.rope.freq_base")
            .and_then(|v| v.as_f32())
            .unwrap_or(1_000_000.0);

        let rms_norm_eps = gguf
            .get_metadata("llama.attention.layer_norm_rms_epsilon")
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-6);

        let context_length = gguf
            .get_metadata("llama.context_length")
            .and_then(|v| v.as_u32())
            .unwrap_or(32768) as usize;

        let head_dim = hidden_dim / n_heads;

        let rope_scale = gguf
            .get_metadata("llama.rope.scale_linear")
            .and_then(|v| v.as_f32())
            .unwrap_or(1.0);

        let rope_freqs: Vec<f32> = (0..head_dim / 2)
            .map(|i| 1.0 / (rope_theta.powf((2 * i) as f32 / head_dim as f32) * rope_scale))
            .collect();

        Ok(Self {
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            intermediate_dim,
            vocab_size,
            rope_theta,
            rms_norm_eps,
            head_dim,
            context_length,
            rope_scale,
            rope_freqs,
        })
    }
}

/// RMSNorm weight vector.
struct RmsNormWeight {
    weight: Vec<f32>,
}

/// A single transformer layer's weights.
struct TransformerLayer {
    // Attention
    attn_norm: RmsNormWeight,
    q_proj: WeightMatrix,
    k_proj: WeightMatrix,
    v_proj: WeightMatrix,
    o_proj: WeightMatrix,
    // Attention biases (Qwen2 uses biases for QKV)
    q_bias: Option<Vec<f32>>,
    k_bias: Option<Vec<f32>>,
    v_bias: Option<Vec<f32>>,
    // FFN
    ffn_norm: RmsNormWeight,
    gate_proj: WeightMatrix,
    up_proj: WeightMatrix,
    down_proj: WeightMatrix,
}

/// Weight matrix stored as raw GGUF data with lazy dequantization.
pub struct WeightMatrix {
    data: Vec<u8>,
    dtype: GgufDtype,
    rows: usize,
    cols: usize,
}

#[allow(clippy::needless_range_loop)]
impl WeightMatrix {
    /// Create a WeightMatrix from raw quantized data. Used for benchmarking.
    #[doc(hidden)]
    pub fn from_raw(data: Vec<u8>, dtype: GgufDtype, rows: usize, cols: usize) -> Self {
        Self {
            data,
            dtype,
            rows,
            cols,
        }
    }

    fn from_gguf(gguf: &GgufFile, name: &str) -> Result<Self> {
        let info = gguf
            .tensor_info(name)
            .ok_or_else(|| TensorError::LlmError(format!("tensor not found: {name}")))?;
        let data = gguf.tensor_data(name)?.to_vec();
        let dtype = info.dtype;

        // GGUF tensors: shape is [cols, rows] (column-major convention)
        let (rows, cols) = match info.shape.len() {
            1 => (1, info.shape[0]),
            2 => (info.shape[1], info.shape[0]),
            _ => {
                return Err(TensorError::LlmError(format!(
                    "expected 1D or 2D tensor for {name}, got {:?}",
                    info.shape
                )))
            }
        };

        Ok(Self {
            data,
            dtype,
            rows,
            cols,
        })
    }

    /// Matrix-vector multiply: output[row] = sum(weight[row][col] * input[col])
    /// Weight is stored row-major after dequantization.
    pub fn matvec(&self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.cols);
        debug_assert_eq!(output.len(), self.rows);

        match self.dtype {
            GgufDtype::Q8_0 => self.matvec_q8_0(input, output),
            GgufDtype::Q4_0 => self.matvec_q4_0(input, output),
            GgufDtype::Q6K => self.matvec_q6_k(input, output),
            GgufDtype::F16 => self.matvec_f16(input, output),
            GgufDtype::F32 => self.matvec_f32(input, output),
            _ => {
                // Fallback: dequantize entire matrix (expensive but correct)
                if let Ok(weights) = dequant_tensor(&self.data, self.dtype, self.rows * self.cols) {
                    let cols = self.cols;
                    #[cfg(feature = "llm")]
                    if self.rows >= MIN_PARALLEL_ROWS {
                        output.par_iter_mut().enumerate().for_each(|(r, out)| {
                            let mut sum = 0.0f32;
                            let row_start = r * cols;
                            for c in 0..cols {
                                sum += weights[row_start + c] * input[c];
                            }
                            *out = sum;
                        });
                        return;
                    }
                    for r in 0..self.rows {
                        let mut sum = 0.0f32;
                        let row_start = r * cols;
                        for c in 0..cols {
                            sum += weights[row_start + c] * input[c];
                        }
                        output[r] = sum;
                    }
                }
            }
        }
    }

    /// Matrix-vector multiply for selected rows only:
    /// `output[i] = dot(weight[row_indices[i]], input)` for each index in `row_indices`.
    pub fn matvec_rows(&self, input: &[f32], output: &mut [f32], row_indices: &[u32]) {
        debug_assert_eq!(input.len(), self.cols);
        debug_assert_eq!(output.len(), row_indices.len());

        match self.dtype {
            GgufDtype::Q8_0 => self.matvec_rows_q8_0(input, output, row_indices),
            GgufDtype::Q4_0 => self.matvec_rows_q4_0(input, output, row_indices),
            GgufDtype::Q6K => self.matvec_rows_q6_k(input, output, row_indices),
            _ => {
                // Fallback: use full dequant for the selected rows
                if let Ok(weights) = dequant_tensor(&self.data, self.dtype, self.rows * self.cols) {
                    let cols = self.cols;
                    #[cfg(feature = "llm")]
                    if output.len() >= MIN_PARALLEL_ROWS {
                        output
                            .par_iter_mut()
                            .zip(row_indices.par_iter())
                            .for_each(|(out, &r)| {
                                let r = r as usize;
                                let mut sum = 0.0f32;
                                let row_start = r * cols;
                                for c in 0..cols {
                                    sum += weights[row_start + c] * input[c];
                                }
                                *out = sum;
                            });
                        return;
                    }
                    for (out_idx, &r) in row_indices.iter().enumerate() {
                        let r = r as usize;
                        let mut sum = 0.0f32;
                        let row_start = r * cols;
                        for c in 0..cols {
                            sum += weights[row_start + c] * input[c];
                        }
                        output[out_idx] = sum;
                    }
                }
            }
        }
    }

    /// Q8_0 matvec for selected rows (scalar).
    fn matvec_rows_q8_0(&self, input: &[f32], output: &mut [f32], row_indices: &[u32]) {
        const BLOCK_SIZE: usize = 32;
        const BLOCK_BYTES: usize = 34;
        let blocks_per_row = self.cols / BLOCK_SIZE;
        let data = &self.data;

        let compute_row = |r: u32, out: &mut f32| {
            let r = r as usize;
            let mut sum = 0.0f32;
            let row_offset = r * blocks_per_row * BLOCK_BYTES;
            for b in 0..blocks_per_row {
                let block = &data[row_offset + b * BLOCK_BYTES..];
                let scale = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
                let input_offset = b * BLOCK_SIZE;
                let mut block_sum = 0.0f32;
                for j in 0..BLOCK_SIZE {
                    let val = block[2 + j] as i8;
                    block_sum += val as f32 * input[input_offset + j];
                }
                sum += scale * block_sum;
            }
            *out = sum;
        };

        #[cfg(feature = "llm")]
        if output.len() >= MIN_PARALLEL_ROWS {
            output
                .par_iter_mut()
                .zip(row_indices.par_iter())
                .for_each(|(out, &r)| compute_row(r, out));
            return;
        }
        for (out_idx, &r) in row_indices.iter().enumerate() {
            compute_row(r, &mut output[out_idx]);
        }
    }

    /// Q4_0 matvec for selected rows (scalar).
    fn matvec_rows_q4_0(&self, input: &[f32], output: &mut [f32], row_indices: &[u32]) {
        const BLOCK_SIZE: usize = 32;
        const BLOCK_BYTES: usize = 18;
        let blocks_per_row = self.cols / BLOCK_SIZE;
        let data = &self.data;

        let compute_row = |r: u32, out: &mut f32| {
            let r = r as usize;
            let mut sum = 0.0f32;
            let row_offset = r * blocks_per_row * BLOCK_BYTES;
            for b in 0..blocks_per_row {
                let block = &data[row_offset + b * BLOCK_BYTES..];
                let scale = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
                let input_offset = b * BLOCK_SIZE;
                let mut block_sum = 0.0f32;
                for j in 0..16 {
                    let byte = block[2 + j];
                    let lo = (byte & 0x0F) as i32 - 8;
                    let hi = ((byte >> 4) & 0x0F) as i32 - 8;
                    block_sum += lo as f32 * input[input_offset + j * 2];
                    block_sum += hi as f32 * input[input_offset + j * 2 + 1];
                }
                sum += scale * block_sum;
            }
            *out = sum;
        };

        #[cfg(feature = "llm")]
        if output.len() >= MIN_PARALLEL_ROWS {
            output
                .par_iter_mut()
                .zip(row_indices.par_iter())
                .for_each(|(out, &r)| compute_row(r, out));
            return;
        }
        for (out_idx, &r) in row_indices.iter().enumerate() {
            compute_row(r, &mut output[out_idx]);
        }
    }

    /// Q8_0 matvec dispatch — delegates to kernels module for SIMD + parallelism.
    fn matvec_q8_0(&self, input: &[f32], output: &mut [f32]) {
        super::kernels::q8_0_matvec(&self.data, input, output, self.rows, self.cols);
    }

    /// Q4_0 matvec dispatch — delegates to kernels module for SIMD + parallelism.
    fn matvec_q4_0(&self, input: &[f32], output: &mut [f32]) {
        super::kernels::q4_0_matvec(&self.data, input, output, self.rows, self.cols);
    }

    /// Scalar Q6_K matvec — dequantize and dot-product per super-block.
    fn matvec_q6_k(&self, input: &[f32], output: &mut [f32]) {
        const BLOCK_SIZE: usize = 256;
        const BLOCK_BYTES: usize = 210;
        let blocks_per_row = self.cols / BLOCK_SIZE;
        let data = &self.data;

        let compute_row = |r: usize, out: &mut f32| {
            let mut sum = 0.0f32;
            let row_offset = r * blocks_per_row * BLOCK_BYTES;
            for b in 0..blocks_per_row {
                let block = &data[row_offset + b * BLOCK_BYTES..];
                let ql = &block[0..128];
                let qh = &block[128..192];
                let scales = &block[192..208];
                let d = half::f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();
                let input_offset = b * BLOCK_SIZE;
                for sub in 0..16 {
                    let sc = scales[sub] as i8 as f32;
                    let mut sub_sum = 0.0f32;
                    for j in 0..16 {
                        let idx = sub * 16 + j;
                        let ql_byte = ql[idx / 2];
                        let lo4 = if idx % 2 == 0 {
                            (ql_byte & 0x0F) as i32
                        } else {
                            ((ql_byte >> 4) & 0x0F) as i32
                        };
                        let qh_byte = qh[idx / 4];
                        let shift = (idx % 4) * 2;
                        let hi2 = ((qh_byte >> shift) & 0x03) as i32;
                        let q = ((hi2 << 4) | lo4) - 32;
                        sub_sum += q as f32 * input[input_offset + idx];
                    }
                    sum += d * sc * sub_sum;
                }
            }
            *out = sum;
        };

        #[cfg(feature = "llm")]
        if self.rows >= MIN_PARALLEL_ROWS {
            output
                .par_iter_mut()
                .enumerate()
                .for_each(|(r, out)| compute_row(r, out));
            return;
        }
        for r in 0..self.rows {
            compute_row(r, &mut output[r]);
        }
    }

    /// Q6_K matvec for selected rows (scalar).
    fn matvec_rows_q6_k(&self, input: &[f32], output: &mut [f32], row_indices: &[u32]) {
        const BLOCK_SIZE: usize = 256;
        const BLOCK_BYTES: usize = 210;
        let blocks_per_row = self.cols / BLOCK_SIZE;
        let data = &self.data;

        let compute_row = |r: u32, out: &mut f32| {
            let r = r as usize;
            let mut sum = 0.0f32;
            let row_offset = r * blocks_per_row * BLOCK_BYTES;
            for b in 0..blocks_per_row {
                let block = &data[row_offset + b * BLOCK_BYTES..];
                let ql = &block[0..128];
                let qh = &block[128..192];
                let scales = &block[192..208];
                let d = half::f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();
                let input_offset = b * BLOCK_SIZE;
                for sub in 0..16 {
                    let sc = scales[sub] as i8 as f32;
                    let mut sub_sum = 0.0f32;
                    for j in 0..16 {
                        let idx = sub * 16 + j;
                        let ql_byte = ql[idx / 2];
                        let lo4 = if idx % 2 == 0 {
                            (ql_byte & 0x0F) as i32
                        } else {
                            ((ql_byte >> 4) & 0x0F) as i32
                        };
                        let qh_byte = qh[idx / 4];
                        let shift = (idx % 4) * 2;
                        let hi2 = ((qh_byte >> shift) & 0x03) as i32;
                        let q = ((hi2 << 4) | lo4) - 32;
                        sub_sum += q as f32 * input[input_offset + idx];
                    }
                    sum += d * sc * sub_sum;
                }
            }
            *out = sum;
        };

        #[cfg(feature = "llm")]
        if output.len() >= MIN_PARALLEL_ROWS {
            output
                .par_iter_mut()
                .zip(row_indices.par_iter())
                .for_each(|(out, &r)| compute_row(r, out));
            return;
        }
        for (out_idx, &r) in row_indices.iter().enumerate() {
            compute_row(r, &mut output[out_idx]);
        }
    }

    /// F16 matvec.
    fn matvec_f16(&self, input: &[f32], output: &mut [f32]) {
        let cols = self.cols;
        let data = &self.data;

        let compute_row = |r: usize, out: &mut f32| {
            let mut sum = 0.0f32;
            let row_offset = r * cols * 2;
            for c in 0..cols {
                let bits = u16::from_le_bytes([
                    data[row_offset + c * 2],
                    data[row_offset + c * 2 + 1],
                ]);
                let w = half::f16::from_bits(bits).to_f32();
                sum += w * input[c];
            }
            *out = sum;
        };

        #[cfg(feature = "llm")]
        if self.rows >= MIN_PARALLEL_ROWS {
            output
                .par_iter_mut()
                .enumerate()
                .for_each(|(r, out)| compute_row(r, out));
            return;
        }
        for r in 0..self.rows {
            compute_row(r, &mut output[r]);
        }
    }

    /// F32 matvec.
    fn matvec_f32(&self, input: &[f32], output: &mut [f32]) {
        let cols = self.cols;
        let data = &self.data;

        let compute_row = |r: usize, out: &mut f32| {
            let mut sum = 0.0f32;
            let row_offset = r * cols * 4;
            for c in 0..cols {
                let start = row_offset + c * 4;
                let w = f32::from_le_bytes([
                    data[start],
                    data[start + 1],
                    data[start + 2],
                    data[start + 3],
                ]);
                sum += w * input[c];
            }
            *out = sum;
        };

        #[cfg(feature = "llm")]
        if self.rows >= MIN_PARALLEL_ROWS {
            output
                .par_iter_mut()
                .enumerate()
                .for_each(|(r, out)| compute_row(r, out));
            return;
        }
        for r in 0..self.rows {
            compute_row(r, &mut output[r]);
        }
    }
}

/// Pre-allocated scratch buffers for forward passes, avoiding per-token allocations.
pub struct ScratchBuffers {
    x: Vec<f32>,
    xb: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    attn_projected: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_down: Vec<f32>,
    pub logits: Vec<f32>,
    scores: Vec<f32>,
}

impl ScratchBuffers {
    /// Allocate scratch buffers sized for the given model config.
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            x: vec![0.0; config.hidden_dim],
            xb: vec![0.0; config.hidden_dim],
            q: vec![0.0; config.n_heads * config.head_dim],
            k: vec![0.0; config.n_kv_heads * config.head_dim],
            v: vec![0.0; config.n_kv_heads * config.head_dim],
            attn_out: vec![0.0; config.hidden_dim],
            attn_projected: vec![0.0; config.hidden_dim],
            ffn_gate: vec![0.0; config.intermediate_dim],
            ffn_up: vec![0.0; config.intermediate_dim],
            ffn_down: vec![0.0; config.hidden_dim],
            logits: vec![0.0; config.vocab_size],
            scores: vec![0.0; config.context_length],
        }
    }
}

/// Vocabulary-pruned LM head — computes logits only for SQL-compatible tokens.
pub struct ActiveVocab {
    /// Sorted SQL-compatible token IDs (~5K out of ~151K vocab).
    pub token_ids: Vec<u32>,
    /// Reduced logits buffer — one entry per active token.
    reduced_logits: Vec<f32>,
    /// Full-vocab logits buffer — filled with NEG_INFINITY except active positions.
    full_logits: Vec<f32>,
}

impl ActiveVocab {
    /// Build from a set of active token IDs and the full vocab size.
    pub fn new(token_ids: Vec<u32>, vocab_size: usize) -> Self {
        let n_active = token_ids.len();
        Self {
            token_ids,
            reduced_logits: vec![0.0; n_active],
            full_logits: vec![f32::NEG_INFINITY; vocab_size],
        }
    }

    /// Compute reduced logits using only the active rows of the LM head.
    pub fn compute_reduced(&mut self, lm_head: &WeightMatrix, hidden: &[f32]) {
        lm_head.matvec_rows(hidden, &mut self.reduced_logits, &self.token_ids);
    }

    /// Access the reduced logits buffer for external writes (benchmarking).
    #[doc(hidden)]
    pub fn reduced_logits_mut(&mut self) -> &mut [f32] {
        &mut self.reduced_logits
    }

    /// Scatter reduced logits into the full-vocab buffer and return it.
    /// Non-active positions are set to NEG_INFINITY.
    pub fn scatter_to_full(&mut self) -> &mut [f32] {
        self.full_logits.fill(f32::NEG_INFINITY);
        for (i, &token_id) in self.token_ids.iter().enumerate() {
            self.full_logits[token_id as usize] = self.reduced_logits[i];
        }
        &mut self.full_logits
    }
}

/// The full transformer model with all layer weights loaded.
pub struct TransformerModel {
    pub config: ModelConfig,
    embedding: Vec<f32>, // [vocab_size * hidden_dim], always f32
    layers: Vec<TransformerLayer>,
    final_norm: RmsNormWeight,
    output_weight: WeightMatrix, // lm_head / output projection
}

impl TransformerModel {
    /// Load model weights from a parsed GGUF file.
    pub fn from_gguf(gguf: &GgufFile, config: &ModelConfig) -> Result<Self> {
        // Load token embeddings — always dequantize to f32 for lookup
        let emb_info = gguf
            .tensor_info("token_embd.weight")
            .ok_or_else(|| TensorError::LlmError("missing token_embd.weight tensor".into()))?;
        let emb_data = gguf.tensor_data("token_embd.weight")?;
        let embedding = dequant_tensor(
            emb_data,
            emb_info.dtype,
            config.vocab_size * config.hidden_dim,
        )?;

        // Load layers
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let prefix = format!("blk.{i}");
            let layer = TransformerLayer {
                attn_norm: RmsNormWeight {
                    weight: load_1d_f32(gguf, &format!("{prefix}.attn_norm.weight"))?,
                },
                q_proj: WeightMatrix::from_gguf(gguf, &format!("{prefix}.attn_q.weight"))?,
                k_proj: WeightMatrix::from_gguf(gguf, &format!("{prefix}.attn_k.weight"))?,
                v_proj: WeightMatrix::from_gguf(gguf, &format!("{prefix}.attn_v.weight"))?,
                o_proj: WeightMatrix::from_gguf(gguf, &format!("{prefix}.attn_output.weight"))?,
                q_bias: load_optional_1d_f32(gguf, &format!("{prefix}.attn_q.bias")),
                k_bias: load_optional_1d_f32(gguf, &format!("{prefix}.attn_k.bias")),
                v_bias: load_optional_1d_f32(gguf, &format!("{prefix}.attn_v.bias")),
                ffn_norm: RmsNormWeight {
                    weight: load_1d_f32(gguf, &format!("{prefix}.ffn_norm.weight"))?,
                },
                gate_proj: WeightMatrix::from_gguf(gguf, &format!("{prefix}.ffn_gate.weight"))?,
                up_proj: WeightMatrix::from_gguf(gguf, &format!("{prefix}.ffn_up.weight"))?,
                down_proj: WeightMatrix::from_gguf(gguf, &format!("{prefix}.ffn_down.weight"))?,
            };
            layers.push(layer);
        }

        // Final norm
        let final_norm = RmsNormWeight {
            weight: load_1d_f32(gguf, "output_norm.weight")?,
        };

        // Output projection (lm_head). May be tied to embedding weights.
        let output_weight = if gguf.tensor_info("output.weight").is_some() {
            WeightMatrix::from_gguf(gguf, "output.weight")?
        } else {
            // Tied weights — use embedding tensor directly
            let info = gguf.tensor_info("token_embd.weight").unwrap();
            WeightMatrix {
                data: gguf.tensor_data("token_embd.weight")?.to_vec(),
                dtype: info.dtype,
                rows: config.vocab_size,
                cols: config.hidden_dim,
            }
        };

        Ok(Self {
            config: config.clone(),
            embedding,
            layers,
            final_norm,
            output_weight,
        })
    }

    /// Run a forward pass for a single token position, returning logits.
    ///
    /// `token_id`: the input token
    /// `pos`: the sequence position (for RoPE)
    /// `kv_cache`: the KV cache to read from and write to
    pub fn forward(&self, token_id: u32, pos: usize, kv_cache: &mut KvCache) -> Vec<f32> {
        let cfg = &self.config;
        let dim = cfg.hidden_dim;
        let head_dim = cfg.head_dim;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let kv_head_ratio = n_heads / n_kv_heads;

        // Token embedding lookup
        let emb_offset = (token_id as usize) * dim;
        let mut x = self.embedding[emb_offset..emb_offset + dim].to_vec();

        // Scratch buffers (reused across layers)
        let mut xb = vec![0.0f32; dim]; // after attention norm
        let mut q = vec![0.0f32; n_heads * head_dim];
        let mut k = vec![0.0f32; n_kv_heads * head_dim];
        let mut v = vec![0.0f32; n_kv_heads * head_dim];
        let mut attn_out = vec![0.0f32; dim];
        let mut ffn_gate = vec![0.0f32; cfg.intermediate_dim];
        let mut ffn_up = vec![0.0f32; cfg.intermediate_dim];
        let mut ffn_down = vec![0.0f32; dim];

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Pre-attention RMSNorm
            rms_norm(&x, &layer.attn_norm.weight, cfg.rms_norm_eps, &mut xb);

            // QKV projections
            layer.q_proj.matvec(&xb, &mut q);
            layer.k_proj.matvec(&xb, &mut k);
            layer.v_proj.matvec(&xb, &mut v);

            // Add biases if present (Qwen2 uses biases)
            if let Some(ref bias) = layer.q_bias {
                for (qi, bi) in q.iter_mut().zip(bias.iter()) {
                    *qi += bi;
                }
            }
            if let Some(ref bias) = layer.k_bias {
                for (ki, bi) in k.iter_mut().zip(bias.iter()) {
                    *ki += bi;
                }
            }
            if let Some(ref bias) = layer.v_bias {
                for (vi, bi) in v.iter_mut().zip(bias.iter()) {
                    *vi += bi;
                }
            }

            // Apply RoPE to Q and K
            apply_rope(&mut q, n_heads, head_dim, pos, &cfg.rope_freqs);
            apply_rope(&mut k, n_kv_heads, head_dim, pos, &cfg.rope_freqs);

            // Store K, V into cache
            kv_cache.store(layer_idx, pos, &k, &v);

            // Grouped-Query Attention
            let seq_len = pos + 1;
            let scale = 1.0 / (head_dim as f32).sqrt();

            for h in 0..n_heads {
                let kv_h = h / kv_head_ratio; // which KV head this query head uses
                let q_offset = h * head_dim;
                let q_vec = &q[q_offset..q_offset + head_dim];

                // Compute attention scores for all cached positions
                let mut scores = Vec::with_capacity(seq_len);
                for t in 0..seq_len {
                    let k_vec = kv_cache.get_k(layer_idx, t, kv_h, head_dim);
                    let score: f32 = q_vec
                        .iter()
                        .zip(k_vec.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        * scale;
                    scores.push(score);
                }

                // Softmax
                softmax(&mut scores);

                // Weighted sum of values
                let out_slice = &mut attn_out[q_offset..q_offset + head_dim];
                out_slice.fill(0.0);
                for (t, &w) in scores.iter().enumerate().take(seq_len) {
                    let v_vec = kv_cache.get_v(layer_idx, t, kv_h, head_dim);
                    for d in 0..head_dim {
                        out_slice[d] += w * v_vec[d];
                    }
                }
            }

            // Output projection
            let mut attn_projected = vec![0.0f32; dim];
            layer.o_proj.matvec(&attn_out, &mut attn_projected);

            // Residual connection
            for i in 0..dim {
                x[i] += attn_projected[i];
            }

            // Pre-FFN RMSNorm
            rms_norm(&x, &layer.ffn_norm.weight, cfg.rms_norm_eps, &mut xb);

            // SwiGLU FFN
            layer.gate_proj.matvec(&xb, &mut ffn_gate);
            layer.up_proj.matvec(&xb, &mut ffn_up);

            // gate = silu(gate) * up
            super::kernels::silu_inplace(&mut ffn_gate[..cfg.intermediate_dim]);
            for i in 0..cfg.intermediate_dim {
                ffn_gate[i] *= ffn_up[i];
            }

            layer.down_proj.matvec(&ffn_gate, &mut ffn_down);

            // Residual connection
            for i in 0..dim {
                x[i] += ffn_down[i];
            }
        }

        // Final RMSNorm
        rms_norm(&x, &self.final_norm.weight, cfg.rms_norm_eps, &mut xb);

        // LM head: project to vocabulary
        let mut logits = vec![0.0f32; cfg.vocab_size];
        self.output_weight.matvec(&xb, &mut logits);

        logits
    }

    /// Process a batch of tokens (prompt prefill) — runs forward pass for each token.
    /// Returns logits for the last token only.
    ///
    /// # Panics
    /// Panics if `tokens` is empty.
    pub fn forward_batch(
        &self,
        tokens: &[u32],
        start_pos: usize,
        kv_cache: &mut KvCache,
    ) -> Vec<f32> {
        debug_assert!(
            !tokens.is_empty(),
            "forward_batch called with empty token sequence"
        );
        let mut logits = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            logits = self.forward(token, start_pos + i, kv_cache);
        }
        logits
    }

    /// Run transformer layers only (no LM head), writing hidden state to `scratch.xb`.
    pub fn forward_hidden(
        &self,
        token_id: u32,
        pos: usize,
        kv_cache: &mut KvCache,
        scratch: &mut ScratchBuffers,
    ) {
        let cfg = &self.config;
        let dim = cfg.hidden_dim;
        let head_dim = cfg.head_dim;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let kv_head_ratio = n_heads / n_kv_heads;

        // Token embedding lookup
        let emb_offset = (token_id as usize) * dim;
        scratch
            .x
            .copy_from_slice(&self.embedding[emb_offset..emb_offset + dim]);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Pre-attention RMSNorm + QKV projections
            // Fused RMSNorm + Q projection when Q is Q8_0 (saves one hidden_dim read+write)
            if layer.q_proj.dtype == super::gguf::GgufDtype::Q8_0 {
                super::kernels::fused_rmsnorm_q8_0_matvec(
                    &scratch.x,
                    &layer.attn_norm.weight,
                    cfg.rms_norm_eps,
                    &layer.q_proj.data,
                    &mut scratch.q,
                    layer.q_proj.rows,
                    layer.q_proj.cols,
                );
                // K and V still need normalized input
                rms_norm(
                    &scratch.x,
                    &layer.attn_norm.weight,
                    cfg.rms_norm_eps,
                    &mut scratch.xb,
                );
                layer.k_proj.matvec(&scratch.xb, &mut scratch.k);
                layer.v_proj.matvec(&scratch.xb, &mut scratch.v);
            } else {
                rms_norm(
                    &scratch.x,
                    &layer.attn_norm.weight,
                    cfg.rms_norm_eps,
                    &mut scratch.xb,
                );
                layer.q_proj.matvec(&scratch.xb, &mut scratch.q);
                layer.k_proj.matvec(&scratch.xb, &mut scratch.k);
                layer.v_proj.matvec(&scratch.xb, &mut scratch.v);
            }

            // Add biases if present (Qwen2 uses biases)
            if let Some(ref bias) = layer.q_bias {
                for (qi, bi) in scratch.q.iter_mut().zip(bias.iter()) {
                    *qi += bi;
                }
            }
            if let Some(ref bias) = layer.k_bias {
                for (ki, bi) in scratch.k.iter_mut().zip(bias.iter()) {
                    *ki += bi;
                }
            }
            if let Some(ref bias) = layer.v_bias {
                for (vi, bi) in scratch.v.iter_mut().zip(bias.iter()) {
                    *vi += bi;
                }
            }

            // Apply RoPE to Q and K
            apply_rope(&mut scratch.q, n_heads, head_dim, pos, &cfg.rope_freqs);
            apply_rope(&mut scratch.k, n_kv_heads, head_dim, pos, &cfg.rope_freqs);

            // Store K, V into cache
            kv_cache.store(layer_idx, pos, &scratch.k, &scratch.v);

            // Grouped-Query Attention
            let seq_len = pos + 1;
            let scale = 1.0 / (head_dim as f32).sqrt();

            for h in 0..n_heads {
                let kv_h = h / kv_head_ratio;
                let q_offset = h * head_dim;
                let q_vec = &scratch.q[q_offset..q_offset + head_dim];

                // Compute attention scores
                let scores = &mut scratch.scores[..seq_len];
                for (t, score_slot) in scores.iter_mut().enumerate() {
                    let k_vec = kv_cache.get_k(layer_idx, t, kv_h, head_dim);
                    *score_slot = q_vec
                        .iter()
                        .zip(k_vec.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        * scale;
                }

                softmax(scores);

                // Weighted sum of values
                let out_slice = &mut scratch.attn_out[q_offset..q_offset + head_dim];
                out_slice.fill(0.0);
                for (t, &w) in scores.iter().enumerate().take(seq_len) {
                    let v_vec = kv_cache.get_v(layer_idx, t, kv_h, head_dim);
                    for d in 0..head_dim {
                        out_slice[d] += w * v_vec[d];
                    }
                }
            }

            // Output projection
            layer
                .o_proj
                .matvec(&scratch.attn_out, &mut scratch.attn_projected);

            // Residual connection
            for i in 0..dim {
                scratch.x[i] += scratch.attn_projected[i];
            }

            // Pre-FFN RMSNorm + SwiGLU FFN
            // Fused RMSNorm + gate projection when gate_proj is Q8_0
            if layer.gate_proj.dtype == super::gguf::GgufDtype::Q8_0 {
                super::kernels::fused_rmsnorm_q8_0_matvec(
                    &scratch.x,
                    &layer.ffn_norm.weight,
                    cfg.rms_norm_eps,
                    &layer.gate_proj.data,
                    &mut scratch.ffn_gate,
                    layer.gate_proj.rows,
                    layer.gate_proj.cols,
                );
                // up_proj still needs normalized input
                rms_norm(
                    &scratch.x,
                    &layer.ffn_norm.weight,
                    cfg.rms_norm_eps,
                    &mut scratch.xb,
                );
                layer.up_proj.matvec(&scratch.xb, &mut scratch.ffn_up);
            } else {
                rms_norm(
                    &scratch.x,
                    &layer.ffn_norm.weight,
                    cfg.rms_norm_eps,
                    &mut scratch.xb,
                );
                layer.gate_proj.matvec(&scratch.xb, &mut scratch.ffn_gate);
                layer.up_proj.matvec(&scratch.xb, &mut scratch.ffn_up);
            }

            super::kernels::silu_inplace(&mut scratch.ffn_gate[..cfg.intermediate_dim]);
            for i in 0..cfg.intermediate_dim {
                scratch.ffn_gate[i] *= scratch.ffn_up[i];
            }

            layer
                .down_proj
                .matvec(&scratch.ffn_gate, &mut scratch.ffn_down);

            // Residual connection
            for i in 0..dim {
                scratch.x[i] += scratch.ffn_down[i];
            }
        }

        // Final RMSNorm — output goes to scratch.xb
        rms_norm(
            &scratch.x,
            &self.final_norm.weight,
            cfg.rms_norm_eps,
            &mut scratch.xb,
        );
    }

    /// Run a forward pass writing logits into `scratch.logits`. No allocations.
    pub fn forward_into(
        &self,
        token_id: u32,
        pos: usize,
        kv_cache: &mut KvCache,
        scratch: &mut ScratchBuffers,
    ) {
        self.forward_hidden(token_id, pos, kv_cache, scratch);
        self.output_weight.matvec(&scratch.xb, &mut scratch.logits);
    }

    /// Process a batch of tokens using scratch buffers.
    /// Skips LM head for all tokens except the last (intermediate logits are discarded).
    pub fn forward_batch_into(
        &self,
        tokens: &[u32],
        start_pos: usize,
        kv_cache: &mut KvCache,
        scratch: &mut ScratchBuffers,
    ) {
        debug_assert!(
            !tokens.is_empty(),
            "forward_batch_into called with empty token sequence"
        );
        let last_idx = tokens.len() - 1;
        // Process all but last token — hidden state only (skip LM head)
        for (i, &token) in tokens[..last_idx].iter().enumerate() {
            self.forward_hidden(token, start_pos + i, kv_cache, scratch);
        }
        // Last token: full forward with LM head
        self.forward_into(tokens[last_idx], start_pos + last_idx, kv_cache, scratch);
    }

    /// Forward pass with vocabulary-pruned LM head.
    /// Computes logits only for active (SQL-compatible) tokens.
    /// Returns a mutable reference to the full-vocab logits buffer
    /// where non-active positions are NEG_INFINITY.
    pub fn forward_active_vocab<'a>(
        &self,
        token_id: u32,
        pos: usize,
        kv_cache: &mut KvCache,
        scratch: &mut ScratchBuffers,
        active_vocab: &'a mut ActiveVocab,
    ) -> &'a mut [f32] {
        self.forward_hidden(token_id, pos, kv_cache, scratch);
        active_vocab.compute_reduced(&self.output_weight, &scratch.xb);
        active_vocab.scatter_to_full()
    }

    /// Batch forward with vocabulary-pruned LM head for the final token.
    /// Skips LM head entirely for non-final tokens.
    pub fn forward_batch_active_vocab<'a>(
        &self,
        tokens: &[u32],
        start_pos: usize,
        kv_cache: &mut KvCache,
        scratch: &mut ScratchBuffers,
        active_vocab: &'a mut ActiveVocab,
    ) -> &'a mut [f32] {
        debug_assert!(
            !tokens.is_empty(),
            "forward_batch_active_vocab called with empty token sequence"
        );
        let last_idx = tokens.len() - 1;
        for (i, &token) in tokens[..last_idx].iter().enumerate() {
            self.forward_hidden(token, start_pos + i, kv_cache, scratch);
        }
        self.forward_active_vocab(
            tokens[last_idx],
            start_pos + last_idx,
            kv_cache,
            scratch,
            active_vocab,
        )
    }

    /// Access the output weight matrix (LM head) for external use.
    pub fn output_weight(&self) -> &WeightMatrix {
        &self.output_weight
    }
}

/// KV cache for autoregressive generation.
pub struct KvCache {
    /// k[layer][pos * n_kv_heads * head_dim .. (pos+1) * n_kv_heads * head_dim]
    k: Vec<Vec<f32>>,
    /// v[layer][pos * n_kv_heads * head_dim .. (pos+1) * n_kv_heads * head_dim]
    v: Vec<Vec<f32>>,
    n_kv_heads: usize,
    head_dim: usize,
    pub seq_len: usize,
}

impl KvCache {
    /// Create a new empty KV cache.
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let kv_dim = n_kv_heads * head_dim;
        let layer_size = max_seq_len * kv_dim;
        Self {
            k: (0..n_layers).map(|_| vec![0.0; layer_size]).collect(),
            v: (0..n_layers).map(|_| vec![0.0; layer_size]).collect(),
            n_kv_heads,
            head_dim,
            seq_len: 0,
        }
    }

    /// Store K and V vectors for a given layer and position.
    fn store(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        let kv_dim = self.n_kv_heads * self.head_dim;
        let offset = pos * kv_dim;
        let end = offset + kv_dim;
        if end > self.k[layer].len() {
            return; // Context window exceeded — silently drop
        }
        self.k[layer][offset..end].copy_from_slice(k);
        self.v[layer][offset..end].copy_from_slice(v);
        if pos >= self.seq_len {
            self.seq_len = pos + 1;
        }
    }

    /// Get K vector for a specific layer, position, and KV head.
    fn get_k(&self, layer: usize, pos: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        let kv_dim = self.n_kv_heads * head_dim;
        let offset = pos * kv_dim + kv_head * head_dim;
        &self.k[layer][offset..offset + head_dim]
    }

    /// Get V vector for a specific layer, position, and KV head.
    fn get_v(&self, layer: usize, pos: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        let kv_dim = self.n_kv_heads * head_dim;
        let offset = pos * kv_dim + kv_head * head_dim;
        &self.v[layer][offset..offset + head_dim]
    }

    /// Clone the current KV cache state (for prefix caching).
    pub fn clone_state(&self) -> Self {
        Self {
            k: self.k.clone(),
            v: self.v.clone(),
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            seq_len: self.seq_len,
        }
    }

    /// Reset the cache (reuse allocated memory).
    pub fn reset(&mut self) {
        for layer_k in &mut self.k {
            layer_k.fill(0.0);
        }
        for layer_v in &mut self.v {
            layer_v.fill(0.0);
        }
        self.seq_len = 0;
    }
}

// ── Math primitives ──────────────────────────────────────────────────────

/// RMSNorm: output[i] = weight[i] * (x[i] / rms(x))
fn rms_norm(x: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    super::kernels::rms_norm(x, weight, eps, output);
}

/// In-place softmax.
fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv_sum;
        }
    }
}

/// Apply Rotary Position Embeddings (RoPE) to Q or K vectors.
///
/// For each head, rotate pairs of dimensions by position-dependent angles.
/// `freqs` contains precomputed `1.0 / theta^(2i/head_dim)` for each dimension pair.
#[allow(clippy::needless_range_loop)]
fn apply_rope(qk: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, freqs: &[f32]) {
    let half_dim = head_dim / 2;
    for h in 0..n_heads {
        let offset = h * head_dim;
        for i in 0..half_dim {
            let angle = pos as f32 * freqs[i];
            let cos_val = angle.cos();
            let sin_val = angle.sin();

            let idx = offset + i * 2;
            let x0 = qk[idx];
            let x1 = qk[idx + 1];
            qk[idx] = x0 * cos_val - x1 * sin_val;
            qk[idx + 1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

/// Load a 1D f32 tensor from GGUF (typically norm weights).
fn load_1d_f32(gguf: &GgufFile, name: &str) -> Result<Vec<f32>> {
    let info = gguf
        .tensor_info(name)
        .ok_or_else(|| TensorError::LlmError(format!("tensor not found: {name}")))?;
    let data = gguf.tensor_data(name)?;
    dequant_tensor(data, info.dtype, info.n_elements())
}

/// Load an optional 1D f32 tensor (returns None if not present).
fn load_optional_1d_f32(gguf: &GgufFile, name: &str) -> Option<Vec<f32>> {
    load_1d_f32(gguf, name).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let mut output = vec![0.0; 4];
        rms_norm(&x, &weight, 1e-6, &mut output);

        // RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        let rms = (7.5f32).sqrt();
        for (i, &v) in output.iter().enumerate() {
            let expected = (i as f32 + 1.0) / rms;
            assert!(
                (v - expected).abs() < 1e-4,
                "element {i}: expected {expected}, got {v}"
            );
        }
    }

    #[test]
    fn silu_values() {
        let mut x = vec![0.0f32, 1.0, -1.0];
        crate::ai::kernels::silu_inplace(&mut x);
        assert!((x[0] - 0.0f32).abs() < 1e-6);
        // silu(1.0) = 1.0 / (1 + exp(-1)) ~ 0.7311
        assert!((x[1] - 0.7311f32).abs() < 0.01);
        // silu(-1.0) = -1.0 / (1 + exp(1)) ~ -0.2689
        assert!((x[2] - (-0.2689f32)).abs() < 0.01);
    }

    #[test]
    fn softmax_basic() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum should be 1.0, got {sum}");
        // Largest input should have largest probability
        assert!(x[2] > x[1]);
        assert!(x[1] > x[0]);
    }

    #[test]
    fn softmax_single() {
        let mut x = vec![5.0];
        softmax(&mut x);
        assert!((x[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_empty() {
        let mut x: Vec<f32> = vec![];
        softmax(&mut x);
        assert!(x.is_empty());
    }

    #[test]
    fn rope_preserves_magnitude() {
        // RoPE is a rotation, so it should preserve vector magnitude
        let head_dim = 4;
        let theta = 10000.0f32;
        let freqs: Vec<f32> = (0..head_dim / 2)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32))
            .collect();

        let mut q = vec![1.0, 0.0, 0.0, 1.0]; // 1 head, 4 dims
        let mag_before: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        apply_rope(&mut q, 1, head_dim, 5, &freqs);
        let mag_after: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (mag_before - mag_after).abs() < 1e-5,
            "RoPE should preserve magnitude: {mag_before} vs {mag_after}"
        );
    }

    #[test]
    fn rope_precomputed_matches_inline() {
        // Verify precomputed freqs match the old inline computation
        let head_dim = 8;
        let theta = 10000.0f32;
        let freqs: Vec<f32> = (0..head_dim / 2)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32))
            .collect();

        let mut q1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut q2 = q1.clone();

        // Apply with precomputed freqs
        apply_rope(&mut q1, 1, head_dim, 7, &freqs);

        // Apply manually with inline computation (old behavior)
        for i in 0..head_dim / 2 {
            let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
            let angle = 7.0 * freq;
            let cos_val = angle.cos();
            let sin_val = angle.sin();
            let x0 = q2[i * 2];
            let x1 = q2[i * 2 + 1];
            q2[i * 2] = x0 * cos_val - x1 * sin_val;
            q2[i * 2 + 1] = x0 * sin_val + x1 * cos_val;
        }

        for (a, b) in q1.iter().zip(q2.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "precomputed vs inline mismatch: {a} vs {b}"
            );
        }
    }

    #[test]
    fn kv_cache_store_retrieve() {
        let mut cache = KvCache::new(2, 2, 4, 16);
        let k = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 heads * 4 dims
        let v = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        cache.store(0, 0, &k, &v);
        assert_eq!(cache.seq_len, 1);

        let k0 = cache.get_k(0, 0, 0, 4);
        assert_eq!(k0, &[1.0, 2.0, 3.0, 4.0]);

        let k1 = cache.get_k(0, 0, 1, 4);
        assert_eq!(k1, &[5.0, 6.0, 7.0, 8.0]);

        let v0 = cache.get_v(0, 0, 0, 4);
        assert_eq!(v0, &[0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn kv_cache_clone_state() {
        let mut cache = KvCache::new(1, 1, 4, 8);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];
        cache.store(0, 0, &k, &v);

        let cloned = cache.clone_state();
        assert_eq!(cloned.seq_len, 1);
        assert_eq!(cloned.get_k(0, 0, 0, 4), &[1.0, 2.0, 3.0, 4.0]);
    }

    /// Build a Q8_0 weight matrix from f32 values for testing.
    fn make_q8_0_matrix(values: &[Vec<f32>]) -> WeightMatrix {
        let rows = values.len();
        let cols = values[0].len();
        assert!(
            cols.is_multiple_of(32),
            "cols must be multiple of 32 for Q8_0"
        );
        let blocks_per_row = cols / 32;
        let mut data = Vec::new();

        for row in values {
            for b in 0..blocks_per_row {
                // Find max abs value in block for quantization
                let block_start = b * 32;
                let block_vals = &row[block_start..block_start + 32];
                let max_abs = block_vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };

                // Write scale as f16
                let scale_f16 = half::f16::from_f32(scale);
                data.extend_from_slice(&scale_f16.to_bits().to_le_bytes());

                // Quantize values to int8
                for &v in block_vals {
                    let q = if scale > 0.0 {
                        (v / scale).round().clamp(-128.0, 127.0) as i8
                    } else {
                        0i8
                    };
                    data.push(q as u8);
                }
            }
        }

        WeightMatrix {
            data,
            dtype: GgufDtype::Q8_0,
            rows,
            cols,
        }
    }

    #[test]
    fn matvec_rows_matches_full_matvec() {
        // Create a 4x32 Q8_0 matrix
        let values: Vec<Vec<f32>> = (0..4)
            .map(|r| (0..32).map(|c| (r * 32 + c) as f32 * 0.01).collect())
            .collect();
        let mat = make_q8_0_matrix(&values);
        let input: Vec<f32> = (0..32).map(|i| (i as f32 + 1.0) * 0.1).collect();

        // Full matvec
        let mut full_output = vec![0.0f32; 4];
        mat.matvec(&input, &mut full_output);

        // Selected rows
        let row_indices = vec![0u32, 2, 3];
        let mut rows_output = vec![0.0f32; 3];
        mat.matvec_rows(&input, &mut rows_output, &row_indices);

        // Compare
        assert!(
            (rows_output[0] - full_output[0]).abs() < 0.1,
            "row 0: {} vs {}",
            rows_output[0],
            full_output[0]
        );
        assert!(
            (rows_output[1] - full_output[2]).abs() < 0.1,
            "row 2: {} vs {}",
            rows_output[1],
            full_output[2]
        );
        assert!(
            (rows_output[2] - full_output[3]).abs() < 0.1,
            "row 3: {} vs {}",
            rows_output[2],
            full_output[3]
        );
    }

    #[test]
    fn active_vocab_scatter() {
        let token_ids = vec![2u32, 5, 8];
        let mut av = ActiveVocab::new(token_ids, 10);

        // Simulate some reduced logits
        av.reduced_logits = vec![1.0, 2.0, 3.0];

        let full = av.scatter_to_full();
        assert_eq!(full[2], 1.0);
        assert_eq!(full[5], 2.0);
        assert_eq!(full[8], 3.0);
        assert_eq!(full[0], f32::NEG_INFINITY);
        assert_eq!(full[1], f32::NEG_INFINITY);
        assert_eq!(full[3], f32::NEG_INFINITY);
    }

    #[test]
    fn kernel_dispatch_matches_scalar() {
        // Create a 4x64 Q8_0 matrix (2 blocks per row)
        let values: Vec<Vec<f32>> = (0..4)
            .map(|r| {
                (0..64)
                    .map(|c| ((r * 64 + c) as f32 * 0.01) - 0.5)
                    .collect()
            })
            .collect();
        let mat = make_q8_0_matrix(&values);
        let input: Vec<f32> = (0..64).map(|i| (i as f32 + 1.0) * 0.05).collect();

        // Scalar kernel directly
        let mut scalar_out = vec![0.0f32; 4];
        crate::ai::kernels::scalar::q8_0_matvec(&mat.data, &input, &mut scalar_out, 4, 64);

        // Dispatch (uses best available SIMD kernel)
        let mut dispatch_out = vec![0.0f32; 4];
        mat.matvec_q8_0(&input, &mut dispatch_out);

        for i in 0..4 {
            assert!(
                (scalar_out[i] - dispatch_out[i]).abs() < 0.5,
                "row {i}: scalar={}, dispatch={}",
                scalar_out[i],
                dispatch_out[i]
            );
        }
    }

    /// Build a Q8_0 matrix with 256 rows (above MIN_PARALLEL_ROWS) for parallel tests.
    fn make_large_q8_0_matrix(rows: usize, cols: usize) -> WeightMatrix {
        assert!(cols.is_multiple_of(32));
        let blocks_per_row = cols / 32;
        let mut data = Vec::new();
        for r in 0..rows {
            for b in 0..blocks_per_row {
                let scale = half::f16::from_f32(0.01);
                data.extend_from_slice(&scale.to_bits().to_le_bytes());
                for j in 0..32 {
                    data.push(((r + b + j) % 127) as u8);
                }
            }
        }
        WeightMatrix {
            data,
            dtype: GgufDtype::Q8_0,
            rows,
            cols,
        }
    }

    /// Compute expected dot product for a row of the large Q8_0 matrix.
    fn expected_q8_0_dot(row: usize, cols: usize, input: &[f32]) -> f32 {
        let blocks_per_row = cols / 32;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let scale = half::f16::from_f32(0.01).to_f32();
            let mut block_sum = 0.0f32;
            for j in 0..32 {
                let val = ((row + b + j) % 127) as u8 as i8;
                block_sum += val as f32 * input[b * 32 + j];
            }
            sum += scale * block_sum;
        }
        sum
    }

    #[test]
    fn parallel_matvec_q8_0_correctness() {
        // 256 rows > MIN_PARALLEL_ROWS (128), exercises the parallel path
        let rows = 256;
        let cols = 64;
        let mat = make_large_q8_0_matrix(rows, cols);
        let input: Vec<f32> = (0..cols).map(|i| (i as f32 + 1.0) * 0.01).collect();

        let mut output = vec![0.0f32; rows];
        mat.matvec(&input, &mut output);

        for (r, &actual) in output.iter().enumerate() {
            let expected = expected_q8_0_dot(r, cols, &input);
            assert!(
                (actual - expected).abs() < 0.1,
                "row {r}: expected {expected}, got {actual}",
            );
        }
    }

    #[test]
    fn parallel_matvec_rows_q8_0_correctness() {
        // 256 selected rows > MIN_PARALLEL_ROWS, exercises parallel matvec_rows
        let total_rows = 512;
        let cols = 64;
        let mat = make_large_q8_0_matrix(total_rows, cols);
        let input: Vec<f32> = (0..cols).map(|i| (i as f32 + 1.0) * 0.01).collect();

        // Select 256 rows (every other row)
        let row_indices: Vec<u32> = (0..total_rows as u32).step_by(2).collect();
        let mut output = vec![0.0f32; row_indices.len()];
        mat.matvec_rows(&input, &mut output, &row_indices);

        for (i, &r) in row_indices.iter().enumerate() {
            let expected = expected_q8_0_dot(r as usize, cols, &input);
            assert!(
                (output[i] - expected).abs() < 0.1,
                "row_indices[{i}] (row {r}): expected {expected}, got {}",
                output[i]
            );
        }
    }

    #[test]
    fn parallel_matches_sequential_large_matvec() {
        // Cross-check: run the same large matrix through matvec and compare
        // full matvec output against individual row dot products from matvec_rows.
        let rows = 256;
        let cols = 64;
        let mat = make_large_q8_0_matrix(rows, cols);
        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.003).sin()).collect();

        // Full matvec (parallel path)
        let mut full_output = vec![0.0f32; rows];
        mat.matvec(&input, &mut full_output);

        // matvec_rows for all rows (parallel path)
        let all_indices: Vec<u32> = (0..rows as u32).collect();
        let mut rows_output = vec![0.0f32; rows];
        mat.matvec_rows(&input, &mut rows_output, &all_indices);

        for r in 0..rows {
            assert!(
                (full_output[r] - rows_output[r]).abs() < 1e-3,
                "row {r}: matvec={}, matvec_rows={}",
                full_output[r],
                rows_output[r]
            );
        }
    }
}
