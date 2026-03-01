//! Qwen2/3 transformer runtime — pure-Rust forward pass with KV cache.
//!
//! Implements the full Qwen2 architecture: RMSNorm, Grouped-Query Attention (GQA)
//! with RoPE, SwiGLU FFN, and a KV cache for efficient autoregressive generation.

use crate::error::{Result, TensorError};

use super::gguf::{dequant_tensor, GgufDtype, GgufFile};

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
struct WeightMatrix {
    data: Vec<u8>,
    dtype: GgufDtype,
    rows: usize,
    cols: usize,
}

#[allow(clippy::needless_range_loop)]
impl WeightMatrix {
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
    fn matvec(&self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.cols);
        debug_assert_eq!(output.len(), self.rows);

        match self.dtype {
            GgufDtype::Q8_0 => self.matvec_q8_0(input, output),
            GgufDtype::Q4_0 => self.matvec_q4_0(input, output),
            GgufDtype::F16 => self.matvec_f16(input, output),
            GgufDtype::F32 => self.matvec_f32(input, output),
            _ => {
                // Fallback: dequantize entire matrix (expensive but correct)
                if let Ok(weights) = dequant_tensor(&self.data, self.dtype, self.rows * self.cols) {
                    for r in 0..self.rows {
                        let mut sum = 0.0f32;
                        let row_start = r * self.cols;
                        for c in 0..self.cols {
                            sum += weights[row_start + c] * input[c];
                        }
                        output[r] = sum;
                    }
                }
            }
        }
    }

    /// Optimized Q8_0 matvec — dequantize and dot-product per block.
    fn matvec_q8_0(&self, input: &[f32], output: &mut [f32]) {
        const BLOCK_SIZE: usize = 32;
        const BLOCK_BYTES: usize = 34;

        let blocks_per_row = self.cols / BLOCK_SIZE;

        for r in 0..self.rows {
            let mut sum = 0.0f32;
            let row_offset = r * blocks_per_row * BLOCK_BYTES;

            for b in 0..blocks_per_row {
                let block = &self.data[row_offset + b * BLOCK_BYTES..];
                let scale = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();

                let input_offset = b * BLOCK_SIZE;
                let mut block_sum = 0.0f32;
                for j in 0..BLOCK_SIZE {
                    let val = block[2 + j] as i8;
                    block_sum += val as f32 * input[input_offset + j];
                }
                sum += scale * block_sum;
            }

            output[r] = sum;
        }
    }

    /// Optimized Q4_0 matvec.
    fn matvec_q4_0(&self, input: &[f32], output: &mut [f32]) {
        const BLOCK_SIZE: usize = 32;
        const BLOCK_BYTES: usize = 18;

        let blocks_per_row = self.cols / BLOCK_SIZE;

        for r in 0..self.rows {
            let mut sum = 0.0f32;
            let row_offset = r * blocks_per_row * BLOCK_BYTES;

            for b in 0..blocks_per_row {
                let block = &self.data[row_offset + b * BLOCK_BYTES..];
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

            output[r] = sum;
        }
    }

    /// F16 matvec.
    fn matvec_f16(&self, input: &[f32], output: &mut [f32]) {
        for r in 0..self.rows {
            let mut sum = 0.0f32;
            let row_offset = r * self.cols * 2;
            for c in 0..self.cols {
                let bits = u16::from_le_bytes([
                    self.data[row_offset + c * 2],
                    self.data[row_offset + c * 2 + 1],
                ]);
                let w = half::f16::from_bits(bits).to_f32();
                sum += w * input[c];
            }
            output[r] = sum;
        }
    }

    /// F32 matvec.
    fn matvec_f32(&self, input: &[f32], output: &mut [f32]) {
        for r in 0..self.rows {
            let mut sum = 0.0f32;
            let row_offset = r * self.cols * 4;
            for c in 0..self.cols {
                let start = row_offset + c * 4;
                let w = f32::from_le_bytes([
                    self.data[start],
                    self.data[start + 1],
                    self.data[start + 2],
                    self.data[start + 3],
                ]);
                sum += w * input[c];
            }
            output[r] = sum;
        }
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
            apply_rope(&mut q, n_heads, head_dim, pos, cfg.rope_theta);
            apply_rope(&mut k, n_kv_heads, head_dim, pos, cfg.rope_theta);

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
            for i in 0..cfg.intermediate_dim {
                ffn_gate[i] = silu(ffn_gate[i]) * ffn_up[i];
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
    let n = x.len();
    let ss: f32 = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let rms = (ss + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for i in 0..n {
        output[i] = weight[i] * (x[i] * inv_rms);
    }
}

/// SiLU activation: x * sigmoid(x)
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
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
fn apply_rope(qk: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, theta: f32) {
    for h in 0..n_heads {
        let offset = h * head_dim;
        for i in (0..head_dim).step_by(2) {
            let freq = 1.0 / theta.powf(i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let cos_val = angle.cos();
            let sin_val = angle.sin();

            let x0 = qk[offset + i];
            let x1 = qk[offset + i + 1];
            qk[offset + i] = x0 * cos_val - x1 * sin_val;
            qk[offset + i + 1] = x0 * sin_val + x1 * cos_val;
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
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        // silu(1.0) = 1.0 / (1 + exp(-1)) ≈ 0.7311
        assert!((silu(1.0) - 0.7311).abs() < 0.01);
        // silu(-1.0) = -1.0 / (1 + exp(1)) ≈ -0.2689
        assert!((silu(-1.0) - (-0.2689)).abs() < 0.01);
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
        let mut q = vec![1.0, 0.0, 0.0, 1.0]; // 1 head, 4 dims
        let mag_before: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        apply_rope(&mut q, 1, 4, 5, 10000.0);
        let mag_after: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (mag_before - mag_after).abs() < 1e-5,
            "RoPE should preserve magnitude: {mag_before} vs {mag_after}"
        );
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
}
