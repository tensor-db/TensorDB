//! Portable scalar fallback kernels for quantized matvec, RMSNorm, and SiLU.
//!
//! These implementations work on all architectures and serve as the baseline
//! that SIMD kernels (NEON, AVX2, AVX-512, I8MM) are validated against.

/// Q8_0 matrix-vector multiply (scalar).
///
/// Block layout: 2-byte f16 scale + 32 int8 quants = 34 bytes per block.
/// `data` contains `rows * (cols / 32)` consecutive blocks.
pub fn q8_0_matvec(data: &[u8], input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34;
    let blocks_per_row = cols / BLOCK_SIZE;
    debug_assert_eq!(cols % BLOCK_SIZE, 0, "cols must be a multiple of BLOCK_SIZE");
    debug_assert!(output.len() >= rows, "output too small for {rows} rows");
    debug_assert!(input.len() >= cols, "input too small for {cols} cols");
    debug_assert!(
        data.len() >= rows * blocks_per_row * BLOCK_BYTES,
        "data too small for {rows}x{cols} Q8_0 matrix"
    );

    for (r, out) in output.iter_mut().enumerate().take(rows) {
        let mut sum = 0.0f32;
        let row_offset = r * blocks_per_row * BLOCK_BYTES;
        for b in 0..blocks_per_row {
            let block = &data[row_offset + b * BLOCK_BYTES..];
            let scale =
                half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            let input_offset = b * BLOCK_SIZE;
            let mut block_sum = 0.0f32;
            for j in 0..BLOCK_SIZE {
                // Quants stored as signed int8 in two's complement; u8->i8 reinterprets bits.
                let val = block[2 + j] as i8;
                block_sum += val as f32 * input[input_offset + j];
            }
            sum += scale * block_sum;
        }
        *out = sum;
    }
}

/// Q4_0 matrix-vector multiply (scalar).
///
/// Block layout: 2-byte f16 scale + 16 bytes (32 nibbles) = 18 bytes per block.
/// Each byte packs two 4-bit values: lo nibble first, hi nibble second.
/// Values are unsigned [0,15] biased by -8 to give signed range [-8, 7].
pub fn q4_0_matvec(data: &[u8], input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18;
    let blocks_per_row = cols / BLOCK_SIZE;
    debug_assert_eq!(cols % BLOCK_SIZE, 0, "cols must be a multiple of BLOCK_SIZE");
    debug_assert!(output.len() >= rows, "output too small for {rows} rows");
    debug_assert!(input.len() >= cols, "input too small for {cols} cols");
    debug_assert!(
        data.len() >= rows * blocks_per_row * BLOCK_BYTES,
        "data too small for {rows}x{cols} Q4_0 matrix"
    );

    for (r, out) in output.iter_mut().enumerate().take(rows) {
        let mut sum = 0.0f32;
        let row_offset = r * blocks_per_row * BLOCK_BYTES;
        for b in 0..blocks_per_row {
            let block = &data[row_offset + b * BLOCK_BYTES..];
            let scale =
                half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
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
    }
}

/// RMSNorm: `output[i] = weight[i] * (x[i] * inv_rms)`.
///
/// Where `inv_rms = 1.0 / sqrt(mean(x^2) + eps)`.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let n = x.len();
    let ss: f32 = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let inv_rms = 1.0 / (ss + eps).sqrt();
    for i in 0..n {
        output[i] = weight[i] * (x[i] * inv_rms);
    }
}

/// Fused RMSNorm + Q8_0 matvec: eliminates the intermediate normalized buffer.
/// Computes: output[r] = dot(weights[r], x * inv_rms * norm_weight)
/// without materializing the normalized input to memory.
pub fn fused_rmsnorm_q8_0_matvec(
    x: &[f32],
    norm_weight: &[f32],
    eps: f32,
    weight_data: &[u8],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34;
    let blocks_per_row = cols / BLOCK_SIZE;

    // Compute inv_rms once
    let n = x.len();
    let ss: f32 = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let inv_rms = 1.0 / (ss + eps).sqrt();

    for (r, out) in output.iter_mut().enumerate().take(rows) {
        let mut sum = 0.0f32;
        let row_offset = r * blocks_per_row * BLOCK_BYTES;
        for b in 0..blocks_per_row {
            let block = &weight_data[row_offset + b * BLOCK_BYTES..];
            let scale =
                half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            let input_offset = b * BLOCK_SIZE;
            let mut block_sum = 0.0f32;
            for j in 0..BLOCK_SIZE {
                // Quants stored as signed int8 in two's complement; u8->i8 reinterprets bits.
                let val = block[2 + j] as i8;
                // Fused: apply RMSNorm on-the-fly during dot product
                let normed_input = x[input_offset + j] * inv_rms * norm_weight[input_offset + j];
                block_sum += val as f32 * normed_input;
            }
            sum += scale * block_sum;
        }
        *out = sum;
    }
}

/// SiLU (Sigmoid Linear Unit) activation, applied in-place.
///
/// `x[i] = x[i] / (1.0 + exp(-x[i]))` which is equivalent to `x * sigmoid(x)`.
pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_matvec_basic() {
        // 2 rows, 32 cols (1 block per row)
        let rows = 2;
        let cols = 32;
        let scale = half::f16::from_f32(0.01);
        let scale_bytes = scale.to_bits().to_le_bytes();

        // Build data: each block is [scale_lo, scale_hi, quant0..quant31]
        let mut data = Vec::new();
        for _ in 0..rows {
            data.push(scale_bytes[0]);
            data.push(scale_bytes[1]);
            for j in 0..32u8 {
                data.push(j); // interpreted as i8
            }
        }

        let input: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; rows];

        q8_0_matvec(&data, &input, &mut output, rows, cols);

        // Verify against manual calculation
        let scale_f32 = scale.to_f32();
        let expected: f32 = (0..32).map(|j| j as f32 * j as f32).sum::<f32>() * scale_f32;
        assert!(
            (output[0] - expected).abs() < 1e-2,
            "expected ~{expected}, got {}",
            output[0]
        );
    }

    #[test]
    fn q4_0_matvec_basic() {
        // 1 row, 32 cols (1 block)
        let rows = 1;
        let cols = 32;
        let scale = half::f16::from_f32(1.0);
        let scale_bytes = scale.to_bits().to_le_bytes();

        let mut data = Vec::new();
        data.push(scale_bytes[0]);
        data.push(scale_bytes[1]);
        // 16 bytes of nibbles, all zero nibbles => values = 0 - 8 = -8
        data.extend(std::iter::repeat_n(0x00u8, 16));

        let input = vec![1.0f32; 32];
        let mut output = vec![0.0f32; 1];

        q4_0_matvec(&data, &input, &mut output, rows, cols);

        // All quants are -8, so sum = -8 * 32 * 1.0 = -256
        let expected = -256.0f32;
        assert!(
            (output[0] - expected).abs() < 1e-2,
            "expected {expected}, got {}",
            output[0]
        );
    }

    #[test]
    fn rms_norm_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let mut output = vec![0.0; 4];
        rms_norm(&x, &weight, 1e-6, &mut output);

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
    fn silu_inplace_basic() {
        let mut x = vec![0.0, 1.0, -1.0];
        silu_inplace(&mut x);

        assert!((x[0] - 0.0).abs() < 1e-6);
        // silu(1.0) = 1.0 / (1 + exp(-1)) ~ 0.7311
        assert!((x[1] - 0.7311).abs() < 0.01);
        // silu(-1.0) = -1.0 / (1 + exp(1)) ~ -0.2689
        assert!((x[2] - (-0.2689)).abs() < 0.01);
    }

    #[test]
    fn fused_rmsnorm_q8_0_matches_separate() {
        let rows = 2;
        let cols = 32;
        let x: Vec<f32> = (0..32).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let norm_weight = vec![1.0f32; 32];
        let eps = 1e-6;

        // Build Q8_0 weight data
        let scale = half::f16::from_f32(0.01);
        let scale_bytes = scale.to_bits().to_le_bytes();
        let mut data = Vec::new();
        for _ in 0..rows {
            data.push(scale_bytes[0]);
            data.push(scale_bytes[1]);
            for j in 0..32u8 {
                data.push(j);
            }
        }

        // Fused path
        let mut fused_out = vec![0.0f32; rows];
        fused_rmsnorm_q8_0_matvec(&x, &norm_weight, eps, &data, &mut fused_out, rows, cols);

        // Separate path
        let mut normed = vec![0.0f32; cols];
        rms_norm(&x, &norm_weight, eps, &mut normed);
        let mut separate_out = vec![0.0f32; rows];
        q8_0_matvec(&data, &normed, &mut separate_out, rows, cols);

        for i in 0..rows {
            assert!(
                (fused_out[i] - separate_out[i]).abs() < 1e-4,
                "row {i}: fused={}, separate={}",
                fused_out[i],
                separate_out[i]
            );
        }
    }
}
