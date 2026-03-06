//! AVX2+FMA kernel implementations for x86_64.
//!
//! All functions require AVX2 and FMA support, enforced via `#[target_feature]`.
//! The caller must ensure the CPU supports these features before invoking
//! (handled by the dispatch layer in `mod.rs`).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ── Q8_0 matrix-vector multiply ─────────────────────────────────────────

/// AVX2+FMA Q8_0 matvec -- processes 32 elements per block in 4 groups of 8.
///
/// Block layout: 2-byte f16 scale + 32 int8 quants = 34 bytes per block.
///
/// # Safety
/// Requires AVX2+FMA CPU support. Caller must ensure `data`, `input`, and
/// `output` are correctly sized for the given `rows` x `cols` dimensions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn q8_0_matvec(
    data: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34;
    let blocks_per_row = cols / BLOCK_SIZE;

    for (r, output_r) in output.iter_mut().enumerate().take(rows) {
        let mut acc = _mm256_setzero_ps();
        let row_offset = r * blocks_per_row * BLOCK_BYTES;

        for b in 0..blocks_per_row {
            let block = &data[row_offset + b * BLOCK_BYTES..];
            let scale = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            let scale_v = _mm256_set1_ps(scale);

            let input_offset = b * BLOCK_SIZE;
            let quants = &block[2..];

            // Process 32 elements in 4 groups of 8
            let mut block_acc = _mm256_setzero_ps();
            for g in 0..4 {
                let g_off = g * 8;

                // Load 8 int8 quants and sign-extend to 8 x i32, then convert to f32.
                // We load into a 64-bit lane, then use SSE4.1 cvtepi8_epi32 on each
                // 4-byte half, and combine into a 256-bit register.
                let q_ptr = quants[g_off..].as_ptr();

                // Low 4 bytes -> 4 x i32
                let q_lo_128 = _mm_cvtsi32_si128(std::ptr::read_unaligned(q_ptr as *const i32));
                let q_lo_i32 = _mm_cvtepi8_epi32(q_lo_128);

                // High 4 bytes -> 4 x i32
                let q_hi_128 =
                    _mm_cvtsi32_si128(std::ptr::read_unaligned(q_ptr.add(4) as *const i32));
                let q_hi_i32 = _mm_cvtepi8_epi32(q_hi_128);

                // Combine into 256-bit: [lo0..lo3, hi0..hi3]
                let q_i32_256 = _mm256_set_m128i(q_hi_i32, q_lo_i32);
                let q_f32 = _mm256_cvtepi32_ps(q_i32_256);

                // Load 8 input f32 values
                let inp = _mm256_loadu_ps(input[input_offset + g_off..].as_ptr());

                // FMA: block_acc += q * inp
                block_acc = _mm256_fmadd_ps(q_f32, inp, block_acc);
            }

            // Multiply block sum by scale and accumulate
            acc = _mm256_fmadd_ps(scale_v, block_acc, acc);
        }

        // Horizontal sum of 8-wide accumulator -> single f32
        let hi128 = _mm256_extractf128_ps(acc, 1);
        let lo128 = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo128, hi128);
        let shuf64 = _mm_movehdup_ps(sum128);
        let sum64 = _mm_add_ps(sum128, shuf64);
        let shuf32 = _mm_movehl_ps(sum64, sum64);
        let sum32 = _mm_add_ss(sum64, shuf32);
        *output_r = _mm_cvtss_f32(sum32);
    }
}

// ── Q4_0 matrix-vector multiply ─────────────────────────────────────────

/// AVX2+FMA Q4_0 matvec -- nibble unpacking with FMA accumulation.
///
/// Block layout: 2-byte f16 scale + 16 packed bytes (32 nibbles) = 18 bytes.
/// Each byte packs two 4-bit values: lo nibble first, hi nibble second.
/// Values are unsigned [0,15] biased by -8 to give signed range [-8, 7].
///
/// # Safety
/// Requires AVX2+FMA CPU support. Caller must ensure slices are correctly sized.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn q4_0_matvec(
    data: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18;
    let blocks_per_row = cols / BLOCK_SIZE;

    let lo_mask = _mm256_set1_epi8(0x0F);
    let bias = _mm256_set1_epi16(8);

    for (r, output_r) in output.iter_mut().enumerate().take(rows) {
        let mut acc = _mm256_setzero_ps();
        let row_offset = r * blocks_per_row * BLOCK_BYTES;

        for b in 0..blocks_per_row {
            let block = &data[row_offset + b * BLOCK_BYTES..];
            let scale = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            let scale_v = _mm256_set1_ps(scale);

            let input_offset = b * BLOCK_SIZE;
            let quants = &block[2..];

            // Process 16 bytes = 32 nibbles in 2 groups of 8 bytes (16 nibbles each).
            // Each group produces 16 values which pair with 16 input floats,
            // but we accumulate them as 8 pair-sums into an __m256.
            let mut block_acc = _mm256_setzero_ps();

            for g in 0..2 {
                let g_off = g * 8;
                let base = input_offset + g * 16;
                let q_ptr = quants[g_off..].as_ptr();

                // Load 8 packed bytes into low 64 bits of a 128-bit register
                let raw_64 = std::ptr::read_unaligned(q_ptr as *const i64);
                let raw_128 = _mm_set_epi64x(0, raw_64);

                // Extract lo nibbles: byte & 0x0F
                let lo_bytes = _mm_and_si128(raw_128, _mm256_castsi256_si128(lo_mask));
                // Extract hi nibbles: byte >> 4
                let hi_bytes =
                    _mm_and_si128(_mm_srli_epi16(raw_128, 4), _mm256_castsi256_si128(lo_mask));

                // Zero-extend lo nibbles from u8 -> i16 (8 values)
                let lo_i16 = _mm_cvtepu8_epi16(lo_bytes);
                // Zero-extend hi nibbles from u8 -> i16 (8 values)
                let hi_i16 = _mm_cvtepu8_epi16(hi_bytes);

                // Bias: subtract 8
                let bias_128 = _mm256_castsi256_si128(bias);
                let lo_biased = _mm_sub_epi16(lo_i16, bias_128);
                let hi_biased = _mm_sub_epi16(hi_i16, bias_128);

                // For each of the 8 byte positions j:
                //   pair_sum[j] = lo_val[j] * input[base + 2*j] + hi_val[j] * input[base + 2*j + 1]
                // We compute this by converting to f32 and doing 2 separate FMAs
                // with even/odd input elements.

                // Convert lo_biased (8 x i16) -> two sets of 4 x i32 -> 8 x f32
                let lo_lo_i32 = _mm_cvtepi16_epi32(lo_biased);
                let lo_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128(lo_biased, 8));
                let lo_f32 = _mm256_cvtepi32_ps(_mm256_set_m128i(lo_hi_i32, lo_lo_i32));

                // Convert hi_biased similarly
                let hi_lo_i32 = _mm_cvtepi16_epi32(hi_biased);
                let hi_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128(hi_biased, 8));
                let hi_f32 = _mm256_cvtepi32_ps(_mm256_set_m128i(hi_hi_i32, hi_lo_i32));

                // Gather even inputs: input[base+0], input[base+2], ..., input[base+14]
                // Gather odd inputs:  input[base+1], input[base+3], ..., input[base+15]
                let inp_ptr = input[base..].as_ptr();
                let mut even = [0.0f32; 8];
                let mut odd = [0.0f32; 8];
                for j in 0..8 {
                    even[j] = *inp_ptr.add(j * 2);
                    odd[j] = *inp_ptr.add(j * 2 + 1);
                }
                let even_v = _mm256_loadu_ps(even.as_ptr());
                let odd_v = _mm256_loadu_ps(odd.as_ptr());

                // pair_sum = lo * even + hi * odd
                let pair_sum = _mm256_fmadd_ps(lo_f32, even_v, _mm256_mul_ps(hi_f32, odd_v));
                block_acc = _mm256_add_ps(block_acc, pair_sum);
            }

            acc = _mm256_fmadd_ps(scale_v, block_acc, acc);
        }

        // Horizontal sum
        let hi128 = _mm256_extractf128_ps(acc, 1);
        let lo128 = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo128, hi128);
        let shuf64 = _mm_movehdup_ps(sum128);
        let sum64 = _mm_add_ps(sum128, shuf64);
        let shuf32 = _mm_movehl_ps(sum64, sum64);
        let sum32 = _mm_add_ss(sum64, shuf32);
        *output_r = _mm_cvtss_f32(sum32);
    }
}

// ── RMSNorm ─────────────────────────────────────────────────────────────

/// AVX2+FMA RMSNorm: `output[i] = weight[i] * (x[i] / rms)`.
///
/// Computes the root-mean-square using vectorized sum-of-squares, then applies
/// the normalization and weight scaling.
///
/// # Safety
/// Requires AVX2+FMA CPU support. All slices must have the same length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn rms_norm(x: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let n = x.len();

    // ---- Vectorized sum-of-squares ----
    let mut ss_acc = _mm256_setzero_ps();
    let chunks = n / 8;
    let remainder = n % 8;

    for i in 0..chunks {
        let v = _mm256_loadu_ps(x[i * 8..].as_ptr());
        ss_acc = _mm256_fmadd_ps(v, v, ss_acc);
    }

    // Horizontal sum of ss_acc
    let hi128 = _mm256_extractf128_ps(ss_acc, 1);
    let lo128 = _mm256_castps256_ps128(ss_acc);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf64 = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf64);
    let shuf32 = _mm_movehl_ps(sum64, sum64);
    let sum32 = _mm_add_ss(sum64, shuf32);
    let mut ss = _mm_cvtss_f32(sum32);

    // Handle remainder elements
    for xi in &x[(chunks * 8)..n] {
        ss += xi * xi;
    }

    let inv_rms = 1.0 / (ss / n as f32 + eps).sqrt();
    let inv_rms_v = _mm256_set1_ps(inv_rms);

    // ---- Apply normalization + weight ----
    for i in 0..chunks {
        let off = i * 8;
        let xv = _mm256_loadu_ps(x[off..].as_ptr());
        let wv = _mm256_loadu_ps(weight[off..].as_ptr());
        let normed = _mm256_mul_ps(xv, inv_rms_v);
        let result = _mm256_mul_ps(wv, normed);
        _mm256_storeu_ps(output[off..].as_mut_ptr(), result);
    }

    // Handle remainder elements
    for i in (chunks * 8)..n {
        output[i] = weight[i] * (x[i] * inv_rms);
    }

    let _ = remainder;
}

// ── SiLU ────────────────────────────────────────────────────────────────

/// SiLU (Sigmoid Linear Unit) activation, applied in-place (scalar fallback).
///
/// `x[i] = x[i] / (1.0 + exp(-x[i]))` which is equivalent to `x * sigmoid(x)`.
/// The exp() function is not easily vectorized with pure AVX2 intrinsics, so
/// this uses the scalar implementation.
///
/// # Safety
/// Requires AVX2+FMA CPU support (for target_feature consistency), but the
/// actual computation is scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}
