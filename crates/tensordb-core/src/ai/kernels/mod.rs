//! Kernel dispatch engine with runtime CPU feature detection.
//!
//! Probes CPU capabilities once at startup (via `OnceLock`) and dispatches
//! quantized matvec, RMSNorm, and SiLU operations to the best available
//! kernel tier: I8MM > NEON+DotProd > NEON (aarch64) or
//! AVX-512+VNNI > AVX-512 > AVX2 (x86_64) or scalar fallback.
//!
//! Arch-specific kernel modules are conditionally compiled per target and will
//! be added in subsequent tasks. For now all dispatch routes to scalar.

pub mod scalar;

#[cfg(feature = "llm")]
use rayon::prelude::*;

use std::sync::OnceLock;

/// Minimum number of output rows before engaging Rayon parallelism.
/// Below this threshold the scheduling overhead exceeds the benefit.
const MIN_PARALLEL_ROWS: usize = 128;

// ── CPU feature detection ────────────────────────────────────────────────

/// Detected CPU SIMD capabilities, probed once at startup.
#[derive(Debug, Clone)]
pub struct CpuFeatures {
    // ARM64 features
    pub has_neon: bool,
    pub has_i8mm: bool,
    pub has_dotprod: bool,

    // x86_64 features
    pub has_avx2: bool,
    pub has_fma: bool,
    pub has_avx512f: bool,
    pub has_avx512vnni: bool,
}

static CPU_FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

/// Returns the detected CPU features, probing once on first call.
pub fn cpu_features() -> &'static CpuFeatures {
    CPU_FEATURES.get_or_init(detect_cpu_features)
}

fn detect_cpu_features() -> CpuFeatures {
    CpuFeatures {
        // ARM64 detection
        #[cfg(target_arch = "aarch64")]
        has_neon: std::arch::is_aarch64_feature_detected!("neon"),
        #[cfg(not(target_arch = "aarch64"))]
        has_neon: false,

        #[cfg(target_arch = "aarch64")]
        has_i8mm: std::arch::is_aarch64_feature_detected!("i8mm"),
        #[cfg(not(target_arch = "aarch64"))]
        has_i8mm: false,

        #[cfg(target_arch = "aarch64")]
        has_dotprod: std::arch::is_aarch64_feature_detected!("dotprod"),
        #[cfg(not(target_arch = "aarch64"))]
        has_dotprod: false,

        // x86_64 detection
        #[cfg(target_arch = "x86_64")]
        has_avx2: std::arch::is_x86_feature_detected!("avx2"),
        #[cfg(not(target_arch = "x86_64"))]
        has_avx2: false,

        #[cfg(target_arch = "x86_64")]
        has_fma: std::arch::is_x86_feature_detected!("fma"),
        #[cfg(not(target_arch = "x86_64"))]
        has_fma: false,

        #[cfg(target_arch = "x86_64")]
        has_avx512f: std::arch::is_x86_feature_detected!("avx512f"),
        #[cfg(not(target_arch = "x86_64"))]
        has_avx512f: false,

        #[cfg(target_arch = "x86_64")]
        has_avx512vnni: std::arch::is_x86_feature_detected!("avx512vnni"),
        #[cfg(not(target_arch = "x86_64"))]
        has_avx512vnni: false,
    }
}

// ── Kernel tiers ─────────────────────────────────────────────────────────

/// Kernel implementation tier, ordered from fastest to slowest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelTier {
    /// ARM I8MM (int8 matrix multiply) — best for quantized integer ops on ARM.
    I8mm,
    /// ARM NEON with dot-product instructions.
    NeonDotprod,
    /// ARM NEON baseline.
    Neon,
    /// x86 AVX-512 with VNNI (Vector Neural Network Instructions).
    Avx512Vnni,
    /// x86 AVX-512 baseline.
    Avx512,
    /// x86 AVX2 with FMA.
    Avx2,
    /// Portable scalar fallback — works everywhere.
    Scalar,
}

impl std::fmt::Display for KernelTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelTier::I8mm => write!(f, "I8MM"),
            KernelTier::NeonDotprod => write!(f, "NEON+DotProd"),
            KernelTier::Neon => write!(f, "NEON"),
            KernelTier::Avx512Vnni => write!(f, "AVX-512+VNNI"),
            KernelTier::Avx512 => write!(f, "AVX-512"),
            KernelTier::Avx2 => write!(f, "AVX2+FMA"),
            KernelTier::Scalar => write!(f, "Scalar"),
        }
    }
}

/// Select the best available kernel tier for integer/quantized operations.
pub fn best_int_kernel() -> KernelTier {
    let feat = cpu_features();

    // ARM tiers (most specific first)
    if feat.has_i8mm {
        return KernelTier::I8mm;
    }
    if feat.has_dotprod && feat.has_neon {
        return KernelTier::NeonDotprod;
    }
    if feat.has_neon {
        return KernelTier::Neon;
    }

    // x86 tiers
    if feat.has_avx512vnni && feat.has_avx512f {
        return KernelTier::Avx512Vnni;
    }
    if feat.has_avx512f {
        return KernelTier::Avx512;
    }
    if feat.has_avx2 && feat.has_fma {
        return KernelTier::Avx2;
    }

    KernelTier::Scalar
}

/// Select the best available kernel tier for floating-point operations
/// (RMSNorm, SiLU, softmax, etc.).
pub fn best_float_kernel() -> KernelTier {
    let feat = cpu_features();

    // ARM tiers — dotprod not relevant for float, just NEON
    if feat.has_neon {
        return KernelTier::Neon;
    }

    // x86 tiers
    if feat.has_avx512f {
        return KernelTier::Avx512;
    }
    if feat.has_avx2 && feat.has_fma {
        return KernelTier::Avx2;
    }

    KernelTier::Scalar
}

// ── Dispatch functions ───────────────────────────────────────────────────

/// Q8_0 matrix-vector multiply with automatic parallelism for large row counts.
///
/// Block layout: 2-byte f16 scale + 32 int8 quants = 34 bytes.
pub fn q8_0_matvec(data: &[u8], input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    #[cfg(feature = "llm")]
    if rows >= MIN_PARALLEL_ROWS {
        const BLOCK_SIZE: usize = 32;
        const BLOCK_BYTES: usize = 34;
        let blocks_per_row = cols / BLOCK_SIZE;
        let row_stride = blocks_per_row * BLOCK_BYTES;

        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(r, out)| {
                let row_data_start = r * row_stride;
                let row_data = &data[row_data_start..row_data_start + row_stride];
                let mut single = [0.0f32];
                q8_0_matvec_dispatch(row_data, input, &mut single, 1, cols);
                *out = single[0];
            });
        return;
    }
    q8_0_matvec_dispatch(data, input, output, rows, cols);
}

/// Q4_0 matrix-vector multiply with automatic parallelism for large row counts.
///
/// Block layout: 2-byte f16 scale + 16 bytes (32 nibbles) = 18 bytes.
pub fn q4_0_matvec(data: &[u8], input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    #[cfg(feature = "llm")]
    if rows >= MIN_PARALLEL_ROWS {
        const BLOCK_SIZE: usize = 32;
        const BLOCK_BYTES: usize = 18;
        let blocks_per_row = cols / BLOCK_SIZE;
        let row_stride = blocks_per_row * BLOCK_BYTES;

        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(r, out)| {
                let row_data_start = r * row_stride;
                let row_data = &data[row_data_start..row_data_start + row_stride];
                let mut single = [0.0f32];
                q4_0_matvec_dispatch(row_data, input, &mut single, 1, cols);
                *out = single[0];
            });
        return;
    }
    q4_0_matvec_dispatch(data, input, output, rows, cols);
}

/// RMSNorm: `output[i] = weight[i] * (x[i] * inv_rms)`.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    rms_norm_dispatch(x, weight, eps, output);
}

/// SiLU (Sigmoid Linear Unit) activation, applied in-place.
pub fn silu_inplace(x: &mut [f32]) {
    silu_inplace_dispatch(x);
}

// ── Internal dispatch (scalar only for now; arch kernels added later) ────

fn q8_0_matvec_dispatch(
    data: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    scalar::q8_0_matvec(data, input, output, rows, cols);
}

fn q4_0_matvec_dispatch(
    data: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    scalar::q4_0_matvec(data, input, output, rows, cols);
}

fn rms_norm_dispatch(x: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    scalar::rms_norm(x, weight, eps, output);
}

fn silu_inplace_dispatch(x: &mut [f32]) {
    scalar::silu_inplace(x);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_features_detects_something() {
        let feat = cpu_features();
        // On x86_64 CI, AVX2 is almost always available; on aarch64, NEON is baseline.
        // We just check the struct is populated without panicking.
        let _debug = format!("{feat:?}");
    }

    #[test]
    fn best_kernels_return_valid_tier() {
        let int_tier = best_int_kernel();
        let float_tier = best_float_kernel();
        // Both should be displayable
        let _int = format!("{int_tier}");
        let _float = format!("{float_tier}");
    }

    #[test]
    fn dispatch_q8_0_matches_scalar() {
        let rows = 2;
        let cols = 32;
        let scale = half::f16::from_f32(0.5);
        let scale_bytes = scale.to_bits().to_le_bytes();

        let mut data = Vec::new();
        for _ in 0..rows {
            data.push(scale_bytes[0]);
            data.push(scale_bytes[1]);
            for j in 0..32i8 {
                data.push(j as u8);
            }
        }

        let input: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let mut output_dispatch = vec![0.0f32; rows];
        let mut output_scalar = vec![0.0f32; rows];

        q8_0_matvec(&data, &input, &mut output_dispatch, rows, cols);
        scalar::q8_0_matvec(&data, &input, &mut output_scalar, rows, cols);

        for i in 0..rows {
            assert!(
                (output_dispatch[i] - output_scalar[i]).abs() < 1e-6,
                "row {i}: dispatch={} scalar={}",
                output_dispatch[i],
                output_scalar[i]
            );
        }
    }

    #[test]
    fn dispatch_q4_0_matches_scalar() {
        let rows = 1;
        let cols = 32;
        let scale = half::f16::from_f32(1.0);
        let scale_bytes = scale.to_bits().to_le_bytes();

        let mut data = Vec::new();
        data.push(scale_bytes[0]);
        data.push(scale_bytes[1]);
        // Each byte: lo=5, hi=10 => values 5-8=-3, 10-8=2
        data.extend(std::iter::repeat_n(0xA5u8, 16));

        let input = vec![1.0f32; 32];
        let mut output_dispatch = vec![0.0f32; 1];
        let mut output_scalar = vec![0.0f32; 1];

        q4_0_matvec(&data, &input, &mut output_dispatch, rows, cols);
        scalar::q4_0_matvec(&data, &input, &mut output_scalar, rows, cols);

        assert!(
            (output_dispatch[0] - output_scalar[0]).abs() < 1e-6,
            "dispatch={} scalar={}",
            output_dispatch[0],
            output_scalar[0]
        );
    }

    #[test]
    fn dispatch_rms_norm_matches_scalar() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![0.5, 1.0, 1.5, 2.0];
        let mut out_dispatch = vec![0.0; 4];
        let mut out_scalar = vec![0.0; 4];

        rms_norm(&x, &weight, 1e-6, &mut out_dispatch);
        scalar::rms_norm(&x, &weight, 1e-6, &mut out_scalar);

        for i in 0..4 {
            assert!(
                (out_dispatch[i] - out_scalar[i]).abs() < 1e-6,
                "element {i}: dispatch={} scalar={}",
                out_dispatch[i],
                out_scalar[i]
            );
        }
    }

    #[test]
    fn dispatch_silu_matches_scalar() {
        let mut x_dispatch = vec![0.0, 1.0, -1.0, 5.0, -5.0];
        let mut x_scalar = x_dispatch.clone();

        silu_inplace(&mut x_dispatch);
        scalar::silu_inplace(&mut x_scalar);

        for i in 0..x_dispatch.len() {
            assert!(
                (x_dispatch[i] - x_scalar[i]).abs() < 1e-6,
                "element {i}: dispatch={} scalar={}",
                x_dispatch[i],
                x_scalar[i]
            );
        }
    }
}
