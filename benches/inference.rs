//! Inference engine microbenchmarks — measures the impact of each optimization.
//!
//! Benchmarks:
//! 1. LM head: full vocab (151K) vs active vocab (~5K) matvec
//! 2. Grammar apply: Vec<bool> mask vs HashSet<u32> (simulated)
//! 3. Q8_0 matvec: scalar vs SIMD dispatch (on supported architectures)
//! 4. RoPE: precomputed frequencies vs inline powf
//! 5. Scratch buffer allocation: per-token alloc vs reuse
//! 6. ActiveVocab scatter: NEG_INFINITY fill + scatter
//!
//! Run:
//!   cargo bench --bench inference
//!   cargo bench --bench inference --features simd

use std::collections::HashSet;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use tensordb_core::ai::gguf::GgufDtype;
use tensordb_core::ai::transformer::{ActiveVocab, ScratchBuffers, WeightMatrix};

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Qwen3-0.6B dimensions.
const HIDDEN_DIM: usize = 1024;
const VOCAB_SIZE: usize = 151936;
const ACTIVE_VOCAB_SIZE: usize = 5000;
const HEAD_DIM: usize = 64;
const N_HEADS: usize = 16;
const INTERMEDIATE_DIM: usize = 2816;
const CONTEXT_LENGTH: usize = 2048;

/// Build a Q8_0 weight matrix of given dimensions with synthetic data.
fn make_q8_0_weight(rows: usize, cols: usize) -> WeightMatrix {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34; // 2 (f16 scale) + 32 (int8 quants)

    let blocks_per_row = cols / BLOCK_SIZE;
    let total_bytes = rows * blocks_per_row * BLOCK_BYTES;
    let mut data = vec![0u8; total_bytes];

    // Fill with deterministic pseudo-random pattern
    for r in 0..rows {
        let row_offset = r * blocks_per_row * BLOCK_BYTES;
        for b in 0..blocks_per_row {
            let offset = row_offset + b * BLOCK_BYTES;
            // Scale = 0.01 as f16
            let scale = half::f16::from_f32(0.01);
            data[offset] = scale.to_bits().to_le_bytes()[0];
            data[offset + 1] = scale.to_bits().to_le_bytes()[1];
            // Fill quant values with small ints
            for j in 0..BLOCK_SIZE {
                data[offset + 2 + j] = ((r + b + j) % 127) as u8;
            }
        }
    }

    WeightMatrix::from_raw(data, GgufDtype::Q8_0, rows, cols)
}

/// Generate a synthetic input vector.
fn make_input(dim: usize) -> Vec<f32> {
    (0..dim).map(|i| (i as f32 * 0.001).sin()).collect()
}

/// Generate synthetic active token IDs (evenly distributed across vocab).
fn make_active_token_ids(n_active: usize, vocab_size: usize) -> Vec<u32> {
    let step = vocab_size / n_active;
    (0..n_active).map(|i| (i * step) as u32).collect()
}

// ── Benchmarks ──────────────────────────────────────────────────────────────

/// Benchmark 1: Full-vocab LM head vs active-vocab LM head.
///
/// The LM head is the single most expensive per-token operation.
/// Full: matmul [151K × 1024] × [1024] = 151K dot products
/// Active: matmul [5K × 1024] × [1024] = 5K dot products (~30× fewer)
fn bench_lm_head(c: &mut Criterion) {
    let mut group = c.benchmark_group("lm_head");

    let lm_head = make_q8_0_weight(VOCAB_SIZE, HIDDEN_DIM);
    let input = make_input(HIDDEN_DIM);
    let active_ids = make_active_token_ids(ACTIVE_VOCAB_SIZE, VOCAB_SIZE);

    // Full vocab matvec
    group.throughput(Throughput::Elements(VOCAB_SIZE as u64));
    group.bench_function("full_vocab_151k", |b| {
        let mut output = vec![0.0f32; VOCAB_SIZE];
        b.iter(|| {
            lm_head.matvec(black_box(&input), black_box(&mut output));
        });
    });

    // Active vocab matvec (only ~5K rows)
    group.throughput(Throughput::Elements(ACTIVE_VOCAB_SIZE as u64));
    group.bench_function("active_vocab_5k", |b| {
        let mut output = vec![0.0f32; ACTIVE_VOCAB_SIZE];
        b.iter(|| {
            lm_head.matvec_rows(
                black_box(&input),
                black_box(&mut output),
                black_box(&active_ids),
            );
        });
    });

    group.finish();
}

/// Benchmark 2: Grammar apply — Vec<bool> indexing vs HashSet::contains.
///
/// Old: 151K HashSet::contains() per token = hash + probe per entry.
/// New: 151K Vec<bool> indexed lookups = single array access per entry.
fn bench_grammar_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("grammar_apply");
    group.throughput(Throughput::Elements(VOCAB_SIZE as u64));

    let mut logits = vec![5.0f32; VOCAB_SIZE];

    // Build the Vec<bool> mask (new approach)
    let mut valid_mask = vec![false; VOCAB_SIZE];
    let active_ids = make_active_token_ids(ACTIVE_VOCAB_SIZE, VOCAB_SIZE);
    for &id in &active_ids {
        valid_mask[id as usize] = true;
    }

    // Build the HashSet (old approach, for comparison)
    let valid_set: HashSet<u32> = active_ids.iter().copied().collect();

    group.bench_function("vec_bool_mask", |b| {
        b.iter(|| {
            for (id, logit) in logits.iter_mut().enumerate() {
                if !valid_mask[id] {
                    *logit -= 10.0;
                }
            }
            // Reset
            logits.fill(5.0);
        });
    });

    group.bench_function("hashset_contains", |b| {
        b.iter(|| {
            for (id, logit) in logits.iter_mut().enumerate() {
                if !valid_set.contains(&(id as u32)) {
                    *logit -= 10.0;
                }
            }
            // Reset
            logits.fill(5.0);
        });
    });

    group.finish();
}

/// Benchmark 3: RoPE — precomputed frequencies vs inline powf.
///
/// Old: `1.0 / theta.powf(i as f32 / head_dim as f32)` per dimension pair per head.
/// New: table lookup from precomputed `rope_freqs[i]`.
fn bench_rope(c: &mut Criterion) {
    let mut group = c.benchmark_group("rope");
    let theta = 1_000_000.0f32; // Qwen3 rope_theta

    // Precomputed frequencies (new)
    let freqs: Vec<f32> = (0..HEAD_DIM / 2)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / HEAD_DIM as f32))
        .collect();

    let mut qk = vec![1.0f32; N_HEADS * HEAD_DIM];

    group.bench_function("precomputed_freqs", |b| {
        b.iter(|| {
            for h in 0..N_HEADS {
                let offset = h * HEAD_DIM;
                for (i, &freq) in freqs.iter().enumerate() {
                    let angle = 100.0 * freq;
                    let (sin_val, cos_val) = angle.sin_cos();
                    let idx = offset + i * 2;
                    let x0 = qk[idx];
                    let x1 = qk[idx + 1];
                    qk[idx] = x0 * cos_val - x1 * sin_val;
                    qk[idx + 1] = x0 * sin_val + x1 * cos_val;
                }
            }
            black_box(&qk);
        });
    });

    group.bench_function("inline_powf", |b| {
        b.iter(|| {
            for h in 0..N_HEADS {
                let offset = h * HEAD_DIM;
                for i in (0..HEAD_DIM).step_by(2) {
                    let freq = 1.0 / theta.powf(i as f32 / HEAD_DIM as f32);
                    let angle = 100.0 * freq;
                    let (sin_val, cos_val) = angle.sin_cos();
                    let x0 = qk[offset + i];
                    let x1 = qk[offset + i + 1];
                    qk[offset + i] = x0 * cos_val - x1 * sin_val;
                    qk[offset + i + 1] = x0 * sin_val + x1 * cos_val;
                }
            }
            black_box(&qk);
        });
    });

    group.finish();
}

/// Benchmark 4: Scratch buffer allocation — per-token alloc vs reuse.
///
/// Old: 10 Vec allocations per forward pass (xb, q, k, v, attn_out, ...).
/// New: Zero allocations — all buffers pre-allocated in ScratchBuffers.
fn bench_scratch_alloc(c: &mut Criterion) {
    let mut group = c.benchmark_group("scratch_buffers");

    let config = tensordb_core::ai::transformer::ModelConfig {
        hidden_dim: HIDDEN_DIM,
        n_layers: 28,
        n_heads: N_HEADS,
        n_kv_heads: 2,
        intermediate_dim: INTERMEDIATE_DIM,
        vocab_size: VOCAB_SIZE,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
        head_dim: HEAD_DIM,
        context_length: CONTEXT_LENGTH,
        rope_freqs: (0..HEAD_DIM / 2)
            .map(|i| 1.0 / 1_000_000.0f32.powf((2 * i) as f32 / HEAD_DIM as f32))
            .collect(),
    };

    // Pre-allocated (new)
    group.bench_function("preallocated_reuse", |b| {
        let mut scratch = ScratchBuffers::new(&config);
        b.iter(|| {
            // Simulate clearing buffers (what happens each forward pass)
            scratch.logits.fill(0.0);
            black_box(&mut scratch);
        });
    });

    // Per-token allocation (old)
    group.bench_function("per_token_alloc", |b| {
        b.iter(|| {
            let _xb = vec![0.0f32; HIDDEN_DIM];
            let _q = vec![0.0f32; N_HEADS * HEAD_DIM];
            let _k = vec![0.0f32; 2 * HEAD_DIM]; // n_kv_heads=2
            let _v = vec![0.0f32; 2 * HEAD_DIM];
            let _attn_out = vec![0.0f32; HIDDEN_DIM];
            let _attn_projected = vec![0.0f32; HIDDEN_DIM];
            let _ffn_gate = vec![0.0f32; INTERMEDIATE_DIM];
            let _ffn_up = vec![0.0f32; INTERMEDIATE_DIM];
            let _ffn_down = vec![0.0f32; HIDDEN_DIM];
            let _logits = vec![0.0f32; VOCAB_SIZE];
            black_box((&_xb, &_q, &_k, &_v, &_attn_out, &_ffn_gate, &_logits));
        });
    });

    group.finish();
}

/// Benchmark 5: ActiveVocab scatter — fill + scatter performance.
fn bench_active_vocab_scatter(c: &mut Criterion) {
    let mut group = c.benchmark_group("active_vocab_scatter");

    let active_ids = make_active_token_ids(ACTIVE_VOCAB_SIZE, VOCAB_SIZE);
    let mut av = ActiveVocab::new(active_ids, VOCAB_SIZE);

    group.bench_function("scatter_5k_to_151k", |b| {
        // Pre-fill reduced logits
        for (i, logit) in av.reduced_logits_mut().iter_mut().enumerate() {
            *logit = i as f32 * 0.01;
        }
        b.iter(|| {
            let full = av.scatter_to_full();
            black_box(full);
        });
    });

    group.finish();
}

/// Benchmark 6: Q8_0 matvec at typical layer sizes.
///
/// Tests matvec throughput for the most common weight matrix dimensions
/// in Qwen3-0.6B: q/k/v projections and FFN layers.
fn bench_q8_0_matvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("q8_0_matvec");
    group.sample_size(50); // Large matrices take a while

    // Typical transformer layer dimensions
    let configs = [
        ("qkv_proj_1024x1024", 1024, 1024),
        ("ffn_up_2816x1024", 2816, 1024),
        ("ffn_down_1024x2816", 1024, 2816),
    ];

    for (name, rows, cols) in &configs {
        let mat = make_q8_0_weight(*rows, *cols);
        let input = make_input(*cols);
        let mut output = vec![0.0f32; *rows];

        group.throughput(Throughput::Elements((*rows as u64) * (*cols as u64)));
        group.bench_with_input(BenchmarkId::new("matvec", name), &(), |b, _| {
            b.iter(|| {
                mat.matvec(black_box(&input), black_box(&mut output));
            });
        });
    }

    group.finish();
}

/// Benchmark 7: End-to-end per-token cost comparison.
///
/// Compares the dominant per-token costs:
/// - Old: full 151K matvec + 151K HashSet grammar apply
/// - New: 5K matvec_rows + scatter (grammar apply unnecessary)
fn bench_per_token_cost(c: &mut Criterion) {
    let mut group = c.benchmark_group("per_token_cost");
    group.sample_size(30);

    let lm_head = make_q8_0_weight(VOCAB_SIZE, HIDDEN_DIM);
    let input = make_input(HIDDEN_DIM);
    let active_ids = make_active_token_ids(ACTIVE_VOCAB_SIZE, VOCAB_SIZE);

    // Build grammar structures
    let mut valid_mask = vec![false; VOCAB_SIZE];
    for &id in &active_ids {
        valid_mask[id as usize] = true;
    }
    let valid_set: HashSet<u32> = active_ids.iter().copied().collect();

    // Old path: full matvec + HashSet grammar apply
    group.bench_function("old_full_matvec_plus_hashset", |b| {
        let mut logits = vec![0.0f32; VOCAB_SIZE];
        b.iter(|| {
            lm_head.matvec(black_box(&input), black_box(&mut logits));
            for (id, logit) in logits.iter_mut().enumerate() {
                if !valid_set.contains(&(id as u32)) {
                    *logit -= 10.0;
                }
            }
            black_box(&logits);
        });
    });

    // New path: active vocab matvec_rows + scatter (no grammar apply needed)
    group.bench_function("new_active_vocab_plus_scatter", |b| {
        let mut av = ActiveVocab::new(active_ids.clone(), VOCAB_SIZE);
        b.iter(|| {
            let mut reduced = vec![0.0f32; ACTIVE_VOCAB_SIZE];
            lm_head.matvec_rows(black_box(&input), black_box(&mut reduced), &active_ids);
            // Scatter to full vocab (NEG_INFINITY for non-active)
            av.reduced_logits_mut().copy_from_slice(&reduced);
            let full = av.scatter_to_full();
            black_box(full);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_lm_head,
    bench_grammar_apply,
    bench_rope,
    bench_scratch_alloc,
    bench_active_vocab_scatter,
    bench_q8_0_matvec,
    bench_per_token_cost,
);
criterion_main!(benches);
