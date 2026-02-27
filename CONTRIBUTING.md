# Contributing to SpectraDB

Thanks for your interest in contributing.

## Development Setup

**Prerequisites:**
- Rust stable toolchain (`rustup`, `cargo`)
- Optional: C++ toolchain (`g++` or `clang++`) for native feature path
- Optional: Node.js 18+ for documentation site

**Build and test:**

```bash
# Build (pure Rust, default)
cargo build

# Run all tests (224+ integration tests)
cargo test --workspace --all-targets

# With C++ acceleration
cargo test --workspace --all-targets --features native

# With SIMD bloom filters
cargo test --features simd

# With io_uring (Linux only)
cargo test --features io-uring

# With Parquet support
cargo test --features parquet

# Lint and format
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings

# AI overhead regression gate
./scripts/ai_overhead_gate.sh

# Benchmarks
cargo bench --bench comparative
cargo bench --bench multi_engine
```

## Workspace Structure

| Crate | Purpose |
|-------|---------|
| `spectradb-core` | Database engine: storage, SQL, AI runtime, facets (~15k lines) |
| `spectradb-cli` | Interactive shell with TAB completion and output modes |
| `spectradb-server` | PostgreSQL wire protocol server (pgwire) |
| `spectradb-native` | Optional C++ acceleration via cxx (behind `--features native`) |
| `spectradb-python` | Python bindings via PyO3/maturin |
| `spectradb-node` | Node.js bindings via napi-rs |

Default workspace members: root, core, cli, server. Python and Node crates are excluded from default builds and require their respective build tools (maturin, npm).

## Contribution Guidelines

1. **Pure-Rust first.** The pure-Rust path must always be fully functional. Native optimizations stay behind `--features native`.
2. **Preserve correctness.** Bitemporal + MVCC semantics are the core guarantee. Don't break temporal invariants.
3. **Test your changes.** Add tests for behavioral changes, especially for WAL, recovery, temporal filters, SQL semantics, and write path.
4. **Keep it simple.** Don't add abstractions for single-use cases. Don't add error handling for scenarios that can't happen.
5. **Don't over-document.** Add comments only where the logic isn't self-evident. Don't add docstrings to code you didn't change.

## Pull Request Checklist

1. `cargo fmt --all --check` passes
2. `cargo clippy --workspace --all-targets -- -D warnings` passes
3. `cargo test --workspace --all-targets` passes
4. `cargo test --workspace --all-targets --features native` passes (if C++ toolchain available)
5. New or updated tests cover the change
6. No generated artifacts committed (`target/`, `data/`, `bench_runs/`, `overnight_runs/`)

## Code Style

- Follow existing patterns in the codebase
- Use `thiserror` for error types
- Keep modules focused — one concern per file
- Prefer `Arc<T>` for shared ownership, `RwLock` for read-heavy shared state
- Use `crossbeam_channel` for inter-thread communication
- SQL metadata under `__meta/` prefix, AI data under `__ai/` prefix

## Running the Documentation Site

```bash
cd docs && npm install && npm run dev
# Opens at http://localhost:4321

# Production build
cd docs && npm run build
```
