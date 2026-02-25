# Contributing to SpectraDB

Thanks for contributing.

## Development setup

Prerequisites:
- Rust stable toolchain (`rustup`, `cargo`)
- Optional C++ toolchain for native feature path (`g++`/`clang++`)

Clone and build:

```bash
cargo build
```

Run tests:

```bash
cargo test
cargo test --features native
```

Run formatter:

```bash
cargo fmt --all
```

## Contribution expectations

1. Keep the pure-Rust path fully functional.
2. Any native optimization must stay behind `--features native`.
3. Preserve bitemporal + MVCC correctness semantics.
4. Add tests for behavioral changes (especially WAL, recovery, temporal filters, SQL semantics).
5. Update docs (`README.md`, `design.md`, `perf.md`, `TEST_PLAN.md`) when surface behavior changes.

## Pull request checklist

1. `cargo test` passes.
2. `cargo test --features native` passes (if C++ toolchain available).
3. New/updated tests cover the change.
4. Docs updated where needed.
5. No generated local artifacts committed (e.g. `target/`, `data/`, `bench_runs/`, `overnight_runs/`).
