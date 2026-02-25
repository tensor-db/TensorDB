# Pre-Publish Fixes Implementation Plan

**Goal:** Fix all critical, important, and quality issues identified in the comprehensive code review before publishing on GitHub.

**Architecture:** Surgical fixes to existing files. No new crates or architectural changes. Each task is a focused fix with tests.

**Tech Stack:** Rust, cargo test

---

## Phase 1: Critical Storage/WAL Fixes

### Task 1: WAL truncate seek + CRC mismatch recovery + record size bound
- Fix `wal.rs`: add `seek(0)` after `set_len(0)` in `truncate()`
- Fix `wal.rs`: change CRC mismatch from fatal error to `break` (stop replay)
- Fix `wal.rs`: add `MAX_WAL_RECORD_SIZE` check before allocation
- Add test for multi-record WAL CRC corruption mid-file

### Task 2: Shard flush safety (dir fsync + orphan cleanup + push/pop fix)
- Fix `shard.rs`: add dir fsync after `build_sstable`
- Fix `shard.rs`: add dir fsync after compaction SSTable creation
- Fix `shard.rs`: simplify `flush_active_memtable` push/pop to use `old` directly
- Fix `shard.rs`: clean up orphan SSTable on flush failure

### Task 3: Compaction deduplication
- Fix `compaction.rs`: add `dedup_by` after sort

### Task 4: Memtable approx_bytes accounting
- Fix `memtable.rs`: handle overwrite case in `insert`

### Task 5: Config validation for compaction_l0_threshold
- Fix `config.rs`: reject `compaction_l0_threshold == 0`
- Add `ConfigError` variant to `error.rs`
- Use `ConfigError` for all config validation errors

## Phase 2: SQL Executor Fixes

### Task 6: View pk_filter propagation
- Fix `exec.rs`: carry view's `pk_filter` through `resolve_select_target`

### Task 7: DROP TABLE tombstones index metadata
- Fix `exec.rs`: scan and tombstone `__meta/index/{table}/` on DROP TABLE

### Task 8: PK validation in INSERT
- Add `validate_pk` function in `relational.rs`
- Call it from INSERT in `exec.rs`

### Task 9: Fix parser error message for SELECT projection
- Fix `parser.rs` line 400: correct the error message text

## Phase 3: CI/CD and Project Metadata

### Task 10: Add clippy to CI + fix clippy warnings
- Add clippy step to `ci.yml`
- Add explicit C++ toolchain install

### Task 11: Add Cargo.toml metadata + CHANGELOG + gitignore output/
- Add `repository`, `keywords`, `categories` to Cargo.toml
- Create CHANGELOG.md
- Add `output/` to .gitignore
- Remove unused `rand` dev-dependency

## Phase 4: Quality Improvements

### Task 12: Bloom filter independent hash
- Fix `bloom.rs`: use second hash call instead of derived h2

### Task 13: Fix hasher naming
- Fix `native_bridge.rs`: rename from "rust-fnv64" to "rust-fnv64-mix"

### Task 14: Benchmark setup/teardown isolation
- Fix `benches/basic.rs`: move DB creation outside `b.iter()`

### Task 15: Varint shift overflow fix
- Fix `varint.rs`: use `checked_shl` for safety

---

## Execution: Subagent-Driven (parallel batches)
