# Changelog

All notable changes to SpectraDB will be documented in this file.

## [0.1.0] - 2026-02-25

### Added
- Durable append-only fact ledger with WAL and CRC-framed records.
- MVCC snapshot reads (`AS OF <commit_ts>`).
- Bitemporal filtering (`VALID AT <valid_ts>`).
- Sharded single-writer execution model.
- LSM-style SSTables with bloom filters, block index, and mmap reads.
- RelationalFacet SQL subset: CREATE TABLE/VIEW/INDEX, ALTER TABLE ADD COLUMN, INSERT, SELECT, DROP, SHOW TABLES, DESCRIBE, EXPLAIN, transactions (BEGIN/COMMIT/ROLLBACK).
- Interactive CLI shell with TAB autocomplete, persistent history, and output modes (table/line/json).
- Optional C++ acceleration behind `--features native` via `cxx`.
- Benchmark harness with configurable workload matrix.
- Overnight burn-in script for reliability testing.
