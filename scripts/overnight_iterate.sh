#!/usr/bin/env bash
set -euo pipefail

if [ -f "$HOME/.cargo/env" ]; then
  # shellcheck disable=SC1090
  . "$HOME/.cargo/env"
fi

ROUNDS="${ROUNDS:-12}"
WRITE_OPS="${WRITE_OPS:-150000}"
READ_OPS="${READ_OPS:-100000}"
KEYSPACE="${KEYSPACE:-20000}"
READ_MISS_RATIO="${READ_MISS_RATIO:-0.20}"
SHARD_COUNT="${SHARD_COUNT:-4}"
MEMTABLE_MAX_BYTES="${MEMTABLE_MAX_BYTES:-65536}"
SSTABLE_BLOCK_BYTES="${SSTABLE_BLOCK_BYTES:-16384}"
BLOOM_BITS_PER_KEY="${BLOOM_BITS_PER_KEY:-10}"
WAL_FSYNC_EVERY_N_RECORDS="${WAL_FSYNC_EVERY_N_RECORDS:-128}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT}/overnight_runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

has_cpp=0
if command -v c++ >/dev/null 2>&1 || command -v g++ >/dev/null 2>&1; then
  has_cpp=1
fi

for i in $(seq 1 "$ROUNDS"); do
  round_dir="$OUT_DIR/round_$i"
  mkdir -p "$round_dir"

  {
    echo "=== Round $i / $ROUNDS ==="
    date -u

    echo "[1] cargo test"
    (cd "$ROOT" && cargo test)

    if [ "$has_cpp" -eq 1 ]; then
      echo "[2] cargo test --features native"
      (cd "$ROOT" && cargo test --features native)
    fi

    echo "[3] bench (rust path)"
    db_path="$round_dir/db"
    (cd "$ROOT" && cargo run -q -p spectradb-cli -- --path "$db_path" \
      --shard-count "$SHARD_COUNT" \
      --memtable-max-bytes "$MEMTABLE_MAX_BYTES" \
      --sstable-block-bytes "$SSTABLE_BLOCK_BYTES" \
      --bloom-bits-per-key "$BLOOM_BITS_PER_KEY" \
      --wal-fsync-every-n-records "$WAL_FSYNC_EVERY_N_RECORDS" \
      bench \
      --write-ops "$WRITE_OPS" --read-ops "$READ_OPS" --keyspace "$KEYSPACE" --read-miss-ratio "$READ_MISS_RATIO")

    if [ "$has_cpp" -eq 1 ]; then
      echo "[4] bench (native path)"
      db_native_path="$round_dir/db_native"
      (cd "$ROOT" && cargo run -q -p spectradb-cli --features native -- --path "$db_native_path" \
        --shard-count "$SHARD_COUNT" \
        --memtable-max-bytes "$MEMTABLE_MAX_BYTES" \
        --sstable-block-bytes "$SSTABLE_BLOCK_BYTES" \
        --bloom-bits-per-key "$BLOOM_BITS_PER_KEY" \
        --wal-fsync-every-n-records "$WAL_FSYNC_EVERY_N_RECORDS" \
        bench \
        --write-ops "$WRITE_OPS" --read-ops "$READ_OPS" --keyspace "$KEYSPACE" --read-miss-ratio "$READ_MISS_RATIO")
    fi

    echo "[5] sql smoke"
    (cd "$ROOT" && cargo run -q -p spectradb-cli -- --path "$db_path" sql "CREATE TABLE nightly (pk TEXT PRIMARY KEY);")
    (cd "$ROOT" && cargo run -q -p spectradb-cli -- --path "$db_path" sql "INSERT INTO nightly (pk, doc) VALUES ('k', '{\\\"ok\\\":true}');")
    (cd "$ROOT" && cargo run -q -p spectradb-cli -- --path "$db_path" sql "SELECT doc FROM nightly WHERE pk='k';")
  } >"$round_dir/report.txt" 2>&1

done

echo "Overnight iteration completed. Reports in: $OUT_DIR"
