#!/usr/bin/env bash
set -euo pipefail

if [ -f "$HOME/.cargo/env" ]; then
  # shellcheck disable=SC1090
  . "$HOME/.cargo/env"
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK_DIR="${WORK_DIR:-$(mktemp -d /tmp/tensordb-ai-gate-XXXXXX)}"
mkdir -p "$WORK_DIR"

WRITE_OPS="${WRITE_OPS:-5000}"
READ_OPS="${READ_OPS:-2500}"
KEYSPACE="${KEYSPACE:-1500}"
READ_MISS_RATIO="${READ_MISS_RATIO:-0.10}"
SHARD_COUNT="${SHARD_COUNT:-2}"
MEMTABLE_MAX_BYTES="${MEMTABLE_MAX_BYTES:-65536}"
SSTABLE_BLOCK_BYTES="${SSTABLE_BLOCK_BYTES:-16384}"
BLOOM_BITS_PER_KEY="${BLOOM_BITS_PER_KEY:-10}"
WAL_FSYNC_EVERY_N_RECORDS="${WAL_FSYNC_EVERY_N_RECORDS:-128}"
AI_BATCH_WINDOW_MS="${AI_BATCH_WINDOW_MS:-20}"
AI_BATCH_MAX_EVENTS="${AI_BATCH_MAX_EVENTS:-16}"

MAX_WRITE_DROP_PCT="${MAX_WRITE_DROP_PCT:-90}"
MAX_P95_INCREASE_PCT="${MAX_P95_INCREASE_PCT:-150}"
MAX_P99_INCREASE_PCT="${MAX_P99_INCREASE_PCT:-150}"

OFF_OUT="$WORK_DIR/bench_ai_off.txt"
ON_OUT="$WORK_DIR/bench_ai_on.txt"

run_bench() {
  local out="$1"
  shift
  (cd "$ROOT" && cargo run -q -p tensordb-cli -- "$@" \
    bench --write-ops "$WRITE_OPS" --read-ops "$READ_OPS" --keyspace "$KEYSPACE" --read-miss-ratio "$READ_MISS_RATIO") > "$out"
}

value_of() {
  local key="$1"
  local file="$2"
  awk -F= -v k="$key" '$1==k {print $2; exit}' "$file"
}

pct_delta() {
  local baseline="$1"
  local current="$2"
  awk -v b="$baseline" -v c="$current" 'BEGIN { if (b <= 0) { print 0; } else { print ((c - b) / b) * 100.0; } }'
}

echo "[ai-gate] running benchmark without AI runtime"
run_bench "$OFF_OUT" \
  --path "$WORK_DIR/db_ai_off" \
  --shard-count "$SHARD_COUNT" \
  --memtable-max-bytes "$MEMTABLE_MAX_BYTES" \
  --sstable-block-bytes "$SSTABLE_BLOCK_BYTES" \
  --bloom-bits-per-key "$BLOOM_BITS_PER_KEY" \
  --wal-fsync-every-n-records "$WAL_FSYNC_EVERY_N_RECORDS"

echo "[ai-gate] running benchmark with AI runtime enabled"
run_bench "$ON_OUT" \
  --path "$WORK_DIR/db_ai_on" \
  --shard-count "$SHARD_COUNT" \
  --memtable-max-bytes "$MEMTABLE_MAX_BYTES" \
  --sstable-block-bytes "$SSTABLE_BLOCK_BYTES" \
  --bloom-bits-per-key "$BLOOM_BITS_PER_KEY" \
  --wal-fsync-every-n-records "$WAL_FSYNC_EVERY_N_RECORDS" \
  --ai-auto-insights \
  --ai-batch-window-ms "$AI_BATCH_WINDOW_MS" \
  --ai-batch-max-events "$AI_BATCH_MAX_EVENTS"

off_write="$(value_of write_ops_per_sec "$OFF_OUT")"
on_write="$(value_of write_ops_per_sec "$ON_OUT")"
off_p95="$(value_of read_p95_us "$OFF_OUT")"
on_p95="$(value_of read_p95_us "$ON_OUT")"
off_p99="$(value_of read_p99_us "$OFF_OUT")"
on_p99="$(value_of read_p99_us "$ON_OUT")"
on_ai_enabled="$(value_of ai_enabled "$ON_OUT")"

write_drop_pct="$(awk -v o="$off_write" -v n="$on_write" 'BEGIN { if (o <= 0) print 0; else print ((o - n) / o) * 100.0; }')"
p95_increase_pct="$(pct_delta "$off_p95" "$on_p95")"
p99_increase_pct="$(pct_delta "$off_p99" "$on_p99")"

echo "[ai-gate] off_write_ops_per_sec=$off_write on_write_ops_per_sec=$on_write write_drop_pct=$(printf '%.2f' "$write_drop_pct")"
echo "[ai-gate] off_p95_us=$off_p95 on_p95_us=$on_p95 p95_increase_pct=$(printf '%.2f' "$p95_increase_pct")"
echo "[ai-gate] off_p99_us=$off_p99 on_p99_us=$on_p99 p99_increase_pct=$(printf '%.2f' "$p99_increase_pct")"
echo "[ai-gate] ai_enabled_on_run=$on_ai_enabled"

if [ "$on_ai_enabled" != "true" ]; then
  echo "[ai-gate] FAIL: AI benchmark run did not report ai_enabled=true"
  exit 1
fi

if awk -v v="$write_drop_pct" -v m="$MAX_WRITE_DROP_PCT" 'BEGIN { exit(v > m ? 0 : 1) }'; then
  echo "[ai-gate] FAIL: write throughput drop $(printf '%.2f' "$write_drop_pct")% exceeds threshold ${MAX_WRITE_DROP_PCT}%"
  exit 1
fi

if awk -v v="$p95_increase_pct" -v m="$MAX_P95_INCREASE_PCT" 'BEGIN { exit(v > m ? 0 : 1) }'; then
  echo "[ai-gate] FAIL: read p95 increase $(printf '%.2f' "$p95_increase_pct")% exceeds threshold ${MAX_P95_INCREASE_PCT}%"
  exit 1
fi

if awk -v v="$p99_increase_pct" -v m="$MAX_P99_INCREASE_PCT" 'BEGIN { exit(v > m ? 0 : 1) }'; then
  echo "[ai-gate] FAIL: read p99 increase $(printf '%.2f' "$p99_increase_pct")% exceeds threshold ${MAX_P99_INCREASE_PCT}%"
  exit 1
fi

echo "[ai-gate] PASS"
