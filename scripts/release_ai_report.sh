#!/usr/bin/env bash
set -euo pipefail

if [ -f "$HOME/.cargo/env" ]; then
  # shellcheck disable=SC1090
  . "$HOME/.cargo/env"
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATE_UTC="$(date -u +%Y-%m-%d)"
STAMP_UTC="$(date -u +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$ROOT/output/release_reports}"
mkdir -p "$OUT_DIR"
REPORT_PATH="${REPORT_PATH:-$OUT_DIR/ai_overhead_${STAMP_UTC}.md}"

WORK_DIR="$(mktemp -d /tmp/tensordb-release-ai-report-XXXXXX)"
OFF_OUT="$WORK_DIR/bench_ai_off.txt"
ON_OUT="$WORK_DIR/bench_ai_on.txt"

WRITE_OPS="${WRITE_OPS:-10000}"
READ_OPS="${READ_OPS:-5000}"
KEYSPACE="${KEYSPACE:-3000}"
READ_MISS_RATIO="${READ_MISS_RATIO:-0.10}"
SHARD_COUNT="${SHARD_COUNT:-2}"
MEMTABLE_MAX_BYTES="${MEMTABLE_MAX_BYTES:-65536}"
SSTABLE_BLOCK_BYTES="${SSTABLE_BLOCK_BYTES:-16384}"
BLOOM_BITS_PER_KEY="${BLOOM_BITS_PER_KEY:-10}"
WAL_FSYNC_EVERY_N_RECORDS="${WAL_FSYNC_EVERY_N_RECORDS:-128}"
AI_BATCH_WINDOW_MS="${AI_BATCH_WINDOW_MS:-20}"
AI_BATCH_MAX_EVENTS="${AI_BATCH_MAX_EVENTS:-16}"

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

echo "[release-ai-report] running benchmark (AI off)"
run_bench "$OFF_OUT" \
  --path "$WORK_DIR/db_ai_off" \
  --shard-count "$SHARD_COUNT" \
  --memtable-max-bytes "$MEMTABLE_MAX_BYTES" \
  --sstable-block-bytes "$SSTABLE_BLOCK_BYTES" \
  --bloom-bits-per-key "$BLOOM_BITS_PER_KEY" \
  --wal-fsync-every-n-records "$WAL_FSYNC_EVERY_N_RECORDS"

echo "[release-ai-report] running benchmark (AI on)"
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
off_p50="$(value_of read_p50_us "$OFF_OUT")"
on_p50="$(value_of read_p50_us "$ON_OUT")"
off_p95="$(value_of read_p95_us "$OFF_OUT")"
on_p95="$(value_of read_p95_us "$ON_OUT")"
off_p99="$(value_of read_p99_us "$OFF_OUT")"
on_p99="$(value_of read_p99_us "$ON_OUT")"
off_ai_enabled="$(value_of ai_enabled "$OFF_OUT")"
on_ai_enabled="$(value_of ai_enabled "$ON_OUT")"
on_ai_events="$(value_of ai_events_received "$ON_OUT")"
on_ai_insights="$(value_of ai_insights_written "$ON_OUT")"
on_ai_failures="$(value_of ai_write_failures "$ON_OUT")"

write_delta_pct="$(pct_delta "$off_write" "$on_write")"
p50_delta_pct="$(pct_delta "$off_p50" "$on_p50")"
p95_delta_pct="$(pct_delta "$off_p95" "$on_p95")"
p99_delta_pct="$(pct_delta "$off_p99" "$on_p99")"

cat > "$REPORT_PATH" <<EOF
# AI Overhead Release Report

- Date (UTC): $DATE_UTC
- Generated at (UTC): $(date -u +"%Y-%m-%dT%H:%M:%SZ")
- Workload: write_ops=$WRITE_OPS read_ops=$READ_OPS keyspace=$KEYSPACE read_miss_ratio=$READ_MISS_RATIO
- Engine knobs: shard_count=$SHARD_COUNT memtable_max_bytes=$MEMTABLE_MAX_BYTES sstable_block_bytes=$SSTABLE_BLOCK_BYTES bloom_bits_per_key=$BLOOM_BITS_PER_KEY wal_fsync_every_n_records=$WAL_FSYNC_EVERY_N_RECORDS
- AI knobs: ai_batch_window_ms=$AI_BATCH_WINDOW_MS ai_batch_max_events=$AI_BATCH_MAX_EVENTS

## Metrics

| Metric | AI Off | AI On | Delta % (On vs Off) |
|---|---:|---:|---:|
| write_ops_per_sec | $off_write | $on_write | $(printf "%.2f" "$write_delta_pct") |
| read_p50_us | $off_p50 | $on_p50 | $(printf "%.2f" "$p50_delta_pct") |
| read_p95_us | $off_p95 | $on_p95 | $(printf "%.2f" "$p95_delta_pct") |
| read_p99_us | $off_p99 | $on_p99 | $(printf "%.2f" "$p99_delta_pct") |

## AI Runtime Counters (AI On Run)

- ai_enabled (off run): $off_ai_enabled
- ai_enabled (on run): $on_ai_enabled
- ai_events_received: $on_ai_events
- ai_insights_written: $on_ai_insights
- ai_write_failures: $on_ai_failures

## Raw Outputs

- AI off output: \`$OFF_OUT\`
- AI on output: \`$ON_OUT\`
EOF

echo "[release-ai-report] wrote $REPORT_PATH"
