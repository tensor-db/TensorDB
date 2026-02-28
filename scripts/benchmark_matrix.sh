#!/usr/bin/env bash
set -euo pipefail

if [ -f "$HOME/.cargo/env" ]; then
  # shellcheck disable=SC1090
  . "$HOME/.cargo/env"
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT}/bench_runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

WRITE_OPS="${WRITE_OPS:-50000}"
READ_OPS="${READ_OPS:-25000}"
KEYSPACE="${KEYSPACE:-10000}"
READ_MISS_RATIO="${READ_MISS_RATIO:-0.10}"

read -r -a SHARDS <<< "${SHARDS:-1 4}"
read -r -a MEMTABLES <<< "${MEMTABLES:-65536 1048576}"
read -r -a BLOCKS <<< "${BLOCKS:-4096 16384}"
read -r -a BLOOMS <<< "${BLOOMS:-8 10}"
read -r -a FSYNCS <<< "${FSYNCS:-1 128}"

CSV="$OUT_DIR/results.csv"
echo "mode,shards,memtable,block,bloom,fsync,write_ops_per_sec,read_p50_us,read_p95_us,read_p99_us,requested_read_miss_ratio,observed_read_miss_ratio,bloom_miss_rate,mmap_reads,hasher" > "$CSV"

run_one() {
  local mode="$1"
  local features=()
  if [ "$mode" = "native" ]; then
    features=(--features native)
  fi

  for shard in "${SHARDS[@]}"; do
    for mem in "${MEMTABLES[@]}"; do
      for block in "${BLOCKS[@]}"; do
        for bloom in "${BLOOMS[@]}"; do
          for fsync in "${FSYNCS[@]}"; do
            db="$OUT_DIR/db_${mode}_${shard}_${mem}_${block}_${bloom}_${fsync}"
            out="$OUT_DIR/out_${mode}_${shard}_${mem}_${block}_${bloom}_${fsync}.txt"

            (cd "$ROOT" && cargo run -q -p tensordb-cli "${features[@]}" -- \
              --path "$db" \
              --shard-count "$shard" \
              --memtable-max-bytes "$mem" \
              --sstable-block-bytes "$block" \
              --bloom-bits-per-key "$bloom" \
              --wal-fsync-every-n-records "$fsync" \
              bench --write-ops "$WRITE_OPS" --read-ops "$READ_OPS" --keyspace "$KEYSPACE" --read-miss-ratio "$READ_MISS_RATIO") > "$out"

            w=$(awk -F= '/write_ops_per_sec=/{print $2}' "$out")
            p50=$(awk -F= '/read_p50_us=/{print $2}' "$out")
            p95=$(awk -F= '/read_p95_us=/{print $2}' "$out")
            p99=$(awk -F= '/read_p99_us=/{print $2}' "$out")
            req_miss=$(awk -F= '/requested_read_miss_ratio=/{print $2}' "$out")
            obs_miss=$(awk -F= '/observed_read_miss_ratio=/{print $2}' "$out")
            miss=$(awk -F= '/bloom_miss_rate=/{print $2}' "$out")
            mmap=$(awk -F= '/mmap_reads=/{print $2}' "$out")
            hasher=$(awk -F= '/hasher=/{print $2}' "$out")

            echo "$mode,$shard,$mem,$block,$bloom,$fsync,$w,$p50,$p95,$p99,$req_miss,$obs_miss,$miss,$mmap,$hasher" >> "$CSV"
          done
        done
      done
    done
  done
}

run_one rust

if command -v c++ >/dev/null 2>&1 || command -v g++ >/dev/null 2>&1; then
  run_one native
fi

echo "Benchmark matrix complete: $CSV"
