#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

csv="${1:-}"
if [ -z "$csv" ]; then
  latest_dir="$(ls -1dt "$ROOT"/bench_runs/* 2>/dev/null | head -n1 || true)"
  if [ -z "$latest_dir" ] || [ ! -f "$latest_dir/results.csv" ]; then
    echo "No benchmark CSV found. Pass a CSV path or run ./scripts/benchmark_matrix.sh first."
    exit 1
  fi
  csv="$latest_dir/results.csv"
fi

if [ ! -f "$csv" ]; then
  echo "CSV not found: $csv"
  exit 1
fi

echo "Benchmark summary for: $csv"
echo

for mode in rust native; do
  if ! awk -F, -v m="$mode" 'NR>1 && $1==m {found=1} END{exit(found?0:1)}' "$csv"; then
    continue
  fi

  best_write_line=$(awk -F, -v m="$mode" 'NR>1 && $1==m {print}' "$csv" | sort -t, -k7,7nr | head -n1)
  best_p99_line=$(awk -F, -v m="$mode" 'NR>1 && $1==m {print}' "$csv" | sort -t, -k10,10n | head -n1)

  echo "mode=$mode"
  echo "  best_write: $best_write_line"
  echo "  best_p99:   $best_p99_line"
  echo
 done
