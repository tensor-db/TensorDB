#!/usr/bin/env bash
set -euo pipefail

if ! command -v sqlite3 >/dev/null 2>&1; then
  echo "sqlite3 not found; install sqlite3 to run baseline"
  exit 0
fi

N="${N:-100000}"
DB="${DB:-/tmp/sdb_sqlite_baseline.db}"
rm -f "$DB"

start_ns=$(date +%s%N)
sqlite3 "$DB" <<SQL
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE kv (k TEXT PRIMARY KEY, v TEXT);
BEGIN;
WITH RECURSIVE cnt(x) AS (
  SELECT 1 UNION ALL SELECT x+1 FROM cnt WHERE x < $N
)
INSERT INTO kv(k,v)
SELECT printf('k%08d', x), printf('{"n":%d}', x) FROM cnt;
COMMIT;
SQL
end_ns=$(date +%s%N)

elapsed_s=$(awk -v a="$start_ns" -v b="$end_ns" 'BEGIN { printf "%.6f", (b-a)/1000000000 }')
ops=$(awk -v n="$N" -v s="$elapsed_s" 'BEGIN { printf "%.2f", n/s }')

echo "sqlite_insert_ops_per_sec=$ops"

a() { sqlite3 "$DB" "SELECT v FROM kv WHERE k='k00050000';" >/dev/null; }
for _ in $(seq 1 1000); do a; done

echo "sqlite_point_read_smoke=ok"
