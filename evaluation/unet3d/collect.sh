#!/usr/bin/env bash
# E3 (MLPerf / 3D-UNet) Choreo collection — 42 KiTS19 cases, online serving.
#
# One request in flight (serialize_queries + queue_depth 1), batch 1: MLPerf's
# SingleStream regime, the one where preprocessing/loading cannot be prefetched
# and therefore sits on the per-request critical path. Choreo times the WHOLE
# graph (load -> preprocess -> sliding-window inference); MLPerf times only the
# inference. That gap is prong 2.
#
#   usage: collect.sh <device> [runs] [exp]
#     device : cuda | mps
set -uo pipefail
DEVICE=${1:?device (cuda|mps)}; RUNS=${2:-5}; EXP=${3:-138}
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd); cd "$ROOT"

CFG="$HERE/configs/unet3d_42_${DEVICE}.yml"
[ -f "$CFG" ] || { echo "no config: $CFG" >&2; exit 1; }
OUT="$HERE/results/$DEVICE"; mkdir -p "$OUT"
export CHOREO_PROC_TRACE=1 RADT_TRACE_BACKEND=radt   # bulk (batch) span export
export RADT_PRESENT=True                              # no RADT_LISTENER_* -> no listeners

SUM="$HERE/collect_summary_${DEVICE}.tsv"
[ -f "$SUM" ] || printf 'run\trc\tseconds\tcsv_rows\n' > "$SUM"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
log "E3: 42 cases x $RUNS runs on $DEVICE (exp $EXP)"
fail=0
for r in $(seq 1 "$RUNS"); do
  lab="unet3d_42_${DEVICE}_r${r}"
  start=$(date +%s)
  python main.py "$CFG" -p 0 -e "$EXP" --label "$lab"
  rc=$?; secs=$(( $(date +%s) - start ))
  for ext in csv jsonl; do
    [ -f "evaluation/results/$lab.$ext" ] && mv "evaluation/results/$lab.$ext" "$OUT/"
  done
  rows=$( [ -f "$OUT/$lab.csv" ] && wc -l < "$OUT/$lab.csv" | tr -d ' ' || echo 0 )
  printf '%s\t%s\t%s\t%s\n' "$r" "$rc" "$secs" "$rows" >> "$SUM"
  [ "$rc" -ne 0 ] && fail=$((fail+1))
  log "  $lab rc=$rc ${secs}s rows=$rows"
done
log "E3 done on $DEVICE ($fail failed)"
touch "$HERE/DONE_collect_${DEVICE}"
[ "$fail" -eq 0 ]
