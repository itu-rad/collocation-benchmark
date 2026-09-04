#!/usr/bin/env bash
# E4 (Self-RAG decomposition, prefill/decode split) — timing collection.
#
# Four arms x two tasks per device:
#   monolith (9B) · monolith_4b · decomposed (4B per role) · decomposed_shared
#   factoid · multihop
#
# Quality is NOT recollected here: the 16-cell quality matrix was already
# validated (greedy decoding, judge overturn = 0). This collects TIMING with the
# bulk+proc tracing, for the prefill(compute)/decode(memory) split.
#
#   usage: collect.sh <device> [runs] [exp] [config-glob]
#     device : mlx | cuda     (selects configs/*_<device>.yml)
set -uo pipefail
DEVICE=${1:?device (mlx|cuda)}; RUNS=${2:-3}; EXP=${3:-138}; GLOB=${4:-*}
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd); cd "$ROOT"

OUT="$HERE/results/$DEVICE"; mkdir -p "$OUT"
export SUITE_PROC_TRACE=1 RADT_TRACE_BACKEND=radt   # bulk (batch) span export
export RADT_PRESENT=True                              # no RADT_LISTENER_* -> no listeners

SUM="$HERE/collect_summary_${DEVICE}.tsv"
[ -f "$SUM" ] || printf 'arm\trun\trc\tseconds\tcsv_rows\n' > "$SUM"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

shopt -s nullglob
cfgs=( "$HERE"/configs/${GLOB}_"$DEVICE".yml )
# keep only the four E4 arms (drop the capacity-ladder and backend variants)
keep=(); for c in "${cfgs[@]}"; do
  b=$(basename "$c" .yml)
  case "$b" in
    *ollama*|*vllm*|*_0.8b_*|*_2b_*|*_27b_*) continue ;;
  esac
  keep+=("$c")
done
(( ${#keep[@]} )) || { echo "no configs match ${GLOB}_${DEVICE}" >&2; exit 1; }

log "E4: ${#keep[@]} arm(s) x $RUNS run(s) on $DEVICE (exp $EXP)"
fail=0
for r in $(seq 1 "$RUNS"); do
  for cfg in "${keep[@]}"; do
    name=$(basename "$cfg" .yml)
    lab="e4_${name}_r${r}"
    start=$(date +%s)
    python main.py "$cfg" -p 0 -e "$EXP" --label "$lab"
    rc=$?; secs=$(( $(date +%s) - start ))
    for ext in csv jsonl; do
      [ -f "evaluation/results/$lab.$ext" ] && mv "evaluation/results/$lab.$ext" "$OUT/"
    done
    rows=$( [ -f "$OUT/$lab.csv" ] && wc -l < "$OUT/$lab.csv" | tr -d ' ' || echo 0 )
    printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$r" "$rc" "$secs" "$rows" >> "$SUM"
    [ "$rc" -ne 0 ] && fail=$((fail+1))
    log "  $lab rc=$rc ${secs}s rows=$rows"
  done
done
log "E4 done on $DEVICE ($fail failed)"
touch "$HERE/DONE_collect_${DEVICE}"
[ "$fail" -eq 0 ]
