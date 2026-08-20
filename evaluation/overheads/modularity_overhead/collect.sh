#!/usr/bin/env bash
# E2 (modularity overhead) collection — three arms over the scale sweep.
#
#   baseline : baseline_finetune.py           — the bare monolith (no framework)
#   t0       : main.py, CHOREO_DISABLE_TRACING=1 — Choreo core wrapper, tracing off
#   t2       : main.py, CHOREO_PROC_TRACE=1      — + radt bulk+proc span export
#
# (t1, the old in-process mlflow exporter, is NOT collected — superseded; see
# EXPERIMENTS.md "tracing is infrastructure, not a contribution".)
#
# Both arms run the IDENTICAL workload per cell: the static configs written by
# gen_configs.py and the same (model, weights, batch) on the baseline command
# line. num_workers=0 in both. Choreo runs direct (`-p 0`) with no listeners, so
# nothing perturbs the per-step measurement.
#
#   usage: collect.sh <device> [runs] [exp] [cell-glob]
#     device    : mps | cuda    (selects configs/generated/*_<device>.yml)
#     runs      : repeats per cell per arm (default 10)
#     exp       : res17 experiment id (default 138)
#     cell-glob : restrict to matching cells, e.g. 'meffv2s_b8' (default: all).
#                 Lets the sweep be staged (anchor at high R, curve cells lower).
set -uo pipefail

usage() { sed -n '2,22p' "$0" >&2; exit 2; }
DEVICE=${1:-}; RUNS=${2:-10}; EXP=${3:-138}; CELLGLOB=${4:-*}
[ -n "$DEVICE" ] || usage

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
cd "$ROOT"

RESULTS="$HERE/results"          # E2 keeps its own results dir (existing scheme)
CHOREO_OUT="evaluation/results"  # where main.py writes before we move it
mkdir -p "$RESULTS"

# Force the BULK (batch) span exporter for the t2 arm rather than auto-detection.
export RADT_TRACE_BACKEND=radt
export RADT_PRESENT=True         # end_run + drain on exit; no RADT_LISTENER_* -> no listeners

SUM="$HERE/collect_summary_${DEVICE}.tsv"
[ -f "$SUM" ] || printf 'arm\tcell\trun\trc\tseconds\tcsv_rows\n' > "$SUM"   # append across staged calls

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# Optional CPU pinning. On GB10 the Grace CPU is heterogeneous (10x Cortex-X925
# performance + 10x A725 efficiency cores) and an unpinned workload migrates
# between them, which added several hundred us of run-to-run noise to E1. E2
# steps are ms-scale so the CLUSTER (all X925) is enough here — unlike E1 we must
# NOT pin to a single core, because the dataloader thread and (for t2) the span
# exporter child need their own cores or they would serialise onto the workload.
#   PIN=5-9,15-19 collect.sh cuda ...
# NOTE: expanded as ${PINCMD[@]+"${PINCMD[@]}"} below — macOS bash 3.2 treats a
# plain "${PINCMD[@]}" on an EMPTY array as an unbound variable under `set -u`.
PINCMD=()
if [ -n "${PIN:-}" ] && command -v taskset >/dev/null 2>&1; then
  PINCMD=(taskset -c "$PIN")
  log "pinning workload to cores [$PIN]"
fi

fail=0
shopt -s nullglob
cfgs=( "$HERE"/configs/generated/mod_${CELLGLOB}_"$DEVICE".yml )
(( ${#cfgs[@]} )) || { echo "collect: no configs match mod_${CELLGLOB}_${DEVICE}.yml — run gen_configs.py first" >&2; exit 1; }

log "E2 collection: ${#cfgs[@]} cells x {baseline,t0,t2} x $RUNS runs on $DEVICE (exp $EXP)"

for cfg in "${cfgs[@]}"; do
  base=$(basename "$cfg" .yml)            # mod_m<tag>_b<batch>_<device>
  cell=${base#mod_}; cell=${cell%_"$DEVICE"}   # m<tag>_b<batch>
  # The cell's (model, weights, batch) come from the config so both arms agree.
  read -r MODEL WEIGHTS BATCH MAXB <<<"$(python - "$cfg" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))["pipelines"][0]
dl = next(s for s in c["stages"] if "DataLoader" in s["component"])["config"]
tr = next(s for s in c["stages"] if "Classification" in s["component"])["config"]
print(tr["model"]["component"].rsplit(".", 1)[-1], dl["dataset"]["weights"],
      dl["batch_size"], c["loadgen"]["max_queries"])
PY
)"

  # One arm = one function, so the ORDER of the three arms inside a repetition
  # can be varied. Without that, collect.sh always runs baseline -> t0 -> t2, and
  # any within-repetition warm-up / clock ramp systematically penalises whichever
  # arm goes first (the baseline) — which shows up as a NEGATIVE overhead, the
  # suspected cause of EfficientNetV2-L b8 on mps reading -575.8 us
  # [-737.8, -420.6] consistently across all 3 runs. A wrapper cannot make work
  # faster, so that is apparatus, not a finding.
  #   ALTERNATE=1 -> reverse the whole arm order on even repetitions, so across
  #   repetitions every arm spends equal time in the penalised first slot.
  run_baseline() {
    local r=$1 lab start rc secs rows
    lab="mod_baseline_${cell}_d${DEVICE}_r${r}"
    rm -f "$RESULTS/$lab.csv"            # never append onto a stale baseline CSV
    start=$(date +%s)
    ${PINCMD[@]+"${PINCMD[@]}"} python evaluation/overheads/modularity_overhead/baseline_finetune.py \
      --device "$DEVICE" --model "$MODEL" --weights "$WEIGHTS" \
      --batch-size "$BATCH" --num-workers 0 --max-batches "$MAXB" \
      --label "$lab" --no-radt --run "$r"
    rc=$?; secs=$(( $(date +%s) - start ))
    rows=$( [ -f "$RESULTS/$lab.csv" ] && wc -l < "$RESULTS/$lab.csv" | tr -d ' ' || echo 0 )
    printf 'baseline\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$r" "$rc" "$secs" "$rows" >> "$SUM"
    [ "$rc" -ne 0 ] && { fail=$((fail+1)); log "  !! $lab rc=$rc"; }
    log "  $lab rc=$rc ${secs}s rows=$rows"
  }

  run_choreo() {
    local arm=$1 r=$2 lab start rc secs rows ext
    lab="mod_choreo_${arm}_${cell}_d${DEVICE}_r${r}"
    if [ "$arm" = t0 ]; then
      export CHOREO_DISABLE_TRACING=1; unset CHOREO_PROC_TRACE
    else
      export CHOREO_PROC_TRACE=1; unset CHOREO_DISABLE_TRACING
    fi
    start=$(date +%s)
    ${PINCMD[@]+"${PINCMD[@]}"} python main.py "$cfg" -p 0 -e "$EXP" --label "$lab"
    rc=$?; secs=$(( $(date +%s) - start ))
    for ext in csv jsonl; do
      [ -f "$CHOREO_OUT/$lab.$ext" ] && mv "$CHOREO_OUT/$lab.$ext" "$RESULTS/"
    done
    rows=$( [ -f "$RESULTS/$lab.csv" ] && wc -l < "$RESULTS/$lab.csv" | tr -d ' ' || echo 0 )
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$cell" "$r" "$rc" "$secs" "$rows" >> "$SUM"
    [ "$rc" -ne 0 ] && { fail=$((fail+1)); log "  !! $lab rc=$rc"; }
    log "  $lab rc=$rc ${secs}s rows=$rows"
    unset CHOREO_DISABLE_TRACING CHOREO_PROC_TRACE
  }

  for r in $(seq 1 "$RUNS"); do
    if [ "${ALTERNATE:-0}" = "1" ] && [ $(( r % 2 )) -eq 0 ]; then
      log "  (alternating: t2 -> t0 -> baseline for r$r)"
      run_choreo t2 "$r"; run_choreo t0 "$r"; run_baseline "$r"
    else
      run_baseline "$r"; run_choreo t0 "$r"; run_choreo t2 "$r"
    fi
  done
done

log "E2 collection done on $DEVICE ($fail failed run(s)). CSVs in $RESULTS/"
touch "$HERE/DONE_collect_${DEVICE}"
[ "$fail" -eq 0 ]
