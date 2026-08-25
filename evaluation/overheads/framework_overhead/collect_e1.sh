#!/usr/bin/env bash
# E1 (framework overhead) collection — four arms over the depth sweep.
#
# The point of the four arms is to separate the framework's cost from the cost
# of MEASURING it. E1's historical headline came from the `off` arm, whose
# per-stage timing is produced by CSV log rows -- and a log row is not free:
# measured 7.7 us single-threaded, but ~25x that under stage-thread contention
# on the logging handler lock. So `off` alone cannot say how much of the
# reported dispatch cost is the framework and how much is our own logger.
#
# Arms are named for the ROLE each plays, not the switch that sets them:
#
#   arm             config              CSV logging  spans   what it is for
#   as-reported     ..._mode_ref            on        off    how E1 has always measured
#   uninstrumented  ..._mode_ref_nolog      off       off    the framework, no instrument
#   spans-only      ..._mode_ref_nolog      off       on     spans replacing the logging
#   both            ..._mode_ref            on        on     both instruments running
#
#   as-reported - uninstrumented = what the CSV instrument costs
#   spans-only  - uninstrumented = what the span instrument costs
#
# L_q survives the nolog arms because it comes from the pipeline-level rows,
# which pipeline.py emits unconditionally -- `disable_logs` is a Stage flag and
# Pipeline has none. What nolog loses is the per-STAGE breakdown, which is
# exactly what the spans arm gives back at ~1/6 the per-event cost.
#
# All four arms are otherwise identical: same -p 0 direct path (no radt
# listeners, nothing sampling the machine), same experiment, same run count.
# Only the tracing env var and the config differ.
#
#   usage: collect_e1.sh <device> [runs] [exp] [depth-list]
#     device     : mlx | cuda   (names the results dir analyze_e1.py globs)
#     runs       : repetitions per cell per arm (default 11; run 1 is dropped
#                  as system warm-up -- the first repetition of a cell is
#                  slower for its WHOLE duration, not just its first steps)
#     exp        : res17 experiment id (default 138 = real; 142 = throwaway)
#     depth-list : space-separated depths (default: the full sweep)
set -uo pipefail

usage() { sed -n '2,40p' "$0" >&2; exit 2; }
DEVICE=${1:-}; RUNS=${2:-11}; EXP=${3:-138}
[ -n "$DEVICE" ] || usage
[ $# -gt 3 ] && { shift 3; DEPTHS="$*"; } || DEPTHS="1 2 3 4 5 6 7 8 9 10 16 32 50 64 100 128"

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
cd "$ROOT"

CFG="$HERE/configs"
SHARED="evaluation/results"          # main.py hardcodes its CSV here
OUT="$SHARED/$DEVICE"                # analyze_e1.py globs one dir per device
mkdir -p "$OUT"

export RADT_PRESENT=True             # end_run + drain on exit; no RADT_LISTENER_* -> no listeners
export RADT_TRACE_BACKEND=radt       # force the BULK exporter rather than auto-detection

# Tracking store. E1 defaults to a LOCAL store, unlike the case-study
# collections, for two reasons measured on 2026-08-25 (depth 4, tracing off,
# three runs each):
#
#     res17 : 72 s, 11 s, 132 s
#     local :  5 s,  5 s,   5 s
#
# The wall-time cost alone would turn this sweep into ~14 h instead of ~1 h, but
# the variance is the real problem: it is a sibling process doing network
# uploads for an unpredictable 11-132 s while we time microsecond-scale stage
# transitions. E1 is the one experiment where that noise is fatal.
#
# Nothing measured is lost. Every E1 quantity comes from the monotonic clock
# during the pipeline run, and the tracing arms' workload-side cost is the queue
# put, which is identical whichever store the exporter child later uploads to.
# The local store also puts the span artifacts on this disk, so the spans arm
# needs no download round-trip to analyse.
#   E1_STORE=res17 collect_e1.sh ...   -> use the ambient server instead
# SQLITE, not a file: URI -- mlflow 3.15 hard-refuses a filesystem tracking
# backend ("in maintenance mode ... migrate to a database backend"), so every
# run exits rc=1 and writes no CSV. Artifacts (the span batches) still land as
# plain directories under <repo>/mlruns/, which utils/span_reader.py reads
# directly.
if [ "${E1_STORE:-local}" = "local" ]; then
  STORE_DB="$HERE/mlruns_e1_${DEVICE}.db"
  export MLFLOW_TRACKING_URI="sqlite:///${STORE_DB}"
  unset MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD
  EXP=0                              # the local store's default experiment
  echo "E1: local tracking store at $STORE_DB (E1_STORE=res17 to override)"
fi

SUM="$HERE/collect_e1_summary_${DEVICE}.tsv"
[ -f "$SUM" ] || printf 'arm\tdepth\trun\trc\tseconds\tcsv_rows\tspans\n' > "$SUM"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# Optional CPU pinning. On GB10 the Grace CPU is heterogeneous (10x Cortex-X925
# performance + 10x A725 efficiency) and an unpinned workload migrates between
# them; E1 measures us-scale transitions, so that noise is fatal and the cuda
# collection MUST pin to a single performance core:
#   PIN=19 collect_e1.sh cuda ...
# WARNING: a SINGLE core is wrong for the span-tracing arms. taskset affinity is
# inherited by children, so the stage threads, the span queue's feeder thread and
# the radt exporter PROCESS all land on that one core. Measured on the GB10, the
# span instrument's per-stage cost is flat at ~11 us up to depth 20 and then
# roughly doubles to ~22 us -- pure CPU saturation, not a property of tracing:
# with two cores (PIN=18,19) it stays flat at ~11 us all the way to depth 64.
# The CSV-logging arms are unaffected (in-process file handler, no extra
# process), which is why as-reported and uninstrumented stay flat on one core.
# NOTE: expanded as ${PINCMD[@]+"${PINCMD[@]}"} — macOS bash 3.2 treats a plain
# "${PINCMD[@]}" on an EMPTY array as unbound under `set -u`.
PINCMD=()
if [ -n "${PIN:-}" ] && command -v taskset >/dev/null 2>&1; then
  PINCMD=(taskset -c "$PIN")
  log "pinning workload to core(s) [$PIN]"
fi

ARMS=(as-reported uninstrumented spans-only both)
NARMS=${#ARMS[@]}

# config + tracing env for one arm
arm_config() {
  case $1 in
    as-reported|both)         echo "$CFG/noop_depth_${2}_size_0_mode_ref.yml" ;;
    uninstrumented|spans-only) echo "$CFG/noop_depth_${2}_size_0_mode_ref_nolog.yml" ;;
  esac
}

run_one() {
  local arm=$1 depth=$2 r=$3 cfg lab start rc secs rows spans
  cfg=$(arm_config "$arm" "$depth")
  [ -f "$cfg" ] || { log "  !! missing config $cfg"; return 1; }
  lab="noop_depth_${depth}_size_0_mode_ref_${arm}_${DEVICE}_r${r}"
  [ -f "$OUT/$lab.csv" ] && { log "  [skip] $lab (exists)"; return 0; }

  unset CHOREO_DISABLE_TRACING CHOREO_PROC_TRACE
  case $arm in
    as-reported|uninstrumented) export CHOREO_DISABLE_TRACING=1 ;;
    spans-only|both)            export CHOREO_PROC_TRACE=1 ;;
  esac

  # Capture through a file, not a pipe: `rc=$?` after a pipeline reports the
  # LAST command's status (sed), which is always 0 and would mark every failed
  # run as successful. PIPESTATUS[0] is python's own status.
  local outfile; outfile=$(mktemp)
  start=$(date +%s)
  ${PINCMD[@]+"${PINCMD[@]}"} python main.py "$cfg" -p 0 -e "$EXP" --label "$lab" 2>&1 | tee "$outfile"
  rc=${PIPESTATUS[0]}; secs=$(( $(date +%s) - start ))
  spans=$(sed -n 's/^\[choreo\] spans emitted: //p' "$outfile" | tail -1)
  rm -f "$outfile"
  unset CHOREO_DISABLE_TRACING CHOREO_PROC_TRACE

  for ext in csv jsonl; do
    [ -f "$SHARED/$lab.$ext" ] && mv "$SHARED/$lab.$ext" "$OUT/"
  done
  rows=$( [ -f "$OUT/$lab.csv" ] && wc -l < "$OUT/$lab.csv" | tr -d ' ' || echo 0 )
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$depth" "$r" "$rc" "$secs" "$rows" "${spans:-}" >> "$SUM"
  [ "$rows" -eq 0 ] && { log "  !! $lab produced NO CSV (rc=$rc)"; return 1; }
  log "  $lab rc=$rc ${secs}s rows=$rows spans=${spans:-n/a}"
}

ncells=$(echo "$DEPTHS" | wc -w | tr -d ' ')
log "E1 collection: $ncells depth(s) x $NARMS arms x $RUNS runs on $DEVICE (exp $EXP)"
fail=0
for depth in $DEPTHS; do
  for r in $(seq 1 "$RUNS"); do
    # Rotate the arm order each repetition. Without this every repetition runs
    # off -> nolog -> proc -> spans and any within-repetition warm-up or clock
    # ramp systematically penalises whichever arm goes first, which is exactly
    # the artifact that made an E2 cell read a negative overhead.
    off=$(( (r - 1) % NARMS ))
    for i in $(seq 0 $(( NARMS - 1 ))); do
      run_one "${ARMS[$(( (off + i) % NARMS ))]}" "$depth" "$r" || fail=$((fail+1))
    done
  done
done

log "E1 collection done on $DEVICE ($fail failed run(s)). CSVs in $OUT/"
touch "$HERE/DONE_collect_e1_${DEVICE}"
[ "$fail" -eq 0 ]
