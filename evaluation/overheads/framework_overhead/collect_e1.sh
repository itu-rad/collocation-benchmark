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
#   arm             config              CSV logging  spans   collected?
#   uninstrumented  ..._mode_ref_nolog      off       off    YES - the framework bare
#   spans-only      ..._mode_ref_nolog      off       on     YES - the framework traced
#   as-reported     ..._mode_ref            on        off    no  - diagnostic, answered
#   both            ..._mode_ref            on        on     no  - diagnostic, answered
#
#   spans-only - uninstrumented = what it costs to run with tracing on, which
#   is the only instrument question left, because those two are the only ways
#   the framework is actually run.
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
#     depth-list : space-separated depths (default: powers of two, 1..128)
#
#   E1_PAYLOAD=1 collect_e1.sh <device> [runs] [exp]
#     collects the PAYLOAD sweep instead: depth 10, sizes {0, 1 KiB, 1 MiB,
#     10 MiB} x modes {ref, copy}, `as-reported` arm only (see below).
set -uo pipefail

usage() { sed -n '2,40p' "$0" >&2; exit 2; }
DEVICE=${1:-}; RUNS=${2:-11}; EXP=${3:-138}
[ -n "$DEVICE" ] || usage
# Powers of two only. O(d) is a smooth curve, so 2^0..2^7 already spans the full
# 128x range that shows it flattening; the intermediate depths cost collection
# time and crowded the figures without adding shape.
[ $# -gt 3 ] && { shift 3; DEPTHS="$*"; } || DEPTHS="1 2 4 8 16 32 64 128"

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
cd "$ROOT"

CFG="$HERE/configs"
SHARED="evaluation/results"          # main.py hardcodes its CSV here
OUT="$SHARED/$DEVICE"                # analyze_e1.py globs one dir per device
mkdir -p "$OUT"

export RADT_PRESENT=True             # end_run + drain on exit; no RADT_LISTENER_* -> no listeners
export RADT_TRACE_BACKEND=radt       # force the BULK exporter rather than auto-detection

# Tracking store: the AMBIENT res17 server, with radt's BULK span exporter.
# Do not point this at a local sqlite/file store. The bulk exporter spools spans
# to a gzipped artifact and uploads them to the run on res17, and that artifact
# is the record; a local store makes runs that cannot be cross-referenced with
# every other experiment.
#
# Credentials come from the conda env (MLFLOW_TRACKING_URI /
# MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD), so nothing is set here
# and nothing is committed. Fail loudly rather than silently falling back to a
# local ./mlruns, which is what a missing URI would otherwise produce.
if [ -z "${MLFLOW_TRACKING_URI:-}" ]; then
  echo "collect_e1: MLFLOW_TRACKING_URI is unset -- activate the benchmark conda env" >&2
  echo "  (a missing URI silently writes a local ./mlruns instead of reaching res17)" >&2
  exit 2
fi
case "${MLFLOW_TRACKING_URI}" in
  sqlite:*|file:*|./*|/*)
    echo "collect_e1: MLFLOW_TRACKING_URI is a LOCAL store (${MLFLOW_TRACKING_URI})." >&2
    echo "  E1 records to the res17 server so its span artifacts sit with every" >&2
    echo "  other experiment's. Point it at the server and re-run." >&2
    exit 2 ;;
esac
echo "E1: recording to ${MLFLOW_TRACKING_URI} (experiment $EXP), bulk span export"

SUM="$HERE/collect_e1_summary_${DEVICE}.tsv"
[ -f "$SUM" ] || printf 'arm\tdepth\trun\trc\tseconds\tcsv_rows\tspans\n' > "$SUM"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# Optional CPU pinning. On GB10 the Grace CPU is heterogeneous (10x Cortex-X925
# performance + 10x A725 efficiency) and an unpinned workload migrates between
# them; E1 measures us-scale transitions, so that noise is fatal and the cuda
# collection MUST pin to a single performance core:
#   PIN=18,19 collect_e1.sh cuda ...     <- the GB10 standard, TWO cores
#
# Two cores, not one, and not more. Measured R=5 at depth 128 (us/stage):
#
#   cores   uninstrumented   spans-only
#     1         10.04          33.42   <- exporter starved
#     2         16.05          32.01   <- neither starved
#     3         16.37          48.14
#
# One core is cheapest for uninstrumented, because every stage-to-stage hand-off
# stays on that core with no inter-core wakeup and a warm cache. But taskset
# affinity is INHERITED BY CHILDREN, so the radt exporter process and the span
# queue's feeder thread land on that same core and starve: the span cost is flat
# to depth ~20 and then roughly doubles, which is our apparatus, not tracing.
# Three or more cores is worse again -- hand-offs start crossing cores faster
# than the extra capacity helps.
#
# Both arms therefore get the SAME two cores, so the difference between them is
# attributable to the instrument and not to the CPU they were given.
#
# The M2 Pro has no taskset, so mlx is collected unpinned. That asymmetry is
# real and is why the two devices' absolute per-stage costs are not directly
# comparable.
# NOTE: expanded as ${PINCMD[@]+"${PINCMD[@]}"} — macOS bash 3.2 treats a plain
# "${PINCMD[@]}" on an EMPTY array as unbound under `set -u`.
PINCMD=()
if [ -n "${PIN:-}" ] && command -v taskset >/dev/null 2>&1; then
  PINCMD=(taskset -c "$PIN")
  log "pinning workload to core(s) [$PIN]"
fi

# The PAYLOAD sweep (E1_PAYLOAD=1) is a different cell set, run in the SAME two
# arms as the depth sweep. It used to run in `as-reported` because its metric
# was per-stage self-duration, which needs the per-stage CSV rows. It no longer
# has to:
#
#   uninstrumented - the metric is L_q / depth. The pipeline-level rows are
#                    always written, so this needs no instrument at all, and it
#                    is the framework's true payload behaviour.
#   spans-only     - the same, plus the per-stage breakdown recovered from the
#                    spans themselves: "<stage>.run" start to
#                    "<stage>.push_to_outputs" start IS the stage's
#                    self-duration, which is where a deep copy happens.
PAYLOAD_DEPTH=10
PAYLOAD_SIZES="0 1024 1048576 10485760"
PAYLOAD_MODES="ref copy"

# The TWO ways the framework actually runs, and nothing else. The diagnostic
# arms that carried the CSV instrument are retired: they answered whether E1's
# historical number was mostly instrument (it was, 58-71%), and neither is a
# configuration anyone would ship. E1_ARMS restricts to one of the two.
if [ "${E1_PAYLOAD:-0}" = "1" ]; then
  ARMS=(${E1_ARMS:-uninstrumented spans-only})
else
  # shellcheck disable=SC2206
  ARMS=(${E1_ARMS:-uninstrumented spans-only})
fi
NARMS=${#ARMS[@]}

# config + tracing env for one arm
arm_config() {
  # Both arms use the logs-off config: neither writes per-stage CSV rows.
  echo "$CFG/noop_depth_${2}_size_0_mode_ref_nolog.yml"
}

run_one() {
  local arm=$1 depth=$2 r=$3 cfg lab start rc secs rows spans
  cfg=$(arm_config "$arm" "$depth")
  [ -f "$cfg" ] || { log "  !! missing config $cfg"; return 1; }
  lab="noop_depth_${depth}_size_0_mode_ref_${arm}_${DEVICE}_r${r}"
  [ -f "$OUT/$lab.csv" ] && { log "  [skip] $lab (exists)"; return 0; }

  unset CHOREO_DISABLE_TRACING CHOREO_PROC_TRACE
  case $arm in
    uninstrumented) export CHOREO_DISABLE_TRACING=1 ;;
    spans-only)     export CHOREO_PROC_TRACE=1 ;;
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

run_payload_one() {
  local arm=$1 size=$2 mode=$3 r=$4 cfg lab start rc secs rows spans
  cfg="$CFG/noop_depth_${PAYLOAD_DEPTH}_size_${size}_mode_${mode}_nolog.yml"
  [ -f "$cfg" ] || { log "  !! missing config $cfg"; return 1; }
  lab="noop_depth_${PAYLOAD_DEPTH}_size_${size}_mode_${mode}_${arm}_${DEVICE}_r${r}"
  [ -f "$OUT/$lab.csv" ] && { log "  [skip] $lab (exists)"; return 0; }
  unset CHOREO_DISABLE_TRACING CHOREO_PROC_TRACE
  case $arm in
    uninstrumented) export CHOREO_DISABLE_TRACING=1 ;;
    spans-only)     export CHOREO_PROC_TRACE=1 ;;
  esac
  local outfile; outfile=$(mktemp)
  start=$(date +%s)
  ${PINCMD[@]+"${PINCMD[@]}"} python main.py "$cfg" -p 0 -e "$EXP" --label "$lab" 2>&1 | tee "$outfile"
  rc=${PIPESTATUS[0]}; secs=$(( $(date +%s) - start ))
  spans=$(sed -n 's/^\[choreo\] spans emitted: //p' "$outfile" | tail -1)
  rm -f "$outfile"; unset CHOREO_DISABLE_TRACING CHOREO_PROC_TRACE
  for ext in csv jsonl; do
    [ -f "$SHARED/$lab.$ext" ] && mv "$SHARED/$lab.$ext" "$OUT/"
  done
  rows=$( [ -f "$OUT/$lab.csv" ] && wc -l < "$OUT/$lab.csv" | tr -d ' ' || echo 0 )
  printf '%s\tp%s_%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$size" "$mode" "$r" "$rc" "$secs" "$rows" "${spans:-}" >> "$SUM"
  [ "$rows" -eq 0 ] && { log "  !! $lab produced NO CSV (rc=$rc)"; return 1; }
  log "  $lab rc=$rc ${secs}s rows=$rows"
}

fail=0
if [ "${E1_PAYLOAD:-0}" = "1" ]; then
  nc=$(( $(echo "$PAYLOAD_SIZES" | wc -w) * $(echo "$PAYLOAD_MODES" | wc -w) ))
  log "E1 PAYLOAD sweep: $nc cell(s) x $NARMS arm(s) x $RUNS runs on $DEVICE (exp $EXP)"
  for size in $PAYLOAD_SIZES; do
    for mode in $PAYLOAD_MODES; do
      for r in $(seq 1 "$RUNS"); do
        # Rotate arm order per repetition, as the depth sweep does, so neither
        # arm always runs first.
        off=$(( (r - 1) % NARMS ))
        for i in $(seq 0 $(( NARMS - 1 ))); do
          run_payload_one "${ARMS[$(( (off + i) % NARMS ))]}" "$size" "$mode" "$r" \
            || fail=$((fail+1))
        done
      done
    done
  done
  log "E1 payload sweep done on $DEVICE ($fail failed run(s)). CSVs in $OUT/"
  touch "$HERE/DONE_collect_e1_payload_${DEVICE}"
  exit $([ "$fail" -eq 0 ] && echo 0 || echo 1)
fi

ncells=$(echo "$DEPTHS" | wc -w | tr -d ' ')
log "E1 collection: $ncells depth(s) x $NARMS arms x $RUNS runs on $DEVICE (exp $EXP)"
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
