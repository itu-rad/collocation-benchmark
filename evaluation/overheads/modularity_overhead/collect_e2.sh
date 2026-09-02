#!/usr/bin/env bash
# E2 (modularity overhead) collection — three configurations over the scale sweep.
#
# The question is what wrapping a training loop in Choreo costs against a bare
# hand-written PyTorch loop. Three things are run per cell, named for what they
# are rather than for the switch that produces them:
#
#   monolith        baseline_finetune.py, no framework at all
#   choreo          main.py, CHOREO_DISABLE_TRACING=1
#   choreo-traced   main.py, CHOREO_PROC_TRACE=1
#
# METRIC OF RECORD: TIME PER QUERY (equivalently per batch — one query is one
# batch here), measured start-of-step to start-of-next-step. It covers the whole
# cycle including data loading, and it is anchor-invariant: in steady state the
# same period is measured whether you cut at the pipeline row, the training row,
# or the monolith's step row. That is what makes the monolith and Choreo
# comparable even though they emit different markers.
#
# The previous metric was the training step's own duration, compared across the
# two processes. It is unusable: the framework's cost lands mostly BETWEEN
# steps, which that marker excludes by construction, so it measured a near-zero
# difference against +/-600 us of run-to-run noise and flipped sign between
# repetitions. One cell settled at -575 us, i.e. "the wrapper makes work
# faster".
#
# QUERY LATENCY BREAKDOWN: `choreo-traced` additionally yields the per-stage
# latencies (dataloader, training) and the auxiliary framework overheads
# (entry, handoff, exit, turnaround) from the spans. Those are successive
# instants within ONE query on ONE clock, so unlike any cross-process
# difference they are non-negative by construction and carry no run-level term.
#
# Both Choreo configurations run with `disable_logs: true`, so neither writes
# per-stage CSV rows; the pipeline-level rows that carry the timing are emitted
# unconditionally by pipeline.py. The monolith writes 2 rows/step through the
# same synchronous FileHandler that main.py installs, so the instrument is the
# same on both sides.
#
#   usage: collect_e2.sh <machine> [runs] [cell-glob]
#     machine   : m2pro | gb10   (names the results dir and the run labels; the
#                 torch device string lives in the config, not here)
#     runs      : repetitions per cell per configuration (default 11; run 1 is
#                 dropped as system warm-up — the first repetition of a cell is
#                 slower for its WHOLE duration, not just its first steps)
#     cell-glob : restrict to matching cells, e.g. 'meffv2s_b8' (default: all)
set -uo pipefail

usage() { sed -n '2,44p' "$0" >&2; exit 2; }
MACHINE=${1:-}; RUNS=${2:-11}; CELLGLOB=${3:-*}
[ -n "$MACHINE" ] || usage

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
cd "$ROOT"

# Keep the machine awake for the whole collection, and re-exec under caffeinate
# rather than asking the operator to remember it. A 2026-08-27 m2pro run was
# collecting fine at ~170 s/run until the laptop was left on battery; it then
# spent the night cycling through Deep Idle / DarkWake, and three runs came back
# with 20-25 steps above 2x the median and their MEDIAN moved 4-6%. E2 is trying
# to resolve effects around 0.05% of a step, so those runs were unusable -- and
# they carried rc=0 and exactly the right row count, so nothing but the timing
# said anything was wrong. -d display, -i idle, -m disk, -s system, -u declares
# the user active.
#
# This is prevention, not detection: keep printing per-run seconds in the log so
# a wall-clock anomaly is still visible if something else stalls a run.
if [ "$(uname)" = "Darwin" ] && [ -z "${E2_NO_CAFFEINATE:-}" ] \
   && [ -z "${_E2_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  export _E2_CAFFEINATED=1
  exec caffeinate -dimsu "$0" "$@"
fi

RESULTS="$HERE/results"
# main.py and TerminalCapture write straight into this experiment's
# results/ dir; there is no shared staging directory to sweep.
export BENCH_OUTPUT_DIR="$RESULTS"
mkdir -p "$RESULTS"

export RADT_TRACE_BACKEND=radt       # force the BULK exporter, not auto-detection
# RADT_PRESENT makes main.py enter radt.run.RADTBenchmark, which end_run()s and
# drains on exit. No RADT_LISTENER_* is set, so no macmon/top listener spawns —
# but note __enter__ starts TWO _MLFlowLogger child processes unconditionally
# (radt/run/benchmark.py), which the --no-radt monolith does not have. They wake
# at 0.2 Hz in separate processes against a per-step effect at 1.5-100 Hz, and
# their startup lands inside the dropped warm-up, so the asymmetry is small —
# but it is real and belongs in the methods, not in a comment claiming there is
# nothing there.
export RADT_PRESENT=True

# Tracking store: LOCAL, and deliberately so — the exemption covers the OVERHEAD
# experiments (E1, E2) only; every other experiment records to res17. Measured
# on E1 at depth 8, same code, same runs: recording to res17 left the framework's
# own number alone (-2.8%, noise) but inflated the traced configuration by
# +43.5%. That is an artifact of microbenchmarking — a rate of span emission no
# real workload approaches — so measuring against a remote server would report
# the server rather than the framework. E2_STORE=res17 overrides.
if [ "${E2_STORE:-local}" = "res17" ]; then
  [ -n "${MLFLOW_TRACKING_URI:-}" ] || { echo "collect_e2: E2_STORE=res17 but MLFLOW_TRACKING_URI is unset" >&2; exit 2; }
  EXP=${E2_EXPERIMENT:-138}
  echo "E2: recording to ${MLFLOW_TRACKING_URI} (experiment $EXP)"
else
  STORE_DB="$HERE/mlruns_e2_${MACHINE}.db"
  export MLFLOW_TRACKING_URI="sqlite:///${STORE_DB}"
  unset MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD
  EXP=0
  echo "E2: local store $STORE_DB (overhead-experiment exemption; E2_STORE=res17 to override)"
fi

# The radt install must be the PATCHED checkout. A stale site-packages copy
# shadowing it would silently reintroduce the teardown race and lose spans.
python scripts/radt_gate.py || exit 3

# Environment gate. The retired run_modularity.py had one and collect.sh lost
# it, so a mismatched environment could burn a whole collection before anyone
# noticed. Fail loudly here instead.
python - "$MACHINE" <<'PY' || exit 3
import sys
machine = sys.argv[1]
try:
    import radt, torch
except Exception as e:
    sys.exit(f"collect_e2: cannot import radt/torch: {e}")
if not hasattr(radt, "trace"):
    sys.exit("collect_e2: this radt has no radt.trace — the traced configuration "
             "would silently collect nothing")
want = {"gb10": "cuda", "m2pro": "mps"}.get(machine)
if want == "cuda" and not torch.cuda.is_available():
    sys.exit("collect_e2: machine is gb10 but torch.cuda.is_available() is False")
if want == "mps" and not torch.backends.mps.is_available():
    sys.exit("collect_e2: machine is m2pro but torch MPS is unavailable")
PY

# Provenance and the durable log live with the harness, not with the caller and
# not in /tmp — an E1 log written to /tmp was lost to a reboot and completeness
# had to be reconstructed from the CSVs. Both files are stamped with the
# collection's start time rather than appended to: an append-only summary is
# what silently mixed two collection eras in E1, averaging two different span
# counts into a meaningless number that nearly hid a bad deploy.
RUNSTAMP=$(date '+%Y%m%d-%H%M%S')
LOGDIR="$HERE/collect_logs"
mkdir -p "$LOGDIR"
SUM="$LOGDIR/collect_e2_${MACHINE}_${RUNSTAMP}.tsv"
LOG="$LOGDIR/collect_e2_${MACHINE}_${RUNSTAMP}.log"
printf 'config\tcell\trun\trc\tseconds\tcsv_rows\tspans\n' > "$SUM"

log(){ local m="[$(date '+%m-%d %H:%M:%S')] $*"; echo "$m"; echo "$m" >> "$LOG"; }

{
  echo "# E2 collection"
  echo "# started      : $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "# machine      : $MACHINE"
  echo "# git_commit   : $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "# git_dirty    : $(test -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" && echo yes || echo no)"
  echo "# host         : $(hostname)"
  echo "# platform     : $(python -c 'import platform;print(platform.platform())' 2>/dev/null)"
  echo "# python       : $(python -c 'import sys;print(sys.version.split()[0])' 2>/dev/null)"
  echo "# torch        : $(python -c 'import torch;print(torch.__version__)' 2>/dev/null || echo n/a)"
  echo "# radt         : $(python -c 'import radt;print(getattr(radt, "__version__", "?"))' 2>/dev/null || echo n/a)"
  echo "# mlflow       : $(python -c 'import mlflow;print(mlflow.__version__)' 2>/dev/null || echo n/a)"
  echo "# pin          : ${PIN:-none}"
  echo "# runs         : $RUNS"
} >> "$LOG"

# Optional CPU pinning. On GB10 the Grace CPU is heterogeneous (10x Cortex-X925
# performance + 10x A725 efficiency) and an unpinned workload migrates between
# them. E2 steps are ms-scale so a CLUSTER is right here, unlike E1's us-scale
# transitions — and it must NOT be a single core, because the dataloader thread
# and the span exporter child need their own:
#   PIN=5-9,15-19 collect_e2.sh gb10
# NOTE: expanded as ${PINCMD[@]+"${PINCMD[@]}"} — macOS bash 3.2 treats a plain
# "${PINCMD[@]}" on an EMPTY array as unbound under `set -u`.
PINCMD=()
if [ -n "${PIN:-}" ] && command -v taskset >/dev/null 2>&1; then
  PINCMD=(taskset -c "$PIN")
  log "pinning workload to core(s) [$PIN]"
fi

# Two configurations, not three. The untraced `choreo` arm is dropped: with the
# pipeline's per-query CSV rows gated off (PipelineModel.disable_logs, which the
# generated configs set), it has NO instrument at all -- no spans because
# tracing is off, and no rows -- so it cannot be measured. Keeping the rows just
# for that arm would price tracing by comparing two different instruments.
#
# Nothing is lost: E1 measures the cost of tracing directly and on a clean
# microbenchmark (uninstrumented vs "+ tracing", 9.37 -> 25.31 us/stage on gb10,
# 12.03 -> 42.14 on m3pro). E2's question is what DECOMPOSITION costs, which the
# traced arm answers on its own from spans.
CONFIGS=(${E2_CONFIGS:-monolith choreo-traced})
NCONF=${#CONFIGS[@]}

run_one() {
  local conf=$1 cell=$2 cfg=$3 r=$4 lab start rc secs rows spans outfile
  lab="mod_${cell}_${conf}_${MACHINE}_r${r}"
  [ -f "$RESULTS/$lab.csv" ] && { log "  [skip] $lab (exists)"; return 0; }

  outfile=$(mktemp)
  start=$(date +%s)
  # Never append onto a stale CSV, on EITHER side. main.py appends to an
  # existing label's file, so a run killed part-way leaves a partial session
  # that the next run with the same label concatenates onto. That has now bitten
  # twice: mod_meffv2l_b8_choreo-traced_m2pro_r7 (655 rows, a 57-minute
  # interval) and mod_meffv2s_b64_choreo_gb10_r1 (955 rows, a 4.6-hour one).
  # The skip-if-exists check above only looks in $RESULTS; the Choreo side
  # writes to $CHOREO_OUT first, which nothing was clearing.
  rm -f "$RESULTS/$lab.csv" "$RESULTS/$lab.jsonl"
  if [ "$conf" = monolith ]; then
    ${PINCMD[@]+"${PINCMD[@]}"} python evaluation/overheads/modularity_overhead/baseline_finetune.py \
      --device "$TORCH_DEVICE" --model "$MODEL" --weights "$WEIGHTS" \
      --batch-size "$BATCH" --num-workers 0 --max-batches "$MAXB" \
      --label "$lab" --no-radt 2>&1 | tee "$outfile"
  else
    unset CHOREO_DISABLE_TRACING CHOREO_PROC_TRACE
    case $conf in
      choreo)        export CHOREO_DISABLE_TRACING=1 ;;
      choreo-traced) export CHOREO_PROC_TRACE=1 ;;
    esac
    ${PINCMD[@]+"${PINCMD[@]}"} python main.py "$cfg" -p 0 -e "$EXP" --label "$lab" 2>&1 | tee "$outfile"
  fi
  # rc from PIPESTATUS[0], not $? — after a pipeline $? is the LAST element's
  # status (tee, always 0), which would mark every failed run as successful.
  rc=${PIPESTATUS[0]}; secs=$(( $(date +%s) - start ))
  spans=$(sed -n 's/^\[choreo\] spans emitted: //p' "$outfile" | tail -1)
  rm -f "$outfile"
  unset CHOREO_DISABLE_TRACING CHOREO_PROC_TRACE

  rows=$( [ -f "$RESULTS/$lab.csv" ] && wc -l < "$RESULTS/$lab.csv" | tr -d ' ' || echo 0 )
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$conf" "$cell" "$r" "$rc" "$secs" "$rows" "${spans:-}" >> "$SUM"
  [ "$rows" -eq 0 ] && { log "  !! $lab produced NO CSV (rc=$rc)"; return 1; }
  log "  $lab rc=$rc ${secs}s rows=$rows spans=${spans:-n/a}"
}

shopt -s nullglob
cfgs=( "$HERE"/configs/generated/mod_${CELLGLOB}_"$MACHINE".yml )
(( ${#cfgs[@]} )) || { echo "collect_e2: no configs match mod_${CELLGLOB}_${MACHINE}.yml — run gen_configs.py" >&2; exit 1; }

log "E2 collection: ${#cfgs[@]} cell(s) x $NCONF configurations x $RUNS runs on $MACHINE"
fail=0
for cfg in "${cfgs[@]}"; do
  base=$(basename "$cfg" .yml)                 # mod_m<tag>_b<batch>_<machine>
  cell=${base#mod_}; cell=${cell%_"$MACHINE"}  # m<tag>_b<batch>

  # The cell's (model, weights, batch, steps, torch device) all come FROM THE
  # CONFIG, so the monolith and Choreo provably run the identical workload and
  # the config is the single source of truth.
  read -r MODEL WEIGHTS BATCH MAXB TORCH_DEVICE <<<"$(python - "$cfg" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))["pipelines"][0]
dl = next(s for s in c["stages"] if "DataLoader" in s["component"])["config"]
tr = next(s for s in c["stages"] if "Classification" in s["component"])["config"]
print(tr["model"]["component"].rsplit(".", 1)[-1], dl["dataset"]["weights"],
      dl["batch_size"], c["loadgen"]["max_queries"], tr["device"])
PY
)"

  for r in $(seq 1 "$RUNS"); do
    # Rotate the order every repetition. With a fixed order the configuration
    # that runs first absorbs any within-repetition warm-up or clock ramp, which
    # is the suspected cause of the -575 us cell — the monolith always ran first.
    off=$(( (r - 1) % NCONF ))
    for i in $(seq 0 $(( NCONF - 1 ))); do
      run_one "${CONFIGS[$(( (off + i) % NCONF ))]}" "$cell" "$cfg" "$r" || fail=$((fail+1))
    done
  done
done

log "E2 collection done on $MACHINE ($fail failed run(s)). CSVs in $RESULTS/"
log "log + summary: $LOG"
[ "$fail" -eq 0 ]
