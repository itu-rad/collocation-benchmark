#!/usr/bin/env bash
# 5.2 (collocation types and per-pipeline attribution) collection.
#
# One run is one YAML passed to main.py. The config declares both pipelines --
# the RAG-serve foreground from 5.1 and a single-resource background co-runner --
# and main.py's orchestrator mode launches EACH PIPELINE AS ITS OWN PROCESS
# (one radt schedule row per pipeline, re-exec'd as `main.py <cfg> -p <n>`).
#
# That process separation is what makes the section possible: every pipeline gets
# its own radt run, its own listeners and its own spans, so each number is
# attributed to the pipeline that caused it -- and MPS, which partitions between
# processes, has something to partition.
#
# The collocation mechanism is one key in that same YAML (`collocation:`):
#   ""      time-sliced, processes share the GPU as usual
#   "mps"   radt brings up the CUDA MPS control daemon for the group
# So the time-sliced and MPS cells are the same config differing in one line,
# which is precisely the claim the section makes.
#
# Modes:
#   baseline  foreground alone -- the reference every degradation is measured against
#   types     the collocation axis at ONE calibrated intensity:
#               m3pro  background on GPU / ANE / CPU  (three engines, one memory pool)
#               gb10   GPU time-sliced / GPU under MPS / CPU
#   dose      m3pro only: the memory-stream antagonist at stepped intensities, for
#             the bandwidth dose-response with the AMC per-engine counters
#
#   usage: collect_e5.sh <machine> [runs] [mode] [level]
set -uo pipefail

usage() { sed -n '2,28p' "$0" >&2; exit 2; }
MACHINE=${1:-}; RUNS=${2:-6}; MODE=${3:-types}; LEVEL=${4:-${E5_LEVEL:-L50}}
# The matched-bytes level used by the m3pro types cells (see generate_stage_configs).
MLEVEL=${E5_MATCHED_LEVEL:-B12}
[ -n "$MACHINE" ] || usage
case "$MACHINE" in m3pro|gb10) ;; *) echo "collect_e5: machine must be m3pro or gb10" >&2; exit 2 ;; esac
case "$MODE" in types|baseline|dose) ;; *) echo "collect_e5: mode must be types, baseline or dose" >&2; exit 2 ;; esac

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$ROOT"
case "$MACHINE" in m3pro) ENGINE=mlx ;; gb10) ENGINE=cuda ;; esac

if [ "$MODE" = dose ] && [ "$MACHINE" != m3pro ]; then
  echo "collect_e5: the dose-response runs on m3pro only -- it is carried by the AMC" >&2
  echo "            per-engine DRAM counters, and gb10 has no DRAM counter." >&2
  exit 2
fi

if [ "$(uname)" = "Darwin" ]; then
  export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
fi

# Keep the Mac awake: a sleeping machine moved three E2 repetitions' medians 4-6%
# while reporting rc=0 and the right row counts.
if [ "$(uname)" = "Darwin" ] && [ -z "${E5_NO_CAFFEINATE:-}" ] \
   && [ -z "${_E5_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  export _E5_CAFFEINATED=1
  exec caffeinate -dimsu "$0" "$@"
fi

RESULTS="$HERE/results/$MACHINE"
mkdir -p "$RESULTS"
export SUITE_OUTPUT_DIR="$RESULTS"
export RADT_TRACE_BACKEND=radt
export RADT_PRESENT=True
export SUITE_PROC_TRACE=1

# Bound the span batch uploaded to the tracking server. radt batches spans into
# gzipped artifacts, rolling at RADT_TRACE_BATCH_EVENTS (default 200000) or
# RADT_TRACE_BATCH_SECONDS. Batches spooled before the mlflow run id arrives are
# held and uploaded together, so batch 1 can carry most of a run -- and res17
# rejects an oversized upload with "413 Request Entity Too Large", which kills
# the exporter thread and leaves the workload process wedged in teardown. radt
# then waits for it and the whole cell hangs. Smaller batches, more of them.
export RADT_TRACE_BATCH_EVENTS=${RADT_TRACE_BATCH_EVENTS:-20000}
export RADT_TRACE_BATCH_SECONDS=${RADT_TRACE_BATCH_SECONDS:-30}


: "${MLFLOW_TRACKING_URI:?collect_e5: MLFLOW_TRACKING_URI is unset. Activate the conda env rather than invoking the python binary directly; the credentials are env config vars.}"
EXP=${E5_EXPERIMENT:-138}
echo "5.2: recording to ${MLFLOW_TRACKING_URI} (experiment $EXP)"

case "$MACHINE" in
  m3pro) WANT_LISTENERS="macmon" ;;
  gb10)  WANT_LISTENERS="dcgmi,top" ;;
esac

# radt spawns a listener ONLY if RADT_LISTENER_<NAME>=True is in the environment
# (radt/run/benchmark.py). The config's `listeners:` key alone spawns nothing, and
# the run then finishes clean with no counters at all.
export_listener_env() {
  local IFS=,
  for l in $1; do
    [ -n "$l" ] || continue
    export "RADT_LISTENER_$(echo "$l" | tr 'a-z' 'A-Z')=True"
  done
}
export_listener_env "$WANT_LISTENERS"

python scripts/radt_gate.py --listeners "$WANT_LISTENERS" || exit 3

# Cells: "<cell name>|<config>"
cells=()
case "$MODE" in
  baseline) cells=( "baseline|$HERE/configs/stage_a_B0_${ENGINE}.yml" ) ;;
  types)
    case "$MACHINE" in
      m3pro)
        # Matched on BYTES/S, not on each engine's own capacity: the L-ladder put
        # 51 / 32 / 13 GB/s on the three engines at "L50", which is not one
        # intensity. B12 = 12 GB/s from each, from measured bytes/query.
        cells=( "bg_gpu|$HERE/configs/stage_c_clipgpu_${MLEVEL}_mlx.yml"
                "bg_ane|$HERE/configs/stage_c_clipane_${MLEVEL}_mlx.yml"
                "bg_cpu|$HERE/configs/stage_c_stream_${MLEVEL}_mlx.yml" ) ;;
      gb10)
        # The first two are the SAME config bar one line: `collocation: mps`.
        cells=( "bg_gpu_timesliced|$HERE/configs/stage_c_clipgpu_${LEVEL}_cuda.yml"
                "bg_gpu_mps|$HERE/configs/stage_c_clipgpu_${LEVEL}_cuda_mps.yml"
                "bg_cpu|$HERE/configs/stage_c_stream_${LEVEL}_cuda.yml" ) ;;
    esac ;;
  dose)
    for lv in L25 L50 L75 L100; do
      cells+=( "dose_${lv}|$HERE/configs/stage_c_stream_${lv}_mlx.yml" )
    done ;;
esac

# Every 5.2 config must declare listeners: this section IS the counters, and a
# config without them runs with none, silently.
for c in "${cells[@]}"; do
  cfg=${c#*|}
  [ -f "$cfg" ] || { echo "collect_e5: missing config $cfg -- run generate_stage_configs.py --device $ENGINE" >&2; exit 3; }
  python - "$cfg" "$WANT_LISTENERS" <<'PYCHK' || exit 3
import sys, yaml
cfg, want = sys.argv[1], [w for w in sys.argv[2].split(",") if w]
d = yaml.safe_load(open(cfg))
got = d.get("listeners")
if not got:
    sys.exit(f"collect_e5: {cfg} declares no listeners -- it would collect no "
             f"counters at all. Regenerate it with generate_stage_configs.py.")
missing = [w for w in want if w not in got]
if missing:
    sys.exit(f"collect_e5: {cfg} is missing listener(s) {missing} (has {got})")
PYCHK
done

# The AMC per-engine DRAM counters are the evidence for "separation is not
# isolation". They come from our sampler rather than radt, so they need starting
# per cell, and its default output path points at a results tree that no longer
# exists -- hence the explicit --out below. Apple only; proven before collecting.
AMC="$ROOT/scripts/amc_bandwidth_sampler.py"
USE_AMC=0
if [ "$(uname)" = "Darwin" ] && [ -z "${E5_NO_AMC:-}" ]; then
  if python "$AMC" --duration 2 --interval 0.5 --out /tmp/e5_amc_preflight.csv >/dev/null 2>&1 \
     && [ -s /tmp/e5_amc_preflight.csv ]; then
    USE_AMC=1; rm -f /tmp/e5_amc_preflight.csv
  else
    echo "collect_e5: the AMC bandwidth sampler produced no output. Exhibit 2 rests" >&2
    echo "            on these counters, so this is refused rather than collected" >&2
    echo "            blind. Set E5_NO_AMC=1 to override." >&2
    exit 3
  fi
fi

RUNSTAMP=$(date '+%Y%m%d-%H%M%S')
LOGDIR="$HERE/collect_logs"; mkdir -p "$LOGDIR"
SUM="$LOGDIR/collect_e5_${MACHINE}_${MODE}_${RUNSTAMP}.tsv"
LOG="$LOGDIR/collect_e5_${MACHINE}_${MODE}_${RUNSTAMP}.log"
printf 'mode\tcell\trun\trc\tseconds\tfg_rows\tbg_rows\tspans\n' > "$SUM"
log(){ local m="[$(date '+%m-%d %H:%M:%S')] $*"; echo "$m"; echo "$m" >> "$LOG"; }

{
  echo "# 5.2 collection"
  echo "# started      : $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "# machine      : $MACHINE (engine token: $ENGINE)"
  echo "# mode         : $MODE   level: $LEVEL"
  echo "# git_commit   : $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "# git_dirty    : $(test -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" && echo yes || echo no)"
  echo "# host         : $(hostname)"
  echo "# platform     : $(python -c 'import platform;print(platform.platform())' 2>/dev/null)"
  echo "# radt         : $(python -c 'import radt,os;print(os.path.dirname(radt.__file__))' 2>/dev/null || echo n/a)"
  echo "# store        : $MLFLOW_TRACKING_URI (experiment $EXP)"
  echo "# listeners    : $WANT_LISTENERS"
  echo "# amc_sampler  : $([ "$USE_AMC" = 1 ] && echo "on (per-engine DRAM bytes)" || echo off)"
  echo "# runs         : $RUNS"
  echo "# load         : $(uptime | sed 's/.*load average[s]*: //')"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "# gpu_procs    : $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';' || echo none)"
  fi
} >> "$LOG"

# A foreign job on the GPU is exactly the confound this experiment measures.
if [ -z "${E5_ALLOW_BUSY:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
  if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" -gt 0 ]; then
    echo "collect_e5: the GPU already has process(es) on it:" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
    echo "            Set E5_ALLOW_BUSY=1 to collect anyway and mark the data." >&2
    exit 4
  fi
fi

# Wall-clock cap for one cell, from the foreground's own declared loadgen.
CELL_CAP_S=$(python - "${cells[0]#*|}" <<'PYCAP'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
lg = d["pipelines"][0].get("loadgen", {})
n = float(lg.get("max_queries") or 0); cfg = lg.get("config") or {}
rate = float(cfg.get("rate") or 0); iv = float(cfg.get("interval") or 0)
secs = n / rate if rate > 0 else (n * iv if iv > 0 else 0)
# 4x the foreground plus model-load headroom.
print(int(secs * 4 + 600) if secs > 0 else 5400)
PYCAP
)
[ -n "$CELL_CAP_S" ] || CELL_CAP_S=5400
CELL_CAP_S=${E5_CELL_CAP_S:-$CELL_CAP_S}

run_cell() {
  local cell=$1 cfg=$2 r=$3 lab start rc secs outfile amcpid fg_rows bg_rows spans runpid waited
  lab="e5_${cell}_${MACHINE}_r${r}"
  ls "$RESULTS/${lab}"*.csv >/dev/null 2>&1 && { log "  [skip] $lab (exists)"; return 0; }
  rm -f "$RESULTS/${lab}"*.csv "$RESULTS/${lab}"*_outputs.jsonl

  amcpid=""
  if [ "$USE_AMC" = 1 ]; then
    python "$AMC" --interval 0.5 --out "$RESULTS/${lab}_bandwidth.csv" \
      > "$LOGDIR/${lab}_amc.log" 2>&1 &
    amcpid=$!
  fi

  start=$(date +%s); outfile=$(mktemp)
  # No -p: orchestrator mode, so radt launches one process per pipeline.
  #
  # Bounded by wall clock. radt waits for EVERY pipeline in the config, so one
  # workload that will not finish holds the whole collection: an MPS cell once
  # ran 92 minutes against a 12-minute foreground, with the machine idle-ish and
  # nothing in the log to say so. The cap is generous -- it is a runaway
  # detector, not a schedule.
  python main.py "$cfg" -e "$EXP" --label "$lab" 2>&1 | tee "$outfile" &
  local runpid=$! waited=0
  while kill -0 "$runpid" 2>/dev/null; do
    sleep 10; waited=$(( waited + 10 ))
    if [ "$waited" -gt "$CELL_CAP_S" ]; then
      log "  !! $lab exceeded ${CELL_CAP_S}s (cap) -- killing; the cell is a runaway"
      pkill -P "$runpid" 2>/dev/null; kill -9 "$runpid" 2>/dev/null
      for pat in "radt[ ]run" "main[.]py"; do
        p2=$(pgrep -f "$pat"); [ -n "$p2" ] && kill -9 $p2 2>/dev/null
      done
      rm -f "$ROOT/radtlock"
      break
    fi
  done
  wait "$runpid" 2>/dev/null; rc=$?; secs=$(( $(date +%s) - start ))

  if [ -n "$amcpid" ]; then
    kill -TERM "$amcpid" 2>/dev/null; wait "$amcpid" 2>/dev/null
    [ -s "$RESULTS/${lab}_bandwidth.csv" ] || log "  !! $lab produced no bandwidth trace"
  fi

  spans=$(sed -n 's/^\[[a-z]*\] spans emitted: //p' "$outfile" | tail -1)
  rm -f "$outfile"
  fg_rows=$(cat "$RESULTS/${lab}"*serve*.csv 2>/dev/null | wc -l | tr -d ' ')
  bg_rows=$(cat "$RESULTS/${lab}"*[Bb][Gg]*.csv 2>/dev/null | wc -l | tr -d ' ')
  [ -n "$fg_rows" ] || fg_rows=0; [ -n "$bg_rows" ] || bg_rows=0
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$MODE" "$cell" "$r" "$rc" "$secs" "$fg_rows" "$bg_rows" "${spans:-}" >> "$SUM"
  [ "$rc" -ne 0 ] && { log "  !! $lab FAILED rc=$rc"; return 1; }
  log "  $lab rc=$rc ${secs}s fg_rows=$fg_rows bg_rows=$bg_rows spans=${spans:-n/a}"

  # Prove the counters reached the tracking server after the first cell. A
  # listener that never spawned is invisible otherwise: clean exit, right row
  # counts, no metrics -- and this whole section is the counters.
  if [ -z "${_E5_COUNTERS_VERIFIED:-}" ]; then
    export _E5_COUNTERS_VERIFIED=1
    log "  verifying counters reached $MLFLOW_TRACKING_URI ..."
    if python scripts/check_listener_metrics.py "$lab" >> "$LOG" 2>&1; then
      log "  counters OK for $lab"
    else
      log "  !! $lab recorded NO system/* metrics -- the listeners did not run."
      return 2
    fi
  fi
  return 0
}

log "5.2 collection: ${#cells[@]} cell(s) x $RUNS run(s) on $MACHINE [$MODE], level=$LEVEL, listeners=$WANT_LISTENERS"
fail=0
for r in $(seq 1 "$RUNS"); do
  # Rotate cell order every repetition so no cell always absorbs thermal ramp.
  n=${#cells[@]}; off=$(( (r - 1) % n ))
  for i in $(seq 0 $(( n - 1 ))); do
    IFS='|' read -r cell cfg <<< "${cells[$(( (off + i) % n ))]}"
    run_cell "$cell" "$cfg" "$r"; rc_one=$?
    if [ "$rc_one" -eq 2 ]; then
      log "5.2 collection ABORTED on $MACHINE [$MODE]: counters are not being recorded."
      exit 5
    fi
    [ "$rc_one" -ne 0 ] && fail=$((fail+1))
  done
done

log "5.2 collection done on $MACHINE [$MODE] ($fail failed cell(s)). Results in $RESULTS/"
log "log + summary: $LOG"
[ "$fail" -eq 0 ]
