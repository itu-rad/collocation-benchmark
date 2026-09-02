#!/usr/bin/env bash
# 5.2 (collocation types and per-pipeline attribution) collection.
#
# The foreground is the RAG-serve pipeline from 5.1 under Poisson arrivals. The
# background is a single-resource co-runner. They run as SEPARATE PROCESSES --
# each with its own radt run, its own listeners and its own spans -- which is
# what lets every number be attributed to the pipeline that caused it, and is
# also the only arrangement MPS can partition (it separates processes, not
# threads).
#
# Modes:
#   baseline  foreground alone. The reference every degradation is measured against.
#   types     the collocation axis, at ONE calibrated intensity:
#               m3pro  background on GPU / ANE / CPU  (three engines, one memory pool)
#               gb10   background on GPU time-sliced / GPU under MPS / CPU
#             The GPU and MPS cells are the SAME config; only radt's collocation
#             flag differs, which is the point being demonstrated.
#   dose      m3pro only: the CPU memory-stream antagonist at stepped intensities,
#             for the bandwidth dose-response with the AMC per-engine counters.
#
#   usage: collect_e5.sh <machine> [runs] [mode] [level]
#     machine : m3pro | gb10
#     runs    : repetitions (default 6; repetition 1 is dropped as system warm-up)
#     mode    : types (default) | baseline | dose
#     level   : intensity for `types`, default L50 (override: any of L25 L50 L75 L100)
set -uo pipefail

usage() { sed -n '2,26p' "$0" >&2; exit 2; }
MACHINE=${1:-}; RUNS=${2:-6}; MODE=${3:-types}; LEVEL=${4:-${E5_LEVEL:-L50}}
[ -n "$MACHINE" ] || usage
case "$MACHINE" in m3pro|gb10) ;; *) echo "collect_e5: machine must be m3pro or gb10" >&2; exit 2 ;; esac
case "$MODE" in types|baseline|dose) ;; *) echo "collect_e5: mode must be types, baseline or dose" >&2; exit 2 ;; esac

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$ROOT"
case "$MACHINE" in m3pro) ENGINE=mlx ;; gb10) ENGINE=cuda ;; esac

if [ "$MODE" = dose ] && [ "$MACHINE" != m3pro ]; then
  echo "collect_e5: the dose-response runs on m3pro only -- it is carried by the AMC" >&2
  echo "            per-engine DRAM counters, and gb10 has no DRAM counter (DCGM" >&2
  echo "            profiling fields are unavailable on this stack). Use mode 'types'." >&2
  exit 2
fi

# Homebrew is not on a non-interactive ssh PATH and radt spawns `macmon` by name.
if [ "$(uname)" = "Darwin" ]; then
  export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
fi

# Keep the Mac awake: a sleeping machine moved three E2 repetitions' medians
# 4-6% while reporting rc=0 and the right row counts.
if [ "$(uname)" = "Darwin" ] && [ -z "${E5_NO_CAFFEINATE:-}" ] \
   && [ -z "${_E5_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  export _E5_CAFFEINATED=1
  exec caffeinate -dimsu "$0" "$@"
fi

RESULTS="$HERE/results/$MACHINE"
mkdir -p "$RESULTS"
export BENCH_OUTPUT_DIR="$RESULTS"

export RADT_TRACE_BACKEND=radt
export RADT_PRESENT=True
export CHOREO_PROC_TRACE=1

: "${MLFLOW_TRACKING_URI:?collect_e5: MLFLOW_TRACKING_URI is unset. Activate the conda env rather than invoking the python binary directly; the credentials are env config vars.}"
EXP=${E5_EXPERIMENT:-138}
echo "5.2: recording to ${MLFLOW_TRACKING_URI} (experiment $EXP)"

case "$MACHINE" in
  m3pro) WANT_LISTENERS="macmon" ;;
  gb10)  WANT_LISTENERS="dcgmi,top" ;;
esac

# radt spawns a listener ONLY if RADT_LISTENER_<NAME>=True is in the environment
# (radt/run/benchmark.py). That variable is set by radt's schedule path, which a
# direct `python main.py` invocation never goes through -- so the config's
# `listeners:` key alone spawns nothing, and the run finishes clean with no
# counters. Export one per wanted listener.
export_listener_env() {
  local IFS=,
  for l in $1; do
    [ -n "$l" ] || continue
    export "RADT_LISTENER_$(echo "$l" | tr 'a-z' 'A-Z')=True"
  done
}

export_listener_env "$WANT_LISTENERS"

python scripts/radt_gate.py --listeners "$WANT_LISTENERS" || exit 3

FG="$HERE/configs/fg_ragserve_${ENGINE}.yml"
[ -f "$FG" ] || { echo "collect_e5: missing $FG -- run gen_collocation_configs.py --device $ENGINE" >&2; exit 3; }

# Every 5.2 config must declare listeners. This section IS the counters, and a
# config without a `listeners:` key runs with none -- silently, with a normal exit
# code and the right row counts.
check_listeners() {
  python - "$1" "$WANT_LISTENERS" <<'PYCHK' || exit 3
import sys, yaml
cfg, want = sys.argv[1], [w for w in sys.argv[2].split(",") if w]
d = yaml.safe_load(open(cfg))
got = d.get("listeners")
if not got:
    sys.exit(f"collect_e5: {cfg} declares no listeners -- it would collect no "
             f"counters at all. Run gen_collocation_configs.py to regenerate it.")
missing = [w for w in want if w not in got]
if missing:
    sys.exit(f"collect_e5: {cfg} is missing listener(s) {missing} (has {got})")
PYCHK
}


# Cells: "<cell name>|<background config or - >|<radt collocation or - >"
cells=()
case "$MODE" in
  baseline)
    cells=( "baseline|-|-" )
    ;;
  types)
    case "$MACHINE" in
      m3pro)
        cells=( "bg_gpu|$HERE/configs/bg_clipgpu_${LEVEL}_mlx.yml|-"
                "bg_ane|$HERE/configs/bg_clipane_${LEVEL}_mlx.yml|-"
                "bg_cpu|$HERE/configs/bg_stream_${LEVEL}_mlx.yml|-" )
        ;;
      gb10)
        # Same background config in the first two cells: only the partitioning
        # mechanism radt sets up differs.
        cells=( "bg_gpu_timesliced|$HERE/configs/bg_clipgpu_${LEVEL}_cuda.yml|-"
                "bg_gpu_mps|$HERE/configs/bg_clipgpu_${LEVEL}_cuda.yml|mps"
                "bg_cpu|$HERE/configs/bg_stream_${LEVEL}_cuda.yml|-" )
        ;;
    esac
    ;;
  dose)
    for lv in L25 L50 L75 L100; do
      cells+=( "dose_${lv}|$HERE/configs/bg_stream_${lv}_mlx.yml|-" )
    done
    ;;
esac

# The AMC per-engine DRAM counters are the evidence for "separation is not
# isolation": they are what show the background still moving bytes after it has
# been pushed off the GPU. They come from a sampler of ours, not from radt, so
# they need starting per cell -- and its default output path points at a results
# tree that no longer exists, hence the explicit --out below.
#
# Apple only. Prove it runs now rather than discovering at analysis time that the
# flagship exhibit has no data.
AMC="$ROOT/scripts/amc_bandwidth_sampler.py"
USE_AMC=0
if [ "$(uname)" = "Darwin" ] && [ -z "${E5_NO_AMC:-}" ]; then
  if [ ! -f "$AMC" ]; then
    echo "collect_e5: $AMC not found (set E5_NO_AMC=1 to collect without bandwidth counters)" >&2
    exit 3
  fi
  if python "$AMC" --duration 2 --interval 0.5 --out /tmp/e5_amc_preflight.csv >/dev/null 2>&1 \
     && [ -s /tmp/e5_amc_preflight.csv ]; then
    USE_AMC=1
    rm -f /tmp/e5_amc_preflight.csv
  else
    echo "collect_e5: the AMC bandwidth sampler did not produce output. Exhibit 2" >&2
    echo "            rests on these counters, so this is refused rather than" >&2
    echo "            collected blind. Set E5_NO_AMC=1 to override." >&2
    exit 3
  fi
fi

check_listeners "$FG"
for c in "${cells[@]}"; do
  bg=$(echo "$c" | cut -d'|' -f2)
  [ "$bg" = "-" ] && continue
  [ -f "$bg" ] || { echo "collect_e5: missing background config $bg" >&2; exit 3; }
  check_listeners "$bg"
done

RUNSTAMP=$(date '+%Y%m%d-%H%M%S')
LOGDIR="$HERE/collect_logs"; mkdir -p "$LOGDIR"
SUM="$LOGDIR/collect_e5_${MACHINE}_${MODE}_${RUNSTAMP}.tsv"
LOG="$LOGDIR/collect_e5_${MACHINE}_${MODE}_${RUNSTAMP}.log"
printf 'mode\tcell\trun\tfg_rc\tbg_rc\tseconds\tfg_rows\tfg_spans\n' > "$SUM"
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
  echo "# foreground   : $(basename "$FG")"
  echo "# runs         : $RUNS"
  echo "# load         : $(uptime | sed 's/.*load average[s]*: //')"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "# gpu_procs    : $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';' || echo none)"
  fi
} >> "$LOG"

# A foreign job on the GPU is exactly the confound this experiment measures, so
# it must not be present while measuring it.
if [ -z "${E5_ALLOW_BUSY:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
  if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" -gt 0 ]; then
    echo "collect_e5: the GPU already has process(es) on it:" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
    echo "            Set E5_ALLOW_BUSY=1 to collect anyway and mark the data." >&2
    exit 4
  fi
fi

# MPS. radt can configure MPS itself, but only along its schedule path (a workload
# CSV with a Collocation column); it is NOT reachable from the direct `main.py`
# invocation this script uses, and there is no environment variable for it -- radt
# reads none. So we start the daemon here with exactly the call radt's make_mps()
# makes, and we VERIFY it came up rather than assuming: a control daemon that
# failed to start is invisible at run time, and the MPS cell would silently
# collect a second time-sliced cell and report it as a partitioned one.
mps_up() {
  command -v nvidia-cuda-mps-control >/dev/null 2>&1 || {
    echo "collect_e5: nvidia-cuda-mps-control not found -- cannot run the MPS cell" >&2; return 1; }
  local out
  out=$(nvidia-cuda-mps-control -d 2>&1)
  case "$(echo "$out" | tr 'A-Z' 'a-z')" in
    *"is already running"*) echo "collect_e5: an MPS daemon was already running" >&2; return 1 ;;
  esac
  # Confirm the control daemon answers before anything is measured under it.
  if ! echo get_server_list | timeout 10 nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "collect_e5: MPS control daemon did not answer get_server_list" >&2; return 1
  fi
  return 0
}
mps_down() {
  command -v nvidia-cuda-mps-control >/dev/null 2>&1 || return 0
  echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
}
trap 'mps_down' EXIT

run_cell() {
  local cell=$1 bg=$2 coll=$3 r=$4
  local fg_lab bg_lab start fg_rc bg_rc secs rows spans outfile bgpid
  fg_lab="e5_${cell}_fg_${MACHINE}_r${r}"
  bg_lab="e5_${cell}_bg_${MACHINE}_r${r}"
  [ -f "$RESULTS/$fg_lab.csv" ] && { log "  [skip] $fg_lab (exists)"; return 0; }

  # main.py appends to an existing label's file; a part-way run would otherwise
  # leave a partial session for the next run to concatenate onto.
  rm -f "$RESULTS/$fg_lab.csv" "$RESULTS/${fg_lab}_outputs.jsonl" \
        "$RESULTS/$bg_lab.csv" "$RESULTS/${bg_lab}_outputs.jsonl"

  if [ "$coll" = mps ]; then
    mps_up || { log "  !! $cell skipped -- MPS could not be started"; return 1; }
  else
    mps_down   # never let a previous cell's daemon leak into a time-sliced one
  fi

  start=$(date +%s); bg_rc=0; bgpid=""
  if [ "$bg" != "-" ]; then
    python main.py "$bg" -p 0 -e "$EXP" --label "$bg_lab" \
      > "$LOGDIR/${bg_lab}.log" 2>&1 &
    bgpid=$!
    # Let the background reach steady state before the foreground starts timing.
    sleep "${E5_BG_WARMUP:-20}"
    if ! kill -0 "$bgpid" 2>/dev/null; then
      wait "$bgpid"; bg_rc=$?
      log "  !! $bg_lab exited during warm-up rc=$bg_rc -- cell aborted"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$MODE" "$cell" "$r" "-" "$bg_rc" "0" "0" "" >> "$SUM"
      return 1
    fi
  fi

  amcpid=""
  if [ "$USE_AMC" = 1 ]; then
    python "$AMC" --interval 0.5 --out "$RESULTS/${fg_lab}_bandwidth.csv" \
      > "$LOGDIR/${fg_lab}_amc.log" 2>&1 &
    amcpid=$!
  fi

  outfile=$(mktemp)
  python main.py "$FG" -p 0 -e "$EXP" --label "$fg_lab" 2>&1 | tee "$outfile"
  fg_rc=${PIPESTATUS[0]}
  secs=$(( $(date +%s) - start ))

  if [ -n "$amcpid" ]; then
    kill -TERM "$amcpid" 2>/dev/null; wait "$amcpid" 2>/dev/null
    if [ ! -s "$RESULTS/${fg_lab}_bandwidth.csv" ]; then
      log "  !! $fg_lab produced no bandwidth trace -- see ${fg_lab}_amc.log"
    fi
  fi

  if [ -n "$bgpid" ]; then
    # The foreground defines the measurement window; stop the background with it.
    # The background is deliberately sized to outlast the foreground, so being
    # terminated here is the NORMAL outcome -- 143 (128+SIGTERM) is success, and
    # reporting it as a failure would flag every healthy cell. A background that
    # exited on its own before this point ran dry and left the tail of the
    # foreground uncontended: that is the case worth seeing.
    if ! kill -0 "$bgpid" 2>/dev/null; then
      wait "$bgpid" 2>/dev/null; bg_rc=$?
      log "  !! background exited before the foreground (rc=$bg_rc) -- part of this"
      log "     cell ran UNCONTENDED; check the background's query budget"
    else
      kill -TERM "$bgpid" 2>/dev/null
      wait "$bgpid" 2>/dev/null; bg_rc=$?
      [ "$bg_rc" -eq 143 ] && bg_rc=0
    fi
  fi

  [ "$coll" = mps ] && mps_down

  # Was the foreground actually contended for its whole run? The fixed warm-up
  # only proves the background PROCESS started, not that it was producing load --
  # a background still loading its model contends with nothing. The two traces
  # answer it directly, so check rather than assume: the answer depends on which
  # side loads slower, which differs per cell and per machine.
  if [ "$bg" != "-" ] && [ -s "$RESULTS/$bg_lab.csv" ] && [ -s "$RESULTS/$fg_lab.csv" ]; then
    gapmsg=$(python - "$RESULTS/$bg_lab.csv" "$RESULTS/$fg_lab.csv" <<'PYGAP' 2>&1
import csv, sys
def first_run(path):
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) >= 5 and row[3].strip() == "run" and row[4].strip() == "start":
                return float(row[0])
bg, fg = first_run(sys.argv[1]), first_run(sys.argv[2])
if bg is None or fg is None:
    print("  warm-up check: could not read first query from both traces")
elif bg > fg:
    print(f"  !! WARM-UP: the foreground ran {bg-fg:.0f}s before the background "
          f"produced any load -- that much of this cell is UNCONTENDED. "
          f"Raise E5_BG_WARMUP.")
else:
    print(f"  warm-up ok: background was loaded {fg-bg:.0f}s before the "
          f"foreground's first query")
PYGAP
)
    log "$gapmsg"
  fi

  spans=$(sed -n 's/^\[[a-z]*\] spans emitted: //p' "$outfile" | tail -1)
  rm -f "$outfile"
  rows=$( [ -f "$RESULTS/$fg_lab.csv" ] && wc -l < "$RESULTS/$fg_lab.csv" | tr -d ' ' || echo 0 )
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$MODE" "$cell" "$r" "$fg_rc" "$bg_rc" "$secs" "$rows" "${spans:-}" >> "$SUM"
  [ "$fg_rc" -ne 0 ] && { log "  !! $fg_lab FAILED rc=$fg_rc"; return 1; }
  log "  $fg_lab rc=$fg_rc bg_rc=$bg_rc ${secs}s rows=$rows spans=${spans:-n/a}"

  # Prove the counters reached the tracking server after the first cell. A
  # listener that never spawned is invisible otherwise -- clean exit, right row
  # counts, no metrics -- and this whole section is the counters.
  if [ -z "${_E5_COUNTERS_VERIFIED:-}" ]; then
    export _E5_COUNTERS_VERIFIED=1
    log "  verifying counters reached $MLFLOW_TRACKING_URI ..."
    if python scripts/check_listener_metrics.py "$fg_lab" >> "$LOG" 2>&1; then
      log "  counters OK for $fg_lab"
    else
      log "  !! $fg_lab recorded NO system/* metrics -- the listeners did not run."
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
    IFS='|' read -r cell bg coll <<< "${cells[$(( (off + i) % n ))]}"
    run_cell "$cell" "$bg" "$coll" "$r"; rc_one=$?
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
