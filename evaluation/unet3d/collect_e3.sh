#!/usr/bin/env bash
# E3 (MLPerf 3D-UNet / KiTS19) collection.
#
# E3 has two prongs and this harness serves both:
#
#   1. PARITY, GB10 only. Choreo's port of the 3D-UNet/KiTS19 workload against
#      MLPerf's OWN reference harness on the SAME machine, on accuracy (DICE)
#      and on performance. Without it, prong 2 is a strawman.
#   2. THE MEASUREMENT BOUNDARY, both machines. MLPerf preprocesses the dataset
#      offline -- its QSL preloads .pkl files -- and times only inference. That
#      is valid for offline batch. Online, a request arrives with its own raw
#      volume, so there is nothing to prefetch and load+preprocess sit on the
#      per-request critical path. Choreo times the whole graph.
#
# TWO CONFIGS, and they are never mixed:
#
#   perf   3 stages, nothing written per query, every number from spans.
#          This is the ONLY config that is ever timed.
#   acc    the same 3 stages plus KiTS19DiceScore, which writes one row per
#          case. R=1 and never timed -- MLPerf separates AccuracyOnly from
#          PerformanceOnly for exactly this reason, and scoring reads the whole
#          volume twice per class.
#
# DO NOT PIN THIS EXPERIMENT. E2 ended up single-core-pinned on GB10 because it
# was cleaner and faster there; E3 must not inherit that, and the reason is bias
# rather than hygiene. E3's claim is the RATIO of CPU preprocessing to GPU
# inference. Pinning throttles scipy.ndimage.zoom -- the dominant preprocessing
# cost -- while leaving GPU inference untouched, which inflates the very share
# we are arguing is larger than MLPerf admits. Choreo and the MLPerf reference
# must see identical CPU conditions, and that condition is unpinned. The harness
# refuses a PIN rather than honouring it.
#
#   usage: collect_e3.sh <machine> [runs] [mode]
#     machine : m3pro | gb10   (names the results dir and the run labels; the
#               torch device string lives in the config, not here)
#     runs    : timed repetitions (default 6; repetition 1 is dropped as system
#               warm-up, so 6 collected leaves 5 usable -- the first repetition
#               of a cell is slower for its WHOLE duration)
#     mode    : perf (default) | acc   (acc forces runs=1)
set -uo pipefail

usage() { sed -n '2,37p' "$0" >&2; exit 2; }
MACHINE=${1:-}; RUNS=${2:-6}; MODE=${3:-perf}
[ -n "$MACHINE" ] || usage
case "$MACHINE" in m3pro|gb10) ;; *) echo "collect_e3: machine must be m3pro or gb10" >&2; exit 2 ;; esac
case "$MODE" in perf) ;; acc) RUNS=1 ;; *) echo "collect_e3: mode must be perf or acc" >&2; exit 2 ;; esac

if [ -n "${PIN:-}" ]; then
  echo "collect_e3: PIN is set to '$PIN'. E3 must run UNPINNED -- see the header." >&2
  echo "            Pinning throttles CPU preprocessing but not GPU inference," >&2
  echo "            which manufactures the result this experiment reports." >&2
  exit 2
fi

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$ROOT"

# Keep the machine awake for the whole collection, and re-exec under caffeinate
# rather than asking the operator to remember it. A Mac left on battery during
# an E2 collection spent the night cycling through Deep Idle / DarkWake, and the
# three runs spanning that window came back with their MEDIAN moved 4-6% while
# reporting rc=0 and exactly the right row count.
if [ "$(uname)" = "Darwin" ] && [ -z "${E3_NO_CAFFEINATE:-}" ] \
   && [ -z "${_E3_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  export _E3_CAFFEINATED=1
  exec caffeinate -dimsu "$0" "$@"
fi

RESULTS="$HERE/results/$MACHINE"
# main.py and TerminalCapture write straight into this experiment's
# results/ dir; there is no shared staging directory to sweep.
export BENCH_OUTPUT_DIR="$RESULTS"
mkdir -p "$RESULTS"

export RADT_TRACE_BACKEND=radt       # force the BULK exporter, not auto-detection
export RADT_PRESENT=True
export CHOREO_PROC_TRACE=1

# Tracking store: res17, NOT a local sqlite file. The local-store exemption
# covers the OVERHEAD experiments (E1, E2) only -- they emit spans at a rate no
# real workload approaches, so a remote server would be measuring itself. E3 is
# a real workload at ~6 s/query and emits a handful of spans per query, which is
# exactly what the remote server is there to carry.
: "${MLFLOW_TRACKING_URI:?collect_e3: MLFLOW_TRACKING_URI is unset. E3 records to res17; the credentials live as conda env config vars, so activate the env rather than invoking the python binary directly.}"
EXP=${E3_EXPERIMENT:-138}
echo "E3: recording to ${MLFLOW_TRACKING_URI} (experiment $EXP)"

CFG="$HERE/configs/unet3d_42_${MODE}_${MACHINE}.yml"
[ -f "$CFG" ] || { echo "collect_e3: no config at $CFG" >&2; exit 1; }

# Environment gate, and a gate on the config's own flags. A perf config that
# lost one of its two disable_logs flags would collect a full CSV inside the
# measured pipeline and nothing would say so until analysis.
python - "$MACHINE" "$CFG" "$MODE" <<'PY' || exit 3
import sys, yaml
machine, cfg_path, mode = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    import radt, torch
except Exception as e:
    sys.exit(f"collect_e3: cannot import radt/torch: {e}")
if not hasattr(radt, "trace"):
    sys.exit("collect_e3: this radt has no radt.trace — every E3 number comes "
             "from spans, so this would collect nothing at all")
want = {"gb10": "cuda", "m3pro": "mps"}[machine]
if want == "cuda" and not torch.cuda.is_available():
    sys.exit("collect_e3: machine is gb10 but torch.cuda.is_available() is False")
if want == "mps" and not torch.backends.mps.is_available():
    sys.exit("collect_e3: machine is m3pro but torch MPS is unavailable")

p = yaml.safe_load(open(cfg_path))["pipelines"][0]
dev = next(s for s in p["stages"] if "UNet3DInference" in s["component"])["config"]["device"]
if dev != want:
    sys.exit(f"collect_e3: {cfg_path} runs on {dev!r} but {machine} wants {want!r}")
if mode == "perf":
    if not p.get("disable_logs"):
        sys.exit(f"collect_e3: {cfg_path} has no pipeline-level disable_logs — the "
                 f"per-query CSV write would sit inside the measured `exit` term")
    bad = [s["name"] for s in p["stages"] if not s.get("disable_logs")]
    if bad:
        sys.exit(f"collect_e3: stage(s) without disable_logs in {cfg_path}: {bad}")
    if any("TerminalCapture" in s["component"] or "DiceScore" in s["component"]
           for s in p["stages"]):
        sys.exit(f"collect_e3: {cfg_path} carries an output stage; output "
                 f"serialisation must not sit inside a timed pipeline")
    if len(p["stages"]) != 3:
        sys.exit(f"collect_e3: perf config must have exactly 3 stages, has {len(p['stages'])}")
elif not any("DiceScore" in s["component"] for s in p["stages"]):
    sys.exit(f"collect_e3: accuracy config has no scorer, so it would produce no DICE")
PY

# ---------------------------------------------------------------------------
# Machine occupancy: record it, and refuse to start on a busy machine.
#
# On 2026-09-01 a gb10 collection ran for 40 minutes alongside a MobileNetV2
# training job that started 26 seconds after it and held ~97 GB of GPU memory.
# Nothing said so: rc=0, the right row counts, the right span counts. E3
# measures the RATIO of CPU preprocessing to GPU inference, so a co-resident
# GPU job inflates the inference stage specifically -- it biases prong 2
# conservatively and breaks prong 1 outright, because the reference it is
# compared against was measured on an idle machine. The whole collection had to
# be thrown away.
#
# Provenance headers recorded the git commit and the library versions but never
# WHO ELSE WAS ON THE MACHINE, which is the one thing that mattered. So: record
# it always, and make a busy machine a refusal rather than a footnote.
# E3_ALLOW_BUSY=1 overrides, deliberately -- it should be a decision someone
# makes, not a default.
occupancy_report() {
  echo "# load         : $(uptime | sed 's/.*load average[s]*: //')"
  if command -v nvidia-smi >/dev/null 2>&1; then
    local apps
    apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
    echo "# gpu_procs    : ${apps:-none}"
  fi
  echo "# other_python : $(pgrep -af python 2>/dev/null | grep -v "$$" \
        | grep -vE "main\.py|collect_e3|nvidia-smi" | wc -l | tr -d ' ')"
}

busy_check() {
  local others
  # Anything else holding GPU memory. Our own run has not started yet at this
  # point, so any process here is someone else's.
  if command -v nvidia-smi >/dev/null 2>&1; then
    others=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
             | grep -c . || true)
    if [ "${others:-0}" -gt 0 ]; then
      echo "collect_e3: the GPU already has ${others} process(es) on it:" >&2
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null >&2
      echo "            E3 measures the ratio of CPU preprocessing to GPU inference." >&2
      echo "            A co-resident GPU job inflates the inference stage and makes" >&2
      echo "            both prongs unusable. Wait for the machine, or set" >&2
      echo "            E3_ALLOW_BUSY=1 to collect anyway and mark the data." >&2
      return 1
    fi
  fi
  return 0
}

if [ -z "${E3_ALLOW_BUSY:-}" ]; then
  busy_check || exit 4
fi

RUNSTAMP=$(date '+%Y%m%d-%H%M%S')
LOGDIR="$HERE/collect_logs"
mkdir -p "$LOGDIR"
SUM="$LOGDIR/collect_e3_${MACHINE}_${MODE}_${RUNSTAMP}.tsv"
LOG="$LOGDIR/collect_e3_${MACHINE}_${MODE}_${RUNSTAMP}.log"
printf 'mode\trun\trc\tseconds\tcsv_rows\tspans\n' > "$SUM"

log(){ local m="[$(date '+%m-%d %H:%M:%S')] $*"; echo "$m"; echo "$m" >> "$LOG"; }

{
  echo "# E3 collection"
  echo "# started      : $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "# machine      : $MACHINE"
  echo "# mode         : $MODE"
  echo "# config       : $CFG"
  echo "# git_commit   : $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "# git_dirty    : $(test -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" && echo yes || echo no)"
  echo "# host         : $(hostname)"
  echo "# platform     : $(python -c 'import platform;print(platform.platform())' 2>/dev/null)"
  echo "# python       : $(python -c 'import sys;print(sys.version.split()[0])' 2>/dev/null)"
  echo "# torch        : $(python -c 'import torch;print(torch.__version__)' 2>/dev/null || echo n/a)"
  echo "# radt         : $(python -c 'import radt;print(getattr(radt, "__version__", "?"))' 2>/dev/null || echo n/a)"
  echo "# mlflow       : $(python -c 'import mlflow;print(mlflow.__version__)' 2>/dev/null || echo n/a)"
  echo "# store        : $MLFLOW_TRACKING_URI (experiment $EXP)"
  echo "# pin          : none (enforced)"
  echo "# runs         : $RUNS"
  echo "# allow_busy   : ${E3_ALLOW_BUSY:-no}"
  occupancy_report
} >> "$LOG"

run_one() {
  local r=$1 lab start rc secs rows spans outfile
  lab="unet3d_42_${MODE}_${MACHINE}_r${r}"
  [ -f "$RESULTS/$lab.csv" ] && { log "  [skip] $lab (exists)"; return 0; }

  outfile=$(mktemp)
  start=$(date +%s)
  # Never append onto a stale CSV. main.py appends to an existing label's file,
  # so a run killed part-way leaves a partial session that the next run with the
  # same label concatenates onto -- which produced a 4.6-hour "interval" in E2.
  rm -f "$RESULTS/$lab.csv" "$RESULTS/${lab}_outputs.jsonl"
  python main.py "$CFG" -p 0 -e "$EXP" --label "$lab" 2>&1 | tee "$outfile"
  # rc from PIPESTATUS[0], not $? — after a pipeline $? is the LAST element's
  # status (tee, always 0), which would mark every failed run as successful.
  rc=${PIPESTATUS[0]}; secs=$(( $(date +%s) - start ))
  spans=$(sed -n 's/^\[choreo\] spans emitted: //p' "$outfile" | tail -1)
  rm -f "$outfile"

  rows=$( [ -f "$RESULTS/$lab.csv" ] && wc -l < "$RESULTS/$lab.csv" | tr -d ' ' || echo 0 )
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$MODE" "$r" "$rc" "$secs" "$rows" "${spans:-}" >> "$SUM"
  [ "$rc" -ne 0 ] && { log "  !! $lab FAILED rc=$rc"; return 1; }
  log "  $lab rc=$rc ${secs}s rows=$rows spans=${spans:-n/a}"
}

log "E3 collection: mode=$MODE, $RUNS run(s) of 42 cases on $MACHINE, unpinned"
fail=0
for r in $(seq 1 "$RUNS"); do
  run_one "$r" || fail=$((fail+1))
done

if [ "$MODE" = acc ]; then
  DICE="$HERE/results/dice_${MACHINE}.csv"
  if [ -f "$DICE" ]; then
    log "DICE rows: $(( $(wc -l < "$DICE") - 1 )) in $DICE"
  else
    log "!! no DICE file at $DICE — the accuracy pass produced no score"
    fail=$((fail+1))
  fi
fi

log "E3 collection done on $MACHINE ($fail failed run(s)). Results in $RESULTS/"
log "log + summary: $LOG"
[ "$fail" -eq 0 ]
