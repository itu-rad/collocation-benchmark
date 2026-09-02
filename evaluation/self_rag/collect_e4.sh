#!/usr/bin/env bash
# 5.1 (Self-RAG execution strategies) collection.
#
# Four execution strategies differing ONLY in YAML -- monolithic prompt,
# one model shared behind a lock, per-role copies, and one model behind a
# server with continuous batching -- across two tasks and two devices.
#
# TWO PASSES, and they are not interchangeable:
#
#   serial   the latency/quality pass, listeners OFF. Do not re-collect it
#            under listeners: that would fold an observer cost into the
#            headline table.
#            gb10 is already collected at R=4 and those runs are REUSED.
#            m3pro is NOT: the pre-existing "mlx" serial runs were collected on
#            the older 16 GB Mac, not on this 24 GB one (that machine had
#            neither sentence-transformers nor the embedding model cached, so
#            the pipeline had never run there). Reusing them would have put a
#            16 GB machine's latencies in a table whose memory column claims
#            24 GB, so m3pro's serial pass is collected fresh.
#   obs      the counter pass. Listener-ON twins of the same configs, shorter
#            (R=2), sourcing only the power/energy/memory columns. One paired
#            on/off cell bounds what the listeners themselves cost.
#
# The split is stated in the methods section; see self_rag.md.
#
#   usage: collect_e4.sh <machine> [runs] [mode] [config-glob]
#     machine : m3pro | gb10
#     runs    : repetitions (default 2 for obs, 6 for serial -- repetition 1 is
#               dropped at analysis as system warm-up)
#     mode    : obs (default) | serial
#     glob    : restrict to matching configs, e.g. 'factoid_monolith_4b'
set -uo pipefail

usage() { sed -n '2,28p' "$0" >&2; exit 2; }
MACHINE=${1:-}; RUNS=${2:-}; MODE=${3:-obs}; GLOB=${4:-*}
[ -n "$MACHINE" ] || usage
case "$MACHINE" in m3pro|gb10) ;; *) echo "collect_e4: machine must be m3pro or gb10" >&2; exit 2 ;; esac
case "$MODE" in obs|serial) ;; *) echo "collect_e4: mode must be obs or serial" >&2; exit 2 ;; esac
[ -n "$RUNS" ] || { [ "$MODE" = obs ] && RUNS=2 || RUNS=6; }

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$ROOT"

# The configs still carry the engine token (mlx/cuda); the machine token is what
# labels the results. Map once, here, rather than in every path below.
case "$MACHINE" in m3pro) ENGINE=mlx ;; gb10) ENGINE=cuda ;; esac

# Homebrew is NOT on the PATH of a non-interactive ssh session, and radt's
# macmon listener spawns `macmon` by name. Without this the listener fails to
# start and the run finishes with no counters and no error -- the exact failure
# that left the profiling contribution with no data. The gate below re-checks.
if [ "$(uname)" = "Darwin" ]; then
  export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
fi

# Keep the machine awake for the whole collection (see the E2 incident: a Mac on
# battery slept mid-run and moved three repetitions' medians 4-6% while
# reporting rc=0 and the right row counts).
if [ "$(uname)" = "Darwin" ] && [ -z "${E4_NO_CAFFEINATE:-}" ] \
   && [ -z "${_E4_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  export _E4_CAFFEINATED=1
  exec caffeinate -dimsu "$0" "$@"
fi

RESULTS="$HERE/results/$MACHINE"
mkdir -p "$RESULTS"
export BENCH_OUTPUT_DIR="$RESULTS"   # runs are written straight here

export RADT_TRACE_BACKEND=radt       # force the BULK exporter, not auto-detection
export RADT_PRESENT=True
export CHOREO_PROC_TRACE=1

# 5.1 records to res17 -- the local-store exemption covers the overhead
# experiments only.
: "${MLFLOW_TRACKING_URI:?collect_e4: MLFLOW_TRACKING_URI is unset. Activate the conda env rather than invoking the python binary directly; the credentials are env config vars.}"
EXP=${E4_EXPERIMENT:-138}
echo "5.1: recording to ${MLFLOW_TRACKING_URI} (experiment $EXP)"

if [ "$MODE" = obs ]; then
  PATTERN="$HERE/configs/${GLOB}_serial_obs_${ENGINE}.yml"
  case "$MACHINE" in m3pro) WANT_LISTENERS="macmon" ;; gb10) WANT_LISTENERS="dcgmi,top" ;; esac
else
  PATTERN="$HERE/configs/${GLOB}_serial_${ENGINE}.yml"
  WANT_LISTENERS=""
fi

# radt must be the PATCHED checkout, and every listener the configs ask for must
# actually be spawnable. Both are silent failures otherwise.
python scripts/radt_gate.py ${WANT_LISTENERS:+--listeners "$WANT_LISTENERS"} || exit 3

# Environment + config gate.
python - "$MACHINE" "$MODE" $(ls $PATTERN 2>/dev/null) <<'PY' || exit 3
import sys, yaml
machine, mode, cfgs = sys.argv[1], sys.argv[2], sys.argv[3:]
if not cfgs:
    sys.exit("collect_e4: no configs matched -- run gen_listener_configs.py?")
try:
    import radt, torch
except Exception as e:
    sys.exit(f"collect_e4: cannot import radt/torch: {e}")
want = {"gb10": "cuda", "m3pro": "mps"}[machine]
if want == "cuda" and not torch.cuda.is_available():
    sys.exit("collect_e4: machine is gb10 but torch.cuda.is_available() is False")
if want == "mps" and not torch.backends.mps.is_available():
    sys.exit("collect_e4: machine is m3pro but torch MPS is unavailable")
for c in cfgs:
    cfg = yaml.safe_load(open(c))
    listeners = cfg.get("listeners", ["macmon"])
    if mode == "obs" and not listeners:
        sys.exit(f"collect_e4: {c} is an obs config with no listeners")
    if mode == "serial" and listeners:
        sys.exit(f"collect_e4: {c} is the latency pass but declares listeners "
                 f"{listeners} -- that pass must stay listener-free")
print(f"collect_e4: {len(cfgs)} config(s) OK for {machine} [{mode}]")
PY

RUNSTAMP=$(date '+%Y%m%d-%H%M%S')
LOGDIR="$HERE/collect_logs"; mkdir -p "$LOGDIR"
SUM="$LOGDIR/collect_e4_${MACHINE}_${MODE}_${RUNSTAMP}.tsv"
LOG="$LOGDIR/collect_e4_${MACHINE}_${MODE}_${RUNSTAMP}.log"
printf 'mode\tcell\trun\trc\tseconds\tcsv_rows\tspans\n' > "$SUM"
log(){ local m="[$(date '+%m-%d %H:%M:%S')] $*"; echo "$m"; echo "$m" >> "$LOG"; }

{
  echo "# 5.1 collection"
  echo "# started      : $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "# machine      : $MACHINE (engine token: $ENGINE)"
  echo "# mode         : $MODE"
  echo "# git_commit   : $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "# git_dirty    : $(test -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" && echo yes || echo no)"
  echo "# host         : $(hostname)"
  echo "# platform     : $(python -c 'import platform;print(platform.platform())' 2>/dev/null)"
  echo "# python       : $(python -c 'import sys;print(sys.version.split()[0])' 2>/dev/null)"
  echo "# torch        : $(python -c 'import torch;print(torch.__version__)' 2>/dev/null || echo n/a)"
  echo "# radt         : $(python -c 'import radt,os;print(os.path.dirname(radt.__file__))' 2>/dev/null || echo n/a)"
  echo "# store        : $MLFLOW_TRACKING_URI (experiment $EXP)"
  echo "# listeners    : ${WANT_LISTENERS:-none}"
  echo "# runs         : $RUNS"
  # Who else is on the machine -- the E3 incident (a foreign training job
  # inflated four runs 3.4x) was invisible in rc, row counts and span counts.
  echo "# load         : $(uptime | sed 's/.*load average[s]*: //')"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "# gpu_procs    : $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';' || echo none)"
  fi
} >> "$LOG"

# Refuse to start on a busy GPU: 5.1 compares strategies on latency and power,
# and a co-resident job perturbs exactly that.
if [ -z "${E4_ALLOW_BUSY:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
  if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" -gt 0 ]; then
    echo "collect_e4: the GPU already has process(es) on it:" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
    echo "            Set E4_ALLOW_BUSY=1 to collect anyway and mark the data." >&2
    exit 4
  fi
fi

run_one() {
  local cfg=$1 cell=$2 r=$3 lab outfile start rc secs rows spans
  lab="e4_${cell}_${MACHINE}_r${r}"
  [ -f "$RESULTS/$lab.csv" ] && { log "  [skip] $lab (exists)"; return 0; }
  outfile=$(mktemp); start=$(date +%s)
  # main.py appends to an existing label's file, so a run killed part-way leaves
  # a partial session the next run concatenates onto (this produced a 4.6-hour
  # "interval" in E2). Clear both artifacts first.
  rm -f "$RESULTS/$lab.csv" "$RESULTS/${lab}_outputs.jsonl"
  python main.py "$cfg" -p 0 -e "$EXP" --label "$lab" 2>&1 | tee "$outfile"
  rc=${PIPESTATUS[0]}; secs=$(( $(date +%s) - start ))
  spans=$(sed -n 's/^\[[a-z]*\] spans emitted: //p' "$outfile" | tail -1)
  rm -f "$outfile"
  rows=$( [ -f "$RESULTS/$lab.csv" ] && wc -l < "$RESULTS/$lab.csv" | tr -d ' ' || echo 0 )
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$MODE" "$cell" "$r" "$rc" "$secs" "$rows" "${spans:-}" >> "$SUM"
  [ "$rc" -ne 0 ] && { log "  !! $lab FAILED rc=$rc"; return 1; }
  log "  $lab rc=$rc ${secs}s rows=$rows spans=${spans:-n/a}"
}

shopt -s nullglob
cfgs=( $PATTERN )
(( ${#cfgs[@]} )) || { echo "collect_e4: no configs match $PATTERN" >&2; exit 1; }

log "5.1 collection: ${#cfgs[@]} config(s) x $RUNS run(s) on $MACHINE [$MODE], listeners=${WANT_LISTENERS:-none}"
fail=0
for r in $(seq 1 "$RUNS"); do
  # Rotate strategy order every repetition: with a fixed order the first arm
  # always absorbs thermal ramp, and the existing data has arm order collinear
  # with a 7.5 h soak.
  n=${#cfgs[@]}; off=$(( (r - 1) % n ))
  for i in $(seq 0 $(( n - 1 ))); do
    cfg="${cfgs[$(( (off + i) % n ))]}"
    cell=$(basename "$cfg" .yml); cell=${cell%_$ENGINE}; cell=${cell%_obs}; cell=${cell%_serial}
    run_one "$cfg" "$cell" "$r" || fail=$((fail+1))
  done
done

log "5.1 collection done on $MACHINE [$MODE] ($fail failed run(s)). Results in $RESULTS/"
log "log + summary: $LOG"
[ "$fail" -eq 0 ]
