#!/usr/bin/env bash
# Canonical collection loop (replaces run_collection.py's orchestration).
#
# The execution model is deliberately minimal (see EXPERIMENTS.md "Execution
# model" + EXPERIMENT_PLAN.md P0): static, fully-explicit per-(variant x device)
# YAMLs, one bash loop over `main.py`, and radt does ALL orchestration. We pass
# NO `-p`, so main.py routes through radt.schedule_external, which spawns the
# listeners and (for multi-pipeline collocation configs) co-runs the pipelines.
#
# Tracing is the radt-owned bulk+proc exporter (CHOREO_PROC_TRACE=1): the
# workload only emits lightweight span events; a radt child process spools every
# span into one gzipped artifact per run on the tracking server. Backend is
# auto-detected (res17 -> "radt" bulk artifacts). See ENVIRONMENTS.md.
#
#   usage: collect.sh <config-glob> <device> [runs] [exp]
#     config-glob : quoted glob of configs, e.g. 'configs/*_mlx.yml'
#     device      : mlx | cuda  (only used to name the per-device results dir)
#     runs        : repeats per config (default 5)
#     exp         : res17 experiment id (default 138 = real collection; 142 = throwaway)
#
#   example:
#     conda activate benchmark_macos
#     evaluation/collect/collect.sh 'evaluation/overheads/framework_overhead/configs/noop_*_mode_ref.yml' mlx 10 138
set -uo pipefail

usage() { sed -n '2,20p' "$0" >&2; exit 2; }
GLOB=${1:-}; DEVICE=${2:-}; RUNS=${3:-5}; EXP=${4:-138}
[ -n "$GLOB" ] && [ -n "$DEVICE" ] || usage

# The repo root is two levels up from evaluation/collect/. main.py hardcodes its
# CSV output to evaluation/results/<label>.csv, so run from the repo root and
# sort the per-run outputs into evaluation/results/<device>/ afterwards.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

export CHOREO_PROC_TRACE=1        # radt-owned bulk+proc tracing (ENVIRONMENTS.md)

RESULTS="evaluation/results"
OUT="$RESULTS/$DEVICE"
mkdir -p "$OUT"

shopt -s nullglob
cfgs=( $GLOB )
(( ${#cfgs[@]} )) || { echo "collect: no configs match: $GLOB" >&2; exit 1; }

echo "collect: ${#cfgs[@]} config(s) x $RUNS run(s) -> exp $EXP  [$DEVICE]"
fail=0
for cfg in "${cfgs[@]}"; do
  name=$(basename "$cfg" .yml)
  for r in $(seq 1 "$RUNS"); do
    label="${name}_r${r}"
    echo "[$(date +%H:%M:%S)] $label  ($cfg)"
    python main.py "$cfg" -e "$EXP" --label "$label"   # no -p -> radt orchestrates
    rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "  !! rc=$rc for $label" >&2
      fail=$((fail + 1))
    fi
    # main.py writes evaluation/results/<label>.{csv,jsonl}; sort into the
    # per-device tree so analyze.py can glob one directory per device.
    for ext in csv jsonl; do
      [ -f "$RESULTS/$label.$ext" ] && mv "$RESULTS/$label.$ext" "$OUT/"
    done
  done
done

echo "collect: done ($fail failed run(s)). Outputs in $OUT/ ; spans on exp $EXP."
[ "$fail" -eq 0 ]
