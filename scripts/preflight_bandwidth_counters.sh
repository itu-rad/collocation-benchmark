#!/usr/bin/env bash
# ============================================================================
# Pre-flight check for the E3'/E6' contention experiments
# (see CONTENTION_EXPERIMENTS_REDESIGN.md, "Counters (VERIFY FIRST)").
#
# Question this script answers, per machine:
#   Can we measure DRAM bandwidth utilization (or a defensible proxy) here,
#   and with which tool/field? -> counter-backed vs proxy-backed vs estimate-only.
#
# Run on BOTH DUTs:
#   Apple silicon (darwin):  bash scripts/preflight_bandwidth_counters.sh
#                     (some probes need root; if passwordless sudo is absent the
#                      script prints the exact sudo commands to run manually)
#   GB10  (linux):    bash scripts/preflight_bandwidth_counters.sh
#
# Read-only: samples counters for a few seconds; changes nothing.
# ============================================================================
set -u

BOLD=$(tput bold 2>/dev/null || true); RST=$(tput sgr0 2>/dev/null || true)
hdr()  { printf '\n%s== %s ==%s\n' "$BOLD" "$*" "$RST"; }
note() { printf '  %s\n' "$*"; }
VERDICT=()

# Portable timeout (macOS has no coreutils `timeout` by default)
run_to() { # run_to <seconds> <cmd...>
  local secs=$1; shift
  ( "$@" ) & local pid=$!
  ( sleep "$secs"; kill "$pid" 2>/dev/null ) & local wpid=$!
  wait "$pid" 2>/dev/null; local rc=$?
  kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null
  return $rc
}

OS=$(uname -s)

# ============================================================================
if [ "$OS" = "Darwin" ]; then
  hdr "Machine"
  note "chip:   $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo '?')"
  note "macOS:  $(sw_vers -productVersion 2>/dev/null || echo '?') (Darwin $(uname -r))"
  note "memory: $(($(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824)) GB"

  hdr "sudo availability (powermetrics requires root)"
  if sudo -n true 2>/dev/null; then SUDO_OK=1; note "passwordless sudo: YES"; else SUDO_OK=0; note "passwordless sudo: NO (root probes will be skipped; commands printed at the end)"; fi

  # --------------------------------------------------------------------------
  hdr "Probe 1: powermetrics sampler inventory"
  if command -v powermetrics >/dev/null; then
    SAMPLERS=$(powermetrics -h 2>&1 | awk '/supported by --samplers/{f=1;next} /^ *-/{f=0} f && NF')
    printf '%s\n' "$SAMPLERS" | sed 's/^/  | /'
    if printf '%s' "$SAMPLERS" | grep -qi 'bandwidth'; then
      note "-> 'bandwidth' sampler ADVERTISED in help."
      PM_BW_ADVERTISED=1
    else
      note "-> no 'bandwidth' sampler in help (Apple removed DRAM bandwidth from powermetrics after Monterey/M1 on most builds)."
      PM_BW_ADVERTISED=0
    fi
  else
    note "powermetrics NOT FOUND (unexpected on macOS)"; PM_BW_ADVERTISED=0
  fi

  hdr "Probe 2: powermetrics live sample (root)"
  if [ "$SUDO_OK" = 1 ] && command -v powermetrics >/dev/null; then
    # (a) explicit bandwidth sampler, if the build accepts it
    if [ "$PM_BW_ADVERTISED" = 1 ]; then
      OUT=$(run_to 15 sudo -n powermetrics --samplers bandwidth -i 500 -n 2 2>&1)
      if printf '%s' "$OUT" | grep -qiE 'bandwidth|DCS|rd.*wr'; then
        note "bandwidth sampler RUNS. Sample lines:"
        printf '%s\n' "$OUT" | grep -iE 'bandwidth|DCS|GB/s|MB/s' | head -6 | sed 's/^/  | /'
        VERDICT+=("M2: powermetrics --samplers bandwidth WORKS -> counter-backed")
      else
        note "bandwidth sampler advertised but produced nothing useful:"
        printf '%s\n' "$OUT" | head -4 | sed 's/^/  | /'
      fi
    fi
    # (b) default samplers: look for any DRAM/memory/bandwidth section + DRAM power
    OUT=$(run_to 20 sudo -n powermetrics -i 500 -n 1 2>&1)
    HITS=$(printf '%s' "$OUT" | grep -icE 'bandwidth|dram' || true)
    note "default-sample lines mentioning bandwidth/DRAM: ${HITS}"
    printf '%s' "$OUT" | grep -iE 'bandwidth|dram' | head -8 | sed 's/^/  | /'
    if printf '%s' "$OUT" | grep -qiE 'DRAM.*(power|energy)|ane power|gpu power'; then
      note "-> per-rail power available (DRAM/GPU/ANE) -> usable as monotone PROXY."
      VERDICT+=("M2: powermetrics per-rail power available -> proxy-backed at minimum")
    fi
  else
    note "SKIPPED (no passwordless sudo)."
  fi

  # --------------------------------------------------------------------------
  hdr "Probe 3: macmon (repo's listener) field inventory"
  if command -v macmon >/dev/null; then
    note "macmon: $(macmon --version 2>/dev/null || echo present)"
    OUT=$(run_to 10 macmon pipe -s 1 2>&1)
    if printf '%s' "$OUT" | head -c1 | grep -q '{'; then
      note "JSON keys of one sample:"
      printf '%s\n' "$OUT" | head -1 | python3 -c '
import json,sys
def walk(d,p=""):
    if isinstance(d,dict):
        for k,v in d.items(): walk(v,f"{p}.{k}" if p else k)
    else: print(f"  | {p} = {d!r}"[:100])
walk(json.loads(sys.stdin.readline()))' 2>/dev/null | head -30
      if printf '%s' "$OUT" | grep -qiE 'bandwidth|dcs|rd_|wr_'; then
        note "-> macmon exposes a bandwidth-like field."
        VERDICT+=("M2: macmon exposes bandwidth-like field -> counter-backed via existing listener")
      elif printf '%s' "$OUT" | grep -q 'ram_power'; then
        note "-> no bandwidth field, but ram_power (DRAM rail) present -> monotone power PROXY, no root."
        VERDICT+=("M2: macmon ram_power (DRAM rail) -> proxy-backed via existing listener")
      else
        note "-> no bandwidth field in macmon output (power/usage only)."
      fi
    else
      note "macmon pipe failed:"; printf '%s\n' "$OUT" | head -3 | sed 's/^/  | /'
    fi
  else
    note "macmon NOT on PATH (repo lists it as the Apple listener — install to use as proxy)."
  fi

  # --------------------------------------------------------------------------
  hdr "Probe 4: per-engine DRAM counters via IOReport (no root) - the decisive check"
  # Two backends exist, and which one works is a property of the SoC generation:
  #   M2-family: "AMC Stats / Perf Counters" per-requestor RD/WR BYTE counters
  #   M3-family: "PMP / DCS BW / <REQ>" bandwidth HISTOGRAMS -- there the AMC
  #              group still enumerates 128 channels, but its driver refuses the
  #              subscription and they never reach a sample.
  # So EXISTENCE IS NOT READABILITY: the check below is that channels survive
  # into a SAMPLE, not that they appear in the channel list. Checking the latter
  # is what once made the M3 look like "counters that exist and read zero".
  # See docs/amc-m3-counters-plan.md.
  PROBE_SRC="$(dirname "$0")/ioreport_bw_probe.c"
  if command -v clang >/dev/null && [ -f "$PROBE_SRC" ]; then
    PROBE_BIN=$(mktemp -t bwprobe)
    if clang -O2 -o "$PROBE_BIN" "$PROBE_SRC" -framework CoreFoundation 2>/dev/null; then
      OUT=$(run_to 30 "$PROBE_BIN" 2>/dev/null)
      note "channel survival (COPIED -> SUBSCRIBED -> SAMPLED):"
      printf '%s\n' "$OUT" | sed -n '/per-group survival/,/deltas for flagged/p' \
        | grep -E 'AMC|PMP|COPIED|DROPPED' | sed 's/^/  | /'
      if printf '%s' "$OUT" | grep -q 'DROPPED'; then
        note "-> a group enumerates but never samples (expected on M3-family for AMC Stats):"
        note "   its channels are UNREADABLE, not zero. The PMP backend covers this."
      fi
      note "live bandwidth channels (1s deltas):"
      printf '%s\n' "$OUT" | sed -n '/deltas for flagged/,$p' \
        | grep -E 'DCS BW|DCS (RD|WR)' | head -6 | sed 's/^/  | /'
      rm -f "$PROBE_BIN"
    else
      note "probe failed to compile (need Xcode CLT)."
    fi
  else
    note "clang or $PROBE_SRC missing - skipped."
  fi

  hdr "Probe 5: the sampler we actually collect with"
  # The authoritative check: whichever backend this picks is what a run records.
  SAMP_SRC="$(dirname "$0")/amc_bandwidth_sampler.c"
  if command -v clang >/dev/null && [ -f "$SAMP_SRC" ]; then
    SAMP_BIN=$(mktemp -t amcsamp)
    if clang -O2 -o "$SAMP_BIN" "$SAMP_SRC" -framework CoreFoundation 2>/dev/null; then
      SAMP_CSV=$(mktemp -t amccsv)
      ERR=$(run_to 30 "$SAMP_BIN" -i 500 -n 2 -o "$SAMP_CSV" 2>&1 >/dev/null); RC=$?
      printf '%s\n' "$ERR" | sed 's/^/  | /'
      BACKEND=$(printf '%s' "$ERR" | sed -n 's/.*backend=\([a-z]*\).*/\1/p' | head -1)
      if [ "$RC" -eq 3 ]; then
        note "-> rc=3: NEITHER backend is readable here. E3'/E6' per-engine bytes are"
        note "   unavailable on this machine; escalate before collection."
      elif [ "$BACKEND" = "amc" ]; then
        note "-> backend=amc: exact per-requestor DRAM BYTE counters, no root."
        VERDICT+=("$(sysctl -n machdep.cpu.brand_string): IOReport AMC byte counters -> COUNTER-BACKED, exact per-engine attribution")
      elif [ "$BACKEND" = "pmp" ]; then
        note "-> backend=pmp: per-engine bandwidth HISTOGRAMS; bytes are derived."
        note "   Accurate for smooth loads below the 32 GB/s per-requestor ceiling;"
        note "   above it rows are a lower bound (check the CSV's 'saturated' column)."
        VERDICT+=("$(sysctl -n machdep.cpu.brand_string): IOReport PMP DCS BW histograms -> COUNTER-BACKED, per-engine attribution (derived bytes, 32 GB/s ceiling)")
      else
        note "-> could not determine backend; sampler said rc=$RC."
      fi
      [ -s "$SAMP_CSV" ] && { note "first data row:"; sed -n '2p' "$SAMP_CSV" | sed 's/^/  | /'; }
      rm -f "$SAMP_BIN" "$SAMP_CSV"
    else
      note "sampler failed to compile (need Xcode CLT)."
    fi
  else
    note "clang or $SAMP_SRC missing - skipped."
  fi

  # --------------------------------------------------------------------------
  hdr "VERDICT (this machine)"
  if [ ${#VERDICT[@]} -gt 0 ]; then printf '  * %s\n' "${VERDICT[@]}"; else
    note "No counter or proxy confirmed yet."
  fi
  if [ "$SUDO_OK" = 0 ]; then
    cat <<'EOF'

  Root probes were skipped. To finish the check, run manually:
    sudo powermetrics -i 500 -n 1 | grep -iE 'bandwidth|dram'
    sudo powermetrics --samplers bandwidth -i 500 -n 2   # if advertised in -h
  Interpretation:
    - bandwidth sampler works        -> counter-backed E3' mechanism claim
    - only DRAM/GPU/ANE power lines  -> proxy-backed (power as monotone proxy)
    - neither                        -> try socpowerbud/macpm (IOReport DCS rd/wr),
                                        else estimate-only (model-based GB/s)
EOF
  fi

# ============================================================================
elif [ "$OS" = "Linux" ]; then
  hdr "Machine"
  note "kernel: $(uname -r) ($(uname -m))"
  command -v nvidia-smi >/dev/null && note "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

  hdr "Probe 1: nvidia-smi memory-controller utilization (%)"
  if command -v nvidia-smi >/dev/null; then
    OUT=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv 2>&1)
    printf '%s\n' "$OUT" | sed 's/^/  | /'
    if printf '%s' "$OUT" | grep -q '%'; then
      note "-> utilization.memory (fraction of time the memory controller is busy) is a usable PROXY."
      VERDICT+=("GB10: nvidia-smi utilization.memory works -> proxy-backed at minimum")
    fi
    note "dmon variant (mem column):"
    run_to 10 nvidia-smi dmon -s u -c 2 2>&1 | head -5 | sed 's/^/  | /'
  else
    note "nvidia-smi NOT FOUND."
  fi

  hdr "Probe 2: DCGM profiling field DRAM_ACTIVE (field 1005 — the real counter)"
  if command -v dcgmi >/dev/null; then
    note "dcgmi present: $(dcgmi --version 2>/dev/null | head -1)"
    OUT=$(run_to 20 dcgmi dmon -e 1005,1002 -c 2 2>&1)
    printf '%s\n' "$OUT" | head -8 | sed 's/^/  | /'
    if printf '%s' "$OUT" | grep -qE '[0-9]\.[0-9]+'; then
      note "-> PROF_DRAM_ACTIVE reads on this GPU: counter-backed."
      VERDICT+=("GB10: DCGM PROF_DRAM_ACTIVE (1005) works -> counter-backed")
    else
      note "-> profiling fields not returning values (DCP unsupported on this SKU, or nv-hostengine not running: try 'sudo nv-hostengine' first)."
    fi
  else
    note "dcgmi NOT FOUND (datacenter-gpu-manager not installed)."
  fi

  hdr "Probe 3: Grace-side (CPU/LPDDR) uncore counters via perf"
  if command -v perf >/dev/null; then
    HITS=$(perf list 2>/dev/null | grep -icE 'scf|cmn|ddr|mem_?bw|bandwidth' || true)
    note "perf events matching scf/cmn/ddr/bandwidth: ${HITS}"
    perf list 2>/dev/null | grep -iE 'scf|cmn|ddr|mem_?bw|bandwidth' | head -8 | sed 's/^/  | /'
    if [ "${HITS:-0}" -gt 0 ]; then
      note "-> Grace uncore bandwidth events exist: CPU-side traffic (co-runner C3) measurable."
      VERDICT+=("GB10: perf uncore events present -> Grace-side traffic measurable")
    else
      note "-> none visible (may need root or kernel PMU driver)."
    fi
  else
    note "perf NOT FOUND."
  fi

  hdr "VERDICT (this machine)"
  if [ ${#VERDICT[@]} -gt 0 ]; then printf '  * %s\n' "${VERDICT[@]}"; else
    note "No counter or proxy confirmed. E3' on this DUT would be estimate-only — escalate before collection."
  fi

else
  echo "Unsupported OS: $OS"; exit 1
fi

echo
echo "Done. Paste this output into CONTENTION_EXPERIMENTS_REDESIGN.md §3 (checklist item 1)."
