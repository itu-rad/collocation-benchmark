#!/usr/bin/env bash
# Stop every collection process on this machine, safely.
#
# WHY THIS IS A FILE. `pgrep -f <pattern>` matches the command line doing the
# searching, so a one-liner like
#
#     ssh host 'pkill -f collect_e5'
#
# matches the ssh/bash running it, kills the session, and returns 255 while the
# actual collection keeps running. That happened four times in one day: each
# apparent "cleared" left the run alive, and one of those survivors overlapped a
# freshly launched collection for 20 minutes and contaminated it. Invoking a
# script by path keeps the pattern out of the invoking command line.
#
#     scripts/stop_collection.sh            # stop, then report what is left
#     scripts/stop_collection.sh --check    # report only, kill nothing
set -uo pipefail

PATTERNS=("collect_e1" "collect_e2" "collect_e3" "collect_e4" "collect_e5"
          "radt run" "main.py" "amc_bandwidth_sampler" "caffeinate -dimsu")

# Never kill ourselves OR ANY ANCESTOR. Excluding only $$ is not enough: this is
# usually invoked over ssh, and the ssh/bash command line contains the very
# pattern being matched, so pgrep returns the session and killing it takes down
# the caller (exit 255) while the target survives -- which looked exactly like a
# successful stop, twice.
SELF_CHAIN=""
_pid=$$
while [ -n "$_pid" ] && [ "$_pid" -gt 1 ]; do
  SELF_CHAIN="$SELF_CHAIN $_pid"
  _pid=$(ps -o ppid= -p "$_pid" 2>/dev/null | tr -d ' ')
done

not_self() {
  local pid
  while read -r pid; do
    case " $SELF_CHAIN " in *" $pid "*) continue ;; esac
    echo "$pid"
  done
}

alive() {
  local p n total=0
  for p in "${PATTERNS[@]}"; do
    n=$(pgrep -f "$p" 2>/dev/null | not_self | wc -l | tr -d ' ')
    total=$((total + n))
  done
  echo "$total"
}

if [ "${1:-}" = "--check" ]; then
  echo "collection processes running: $(alive)"
  ps -eo pid,etime,args 2>/dev/null | grep -E "collect_e[0-9]|radt run" | grep -v grep | cut -c1-110 || true
  exit 0
fi

# SIGTERM first so a run can flush its spans, then SIGKILL what ignores it.
for round in 1 2; do
  for p in "${PATTERNS[@]}"; do
    pids=$(pgrep -f "$p" 2>/dev/null | not_self)
    [ -n "$pids" ] && kill $([ "$round" -eq 2 ] && echo -9) $pids 2>/dev/null
  done
  sleep 3
done

# A killed schedule leaves this behind and the next run then fails its gate.
rm -f "$(cd "$(dirname "$0")/.." && pwd)/radtlock"

left=$(alive)
echo "remaining collection processes: $left"
[ "$left" -eq 0 ] || { echo "STILL RUNNING -- investigate before launching anything" >&2; exit 1; }
