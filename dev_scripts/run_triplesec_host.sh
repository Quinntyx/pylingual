#!/bin/bash
# Run this host's approved lanes in order (from the plan's "lanes" entry), for use inside tmux.
# A lane that cannot start yet (e.g. a LoRA waiting for the other host's backbone) or that fails
# after its own retries is retried every 10 minutes; every retry resumes from the Hub.
# Usage: run_triplesec_host.sh PLAN [ROOT]; each plan needs its own ROOT (v1: ~/pylingual-runs),
# because a lane skips stages whose result already exists under ROOT.
set -u
plan=${1:?plan JSON}
root=${2:-$HOME/pylingual-runs}
cd "$(dirname "$0")/.."
host=$(hostname)
lanes=$(.venv/bin/python -m dev_scripts.run_triplesec_lane --plan "$plan" --print-lanes "$host")
mkdir -p "$root/logs"
log="$root/logs/host.log"
for lane in $lanes; do
  for attempt in $(seq 1 300); do
    echo "$(date -Is) $host lane $lane attempt $attempt" | tee -a "$log"
    if .venv/bin/python -u -m dev_scripts.run_triplesec_lane --plan "$plan" --root "$root" --lane "$lane" >> "$log" 2>&1; then
      echo "$(date -Is) $host lane $lane complete" | tee -a "$log"
      continue 2
    fi
    sleep 600
  done
  echo "$(date -Is) $host lane $lane gave up" | tee -a "$log"
  exit 1
done
