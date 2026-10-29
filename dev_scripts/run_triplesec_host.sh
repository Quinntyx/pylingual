#!/bin/bash
# Run this host's approved lanes in order (from the plan's "lanes" entry), for use inside tmux.
# A lane that cannot start yet (e.g. a LoRA waiting for the other host's backbone) or that fails
# after its own retries is retried every 10 minutes; every retry resumes from the Hub.
set -u
plan=${1:?plan JSON}
cd "$(dirname "$0")/.."
host=$(hostname)
lanes=$(.venv/bin/python -m dev_scripts.run_triplesec_lane --plan "$plan" --print-lanes "$host")
mkdir -p ~/pylingual-runs/logs
for lane in $lanes; do
  for attempt in $(seq 1 300); do
    echo "$(date -Is) $host lane $lane attempt $attempt" | tee -a ~/pylingual-runs/logs/host.log
    if .venv/bin/python -u -m dev_scripts.run_triplesec_lane --plan "$plan" --lane "$lane" >> ~/pylingual-runs/logs/host.log 2>&1; then
      echo "$(date -Is) $host lane $lane complete" | tee -a ~/pylingual-runs/logs/host.log
      continue 2
    fi
    sleep 600
  done
  echo "$(date -Is) $host lane $lane gave up" | tee -a ~/pylingual-runs/logs/host.log
  exit 1
done
