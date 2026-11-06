#!/bin/bash
# Run one host's share of the shared v1 benchmark, for use inside tmux:
#   run_shared_v1_benchmark.sh VERSION [RELEASE...]
# Both arms of a release run side by side, so they see the same load. The native 3.14.6 and
# 3.15.0b2 verification compilers must already be in ~/pylingual-bench/compilers.
set -u
version=${1:?Python version}
shift
releases=${*:-v1 v2}
root=~/pylingual-bench
cd "$(dirname "$0")/.."
mkdir -p "$root/logs"
log="$root/logs/host.log"
python=.venv/bin/python
export PYTHONPATH=. TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
# torch 2.14 routes some eager attention matmuls to Triton kernels that need a C compiler, which
# these hosts lack; generation then fails and every statement silently becomes a placeholder.
export TORCH_DISABLE_NATIVE_JIT=1

# A fresh metadata file per launch, so a stale port can never be picked up.
compilers_json="$root/logs/compilers-$(date +%Y%m%dT%H%M%S).json"
$python -u -m dev_scripts.serve_benchmark_compilers --python314 "$root/compilers/cpython-3.14.6-linux-x86_64-gnu/bin/python3.14" \
  --python315 "$root/compilers/cpython-3.15.0b2-linux-x86_64-gnu/bin/python3.15" --metadata "$compilers_json" >> "$root/logs/compilers.log" 2>&1 &
server=$!
trap 'kill $server' EXIT
for _ in $(seq 1 60); do test -s "$compilers_json" && break; sleep 2; done
export PYLINGUAL_COMPILE_GATEWAY_PORT=$($python -m dev_scripts.benchmark_shared_v1 compiler-port --compilers "$compilers_json") || exit 1

echo "$(date -Is) prepare" | tee -a "$log"
$python -u -m dev_scripts.benchmark_shared_v1 prepare --root "$root" >> "$log" 2>&1 || { echo "$(date -Is) prepare failed" | tee -a "$log"; exit 1; }
for release in $releases; do
  pids=()
  for arm in no_adapter statement_adapter; do
    (
      for attempt in 1 2 3; do
        echo "$(date -Is) $release $arm $version attempt $attempt" | tee -a "$log"
        if $python -u -m dev_scripts.benchmark_shared_v1 unit --root "$root" --compilers "$compilers_json" --release "$release" --arm "$arm" --version "$version" \
             >> "$root/logs/$release-$arm-$version.log" 2>&1; then
          echo "$(date -Is) $release $arm $version complete" | tee -a "$log"
          exit 0
        fi
        sleep 60
      done
      echo "$(date -Is) $release $arm $version gave up" | tee -a "$log"
      exit 1
    ) &
    pids+=($!)
  done
  wait "${pids[@]}"
done
echo "$(date -Is) benchmark $version complete" | tee -a "$log"
