#!/bin/bash
# Paired TTFT on one GPU: native vLLM, then this engine (native + transfer), one process at a time, so the two engines
# never share the GPU and drift shows between the engine's first and last pass.
#   VLLM_PY=path/to/vllm-env/bin/python HF_HOME=... bash scripts/vllm_baseline/run.sh OUT_DIR
set -u
cd "$(dirname "$0")/../.."
OUT=${1:?usage: run.sh OUT_DIR}
mkdir -p "$OUT"
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VLLM_PY=${VLLM_PY:?point VLLM_PY at the python of an environment with vllm installed}
M27=$(python -c 'from xkv.experiments import resolve; print(resolve("27b"))')
B=scripts/vllm_baseline
step() { local name=$1; shift; echo "[run] $(date -u +%T) $name"; "$@" > "$OUT/$name.log" 2>&1 || echo "[run] $name FAILED (see $OUT/$name.log)"; }

step prompts python $B/make_prompts.py --out "$OUT/prompts.json"
step vllm "$VLLM_PY" $B/bench_vllm.py --model "$M27" --ids "$OUT/prompts.json" --out "$OUT/vllm.json"
step engine-random-1 python $B/bench_engine.py --mode random --out "$OUT/engine-random-1.json"
for i in 0 1 2 3 4 5; do
  step engine-real-$i python $B/bench_engine.py --mode real --ids "$OUT/prompts.json" --index $i --out "$OUT/engine-real-$i.json"
done
step engine-random-2 python $B/bench_engine.py --mode random --out "$OUT/engine-random-2.json"
echo "[run] $(date -u +%T) done: $OUT"
