# clone/: kvx, the earlier OneTriangle clone

Code for the open clone of OneTriangle's cross-model KV transfer for plain-attention models (Qwen3, Qwen2.5, Gemma,
Llama, Phi, OLMo). It is separate from the `xkv` engine in `src/xkv`, which handles the hybrid
Qwen3.5-4B → Qwen3.8-27B pair. Results are in [../docs/onetriangle-clone.md](../docs/onetriangle-clone.md).

**Status: TODO.** Still being worked on, to beat all of OneTriangle's published benchmarks.

```
kvx/core.py             RoPE strip, Gram statistics, layer choice, ridge fit, key restore, KL eval
kvx/vllm_connector.py   vLLM v1 KV connector (CrossModelConnector), tensor-parallel aware
kvx/ttft.py, bench.py   TTFT and stage benchmarks
kvx/{fa,fp8,mlp,recompute,distill,triton_kernels}.py   source prefill, fp8, MLP value map, recompute, kernels
modal_*.py, run_local.py   Modal entrypoints and a local runner
results/                raw result files behind every number in the clone notes
tests/                  CPU unit tests (test_core_cpu.py) and Modal-run checks
```

Run from this directory (`cd clone`). CPU tests: `python -m pytest tests/test_core_cpu.py -q`.
The Modal volume is named `kvx`.
