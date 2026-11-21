# TTFT against native vLLM

The speed tables in [docs/speed.md](../../docs/speed.md) divide each transfer time by this engine's own native
time. This folder adds the baseline a serving team would use instead: native Qwen3.8-27B in vLLM, on the same GPU,
with the same prompts, one engine at a time.

| file | what it does |
|---|---|
| `make_prompts.py` | six real ~32K-token LongHealth prompts (one patient's letters plus other patients' as distractors, seeded order), as token ids |
| `bench_vllm.py` | native TTFT in vLLM: wall time of `generate(max_tokens=1)` from token ids, prefix caching off, one warm-up and 5 timed requests per prompt; decode speed on the first real prompt |
| `bench_engine.py` | this engine's native and transfer TTFT with the calls of `perf.task_ttft_repair` (repair 0), arms interleaved after one warm-up each |
| `run.sh` | the paired sequence: prompts, vLLM, engine (random ids), engine (each real prompt in its own process), engine again |

## Run

vLLM pins its own torch, so it gets its own environment beside the engine's (the repo's Modal image pins):

```bash
uv venv vllm-env && VIRTUAL_ENV=vllm-env uv pip install vllm==0.34.0
VLLM_PY=$PWD/vllm-env/bin/python HF_HOME=<weights> bash scripts/vllm_baseline/run.sh out/vllm-baseline
```

Maps load from `/artifacts/maps` as everywhere else in the repo. On an 80 GB GPU vLLM needs `max_num_seqs` at most
325 for this model (its default of 1024 exceeds the DeltaNet state slots that fit beside the weights);
`bench_vllm.py` uses 128, which does not change a batch-1 prefill.
