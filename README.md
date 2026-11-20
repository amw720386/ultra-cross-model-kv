# ultra-cross-model-kv

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Models: Qwen3.5-4B → Qwen3.8-27B](https://img.shields.io/badge/models-Qwen3.5--4B%20%E2%86%92%20Qwen3.8--27B-6f42c1.svg)](docs/method.md)
[![GPU: H100 / H200](https://img.shields.io/badge/GPU-H100%20%2F%20H200-76b900.svg)](#reproduce)

Prefill Qwen3.8-27B with Qwen3.5-4B. The 4B reads the prompt; learned maps turn its activations into the 27B's
full-attention KV **and** Gated-DeltaNet recurrent states; the 27B decodes from there. One GPU, no cache over the wire.

- **2.4–3.1× faster TTFT** at 8K–32K, no decode slowdown
- **~2× goodput** under load (TTFT ≤ 5 s)
- **Level with or ahead of [OneTriangle](https://onetriangle.ai/blog/transfer-benchmarks?pair=qwen-27)** on every
  long-document and coding benchmark both ran
- 5× at 128K, 21.5× at 1M, but exact recall is not preserved beyond 32K

![TTFT speedup](docs/img/ttft_speedup.png)

![share of native](docs/img/share_of_native.png)

## Results

| | ours | OneTriangle |
|---|---:|---:|
| TTFT speedup, 8K / 16K / 32K | 2.4–2.8 / 2.8–3.0 / 2.95–3.1× | 2.6 / 2.7 / 2.8× |
| TTFT speedup, 128K / 1M | 5.0 / 21.5× (speed only) | 3.3 / 3.9× (projected) |
| decode (TPOT) | = native | not published |
| serving goodput, TTFT ≤ 5 s | 1.9× native | not published |
| ContractNLI | **0.757** | 0.61 |
| LongHealth | **0.845** | 0.76 |
| HumanEval / MBPP / BigCodeBench | 0.927 / 0.875 / 0.56 | 0.91 / 0.88 / 0.55 |
| 2B → 27B QuALITY | **0.79** | 0.71 |
| 2B → 27B RULER (recall mode) | **0.905** | 0.80 |
| τ2 retail (lighter agent setting) | **0.78** | 0.41 |
| τ2 telecom (lighter agent setting) | 0.22 (native 0.68) | **0.47** (native 0.86) [*] |
| Terminal-Bench 2.0 | 0.10 (native 0.70) | 0.10 |

[*] Their native 27B scores 0.18 higher than ours on telecom. Measured as the drop from each side's own native, the
gap is much closer: −0.46 for us, −0.39 for them (17 tasks, ±0.23).

**Where it loses:** agents with long rule-heavy policies (τ2 telecom, Terminal-Bench) only match native in a setting
that is slower than native; exact recall beyond 32K; MMLU with a fully mapped prompt.

Details:

- [docs/benchmarks.md](docs/benchmarks.md): scoreboard, 4B and 2B suites, held-out gate, long context
- [docs/speed.md](docs/speed.md): TTFT, recall-mode cost, decode, serving
- [docs/agents.md](docs/agents.md): τ2-bench and Terminal-Bench
- [docs/method.md](docs/method.md): how it works, glossary, evaluation caveats, limitations
- [results/](results/README.md): the result file behind every number

## Reproduce

Runs on [Modal](https://modal.com). Install the client, authenticate, then:

```bash
pip install -e .            # xkv package, for local tests
cd modal_jobs
export PYTHONPATH=../src

modal run prefetch.py                         # model weights into a Modal volume
modal run lab.py --task selftest --gpu H100   # engine parity vs Hugging Face (0.8B → 2B)

# TTFT, 4B → 27B (fast mode + recall mode)
modal run lab.py --task ttft_repair --gpu H100 --args '{"repairs":[0,1024,2048]}'

# quality: native vs transfer on ContractNLI and LongHealth
modal run lab.py --task gen_evals --gpu H200 --args '{"mapper_name":"4b-27b-fid+4b-27b-fid-attn","full_until":32768,"benches":["contract_nli","longhealth"],"strategies":["target","transfer@auto"]}'

# decode and serving
modal run lab.py --task decode_bench --gpu H200 --args '{"lengths":[2048,32768],"batches":[1,2,3,4,8,12,16,32]}'
modal run lab.py --task serve --gpu H200 --args '{}'
```

Every other run (2B pair, recall sweeps, τ2, Terminal-Bench, map training) is in
[docs/reproduce.md](docs/reproduce.md). Each file in `results/` records the exact `args` of its run.

CPU tests: `PYTHONPATH=src python -m pytest tests -q`

## Layout

```
src/xkv/qwen35.py       hybrid forward: chunked prefill, capture hooks, ring cache, scattered re-encode
src/xkv/mapper.py       ridge stats, per-layer maps, transfer prefill (native spans, adaptive tail, repair)
src/xkv/attnmap.py      multi-layer + embedding attention-KV maps
src/xkv/decode.py       batched graphed decoding, speculative drafting, tail engine
src/xkv/kernels.py      fused causal depthwise conv + SiLU (Triton)
src/xkv/genevals.py     generation benchmarks and scorers
src/xkv/genrun.py       batched generation for native / transfer / source arms
src/xkv/agentbench.py   OpenAI-compatible local server; τ2 and Terminal-Bench drivers
src/xkv/perf.py         TTFT / TPOT / serving benchmarks
src/xkv/experiments.py  selftest, calibrate, distill, quality, longctx, bench
modal_jobs/             Modal images and entrypoints
results/                result files behind every number
docs/                   detailed results and method
```

## License

[MIT](LICENSE). Qwen3.5 and Qwen3.8 weights are under their own licenses.
