# ultra-cross-model-kv

**A 4B model reads the prompt so a 27B doesn't have to: Qwen3.8-27B starts answering 3× sooner at 32K, level with or
ahead of OneTriangle's cross-model transfer on every long-document and coding benchmark both of us ran, with no
decode slowdown.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Models: Qwen3.5-4B → Qwen3.8-27B](https://img.shields.io/badge/models-Qwen3.5--4B%20%E2%86%92%20Qwen3.8--27B-6f42c1.svg)](#how-it-works)
[![GPU: H100 / H200](https://img.shields.io/badge/GPU-H100%20%2F%20H200-76b900.svg)](#reproduce)

[Results](#results) · [How it works](#how-it-works) · [Where it falls short](#where-it-falls-short) · [Reproduce](#reproduce) · [Full results](docs/RESULTS.md)

![Transfer score as a share of each side's own native 27B score, ours against OneTriangle, per benchmark](docs/share_simple.png)

- **2.4–3.1× faster time to first token at 8K–32K** on one H100 (32K: 4.2 s → 1.4 s), level with OneTriangle at 8K
  and ahead from 16K.
- **Level with or ahead of OneTriangle on every long-document and coding benchmark both of us ran**, with
  ContractNLI at 0.757 against their 0.61 and LongHealth at 0.845 against 0.76.
- **Decoding costs exactly what native does**, because the transferred cache has the native cache's exact shape.
- **About 2× the requests served within a 5 s TTFT target** under Poisson load on one H200.
- **21.5× faster at 1M tokens** (873 s → 40.7 s), as a speed result: exact recall is not preserved beyond 32K.

`ultra-cross-model-kv` turns a Qwen3.5-4B prefill into a ready-to-decode cache for Qwen3.8-27B. These are hybrid
models, so three of every four layers keep a Gated DeltaNet recurrent state instead of a KV cache, and a KV-only
transfer can't work. We synthesize both the full-attention KV and the DeltaNet states from the 4B's residual stream,
so the 27B computes only a short tail of the prompt itself and answers from the mapped cache. Everything runs on one
GPU with no cache sent over the wire, and every number below comes from a file in [`results/`](results/).

## Results

### Time to first token

Batch 1, the same engine and GPU for both arms. The comparable figure is each side's speedup over its own native 27B,
since absolute times also reflect engine and hardware (OneTriangle ran Hugging Face forwards on 2×H100 at 32K).

| prompt | native 27B | ours | speedup | OneTriangle speedup |
|---:|---:|---:|---:|---:|
| 8K | 0.90–0.93 s | 0.33–0.38 s | **2.4–2.8×** | 2.6× |
| 16K | 1.89–1.94 s | 0.63–0.68 s | **2.8–3.0×** | 2.7× |
| 32K | 4.14–4.25 s | 1.33–1.43 s | **2.95–3.1×** | 2.8× |
| 64K | 9.6–10.0 s | 2.4–3.0 s | **3.4–4.0×** | 3.0× |
| 128K | 25.5 s | 5.07 s | **5.0×** | 3.3×¹ |
| 1M | 873.2 s | 40.7 s | **21.5×** | 3.9×¹ |

8K–32K: H100, six runs, median of 5 each. 64K and above: H100 / H200 on an earlier engine build, with the 4B on an 8K
sliding window, which is what makes these lengths fast and also why exact recall is lost there.
¹ OneTriangle's values above 64K are projections.

With a Qwen3.5-2B source the speedup is 3.94× / 4.47× / 4.94× at 8K / 16K / 32K. Decode time per token matches
native at every batch size measured (1–32 at 2K, 1–12 at 32K), and under serving load on one H200 the transfer serves
1.9–2.2× the requests native does within a 5 s TTFT target.

### Quality against OneTriangle

Same prompts for native and transfer, greedy, bf16 on both models with no quantization. OneTriangle's prompts and
scorers aren't public, so we reimplemented every benchmark, and their native column shows where harnesses differ.

| benchmark | n | native 27B | ours | OneTriangle (their native) |
|---|---:|---:|---:|---:|
| ContractNLI | 300 | 0.80 | **0.757** | 0.61 (0.79) |
| LongHealth | 200 | 0.895 | **0.845** | 0.76 (0.72) |
| BAMBOO PaperQA | 84 | 0.869 | 0.881 | 0.89 (0.88) |
| HumanEval | 164 | 0.933 | 0.927 | 0.91 (0.93) |
| MBPP | 257 | 0.875 | 0.875 | 0.88 (0.87) |
| BigCodeBench | 100 | 0.56 | 0.56 | 0.55 (0.60) |
| QuALITY (2B source) | 100 | 0.91 | **0.79** | 0.71 (0.94) |
| ZebraLogic, thinking on (2B source) | 100 | 0.95 | **0.75** | 0.66 (0.91) |
| RULER 4–16K (2B source, recall mode) | 90 | 0.97 | **0.905** | 0.80 (1.00) |
| MRCR 4–32K (2B source, recall mode, held-out) | 120 | 0.972 | **0.681** | up to 0.66 |

Bold marks a lead; most cells hold 50–300 items (±0.06–0.13 at 95%), so the QuALITY, ZebraLogic and MRCR leads are
within noise. ARC-Chat and MATH-500 with thinking on are ties, LongBench v2 ties in recall mode, and QMSum keeps
84% of native on both sides.

**Recall mode** (`+repN`) is the second setting: the 4B's attention picks the parts of the prompt that matter for the
question and the 27B re-encodes them natively. RULER, MRCR and LongBench v2 need it, at 1.4–3.6× TTFT for the 2B pair
instead of 3.9–4.9× in the default fast mode. Every row in [docs/RESULTS.md](docs/RESULTS.md) names its mode.

## How it works

1. **Source prefill.** The 4B runs over the prompt, and we read only its normalized residual stream.
2. **Per-layer maps.** For each 27B layer, a learned linear map from depth-matched 4B layers predicts what that layer
   caches. Attention layers get per-token `k` and `v` (the 27B's own RoPE applied afterwards), and DeltaNet layers get
   their `k`, `v` and gate inputs, which then run through **the 27B's own** causal conv and gated delta-rule scan to
   give exactly the recurrent state the 27B expects.
3. **Native tail.** The 27B runs the last prompt tokens natively on the synthesized cache and emits the first token,
   using 512 tokens for prompts up to 4K, tapering to 128 at 8K and above.
4. **Recall mode.** Optionally, the source scores 64-token blocks or whole chat messages by attention from the tail,
   and the 27B re-encodes the top N tokens in one batched pass, keeping only the attention KV it writes.
5. **Fit, then distill.** Maps start as closed-form ridge regressions on 1M calibration tokens, then are fine-tuned
   end to end on KL(native 27B ‖ transferred 27B), with gradients through the frozen 27B.

The engine adds a fused Triton causal-conv kernel, a state-only delta-rule scan, a CUDA-graphed persistent tail,
graph-batched decoding, and optional 4B-drafted speculative decoding with exact recurrent-state rollback.

## Where it falls short

- **Agents with long, rule-heavy policies.** On τ2 retail the transfer keeps 93% of native (0.78, against
  OneTriangle's 0.41), but on telecom it scores 0.22 against native's 0.68 when most of the prompt is mapped. Matching
  native on telecom needs a setting that computes most of the prompt natively, which is slower than native. On
  Terminal-Bench 2.0, where prompts grow to 75K tokens and every turn re-transfers them, the transfer solves 1 of 10
  tasks against native's 7.
- **Exact recall beyond 32K.** Above 32K the 4B reads through an 8K sliding window, so RULER at 64K scores 0.55 against
  native's 0.92, and the long-context speedups are speed results only.
- **Varying prompt lengths.** The graphed tail is captured per prompt length, so a serving mix of new lengths should
  run with `XKV_NO_GRAPH_TAIL=1`.
- **Tuning on evaluation items.** The maps, tail and recall sizes were developed while looking at these benchmarks,
  so treat them as development-set results. A held-out gate (FineWeb-Edu and five multiple-choice sets never used
  for selection) and a held-out MRCR sample guard against overfitting, and both are in
  [docs/RESULTS.md](docs/RESULTS.md).

## Reproduce

Everything runs on [Modal](https://modal.com). Install the client, authenticate, and run from `modal_jobs/`:

```bash
pip install -e .            # the xkv package and its dependencies, for local tests
cd modal_jobs
export PYTHONPATH=../src

modal run prefetch.py                                   # Qwen3.5 0.8B / 2B / 4B and Qwen3.8-27B into a Modal volume
modal run lab.py --task selftest --gpu H100             # engine parity against Hugging Face (0.8B → 2B)

# Headline TTFT, 8K–32K, fast mode and recall mode
modal run lab.py --task ttft_repair --gpu H100 --args '{"repairs":[0,1024,2048]}'

# Quality, 4B → 27B
modal run lab.py --task gen_evals --gpu H200 --args '{"mapper_name":"4b-27b-fid+4b-27b-fid-attn","full_until":32768,"benches":["contract_nli","longhealth"],"strategies":["target","transfer@auto"]}'

# Decode and serving
modal run lab.py --task decode_bench --gpu H200 --args '{"lengths":[2048,32768],"batches":[1,2,3,4,8,12,16,32]}'
modal run lab.py --task serve --gpu H200 --args '{}'
```

Maps are built with the `calibrate` and `calibrate_attn` tasks (ridge fit) and then `distill`, whose signatures are in
`src/xkv/experiments.py` and `src/xkv/attnmap.py`, and they are written to the `xkv-artifacts` volume under the name
passed as `out_name`. Each result file in `results/` records the exact `args` of its run, and the command for every
table, agents included, is in [docs/RESULTS.md](docs/RESULTS.md#reproduce). The official τ2 user simulator and
Terminal-Bench need the Modal secrets `xkv-openai` (`OPENAI_API_KEY`) and `xkv-harbor-modal`.

CPU tests: `PYTHONPATH=src python -m pytest tests -q`.

## Layout

```
src/xkv/qwen35.py       hybrid forward: chunked prefill, capture hooks, ring cache, scattered-position re-encode
src/xkv/mapper.py       ridge statistics, per-layer maps, transfer prefill (native spans, adaptive tail, repair)
src/xkv/attnmap.py      multi-layer + embedding attention-KV maps
src/xkv/decode.py       batched graphed decoding, speculative drafting, tail engine
src/xkv/kernels.py      fused causal depthwise conv + SiLU (Triton)
src/xkv/genevals.py     generation benchmarks (ARC-Chat … MRCR) and their scorers
src/xkv/genrun.py       batched generation for native / transfer / source arms
src/xkv/agentbench.py   OpenAI-compatible local server; tau2-bench and Terminal-Bench drivers
src/xkv/perf.py         TTFT / TPOT / serving benchmarks and profiling
src/xkv/experiments.py  selftest, calibrate, distill, quality, longctx, bench
modal_jobs/             Modal images and entrypoints
results/                outputs of every run quoted here, indexed in results/README.md
docs/RESULTS.md         every table, note and caveat, with sample sizes and sources
```

## License

[MIT](LICENSE). Qwen3.5 and Qwen3.8 weights are under their own licenses.
