# ultra-cross-model-kv

**Prefill a 27B model with a 4B model: 2.4–3.1× faster time to first token at 8K–32K, 5× at 128K and 21.5× at 1M,
about 2× more requests served within a 5 s TTFT target, and no decode slowdown.**

`ultra-cross-model-kv` turns a Qwen3.5-4B prefill into a ready-to-decode cache for Qwen3.8-27B. It covers the
full-attention KV *and* the Gated-DeltaNet recurrent states of these hybrid models, so the 27B answers without
prefilling the prompt itself. One GPU, no cache transfer over the wire.

Comparisons are against [OneTriangle](https://onetriangle.ai/blog/transfer-benchmarks?pair=qwen-27)'s published
Qwen3.5-4B → Qwen3.8-27B results (page and data release dated 2026-11-13). Every number below comes from a file in
`results/`. Where we lose, the tables say so.

## Speed

### Time to first token (batch 1)

The comparable number is the **speedup over each side's own native baseline**; absolute times also reflect engine
and GPU differences (ours: FlashAttention-2 + fla kernels on one GPU; OneTriangle: HF forwards, 2×H100 at 32K).

| prompt | native 27B | ours | **speedup** | OneTriangle speedup | GPU, source |
|---:|---:|---:|---:|---:|---|
| 8K | 0.90–0.92 s | 0.33–0.38 s | **2.4–2.8×** | 2.6× | H100, 3 runs⁴ |
| 16K | 1.89–1.92 s | 0.63–0.68 s | **2.8–3.0×** | 2.7× | H100, 3 runs⁴ |
| 32K | 4.14–4.24 s | 1.33–1.38 s | **3.1×** | 2.8× | H100, 3 runs⁴ |
| 64K | 9.6–10.0 s | 2.4–3.0 s | **3.4–4.0×** | 3.0× | H100 / H200, 3 runs⁵ |
| 128K | 25.5 s | 5.07 s | **5.0×** | 3.3×¹ | H200, median of 3 |
| 256K | 74.7 s | 10.2 s | **7.4×** | 3.6×¹ | H200, 1 run |
| 512K | 243.1 s | 20.3 s | **12.0×** | 3.7×¹ | H200, 1 run |
| **1M** | 873.2 s | 40.7 s | **21.5×** (21.1×²) | 3.9×¹ | H200, 1 run |

¹ OneTriangle's values above 64K are projections (H100); ours are measured on one H200.
² An earlier 1M run: native 833.2 s, ours 39.5 s (`results/ttft_h200_128k-1m_first_run.json`); its 128K point was
not warmed up and is excluded.
⁴ `results/ttft_h100_final_engine_two_runs.json` and `results/ttft_h100_8k-32k_final_engine.json`: final engine,
final map, median of 5 per run. At 8K the three runs gave 2.44×, 2.69× and 2.77×, so at 8K we are level with
OneTriangle, not clearly ahead.
⁵ 64K does not fit next to the persistent graphed tail on an 80 GB H100 in the final build; the three measured
64K runs (`results/ttft_h100_8k-64k.json`, `results/ttft_h100_untuned_lengths_batches.json`,
`results/ttft_h200_64k-128k_reps3.json`) used earlier builds and gave 3.4×, 4.0× and 3.75×.
≥128K rows use an earlier engine build for both arms (`results/ttft_h200_128k-1m.json`).

**Shapes not used for tuning** (`results/ttft_h100_untuned_lengths_batches.json`, `results/ttft_h100_final_engine_two_runs.json`):
12K 2.6×, 24K 3.0×, 40K 3.5×; batch 2 at 16K 2.6×, at 32K 3.2×; batch 3 at 16K 3.2×. An earlier build gave
batch 2/3/4 at 8K 3.0×/3.1×/3.2× and 50K 3.7×. Batch 8 at 8K did not fit in 80 GB for the native arm.

### Decode (TPOT)

The transferred cache has the native cache's exact shape, so decoding costs the same as native at every batch size
measured (1, 2, 3, 4, 8, 12, 16, 32 at 2K; 1–12 at 32K; H200, CUDA graphs; `results/tpot_and_batch_sweeps.json`).
Optionally the 4B drafts tokens for the 27B to verify (this needs the full 4B cache, `keep_src=True`, which adds
the 4B's remaining layers and tail to the prefill):

| context | batch | native TPOT | ours | ours + 4B drafting |
|---:|---:|---:|---:|---:|
| 2K | 1 | 23.6 ms | 23.5 ms | 17.4 ms |
| 32K | 1 | 24.1 ms | 24.1 ms | 22.2 ms |
| 2K | 2 | 24.5 ms | 24.5 ms | 25.2 ms |
| 2K | 32 | 33.4 ms | 33.4 ms | 60.5 ms |

Drafting helps only at batch 1 (one prompt, 128 tokens per cell). Verification is greedy, but batched bf16 numerics
differ from step-by-step decoding: on 6 real prompts at batch 1, 4 produced a different continuation than plain
decoding (`results/spec_decode_row_independence.json`) — the same kind of divergence plain decoding shows between
batch sizes. OneTriangle publishes no TPOT for this pair.

### Serving under load

One H200, Poisson arrivals, prompts drawn from {8K, 16K, 32K}, 256 output tokens each, identical FCFS scheduler for
both arms, one seed, 30 requests per rate (`results/serve_h200_8k-32k.json`, 8 slots, and
`results/serve_h200_8k-32k_slots16_chunked.json`, 16 slots with chunked prefill):

| | native 27B (best of both schedulers) | ours (8 slots) | gain |
|---|---:|---:|---:|
| max throughput | 0.36 req/s | 0.45 req/s | 1.26× |
| best goodput, TTFT ≤ 5 s | 0.17 req/s | 0.37 req/s | 2.2× |
| best goodput, TTFT ≤ 2 s | 0.10 req/s | 0.20 req/s | 2.1× |
| TTFT p50 at 0.2 / 0.3 req/s | 2.0 / 5.2 s | 1.4 / 1.4 s | 1.5× / 3.6× |
| end-to-end p50 at 0.3 req/s | 27.4 s | 14.1 s | 1.9× |
| peak memory (ours incl. the 4B) | 101 GB | 96 GB | |

Throughput gains much less than TTFT because the 256-token decode, identical for both arms, dominates once prefill
is cheap. With 30 requests per point the p99 values in the result files are close to the maximum and the curves are
noisy.

## Quality

Native Qwen3.8-27B under our harness is the reference. Same prompts, greedy decoding, thinking off, for every arm.
The transfer uses map `4b-27b-chat`, a 128-token native tail and, up to 32K, full attention in the 4B.

### OneTriangle's 4B → 27B suite

| benchmark | n | native 27B | ours (95% CI) | 4B alone | OneTriangle mapped (native) |
|---|---:|---:|---:|---:|---:|
| ContractNLI | 300 | 0.80 | **0.75** (0.70–0.80) | 0.73 | 0.61 (0.79) |
| LongHealth | 200 | 0.895 | **0.835** (0.78–0.88) | 0.84 | 0.76 (0.72) |
| BAMBOO PaperQA | 84 | 0.869 | 0.881 (0.80–0.93) | 0.893 | 0.89 (0.88) |
| QMSum (ROUGE-L) | 100 | 0.220 | 0.213 | 0.203 | 0.41 (0.49)³ |
| MATH-500 | 200 | 0.900 | 0.895 | 0.88 | 0.59 (0.57) |
| ARC-Chat | 300 | 0.97 | 0.97 | 0.91 | 0.97 (0.97) |
| HumanEval | 164 | 0.933 | 0.915 (0.86–0.95) | — | 0.91 (0.93) |
| MBPP | 257 | 0.875 | 0.848 (0.80–0.89) | 0.80 | 0.88 (0.87) |
| BigCodeBench | 100 | 0.56 | 0.55 | 0.40 | 0.55 (0.60) |
| CowText | — | — | not run (not public) | — | 0.65 (0.64) |

³ OneTriangle does not state how QMSum is scored; as a share of native we keep 97%, they keep 84%.

- **Ahead:** ContractNLI and LongHealth, where the documents (2–14K tokens) are mostly mapped.
- **Level:** BAMBOO, MATH-500, ARC-Chat, HumanEval and BigCodeBench are within noise of OneTriangle's mapped score.
  MATH-500's large lead reflects harness differences, not transfer quality.
- **Behind:** MBPP (0.848 vs 0.88).
- **Losses vs native:** the transfer is 6 points below native on LongHealth and 5 on ContractNLI, and on
  long-document tasks it is roughly level with the 4B alone.
- **Short prompts are mostly native.** ARC, HumanEval, MBPP, BigCodeBench and MATH-500 prompts have a median of
  70–170 tokens, so most of each prompt is inside the 128-token native tail. With a 16-token tail (old code prompt),
  ARC drops to 0.92, HumanEval to 0.81 and MBPP to 0.67.

### Agents (τ2-bench, text, English)

| domain | native 27B | ours | OneTriangle mapped (native) |
|---|---:|---:|---:|
| airline | 0.74 | 0.46 | 0.59 (0.86) |
| retail | 0.78 | 0.15 | 0.41 (0.80) |
| telecom | not run | not run | 0.47 (0.86) |

We are **behind** on agents. Agent prompts carry a long policy document and tool schemas; mapped, the 27B picks the
wrong tool (e.g. hands off to a human instead of looking the user up). The agent configuration reads the newest
turn (up to 4096 tokens) and the `<tools>` block natively; this improved retail from 0.06 and airline from 0.39 but
not enough. Our user simulator and judges are the native 27B, not the official OpenAI models, so these are not
strictly comparable to OneTriangle's. Terminal-Bench 2.0 is implemented but was not run (its sandboxes need a Modal
token secret).

### Beyond 32K

Above 32K the 4B switches to an 8K sliding window, which is what makes 64K–1M fast. On RULER and MRCR at 32–64K
(4B → 27B, `results/quality_64k_ruler_mrcr.json`):

| | native 27B | ours |
|---|---:|---:|
| RULER (n=27) | 0.92 | 0.55 |
| MRCR, 2 needles (n=30) | 1.00 | 0.15 |

**Exact recall degrades sharply in the long-context configuration.** The speedups above 32K should be read as speed
results only; we have not validated quality there.

### OneTriangle's 2B → 27B tab

With a Qwen3.5-2B source (map `2b-27b-chat` / `2b-27b-chat2`), up to 32K:

| benchmark | n | native 27B | ours | 2B alone | OneTriangle (native) |
|---|---:|---:|---:|---:|---:|
| ARC-Chat | 100 | 0.98 | **0.98** | 0.81 | 0.89 (0.99) |
| QuALITY | 100 | 0.91 | **0.77** | 0.70 | 0.71 (0.94) |
| MATH-500 | 100 | 0.87 | 0.86 | 0.77 | 0.94 (0.96) |
| LongBench v2 (≤32K) | 60 | 0.37 | 0.32 | 0.23 | 0.42 (0.61) |
| RULER 4–16K | 90 | 0.97 | 0.68 | 0.92 | 0.80 (1.00) retrieval |
| MRCR 4–32K | 60 | 0.97 | 0.37 | 0.87 | up to 0.66 |
| LoCoMo (F1) | 100 | 0.58 | 0.45 | 0.39 | per-category only |
| ZebraLogic | 100 | 0.69 | 0.17 | 0.33 | 0.66 (0.91), reasoning on |

The 2B pair is **behind** on exact recall and puzzle solving (RULER, MRCR, ZebraLogic, LongBench v2) and ahead on
ARC-Chat and QuALITY. MATH-500 and ZebraLogic differ in setup: OneTriangle ran ZebraLogic with reasoning on; we run
everything non-thinking. DevEval was not run (needs the repository test environments).

## How it works

Qwen3.5 and Qwen3.8 are hybrid models. Three of every four layers are Gated DeltaNet (a linear-attention RNN with a
128×128 state per head); the fourth is gated full attention. Most of their "cache" is recurrent state, so a KV-only
transfer cannot work. We synthesize every piece the 27B needs:

1. **Source prefill.** The 4B runs over the prompt. We read only its normalized residual stream.
2. **Per-layer maps.** For each 27B layer, a linear map from a depth-matched 4B layer predicts what that layer caches:
   - *attention layers:* per-token `k` (post-norm, pre-RoPE) and `v`; the 27B's own RoPE is applied afterwards.
   - *DeltaNet layers:* the raw `k`, `v` projections and the gate inputs `a`, `b`, which then run through **the
     27B's own** causal conv and gated delta-rule scan to give the recurrent state the 27B expects.
3. **Tail repair.** The 27B natively runs the last 128 prompt tokens on the synthesized cache and emits the first
   token. Optional *native spans* let the 27B compute selected regions in place (used for tool schemas).
4. **Fit, then distill.** Maps start as closed-form ridge regressions on 1M calibration tokens, then are fine-tuned
   end to end on KL(native 27B ‖ transferred 27B) with gradients through the frozen 27B. Distillation data: FineWeb-Edu
   text, synthetic record lookups (close in style to needle-in-a-haystack tests), and public instruction data
   (LongAlign-10k, Tulu-3 SFT). These were not decontaminated against the evaluation sets.

**Engine.** Fused Triton causal-conv kernel, state-only delta-rule scan, a CUDA-graphed persistent tail (prompts
≤32K), CUDA-graph batched decoding with per-sequence lengths, and source-drafted speculative decoding with exact
recurrent-state rollback (rejected tokens re-scanned with β = 0, g = 0). fla's norm kernels re-autotune for every new
sequence length; we freeze those keys, which made generation evals about 4× faster.

## How the evaluation was run, and what to be careful about

- **Configuration was chosen on evaluation items.** Using full 4B attention up to 32K, the chat-distilled map, and the
  free-form code prompt were each adopted after trying them on LongHealth, ContractNLI and HumanEval items that are
  also part of the final samples. Treat those numbers as development-set results.
- **Harness.** Every benchmark was reimplemented (`src/xkv/genevals.py`); OneTriangle's prompts and scorers are not
  public. Code benchmarks use a free-form "answer in a code block" prompt (a forced code prefix cost the native 27B
  6 points on HumanEval); MBPP shows its test asserts in the prompt, as is standard; MATH-500 allows 3072 new tokens.
- **Greedy continuations diverge.** Next-token agreement with native is 92–94% teacher-forced, but free-running greedy
  outputs drift apart early (3% positional token agreement at 2K, 25% at 32K on FineWeb). Task scores above are the
  meaningful measure.
- **Parity.** The engine's parity check against Hugging Face (99.6% top-1, `results/selftest_parity.json`) was run
  on the 0.8B → 2B pair, not on the 27B.
- **Timing.** The transfer reuses a preallocated graphed tail; the native path allocates its cache inside the timed
  region (well under 20 ms). Both arms run on the same engine and GPU.
- bf16 throughout. Sample sizes are 60–300; differences of 2–3 points are within noise.

## Layout

```
src/xkv/qwen35.py       hybrid forward: chunked prefill, capture hooks, sliding-window ring cache
src/xkv/mapper.py       ridge statistics, per-layer maps, transfer prefill (native spans, graphed tail)
src/xkv/decode.py       batched graphed decoding, speculative drafting, tail engine
src/xkv/kernels.py      fused causal depthwise conv + SiLU (Triton)
src/xkv/genevals.py     generation benchmarks (ARC-Chat … MRCR) and their scorers
src/xkv/genrun.py       batched generation for native / transfer / source arms
src/xkv/agentbench.py   OpenAI-compatible local server; tau2-bench and Terminal-Bench drivers
src/xkv/perf.py         TTFT / TPOT / serving benchmarks and profiling
src/xkv/experiments.py  selftest, calibrate, distill, quality, longctx, bench
modal_jobs/             Modal images and entrypoints
results/                outputs of the runs quoted above
```

## Reproduce

From `modal_jobs/` with `PYTHONPATH=../src`:

```bash
modal run lab.py --task bench_matrix --gpu H100 --args '{"mapper_name":"4b-27b-chat","tail":128,"window":8192,"full_until":32768,"reps":5,"cells":[{"lengths":[8192,16384,32768]}]}'
modal run lab.py --task decode_bench --gpu H200 --args '{"lengths":[2048,32768],"batches":[1,2,3,4,8,12,16,32]}'
modal run lab.py --task serve --gpu H200 --args '{}'
modal run lab.py --task gen_evals --gpu H200 --args '{"mapper_name":"4b-27b-chat","full_until":32768,"benches":["contract_nli","longhealth"],"strategies":["target","transfer","source"]}'
modal run agent_lab.py --task tau2 --gpu H200 --args '{"mapper_name":"4b-27b-chat","full_until":32768,"strategies":["target","transfer"],"turn_tail_max":4096,"native_tools":true}'
```

Code benchmarks use `"env":{"XKV_CODE_PROMPT":"free"}`; BigCodeBench executes in the official evaluation image
(`modal deploy exec_app.py`, then `"exec_modal":true`).
