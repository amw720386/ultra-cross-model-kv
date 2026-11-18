# ultra-cross-model-kv

**Prefill a 27B model with a 4B model: 2.7–3.0× faster time to first token at 8K–32K, about 2× more requests served
within a 5 s TTFT target, and no decode slowdown. Beyond 32K it is 5× faster at 128K and 21.5× at 1M, but exact
recall is not preserved there.**

`ultra-cross-model-kv` turns a Qwen3.5-4B prefill into a ready-to-decode cache for Qwen3.8-27B. It covers the
full-attention KV *and* the Gated-DeltaNet recurrent states of these hybrid models, so the 27B answers without
prefilling the prompt itself. One GPU, no cache transfer over the wire.

Comparisons are against [OneTriangle](https://onetriangle.ai/blog/transfer-benchmarks?pair=qwen-27)'s published
Qwen3.5-4B → Qwen3.8-27B results (page and data release dated 2026-11-13). Every number below comes from a file in
`results/`. Where we lose, the tables say so.

## Two modes

| | **fast mode** (default) | **recall mode** (`+repN`) |
|---|---|---|
| what the 27B computes natively | the last 128 prompt tokens (512 for prompts ≤ 4K, tapering to 128 at 8K) | the same, plus N prompt tokens picked by the 4B's attention from the end of the prompt |
| TTFT speedup, 8K / 16K / 32K | **2.66× / 2.86× / 2.96×** | N=1024: 1.59× / 2.06× / 2.40×; N=2048: 1.37× / 1.81× / 2.17×; N=4096: 0.97× / 1.37× / 1.74× |
| use it for | everything below unless labelled | exact recall and agents |

Every quality number below names its mode. OneTriangle publishes one configuration per pair; we compare their
numbers against both of our modes and say which mode a win needs. Agent runs (τ2) are a third configuration: the
newest turn and the tool schemas are always computed natively, so their TTFT is neither of the two above and was not
measured.

## Scoreboard

"Ahead" means outside the noise of the two samples; "ahead within noise" means a lead that is not. n = 50–300 per
cell. Where several recall sizes or runs were tried, the cell shows the best one and says so.

| metric | ours, fast | ours, recall | OneTriangle | verdict |
|---|---:|---:|---:|---|
| TTFT speedup 8K / 16K / 32K | 2.66 / 2.86 / 2.96× | 1.37–2.40× | 2.6 / 2.7 / 2.8× | fast mode level at 8K, ahead at 16–32K |
| TTFT speedup 64K / 128K / 1M | 3.4–4.0 / 5.0 / 21.5× | — | 3.0 / 3.3 / 3.9×¹ | ahead (speed only, see *Beyond 32K*) |
| TPOT | = native at every batch size measured | = native | not published | — |
| serving goodput, TTFT ≤ 5 s | 2.2× native | not measured | not published | — |
| ContractNLI | **0.757** | | 0.61 | ahead (95% vs 77% of each side's native) |
| LongHealth | **0.845** | | 0.76 | ahead on score, but their mapped (0.76) beats their native (0.72): harness difference likely |
| MATH-500 | 0.89 | | 0.59 | not comparable (their native 0.57, ours 0.90) |
| BAMBOO PaperQA | 0.881 | | 0.89 | tie |
| ARC-Chat | 0.97 | | 0.97 | tie |
| HumanEval | 0.927 | | 0.91 | tie |
| MBPP | 0.875 | | 0.88 | tie |
| BigCodeBench | 0.56 | | 0.55 | tie |
| QMSum (ROUGE-L) | 0.219 | | 0.41 | behind (scoring differs, see note) |
| τ2 airline² | 0.26 | 0.60 | 0.59 | tie in one recall run (other recall runs: 0.32, 0.40) |
| τ2 retail² | 0.60 | 0.70 | 0.41 | ahead in recall; ahead within noise without repair |
| τ2 telecom² | not run | 0.18 | 0.47 | **behind** (also as a share of native: 30% vs 55%) |
| 2B → 27B RULER³ | 0.82 | **0.905** (+rep1024) | 0.80 | tie without repair, ahead in recall |
| 2B → 27B QuALITY | 0.79 | | 0.71 | ahead within noise |
| 2B → 27B MATH-500⁴ | 0.86 | | 0.94 | **behind** |
| 2B → 27B MRCR³ | 0.52 | 0.62 (+rep4096; +rep1024 0.50, +rep2048 0.40) | up to 0.66 | **behind**; +rep4096 is no faster than native at 8K |
| 2B → 27B LongBench v2³ | 0.35 | 0.42 (+rep1024; +rep2048 0.375) | 0.42 | tie in recall mode only |
| 2B → 27B ZebraLogic | 0.63 | 0.61 | 0.66 (reasoning on) | tie (they think, we don't) |
| CowText, DevEval, Terminal-Bench | not run | | published | not compared |

¹ OneTriangle's values above 64K are projections.
² τ2 arms compute the newest turn (up to 4096 tokens) and the `<tools>` block natively; "fast" here means no repair.
³ Fixed 128-token tail rather than the adaptive one: identical to fast mode for prompts ≥ 8K, a smaller native tail
than fast mode for 4–8K prompts.
⁴ Phase-2 map `2b-27b-chat`.

## Speed

### Time to first token (batch 1)

The comparable number is the **speedup over each side's own native baseline**; absolute times also reflect engine
and GPU differences (ours: FlashAttention-2 + fla kernels on one GPU; OneTriangle: HF forwards, 2×H100 at 32K).

| prompt | native 27B | ours, fast mode | **speedup** | OneTriangle speedup | GPU, source |
|---:|---:|---:|---:|---:|---|
| 8K | 0.90–0.93 s | 0.33–0.38 s | **2.4–2.8×** | 2.6× | H100, 5 runs⁴ |
| 16K | 1.89–1.94 s | 0.63–0.68 s | **2.8–3.0×** | 2.7× | H100, 5 runs⁴ |
| 32K | 4.14–4.25 s | 1.33–1.43 s | **2.96–3.1×** | 2.8× | H100, 5 runs⁴ |
| 64K | 9.6–10.0 s | 2.4–3.0 s | **3.4–4.0×** | 3.0× | H100 / H200, 3 runs⁵ |
| 128K | 25.5 s | 5.07 s | **5.0×** | 3.3×¹ | H200, median of 3 |
| 256K | 74.7 s | 10.2 s | **7.4×** | 3.6×¹ | H200, 1 run |
| 512K | 243.1 s | 20.3 s | **12.0×** | 3.7×¹ | H200, 1 run |
| **1M** | 873.2 s | 40.7 s | **21.5×** (21.1×²) | 3.9×¹ | H200, 1 run |

² An earlier 1M run: native 833.2 s, ours 39.5 s (`results/ttft_h200_128k-1m_first_run.json`); its 128K point was
not warmed up and is excluded.
⁴ `results/ttft_h100_final_engine_two_runs.json`, `results/ttft_h100_8k-32k_final_engine.json` and the rep0 arms of
`results/ttft_h100_recall_mode.json`; median of 5 per run. At 8K the runs gave 2.44×, 2.66×, 2.68×, 2.69× and
2.76×, so at 8K we are level with OneTriangle, not clearly ahead.
⁵ 64K does not fit next to the persistent graphed tail on an 80 GB H100 in the final build; the three measured
64K runs (`results/ttft_h100_8k-64k.json`, `results/ttft_h100_untuned_lengths_batches.json`,
`results/ttft_h200_64k-128k_reps3.json`) used earlier builds and gave 3.4×, 4.0× and 3.75×.
The 128K row is `results/ttft_h200_64k-128k_reps3.json`; the 256K–1M rows are `results/ttft_h200_128k-1m.json`
(whose own 128K point was not warmed up and is excluded). Both used an earlier engine build and the phase-2
long-context map for both arms, with the 4B on an 8K sliding window. All TTFT and TPOT runs use random token ids.

**Recall mode** (`results/ttft_h100_recall_mode.json`, H100, same native baseline):

| prompt | fast | +rep512 | +rep1024 | +rep2048 | +rep4096 |
|---:|---:|---:|---:|---:|---:|
| 8K | 2.66–2.68× | 1.72× | 1.59× | 1.37× | 0.97× |
| 16K | 2.86× | 2.19× | 2.06× | 1.81× | 1.37× |
| 32K | 2.96–2.98× | 2.55× | 2.40× | 2.17× | 1.74× |

Recall mode is **slower than OneTriangle's transfer** at every length we measured. The timed region includes
everything a request pays: the 4B pass over the native tail, the scoring pass, and the 27B forward over the chosen
tokens. One run (median of 5) per size.

**Short prompts.** In fast mode prompts under 4K get a 512-token native tail, and prompts shorter than that run
fully native (no speedup, native quality). OneTriangle does not publish TTFT below 8K.

**Shapes not used for tuning** (`results/ttft_h100_untuned_lengths_batches.json`, `results/ttft_h100_final_engine_two_runs.json`):
12K 2.6×, 24K 3.0×, 40K 3.5×; batch 2 at 16K 2.6×, at 32K 3.2×; batch 3 at 16K 3.2×. An earlier build gave
batch 2/3/4 at 8K 3.0×/3.1×/3.2× and 50K 3.7×. Batch 8 at 8K did not fit in 80 GB for the native arm.

### Decode (TPOT)

The transferred cache has the native cache's exact shape, so decoding costs the same as native at every batch size
measured (1, 2, 3, 4, 8, 12, 16, 32 at 2K; 1–12 at 32K; H200, CUDA graphs; `results/tpot_and_batch_sweeps.json`).
Recall mode writes into the same cache, so its decode cost is the same by construction; we did not measure it
separately. Optionally the 4B drafts tokens for the 27B to
verify (this needs the full 4B cache, `keep_src=True`, which adds the 4B's remaining layers and tail to the prefill):

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

### Serving under load (fast mode)

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
noisy. These runs used the phase-2 map; serving in recall mode was not measured.

## Quality

Native Qwen3.8-27B under our harness is the reference. Same prompts, greedy decoding, thinking off, for every arm.
Unless marked otherwise the transfer uses the fidelity maps (`4b-27b-fid` + `4b-27b-fid-attn`, or the 2B
equivalents) and, up to 32K, full attention in the source.

### OneTriangle's 4B → 27B suite (fast mode)

| benchmark | n | native 27B | ours (95% CI) | 128-token tail only | 4B alone | OneTriangle mapped (native) |
|---|---:|---:|---:|---:|---:|---:|
| ContractNLI | 300 | 0.80 | **0.757** (0.71–0.80) | | 0.73 | 0.61 (0.79) |
| LongHealth | 200 | 0.895 | **0.845** (0.79–0.89) | | 0.84 | 0.76 (0.72) |
| BAMBOO PaperQA | 84 | 0.869 | 0.881 (0.80–0.93) | | 0.893 | 0.89 (0.88) |
| QMSum (ROUGE-L) | 100 | 0.220 | 0.219 | | 0.203 | 0.41 (0.49)³ |
| MATH-500 | 200 | 0.900 | **0.89** | 0.885 | 0.88 | 0.59 (0.57) |
| ARC-Chat | 300 | 0.97 | 0.97 | | 0.91 | 0.97 (0.97) |
| HumanEval | 164 | 0.933 | 0.927 (0.88–0.96) | 0.927 | — | 0.91 (0.93) |
| MBPP | 257 | 0.875 | 0.875 (0.83–0.91) | 0.848 | 0.80 | 0.88 (0.87) |
| BigCodeBench | 100 | 0.56 | 0.56 | 0.52 | 0.40 | 0.55 (0.60) |
| CowText | — | — | not run (not public) | | — | 0.65 (0.64) |

`results/quality_4b_fid.json` (fidelity map); native and 4B-alone columns from `results/quality_4b_suite.json`.
The HumanEval, MBPP, BigCodeBench and MATH-500 fidelity-map rows come from job logs: the job hit its time limit
before writing its result file, and the per-benchmark lines were copied from the log.
³ OneTriangle does not state how QMSum is scored; as a share of native we keep 99%, they keep 84%.

- **Ahead:** ContractNLI (prompts 0.4–8K tokens, median 2K, so about three quarters of a median prompt is mapped)
  and LongHealth (longer documents). LongHealth's lead partly reflects harness differences: OneTriangle's mapped
  score is above their native one.
- **Level:** everything else that both sides ran. MATH-500's large lead reflects harness differences, not transfer
  quality: our native 27B scores 0.90 where theirs scores 0.57.
- **Short prompts are mostly native.** ARC, HumanEval, MBPP, BigCodeBench and MATH-500 prompts have a median of
  70–170 tokens, so in fast mode they run fully or almost fully native. The "128-token tail only" column shows the
  same map with the plain 128-token tail; with a 16-token tail (phase-2 map) ARC drops to 0.92, HumanEval to 0.81
  and MBPP to 0.67.
- **Losses vs native:** 4–5 points on LongHealth and ContractNLI, where the transfer is roughly level with the 4B alone.

### Agents (τ2-bench, text, English)

| domain | native 27B | ours, no repair | ours, recall +rep2048 | mostly native⁶ | OneTriangle mapped (native) |
|---|---:|---:|---:|---:|---:|
| airline | 0.74 | 0.26 | **0.60**⁷ | 0.35 | 0.59 (0.86) |
| retail | 0.78 | 0.60 | **0.70** | 0.74 | 0.41 (0.80) |
| telecom | 0.60 | not run | 0.18 | not run | 0.47 (0.86) |

`results/tau2_fid.json`. The no-repair and recall arms read the newest turn (up to 4096 tokens) and the `<tools>`
block natively; everything else (system policy, earlier turns) is mapped. n = 50 tasks per domain, so ±0.13 is a 95%
interval. The native airline/retail run comes from a job log (no result file was saved).
⁶ Only the system prompt and first user turn are mapped; every later turn is appended natively to the kept cache,
so most of each conversation is computed by the 27B. Shown for completeness; not a transfer result.
⁷ Two runs with 1024 repair tokens gave 0.40 and 0.32; the 2048 result is a single run. Airline is a tie at best.

Telecom is a clear loss. Our native 27B also scores far below OneTriangle's native there (0.60 vs 0.86): telecom
needs the simulated user to operate a phone through tools, and our user simulator is the 27B itself.

Our user simulator and judges are the native 27B, not the official OpenAI models, so these numbers are not strictly
comparable to OneTriangle's. Terminal-Bench 2.0 is implemented but was not run (its sandboxes need a Modal token
secret).

### OneTriangle's 2B → 27B tab

With a Qwen3.5-2B source, prompts up to 32K (`results/quality_2b_fid.json`; native and 2B-alone columns from
`results/quality_2b_suite.json`):

| benchmark | n | native 27B | ours, no repair | ours, recall | 2B alone | OneTriangle (native) |
|---|---:|---:|---:|---:|---:|---:|
| ARC-Chat⁸ | 100 | 0.98 | **0.98** | | 0.81 | 0.89 (0.99) |
| QuALITY | 100 | 0.91 | **0.79** | | 0.70 | 0.71 (0.94) |
| MATH-500⁸ | 100 | 0.87 | 0.86 | | 0.77 | 0.94 (0.96) |
| RULER 4–16K | 90 | 0.97 | 0.82 | **0.905** (+rep1024) | 0.92 | 0.80 (1.00) retrieval |
| MRCR 4–32K | 60 | 0.97 | 0.52 (adaptive tail 0.54) | 0.50 (+rep1024), 0.62 (+rep4096) | 0.87 | up to 0.66 |
| LongBench v2 (≤32K) | 200 | 0.37 | 0.35 (n=60) | 0.42 (+rep1024), 0.375 (+rep2048) | 0.23 (n=60) | 0.42 (0.61) |
| LoCoMo (F1) | 100 | 0.58 | 0.457 | | 0.39 | per-category only |
| ZebraLogic | 100 | 0.69 | 0.63 | 0.61 (+rep2048) | 0.33 | 0.66 (0.91), reasoning on |

⁸ Phase-2 map `2b-27b-chat`; not re-run with the fidelity map.

RULER, MRCR, LongBench v2 and LoCoMo "no repair" cells use a fixed 128-token tail (see ³ above). On MRCR the native, 2B-alone, adaptive-tail (0.54), +rep2048 and +rep4096 arms were re-run on the identical 60-item sample as the no-repair arm (native 0.970, 2B alone 0.872, matching the values shown); the +rep1024 value comes from a different 60-item sample.

The 2B pair is **behind on MRCR** in both modes, and the +rep4096 setting that gets closest runs no faster than
native at 8K. LongBench v2 is a tie only with the better of the two repair sizes we ran on the 200-item sample (0.42 vs 0.375; earlier 60-item runs with +rep1024 gave 0.38–0.42), and it is above our native 27B (0.37), which says more about the noise of this benchmark than about the transfer. ZebraLogic is close, but OneTriangle ran it with
reasoning on and we run everything non-thinking (our native 27B scores 0.69 non-thinking vs their 0.91 thinking).
DevEval was not run (needs the repository test environments).

### Held-out fidelity gate

The fidelity maps, adaptive tail and repair were developed while looking at the benchmarks above (see *How the
evaluation was run*). To check that this did not just fit those benchmarks, we compared the phase-2 map and the
fidelity map on data neither was selected on (`results/heldout_fidelity_gate.json`, 4B → 27B):

| | native 27B | phase-2 map | fidelity map | 4B alone |
|---|---:|---:|---:|---:|
| FineWeb-Edu held-out, next-token agreement (4K prefix, 1-token tail) | 1 | 0.896 | 0.900 | 0.736 |
| same, KL to native (nats/token) | 0 | 0.059 | 0.057 | 0.367 |
| same, 128-token tail: agreement / KL | 1 / 0 | 0.925 / 0.031 | 0.925 / 0.030 | |
| HellaSwag (length-normalized acc) | 0.837 | 0.787 | 0.78 | 0.73 |
| PIQA (acc) | 0.803 | 0.803 | 0.797 | 0.797 |
| ARC-Challenge (acc) | 0.543 | 0.507 | 0.527 | 0.517 |
| MMLU (acc) | 0.853 | 0.687 | 0.673 | 0.74 |
| Winogrande (acc) | 0.76 | 0.747 | 0.75 | 0.743 |

Multiple choice: 300 items each, log-likelihood scoring, 1-token native tail (the question is almost fully mapped).

- The fidelity map is **within noise (±0.02) of the phase-2 map** on every row: slightly better on FineWeb, ARC and
  Winogrande, slightly worse on HellaSwag, PIQA and MMLU. The gains on the targeted benchmarks did not cost general
  fidelity.
- FineWeb-Edu is also part of the distillation data; this is its held-out split, i.e. unseen text from the training
  distribution. The multiple-choice sets were never used for training or selection.
- **MMLU with an almost fully mapped prompt is worse than the 4B alone** (0.67–0.69 vs 0.74). In fast mode such
  short prompts run natively, so this does not show up in the tables above, but it is a real weakness of the maps
  on short, knowledge-heavy contexts.

### Beyond 32K

Above 32K the 4B switches to an 8K sliding window, which is what makes 64K–1M fast. On RULER and MRCR at 40–64K
(4B → 27B, phase-2 map, `results/quality_64k_ruler_mrcr.json`):

| | native 27B | ours |
|---|---:|---:|
| RULER (n=27) | 0.92 | 0.55 |
| MRCR, 2 needles (n=30) | 1.00 | 0.15 |

**Exact recall degrades sharply in the long-context configuration.** The speedups above 32K should be read as speed
results only; we have not validated quality there.

### Where recall is lost

Swapping the native 27B's attention KV into a transferred cache restores exact recall (4B → 27B, phase-2 map
`4b-27b-chat`, prompts up to 64K: RULER 0.72 → 1.00, n=27; MRCR 0.30 → 0.97, n=30); swapping in the native DeltaNet
states does not (0.69, 0.26). On 2B ZebraLogic (40 puzzles) the attention swap gives 0.125 → 0.65 vs 0.225 for the
DeltaNet swap, but a plain 512-token native tail also reaches 0.65 there, so for short puzzles a longer tail is an
equally good fix (it is what the adaptive tail does). `results/oracle_diagnosis.json`. This is why the fidelity maps add a dedicated attention-KV
map and why recall mode re-encodes attention KV only.

## How it works

Qwen3.5 and Qwen3.8 are hybrid models. Three of every four layers are Gated DeltaNet (a linear-attention RNN with a
128×128 state per head); the fourth is gated full attention. Most of their "cache" is recurrent state, so a KV-only
transfer cannot work. We synthesize every piece the 27B needs:

1. **Source prefill.** The 4B runs over the prompt. We read only its normalized residual stream.
2. **Per-layer maps.** For each 27B layer, a linear map from a depth-matched 4B layer predicts what that layer caches:
   - *attention layers:* per-token `k` (post-norm, pre-RoPE) and `v`, predicted from several 4B layers plus the
     token embedding (`src/xkv/attnmap.py`); the 27B's own RoPE is applied afterwards.
   - *DeltaNet layers:* the raw `k`, `v` projections and the gate inputs `a`, `b`, which then run through **the
     27B's own** causal conv and gated delta-rule scan to give the recurrent state the 27B expects.
3. **Native tail.** The 27B natively runs the last prompt tokens on the synthesized cache and emits the first token:
   512 tokens for prompts ≤ 4K, tapering linearly to 128 at ≥ 8K. Optional *native spans* let the 27B compute
   selected regions in place (used for tool schemas and the newest agent turn).
4. **Recall mode (`+repN`).** The 4B reads the tail over its own cache; each 64-token block of the mapped prefix is
   scored by the 4B's attention from the tail (max over tail queries, summed over heads and over the 4B attention
   layers up to the deepest mapped one, first 4 positions excluded). The 27B re-encodes the top-N tokens natively in
   one batched pass (absolute-position causal attention over the full cache; each contiguous span's DeltaNet scan
   starts from the mapped end-of-prefix state, via variable-length kernels). Only the attention KV it writes is kept.
5. **Fit, then distill.** Maps start as closed-form ridge regressions on 1M calibration tokens, then are fine-tuned
   end to end on KL(native 27B ‖ transferred 27B) with gradients through the frozen 27B. Distillation data: FineWeb-Edu
   text, synthetic record lookups (close in style to needle-in-a-haystack tests), synthetic tool-use dialogues, and
   public instruction data (LongAlign-10k, Tulu-3 SFT). These were not decontaminated against the evaluation sets.

**Engine.** Fused Triton causal-conv kernel, state-only delta-rule scan, a CUDA-graphed persistent tail (prompts
≤32K), CUDA-graph batched decoding with per-sequence lengths, and source-drafted speculative decoding with exact
recurrent-state rollback (rejected tokens re-scanned with β = 0, g = 0). fla's norm kernels re-autotune for every new
sequence length; we freeze those keys, which made generation evals about 4× faster.

## How the evaluation was run, and what to be careful about

- **Configuration was chosen on evaluation items.** Full 4B attention up to 32K, the chat-distilled map and the
  free-form code prompt were adopted after trying them on LongHealth, ContractNLI and HumanEval items that are also
  in the final samples. The fidelity maps were selected on RULER, MRCR and ZebraLogic subsets; the adaptive tail on
  ZebraLogic and MBPP; recall mode and its size on 2B RULER/MRCR/LongBench v2 and τ2 airline/retail. Treat all of
  those numbers as development-set results. The held-out gate above is the guard against general regression.
- **Mixed configurations.** The scoreboard takes each benchmark from the mode named in its column. No single
  configuration wins everywhere: recall mode buys τ2 airline and RULER at the cost of TTFT.
- **Harness.** Every benchmark was reimplemented (`src/xkv/genevals.py`); OneTriangle's prompts and scorers are not
  public. Code benchmarks use a free-form "answer in a code block" prompt (a forced code prefix cost the native 27B
  6 points on HumanEval); MBPP shows its test asserts in the prompt, as is standard; MATH-500 allows 3072 new tokens.
- **Greedy continuations diverge.** Next-token agreement with native is 90–93% teacher-forced, but free-running greedy
  outputs drift apart early. Task scores are the meaningful measure.
- **Parity.** The engine's parity check against Hugging Face (99.6% top-1, `results/selftest_parity.json`) was run
  on the 0.8B → 2B pair, not on the 27B.
- **Timing.** The transfer reuses a preallocated graphed tail; the native path allocates its cache inside the timed
  region (well under 20 ms). Both arms run on the same engine and GPU.
- bf16 throughout, both models; no quantization on either side. Sample sizes are 50–300; differences of 2–3 points
  (13 on τ2) are within noise.

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
results/                outputs of the runs quoted above
```

## Reproduce

From `modal_jobs/` with `PYTHONPATH=../src`:

```bash
modal run lab.py --task ttft_repair --gpu H100 --args '{"repairs":[0,1024,2048]}'
modal run lab.py --task decode_bench --gpu H200 --args '{"lengths":[2048,32768],"batches":[1,2,3,4,8,12,16,32]}'
modal run lab.py --task serve --gpu H200 --args '{}'
modal run lab.py --task gen_evals --gpu H200 --args '{"mapper_name":"4b-27b-fid+4b-27b-fid-attn","full_until":32768,"benches":["contract_nli","longhealth"],"strategies":["target","transfer@auto"]}'
modal run lab.py --task gen_evals --gpu H200 --args '{"src":"2b","mapper_name":"2b-27b-fid+2b-27b-fid-attn","full_until":32768,"max_ctx":32768,"benches":["ruler","mrcr"],"strategies":["transfer@auto","transfer@auto+rep1024"]}'
modal run agent_lab.py --task tau2 --gpu H200 --args '{"mapper_name":"4b-27b-fid+4b-27b-fid-attn","full_until":32768,"strategies":["transfer+rep2048"],"domains":["airline"],"turn_tail_max":4096,"native_tools":true}'
modal run lab.py --task quality --gpu H100 --args '{"mapper_name":"4b-27b-fid+4b-27b-fid-attn","n_seqs":32,"prefix":4096,"tails":[1,128],"mc_limit":300,"mc_tails":[1]}'
```

Code benchmarks use `"env":{"XKV_CODE_PROMPT":"free"}`; BigCodeBench executes in the official evaluation image
(`modal deploy exec_app.py`, then `"exec_modal":true`).
