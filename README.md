# ultra-cross-model-kv

**Prefill a 27B model with a 4B model: 2.4–3.1× faster time to first token at 8K–32K (level with OneTriangle at 8K,
ahead at 16–32K), about 2× more requests served
within a 5 s TTFT target, and no decode slowdown. Beyond 32K it is 5× faster at 128K and 21.5× at 1M, but exact
recall is not preserved there.**

`ultra-cross-model-kv` turns a Qwen3.5-4B prefill into a ready-to-decode cache for Qwen3.8-27B. It covers the
full-attention KV *and* the Gated-DeltaNet recurrent states of these hybrid models, so the 27B answers without
prefilling the prompt itself. One GPU, no cache transfer over the wire.

Comparisons are against [OneTriangle](https://onetriangle.ai/blog/transfer-benchmarks?pair=qwen-27)'s published
Qwen3.5-4B → Qwen3.8-27B results (page and data release dated 2026-11-13). Every number below comes from a file in
`results/`. Where we lose, the tables say so.

**In one paragraph:** on OneTriangle's non-agent benchmarks we are level or ahead, both on raw score and on how much
of the native 27B's score the transfer keeps, and faster from 16K up. Agents are the exception. On τ2 we only match
native in the agent setting, which computes most of the prompt natively and is slower than native. In the lighter
agent setting (23% of the telecom prompt native, 58% of retail; faster than native only on telecom with the graphed
tail off) we win retail, lose telecom and are behind within noise on airline. Both settings are defined in the
glossary. On
Terminal-Bench our transfer is far below our own native. See *Known limitations* and *Notes on OneTriangle's agent
numbers* at the end.

## How to read the numbers

- **Transfer vs native.** Every quality number is a transfer score unless it says native (see the glossary).
- **Share of native** = transfer score ÷ the same side's native score under its own harness. It removes harness
  differences (our native 27B scores lower than theirs on some benchmarks and higher on others) but flatters a side
  whose native is weak; the raw scores stay in the tables.
- **Modes** trade speed for exactness, and every number names its mode: *fast mode* (default), *recall mode*
  (`+repN`) and, for agents, the *lighter agent setting* and the slower *agent setting*.
- **Noise:** most of our cells are 50–300 items (±0.06–0.13 at 95%); OneTriangle's τ2 cells are 17 tasks (±0.23).
  "Ahead within noise" means the lead is smaller than that.
- **Notes** are numbered [1], [2], … and listed under the table that first uses them.

### Glossary

| term | meaning |
|---|---|
| **transfer / native** | *Transfer*: the 4B reads the prompt and the 27B answers from the mapped cache. *Native*: the 27B reads the prompt itself. |
| **mapped** | A prompt token whose 27B cache entries are predicted from the 4B by the maps, not computed by the 27B. |
| **fast mode, adaptive tail** | The default. The 27B computes only a short *native tail* at the end of the prompt: 512 tokens for prompts ≤ 4K, tapering to 128 at 8K and above (the *adaptive tail*). Some rows use a fixed 128-token tail instead and say so. |
| **recall mode, repair** | *Recall mode* adds *repair*: the 4B's attention picks the prompt parts that matter for the question and the 27B re-encodes them natively. `+repN` repairs N tokens in scattered 64-token blocks; `+repNm` (*whole-message recall*) repairs whole chat messages. |
| **native span** | A region the 27B computes natively in place, e.g. the `<tools>` block of an agent prompt. |
| **graphed tail** | The native tail run as a CUDA graph. A graph is captured once per prompt length, so it pays off when lengths repeat and costs time when every request has a new length (*Speed*). |
| **Gated DeltaNet state** | The recurrent (linear-attention) state that three of every four layers of these hybrid models keep instead of a KV cache. |
| **fidelity map / phase-2 map / agent map** | The learned 4B → 27B maps. *Fidelity map* (`4b-27b-fid`, `2b-27b-fid`): the current maps, used unless stated. *Phase-2 map* (`4b-27b-chat`, `2b-27b-chat`): earlier chat-distilled maps. *Agent map* (`4b-27b-fid-agent`): the fidelity map fine-tuned with τ2 airline traces mixed into its training data. |
| **lighter agent setting / agent setting** | Two τ2 / Terminal-Bench configurations, both with `<tools>` as a native span. *Lighter agent setting*: last 3 messages native, no repair, agent map; faster than native only on telecom with the graphed tail off. *Agent setting*: last 10 messages native plus whole-message recall, fidelity map; slower than native. Details in *Agents*. |
| **policy-deleted control** (`target-drop`) | The native 27B sees only what the agent setting computes natively; the policy and older history are deleted instead of transferred. Shows what a run scores without the policy. |

## Two modes

| | **fast mode** (default) | **recall mode** (`+repN`) |
|---|---|---|
| what the 27B computes natively | the last 128 prompt tokens (512 for prompts ≤ 4K, tapering to 128 at 8K) | the same, plus N prompt tokens picked by the source's attention from the end of the prompt: scattered 64-token blocks (`+repN`) or whole messages (`+repNm`) |
| TTFT speedup, 8K / 16K / 32K (4B → 27B) | **2.4–2.8× / 2.8–3.0× / 2.95–3.1×** | 0.97–2.55×, depending on N (*Speed*) |
| TTFT speedup, 8K / 16K / 32K (2B → 27B) | 3.94× / 4.47× / 4.94× | `+rep1024`: 2.03× / 2.85× / 3.57×; 1.36–3.14× for `+rep2048m` and `+rep3072m` (*Speed*) |
| use it for | everything below unless labelled | exact recall |

Every quality number below names its mode. OneTriangle publishes one configuration per pair; we compare their
numbers against both of our modes and say which mode a win needs. Agents (τ2, Terminal-Bench) use two further
settings, the lighter agent setting and the agent setting; their native share, map and TTFT are in one table in
*Agents*.

## Scoreboard

"Ahead" means outside the noise of the two samples; "ahead within noise" means a lead that is not. n = 50–300 per
cell for most rows (Terminal-Bench: 10 tasks; OneTriangle's τ2 mapped arms: 17 per domain). Where several recall
sizes or runs were tried, the cell shows the best one and says so.

**Speed**

| metric | ours, fast | ours, recall | OneTriangle | verdict |
|---|---:|---:|---:|---|
| TTFT speedup 8K / 16K / 32K | 2.4–2.8 / 2.8–3.0 / 2.95–3.1× | 0.97–2.55× (4B pair, +rep512 to +rep4096) | 2.6 / 2.7 / 2.8× | fast mode level at 8K, ahead at 16–32K |
| TTFT speedup 64K / 128K / 1M | 3.4–4.0 / 5.0 / 21.5× | — | 3.0 / 3.3 / 3.9× [1] | ahead (speed only, see *Beyond 32K*) |
| TPOT | = native at every batch size measured | = native | not published | — |
| serving goodput, TTFT ≤ 5 s | 2.2× native | not measured | not published | — |

**Quality.** "Share of native" is each side's transfer score divided by its own native 27B score under its own
harness (see the charts and caveats below). For τ2 the share cell reads **lighter · agent setting / OneTriangle**;
otherwise **ours / OneTriangle**.

| metric | ours, fast (agents: lighter agent setting, faster than native only on telecom with the graphed tail off) | ours, recall (agents: agent setting, slower than native) | OneTriangle | share of native, ours / theirs | verdict |
|---|---:|---:|---:|---:|---|
| ContractNLI | **0.757** | | 0.61 | 95% / 77% | ahead |
| LongHealth | **0.845** | | 0.76 | 94% / 106% | ahead on score [2] |
| MATH-500 | 0.89 | | 0.59 | 99% / 104% | not comparable [3] |
| BAMBOO PaperQA | 0.881 | | 0.89 | 101% / 101% | tie |
| ARC-Chat | 0.97 | | 0.97 | 100% / 100% | tie |
| HumanEval | 0.927 | | 0.91 | 99% / 98% | tie |
| MBPP | 0.875 | | 0.88 | 100% / 101% | tie |
| BigCodeBench | 0.56 | | 0.55 | 100% / 92% | tie |
| QMSum (LLM judge) [4] | 0.365 | | 0.41 | 84% / 84% | level as a share of native; raw scores not comparable |
| τ2 airline [5] | 0.50 (in-sample) | 0.68 | 0.59 | 71% · 97% / 69% | lighter: level as a share, behind within noise on score; agent setting: ahead within noise |
| τ2 retail [5] | **0.78** | **0.84** | 0.41 | 93% · 100% / 51% | ahead in both settings |
| τ2 telecom [5] | 0.22 | 0.68 | 0.47 | 32% · 100% / 55% | **lighter: behind**; agent setting: ahead within noise |
| 2B → 27B RULER [6] | 0.82 | **0.905** (+rep1024) | 0.80 | 93% / 80% | tie without repair; ahead in recall mode, borderline [7] |
| 2B → 27B QuALITY | **0.79** | | 0.71 | 87% / 76% | ahead within noise |
| 2B → 27B MATH-500, thinking on [8] | 0.93 | | 0.94 | 100% / 98% | tie |
| 2B → 27B MRCR [9] | 0.54 (held-out 0.55) | **0.681** (+rep3072m, held-out n=120; 0.706 on the dev sample) | up to 0.66 | 70% / — | ahead within noise (recall mode) |
| 2B → 27B LongBench v2 [6] | 0.35 | 0.42 (+rep1024; +rep2048 0.375) | 0.42 | 114% / 69% | tie in recall mode only |
| 2B → 27B ZebraLogic, thinking on [8] | **0.75** | | 0.66 | 79% / 73% | ahead within noise |
| Terminal-Bench 2.0 [10] | | agent setting: 0.10 (native 0.70) | 0.10 | 14% / — | **behind our native** |
| CowText, DevEval | not run | | published | — | not compared |

- **[1]** OneTriangle's values above 64K are projections.
- **[2]** Their mapped score (0.76) beats their native (0.72): a harness difference is likely.
- **[3]** Their native scores 0.57, ours 0.90.
- **[4]** OneTriangle scores QMSum with "the campaign's canonical judge" (an LLM). Ours is the native 27B grading
  each answer 0–10 against the reference (`QMSUM_JUDGE` in `src/xkv/genevals.py`), for both its own native answers
  and the transfer's. Their judge model is not named, so the raw numbers are not on the same scale. As a share of
  native both sides keep 84%. With ROUGE-L instead, ours is 0.219 vs native 0.220: ROUGE-L, which barely moves,
  hides that loss.
- **[5]** τ2, read with the settings table in *Agents*:
  - *Sample sizes.* Official τ2 user simulator and judges (OpenAI, as OneTriangle used); 50 tasks per domain for
    us, 17 for OneTriangle's mapped arms.
  - *How each setting was chosen.* The lighter agent setting is the one pre-registered official run of the agent
    map. That map was trained with airline traces, so airline is in-sample; telecom and retail were never trained
    on. The agent setting was developed by replaying failures from the same 50 telecom tasks, so its telecom 0.68 is
    a development-set result. The agent setting is slower than native.
  - *Share column.* Two numbers on our side: lighter agent setting · agent setting, then OneTriangle.
  - The policy-deleted control for these domains is in *Agents*.
- **[6]** Fixed 128-token tail rather than the adaptive one: identical to fast mode for prompts ≥ 8K, a smaller
  native tail than fast mode for 4–8K prompts.
- **[7]** z ≈ 2; the repair size was tuned on RULER.
- **[8]** Thinking enabled for both arms (the model reasons before answering), as OneTriangle's numbers suggest they
  did: their native scores 0.96 / 0.91 here, ours 0.93 / 0.95. Prompts are short (median under 2K tokens), so in
  fast mode most of each prompt runs natively. Non-thinking, ours scored 0.86 on MATH-500 (phase-2 map) and 0.63 on
  ZebraLogic.
- **[9]** OneTriangle's 0.66 is the peak of their MRCR-vs-length curve (around 16K); they do not publish a
  sample-wide average or a native value at that point. Our recall-mode result and its TTFT cost: *OneTriangle's 2B →
  27B tab* and *Speed*.
- **[10]** Same 10 tasks for native 27B and the agent setting, at twice Harbor's default agent time limit. Level with
  OneTriangle's raw score, which has no native value. Details, default-limit results and why the transfer is slow
  here: *Agents → Terminal-Bench 2.0*.

**Share of native, per benchmark.** Each bar is that side's transfer score as a percentage of its own native 27B
score (100% = nothing lost), grouped by mode. OneTriangle publishes one agent number per domain, drawn against both
of our agent settings.

![share of native](docs/share_simple.png)

**The same comparison on raw scores.** Each line runs from a side's native score (hollow) to its transfer score
(filled); a short line means little was lost, and the position of the hollow ring shows how strong each side's
native is.

![transfer vs native](docs/transfer_vs_native.png)

Caveats for these views: a weak native flatters the share (our 2B LongBench v2 transfer beats our own native, 114%,
which is noise); OneTriangle's shares above 100% (LongHealth, MATH-500) are harness effects; MRCR is missing because
OneTriangle publishes no native value; OneTriangle's τ2 shares divide a 17-task mapped score by a 50-task native
score.

## Speed

### Time to first token (batch 1)

The comparable number is the **speedup over each side's own native baseline**; absolute times also reflect engine
and GPU differences (ours: FlashAttention-2 + fla kernels on one GPU; OneTriangle: HF forwards, 2×H100 at 32K).

| prompt | native 27B | ours, fast mode | **speedup** | OneTriangle speedup | GPU, source |
|---:|---:|---:|---:|---:|---|
| 8K | 0.90–0.93 s | 0.33–0.38 s | **2.4–2.8×** | 2.6× | H100, 6 runs [11] |
| 16K | 1.89–1.94 s | 0.63–0.68 s | **2.8–3.0×** | 2.7× | H100, 6 runs [11] |
| 32K | 4.14–4.25 s | 1.33–1.43 s | **2.95–3.1×** | 2.8× | H100, 6 runs [11] |
| 64K | 9.6–10.0 s | 2.4–3.0 s | **3.4–4.0×** | 3.0× | H100 / H200, 3 runs [12] |
| 128K | 25.5 s | 5.07 s | **5.0×** | 3.3× [1] | H200, median of 3 |
| 256K | 74.7 s | 10.2 s | **7.4×** | 3.6× [1] | H200, 1 run |
| 512K | 243.1 s | 20.3 s | **12.0×** | 3.7× [1] | H200, 1 run |
| **1M** | 873.2 s | 40.7 s | **21.5×** (21.1× [13]) | 3.9× [1] | H200, 1 run |

- **[11]** `results/ttft_h100_final_engine_two_runs.json`, `results/ttft_h100_8k-32k_final_engine.json` and the
  rep0 arms of `results/ttft_h100_recall_mode.json`; median of 5 per run. At 8K the six runs gave 2.44×, 2.62×,
  2.66×, 2.68×, 2.69× and 2.76×, so at 8K we are level with OneTriangle, not clearly ahead.
- **[12]** 64K does not fit next to the persistent graphed tail on an 80 GB H100 in the final build; the three
  measured 64K runs (`results/ttft_h100_8k-64k.json`, `results/ttft_h100_untuned_lengths_batches.json`,
  `results/ttft_h200_64k-128k_reps3.json`) used earlier builds and gave 3.4×, 4.0× and 3.75×.
- **[13]** An earlier 1M run: native 833.2 s, ours 39.5 s (`results/ttft_h200_128k-1m_first_run.json`); its 128K
  point was not warmed up and is excluded.

The 128K row is `results/ttft_h200_64k-128k_reps3.json`; the 256K–1M rows are `results/ttft_h200_128k-1m.json`
(whose own 128K point was not warmed up and is excluded). Both used an earlier engine build and the phase-2
long-context map for both arms, with the 4B on an 8K sliding window. All TTFT and TPOT runs use random token ids.

**Recall mode, 4B → 27B** (`results/ttft_h100_recall_mode.json`, H100, same native baseline):

| prompt | fast | +rep512 | +rep1024 | +rep2048 | +rep4096 |
|---:|---:|---:|---:|---:|---:|
| 8K | 2.62–2.68× | 1.72× | 1.59× | 1.37× | 0.97× |
| 16K | 2.83–2.86× | 2.19× | 2.06× | 1.81× | 1.37× |
| 32K | 2.95–2.98× | 2.55× | 2.40× | 2.17× | 1.74× |

**Recall mode, 2B → 27B** (`ttft_repair-H100-20261118-130157` in the same file, `+rep1024` blocks from `ttft_repair-H100-20261119-183932`, which used plain random ids since block repair ignores message boundaries; random ids
with a message-start token every 400 tokens, so selection sees chat-like structure):

| prompt | fast | +rep1024 | +rep2048 | +rep2048m | +rep3072m |
|---:|---:|---:|---:|---:|---:|
| 8K | 3.94× | 2.03× | 1.64× | 1.72× | 1.36× |
| 16K | 4.47× | 2.85× | 2.36× | 2.43× | 2.10× |
| 32K | 4.94× | 3.57× | 3.07× | 3.14× | 2.78× |

Recall mode on the 4B pair is **slower than OneTriangle's transfer** at every length we measured. The timed region
includes everything a request pays: the 4B pass over the native tail, the scoring pass, and the 27B forward over the
chosen tokens. One run (median of 5) per size.

**Short prompts.** In fast mode prompts under 4K get a 512-token native tail, and prompts shorter than that run
fully native (no speedup, native quality). OneTriangle does not publish TTFT below 8K.

**Shapes not used for tuning** (`results/ttft_h100_untuned_lengths_batches.json`,
`results/ttft_h100_final_engine_two_runs.json`): 12K 2.6×, 24K 3.0×, 40K 3.5×; batch 2 at 16K 2.6×, at 32K 3.2×;
batch 3 at 16K 3.2×. An earlier build gave batch 2/3/4 at 8K 3.0×/3.1×/3.2× and 50K 3.7×. Batch 8 at 8K did not fit
in 80 GB for the native arm.

### Graphed tail and varying prompt lengths

By default the native tail runs as a CUDA graph, and a graph is captured for one prompt length. The TTFT tables
above reuse one prompt length per row, so the graph is captured once. In serving, lengths vary and the graph is
re-captured for every request. Measured on τ2 telecom prompts (7.8K tokens, batch 1, `results/tau2_replay.json`;
graph off with `XKV_NO_GRAPH_TAIL=1`):

| | TTFT |
|---|---:|
| native 27B (two runs) | 1.41–1.46 s |
| fast mode, graphed tail on | 1.49 s |
| fast mode, graphed tail off | 0.93 s |

With the graph off the speedup is about 1.5× here, below the 2.4× of the fixed-length benchmark. The fixed-length
benchmark measures the prefill alone; the replay timing also includes tokenization and decoder setup on both sides.
The agent settings' timings, on and off, are in the settings table in *Agents*.

### Decode (TPOT)

The transferred cache has the native cache's exact shape, so decoding costs the same as native at every batch size
measured (1, 2, 3, 4, 8, 12, 16, 32 at 2K; 1–12 at 32K; H200, CUDA graphs; `results/tpot_and_batch_sweeps.json`).
Recall mode writes into the same cache, so its decode cost is the same by construction; we did not measure it
separately. Optionally the 4B drafts tokens for the 27B to verify (this needs the full 4B cache, `keep_src=True`,
which adds the 4B's remaining layers and tail to the prefill):

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

Native Qwen3.8-27B under our harness is the reference. Same prompts, greedy decoding, thinking off unless a row says
otherwise. Unless marked otherwise the transfer uses the fidelity maps (`4b-27b-fid` + `4b-27b-fid-attn`, or the 2B
equivalents) and, up to 32K, full attention in the source.

### OneTriangle's 4B → 27B suite (fast mode)

| benchmark | n | native 27B | ours (95% CI) | 128-token tail only | 4B alone | OneTriangle mapped (native) |
|---|---:|---:|---:|---:|---:|---:|
| ContractNLI | 300 | 0.80 | **0.757** (0.71–0.80) | | 0.73 | 0.61 (0.79) |
| LongHealth | 200 | 0.895 | **0.845** (0.79–0.89) | | 0.84 | 0.76 (0.72) |
| BAMBOO PaperQA | 84 | 0.869 | 0.881 (0.80–0.93) | | 0.893 | 0.89 (0.88) |
| QMSum (LLM judge) [4] | 100 | 0.432 | 0.365 | | — | 0.41 (0.49) |
| QMSum (ROUGE-L) | 100 | 0.220 | 0.219 | | 0.203 | — |
| MATH-500 | 200 | 0.900 | **0.89** | 0.885 | 0.88 | 0.59 (0.57) |
| ARC-Chat | 300 | 0.97 | 0.97 | | 0.91 | 0.97 (0.97) |
| HumanEval | 164 | 0.933 | 0.927 (0.88–0.96) | 0.927 | — | 0.91 (0.93) |
| MBPP | 257 | 0.875 | 0.875 (0.83–0.91) | 0.848 | 0.80 | 0.88 (0.87) |
| BigCodeBench | 100 | 0.56 | 0.56 | 0.52 | 0.40 | 0.55 (0.60) |
| CowText | — | — | not run (not public) | | — | 0.65 (0.64) |

`results/quality_4b_fid.json` (fidelity map); native and 4B-alone columns from `results/quality_4b_suite.json`.
The HumanEval, MBPP, BigCodeBench and MATH-500 fidelity-map rows come from job logs: the job hit its time limit
before writing its result file, and the per-benchmark lines were copied from the log. QMSum judge: see note [4].

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

**The two agent settings.** Both compute the `<tools>` block natively (a native span).

| | **lighter agent setting** | **agent setting** |
|---|---|---|
| natively computed | tools + last 3 messages (≤ 4096 tokens), no repair | tools + last 10 messages (≤ 8192 tokens) + up to 4096 tokens of whole-message recall (`+rep4096m`) |
| share of prompt tokens native (measured, τ2 requests) | telecom 23%, retail 58% | telecom 27–74%, retail 66–98% (range: repair re-encodes up to its budget) |
| map | agent map `4b-27b-fid-agent` (fidelity map fine-tuned with τ2 airline traces mixed into its training data) | fidelity map `4b-27b-fid` (the default) |
| TTFT telecom (7.8K tokens), graphed tail off | 1.31 s vs native 1.41–1.46 s; 1.09 s vs 1.28–1.31 s in a second run | 1.67 s vs 1.28–1.31 s — slower |
| TTFT retail (5.3K tokens), graphed tail off | 1.13 s vs 1.00–1.01 s — slower (20-prompt sample, 70% native) | 1.44 s vs 1.00–1.01 s — slower |
| TTFT telecom, graphed tail on | 1.80 s vs 1.41–1.46 s — slower | 2.36 s vs 1.41 s — slower |
| TTFT retail, graphed tail on | 1.82 s vs 1.18 s — slower | 1.99 s vs 1.17 s — slower |

So the lighter agent setting is faster than native only on telecom with the graphed tail off (about 1.1–1.2×);
on retail, where 58% of the prompt is computed natively anyway, it is slower with the tail on or off. The agent
setting is slower than native in every measurement. Native timings vary by about 10% between runs (separate
containers), so each comparison is against the native arm of the same run. Timings from `results/tau2_replay.json`; why the graphed tail
matters: *Speed → Graphed tail*.

**With the official user simulator and judges** (OpenAI `gpt-5.5` as the simulated customer, as in the τ2 release;
OneTriangle also used the official setup). 50 tasks per domain (±0.13); OneTriangle's mapped arms 17 (±0.23).

| domain | native 27B | lighter agent setting | agent setting (slower than native) | policy-deleted control | earlier: fidelity map, `+rep2048m`, last 3 messages native | OneTriangle mapped (native) |
|---|---:|---:|---:|---:|---:|---:|
| airline | 0.70 | 0.50 (in-sample) | 0.68 | 0.42 | 0.48 | 0.59 (0.86) |
| retail | 0.84 | **0.78** | **0.84** | 0.72 | 0.78 | 0.41 (0.80) |
| telecom | 0.68 | 0.22 | 0.68 | 0.44 | 0.10 → 0.20 (repair fix, item 1 below) | 0.47 (0.86) |

`results/tau2_fid.json`. The **policy-deleted control** column (`target-drop`): the native
27B sees only what the agent setting computes natively (tool schemas and the last 10 messages); the policy and older
history are removed instead of transferred.

- **The transfer carries real information.** In the agent setting, transferring the policy instead of deleting it
  is worth +0.26 on airline, +0.12 on retail and +0.24 on telecom. But that setting computes much of each prompt
  natively and is slower than native, so it shows quality preserved through transfer, not a speedup.
- **In the lighter agent setting, retail holds and telecom fails.** Retail keeps 93% of native (0.78 vs OneTriangle
  0.41). On telecom the transferred policy does *worse* than no policy (0.22 vs 0.44): it pushes the agent to call
  tools that only the customer may run, and τ2 ends a conversation after repeated tool errors.
- **Why telecom is hard.** The telecom policy (~23K characters) is a long troubleshooting procedure plus rules about
  which tools belong to the agent and which to the customer. Replaying logged requests through diagnostic variants
  (next tool matching native, 160 requests): the 4B alone matches 0.81, the transferred cache without repair 0.66,
  and the transferred cache with the target's exact attention KV swapped in 0.84 (exact DeltaNet states instead:
  0.69). The 4B reads the policy fine; the attention mapping loses the exact rules.

**What we tried, in order** (all in `results/tau2_fid.json` and `results/tau2_replay.json`):

1. *Repair overwrote exact cache.* Query-aware repair could re-encode blocks inside the natively computed `<tools>`
   block. Repair now never selects natively computed spans (`XKV_REPAIR_PROTECT`, on by default). Official telecom
   0.10 → 0.20; RULER (2B, `+rep1024`) unchanged at 0.905.
2. *More native context.* Keeping the last 10 messages native plus 4096 tokens of whole-message recall cut invalid
   customer-tool calls on the 42 replayed error requests from 98% to 7% (native 10%), and official telecom to 0.68:
   the agent setting. Measured afterwards with the graphed tail on, it is about 1.7× slower than native (settings
   table above).
3. *An agent-trained map.* The agent map is the fidelity map fine-tuned with native 27B airline traces in the mix
   (`results/agent_map.json`): KL to native on held-out airline requests 0.235 → 0.116, held-out fidelity gate
   unchanged (`results/heldout_fidelity_gate.json`). On the replay it lifted telecom (0.66 → 0.76) and retail
   (0.87 → 0.94) in the lighter agent setting. The pre-registered official run (no repair) then scored airline /
   retail / telecom 0.50 / 0.78 / 0.22, against 0.48 / 0.78 / 0.20 for the earlier fidelity-map `+rep2048m` runs (airline and retail before the repair fix, telecom after it).
   This map is used only for the lighter agent setting.

The replay (teacher-forced next action, `task_tau2_replay`) turned out to be a weak proxy: it scores whether the next
tool matches native, while τ2 fails whole conversations on policy violations. Deleting the policy scored close to the
agent setting on the replay but 0.24–0.26 lower on official airline and telecom.

**With the native 27B as user simulator and judge** (no API needed; earlier settings):

| domain | native 27B | ours, no repair | recall `+rep2048` (blocks) | recall `+rep2048m` | mostly native [14] |
|---|---:|---:|---:|---:|---:|
| airline | 0.74 | 0.26 | 0.60 [15] | 0.60 | 0.35 |
| retail | 0.78 | 0.60 | 0.70 | 0.735 | 0.74 |
| telecom | 0.60 | not run | 0.18 | 0.32 (`+rep4096m`: 0.20) | not run |

These arms read the newest turn (up to 4096 tokens) and the `<tools>` block natively; the `+rep2048m` arms the last
three messages. They predate the repair fix. Two later native runs with this simulator scored telecom 0.50 / retail
0.74 and airline 0.70, so native varies by about ±0.1 between runs.

- **[14]** Only the system prompt and first user turn are mapped; every later turn is appended natively to the kept
  cache. Shown for completeness; not a transfer result.
- **[15]** Two runs with 1024 block-repair tokens gave 0.40 and 0.32.

#### Terminal-Bench 2.0

Harbor, Terminus-2 agent, official task containers. The same 10 tasks for native 27B and the agent setting, side by
side, at twice Harbor's default per-task agent time limit (`results/terminal_bench_tb2.json`):

- **2× time limit:** native solved 7 of 10 (0.70) in 35 minutes; the agent setting solved 1 of 10 (0.10), with all
  10 trials at the time limit after 2 hours.
- **Default time limit:** native 0.20, agent setting 0.00 (9 timeouts, 1 runtime error).
- **Why the transfer is slow here.** Terminal-Bench prompts grow to about 75K tokens and every turn re-reads them;
  in our stack each agent-setting turn pays the 4B prefill, the mapping and repair, so it completes far fewer turns.
- **OneTriangle:** 0.10 on 10 tasks (their 4B alone 0.09), with no native value, so the share of native cannot be
  compared. Our transfer matches their raw 0.10 while our native reaches 0.70 with the same harness.

### OneTriangle's 2B → 27B tab

With a Qwen3.5-2B source, prompts up to 32K (`results/quality_2b_fid.json`; native and 2B-alone columns from
`results/quality_2b_suite.json`):

| benchmark | n | native 27B | ours, no repair | ours, recall | 2B alone | OneTriangle (native) |
|---|---:|---:|---:|---:|---:|---:|
| ARC-Chat [16] | 100 | 0.98 | **0.98** | | 0.81 | 0.89 (0.99) |
| QuALITY | 100 | 0.91 | **0.79** | | 0.70 | 0.71 (0.94) |
| MATH-500 [16] | 100 | 0.87 | 0.87 | | 0.77 | |
| MATH-500, thinking on | 100 | 0.93 | 0.93 | | | 0.94 (0.96) |
| RULER 4–16K | 90 | 0.97 | 0.82 | **0.905** (+rep1024) | 0.92 | 0.80 (1.00) retrieval |
| MRCR 4–32K, dev | 60 | 0.97 | 0.52 (adaptive tail 0.54) | 0.706 (+rep3072m); full sweep below | 0.87 | up to 0.66 |
| MRCR 4–32K, held-out | 120 | 0.972 | 0.552 (adaptive tail) | **0.681** (+rep3072m), 0.602 (+rep2048m) | 0.931 | |
| LongBench v2 (≤32K) | 200 | 0.37 | 0.35 (n=60) | 0.42 (+rep1024), 0.375 (+rep2048) | 0.23 (n=60) | 0.42 (0.61) |
| LoCoMo (F1) | 100 | 0.58 | 0.457 | | 0.39 | per-category only |
| ZebraLogic | 100 | 0.69 | 0.63 | 0.61 (+rep2048) | 0.33 | |
| ZebraLogic, thinking on | 100 | 0.95 | **0.75** | | | 0.66 (0.91), reasoning on |

- **[16]** Re-run with the fidelity map (`results/quality_2b_fid.json`), native re-run alongside: ARC-Chat 0.98 vs
  0.98, MATH-500 0.87 vs 0.87. The phase-2 map had given 0.98 and 0.86; the 2B-alone column is from the phase-2 run.

RULER, MRCR, LongBench v2 and LoCoMo "no repair" cells use a fixed 128-token tail (note [6]).

**MRCR repair sweep (dev, 60 items).** The unit cap is the longest message, as a fraction of N, that is kept whole
(default 0.5; longer messages fall back to blocks, see *How it works*).

| size | unit | unit cap | score |
|---|---|---|---:|
| +rep1024m | message | default | 0.522 |
| +rep2048m | message | default | 0.642 |
| +rep3072m | message | default | **0.706** |
| +rep1024m | message | 0.9 | 0.564 |
| +rep2560m | message | 0.9 | 0.658 |
| +rep3072m | message | 0.9 | 0.602 |
| +rep1024 | block | — | 0.50 |
| +rep2048 | block | — | 0.40 |
| +rep4096 | block | — | 0.62 |

The native, 2B-alone, adaptive-tail (0.54), block +rep2048 and block +rep4096 arms were re-run on the identical
60-item sample as the no-repair arm (native 0.970, 2B alone 0.872, matching the values shown); the block +rep1024
value (0.50) comes from a different 60-item sample.

**Why MRCR needed whole-message recall.** With scattered 64-token blocks, small repair budgets scored *below* no
repair (0.40 at +rep2048 vs 0.54): the target got fragments of several candidate answers. Re-encoding whole messages
(a matching request together with the answer that follows it) fixed that.

**How the size was picked.** Sizes were compared on the 60-item dev sample (table above), where results move by
about ±0.05 between neighbouring settings (a looser unit cap gave 0.602 at +rep3072m).

**Held-out run.** +rep3072m with the default cap was then fixed and run once on 120 further items never used for
tuning the repair (`offset` 60): 0.681, with native and the 2B alone re-run on the same items. The fidelity maps
were selected on earlier MRCR subsets, which are not guaranteed disjoint from these items. That is ahead of
OneTriangle's best bucket within noise, at 1.36× (8K), 2.10× (16K) and 2.78× (32K) TTFT for this pair.

**LongBench v2** is a tie only with the better of the two repair sizes we ran on the 200-item sample (0.42 vs 0.375;
earlier 60-item runs with +rep1024 gave 0.38–0.42), and it is above our native 27B (0.37), which says more about the
noise of this benchmark than about the transfer.

ZebraLogic and MATH-500 were re-run with thinking on for both arms, as OneTriangle's native scores (0.91, 0.96) imply
they did; non-thinking, our native scores 0.69 and 0.87. DevEval was not run (needs the repository test
environments).

### Held-out fidelity gate

The fidelity maps, adaptive tail and repair were developed while looking at the benchmarks above (see *How the
evaluation was run*). To check that this did not just fit those benchmarks, we compared the phase-2 map and the
fidelity map on data neither was selected on (`results/heldout_fidelity_gate.json`, 4B → 27B):

| | native 27B | phase-2 map | fidelity map | agent map [17] | 4B alone |
|---|---:|---:|---:|---:|---:|
| FineWeb-Edu held-out, next-token agreement (4K prefix, 1-token tail) | 1 | 0.896 | 0.900 | 0.898 | 0.736 |
| same, KL to native (nats/token) | 0 | 0.059 | 0.057 | 0.056 | 0.367 |
| same, 128-token tail: agreement / KL | 1 / 0 | 0.925 / 0.031 | 0.925 / 0.030 | 0.926 / 0.029 | |
| HellaSwag (length-normalized acc) | 0.837 | 0.787 | 0.78 | 0.787 | 0.73 |
| PIQA (acc) | 0.803 | 0.803 | 0.797 | 0.80 | 0.797 |
| ARC-Challenge (acc) | 0.543 | 0.507 | 0.527 | 0.523 | 0.517 |
| MMLU (acc) | 0.853 | 0.687 | 0.673 | 0.673 | 0.74 |
| Winogrande (acc) | 0.76 | 0.747 | 0.75 | 0.75 | 0.743 |

Multiple choice: 300 items each, log-likelihood scoring, 1-token native tail (the question is almost fully mapped).

- **[17]** The agent map: the fidelity map fine-tuned with τ2 airline agent traces (used only for the lighter agent
  setting; *Agents*).

- The fidelity map is **within noise (±0.02) of the phase-2 map** on every row: slightly better on FineWeb, ARC and
  Winogrande, slightly worse on HellaSwag, PIQA and MMLU. The gains on the targeted benchmarks did not cost general
  fidelity.
- FineWeb-Edu is also part of the distillation data; this is its held-out split, i.e. unseen text from the training
  distribution. The multiple-choice sets were never used for training or selection.
- **MMLU with an almost fully mapped prompt is worse than the 4B alone** (0.67–0.69 vs 0.74). In fast mode such
  short prompts run natively, so this does not show up in the tables above, but it is a real weakness of the maps
  on short, knowledge-heavy contexts.

### Beyond 32K

Above 32K the 4B switches to an 8K sliding window, which is what makes 64K–1M fast. On RULER at 64K and MRCR at
40K and above (4B → 27B, phase-2 map, 4B on the 8K sliding window, `results/quality_64k_ruler_mrcr.json`):

| | native 27B | ours |
|---|---:|---:|
| RULER (n=27) | 0.92 | 0.55 |
| MRCR, 2 needles (n=30) | 1.00 | 0.15 |

**Exact recall degrades sharply in the long-context configuration.** The speedups above 32K should be read as speed
results only; we have not validated quality there.

### Where recall is lost

Swapping the native 27B's attention KV into a transferred cache restores exact recall; swapping in the native
DeltaNet states does not (`results/oracle_diagnosis.json`):

- **4B → 27B, phase-2 map `4b-27b-chat`.** RULER 0.72 → 1.00 (n=27), MRCR 0.30 → 0.97 (n=30); with native DeltaNet
  states instead: 0.69 and 0.26. This run is not the one in *Beyond 32K*: its run arguments use RULER at 16K and MRCR
  at 8K and above with prompts capped at 32K (`max_ctx` 32768) and full 4B attention, not the 64K sliding-window
  configuration, so its transfer baselines (0.72, 0.30) differ from the 0.55 and 0.15 above.
- **2B → 27B ZebraLogic (40 puzzles).** The attention swap gives 0.125 → 0.65 vs 0.225 for the DeltaNet swap, but a
  plain 512-token native tail also reaches 0.65 there, so for short puzzles a longer tail is an equally good fix (it
  is what the adaptive tail does).

This is why the fidelity maps add a dedicated attention-KV map and why recall mode re-encodes attention KV only.

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
4. **Recall mode (`+repN`): repair.**
   1. The source reads the tail over its own cache.
   2. Each 64-token block of the mapped prefix is scored by the source's attention from the tail: max over tail
      queries, summed over heads and over the 4B attention layers up to the deepest mapped one, first 4 positions
      excluded.
   3. The 27B re-encodes the top-N tokens natively in one batched pass: absolute-position causal attention over the
      full cache; each contiguous span's DeltaNet scan starts from the mapped end-of-prefix state, via
      variable-length kernels.
   4. Only the attention KV it writes is kept.

   With `+repNm` (whole-message recall) the unit is a whole chat message instead of a block:
   - a message scores its best block;
   - selecting a message also takes the message that follows (a request and its answer);
   - messages longer than N/2 (a long system prompt) fall back to blocks;
   - units are taken greedily by score until N tokens are used.
5. **Fit, then distill.** Maps start as closed-form ridge regressions on 1M calibration tokens, then are fine-tuned
   end to end on KL(native 27B ‖ transferred 27B) with gradients through the frozen 27B. Distillation data: FineWeb-Edu
   text, synthetic record lookups (close in style to needle-in-a-haystack tests), synthetic tool-use dialogues, and
   public instruction data (LongAlign-10k, Tulu-3 SFT). These were not decontaminated against the evaluation sets.

**Engine.** Fused Triton causal-conv kernel, state-only delta-rule scan, a CUDA-graphed persistent tail (prompts
≤32K), CUDA-graph batched decoding with per-sequence lengths, and source-drafted speculative decoding with exact
recurrent-state rollback (rejected tokens re-scanned with β = 0, g = 0). fla's norm kernels re-autotune for every new
sequence length; we freeze those keys, which made generation evals about 4× faster.

## How the evaluation was run, and what to be careful about

- **Configuration was chosen on evaluation items.** What was tuned, and on which items:
  - full 4B attention up to 32K, the chat-distilled (phase-2) map and the free-form code prompt → LongHealth,
    ContractNLI and HumanEval items that are also in the final samples;
  - the fidelity maps → RULER, MRCR and ZebraLogic subsets;
  - the adaptive tail → ZebraLogic and MBPP;
  - recall mode and its size → 2B RULER / MRCR / LongBench v2 and earlier τ2 runs;
  - the agent setting (10 native messages, `+rep4096m`, protected repair) → τ2 telecom replay and official telecom;
  - the lighter agent setting and its map → fixed in advance (written down before the run) and scored in one
    official run; airline is in-sample for that map.

  Treat all of those numbers as development-set results. The held-out gate above is the guard against general
  regression, and the MRCR recall result is also reported on a held-out sample.
- **Mixed configurations.** The scoreboard takes each benchmark from the mode named in its column. No single
  configuration wins everywhere: recall mode buys RULER and MRCR, and the agent setting (mostly native) buys τ2, all
  at the cost of TTFT. The lighter agent setting uses the agent map (`4b-27b-fid-agent`); every other row uses the
  fidelity map `4b-27b-fid` (4B pair) or `2b-27b-fid` (2B pair).
- **Harness.** Every benchmark was reimplemented (`src/xkv/genevals.py`); OneTriangle's prompts and scorers are not
  public. Code benchmarks use a free-form "answer in a code block" prompt (a forced code prefix cost the native 27B
  6 points on HumanEval); MBPP shows its test asserts in the prompt, as is standard; MATH-500 allows 3072 new tokens
  (8192 with thinking on; ZebraLogic 12288). Thinking is off unless a row says otherwise.
- **Greedy continuations diverge.** Next-token agreement with native is 90–93% teacher-forced, but free-running greedy
  outputs drift apart early. Task scores are the meaningful measure.
- **Parity.** The engine's parity check against Hugging Face (99.6% top-1, `results/selftest_parity.json`) was run
  on the 0.8B → 2B pair, not on the 27B.
- **Timing.** The transfer reuses a preallocated graphed tail; the native path allocates its cache inside the timed
  region (well under 20 ms). Both arms run on the same engine and GPU. The headline TTFT rows use one prompt length
  per row; with varying lengths the graphed tail should be turned off (*Speed → Graphed tail*).
- bf16 throughout, both models; no quantization on either side. Sample sizes are mostly 50–300 (Terminal-Bench 10,
  the 64K tables 27–30); differences of 2–3 points (13 on τ2) are within noise.

## Known limitations

- **Agents with long rule-heavy policies.** In the lighter agent setting (23% of the telecom prompt native), the
  transferred τ2 telecom policy misleads the agent (0.22, below the 0.44 of the policy-deleted control). Matching
  native needs the agent setting, which is slower than native. Terminal-Bench is far below our native (0.10 vs
  0.70; *Agents → Terminal-Bench 2.0*).
- **Exact recall beyond 32K.** Beyond 32K the source uses an 8K sliding window, so exact recall is lost there
  (*Beyond 32K*).
- **Graphed tail with varying prompt lengths** costs about 0.5 s per request until it is turned off or bucketed
  (*Speed → Graphed tail*).
- **Some wins need recall mode** (all 2B pair): RULER and LongBench v2 use `+rep1024` (2.0–3.6× TTFT, 8K–32K),
  MRCR `+rep3072m` (1.4–2.8×), instead of 3.9–4.9× in fast mode.

## Notes on OneTriangle's agent numbers

Their benchmark page does not describe the agent setup behind its τ2 and Terminal-Bench numbers, so the comparison
above rests on a few assumptions worth stating:

- **How much of each prompt they compute natively is not stated.** Ours ranges from 23% to 98% depending on the
  setting and domain, and that choice moves τ2 scores by more than the gap between us and them.
- **A reference point, not a conclusion:** our policy-deleted control (*Agents*) scores 0.42 / 0.72 / 0.44 (airline
  / retail / telecom); their mapped scores are 0.59 / 0.41 / 0.47. Harnesses and natives differ, so this does not
  say how much of the policy their transfer keeps; it shows that τ2 scores in this range are reachable without the
  policy.
- **17 tasks per domain** gives ±0.23 at 95%, so each of their τ2 numbers is compatible with a wide range.
- **Their native 27B is stronger than ours on τ2** (0.86 vs 0.68–0.70), which points to a different harness or
  thinking mode; we would expect (but have not tested) that a stronger agent recovers more easily from a degraded
  context.
- **Terminal-Bench** has no native value on their side, so the share of native cannot be compared (*Agents →
  Terminal-Bench 2.0*).
- **Speed for agents** is not reported on their page; their TTFT numbers are for single prefills. Ours shows the
  agent setting slower than native, and the lighter agent setting faster only on telecom with the graphed tail off
  (settings table in *Agents*).

## What we would try next

- **Keep a garbled policy from misleading.** Transfer the conversation but leave out, or down-weight attention to,
  the mapped policy span, which could bring telecom near the policy-deleted 0.44 while staying mostly mapped. It
  would have to be disclosed as "policy not transferred".
- **Cache the policy natively once per domain.** The policy is identical for every request, so in serving it can be
  prefix-cached for both models. Native serving gets the same cache, though, so this only helps if the transferred
  conversation part is still cheaper than computing it natively; measure before claiming a speedup.
- **Train the attention map on varied agent data** (public multi-turn tool-use sets with long system prompts, τ2
  held out). One round on airline traces halved the KL on held-out airline requests but did not move official
  telecom.
- **Thinking mode for τ2**, to close the gap between our native (0.68–0.70) and theirs (0.86).
- **Terminal-Bench turn cost:** reuse the previous turn's transferred cache and map only the new tokens, instead of
  re-transferring a 75K-token prompt every turn.

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
results/README.md       index of every result file, grouped by topic (quality, agents, speed, correctness)
```

## Reproduce

From `modal_jobs/` with `PYTHONPATH=../src`:

```bash
modal run lab.py --task ttft_repair --gpu H100 --args '{"repairs":[0,1024,2048]}'
modal run lab.py --task ttft_repair --gpu H100 --args '{"src":"2b","mapper_name":"2b-27b-fid+2b-27b-fid-attn","repairs":[0,1024]}'
modal run lab.py --task ttft_repair --gpu H100 --args '{"src":"2b","mapper_name":"2b-27b-fid+2b-27b-fid-attn","repairs":[0,2048,"2048m","3072m"],"msg_every":400}'
modal run lab.py --task decode_bench --gpu H200 --args '{"lengths":[2048,32768],"batches":[1,2,3,4,8,12,16,32]}'
modal run lab.py --task serve --gpu H200 --args '{}'
modal run lab.py --task gen_evals --gpu H200 --args '{"mapper_name":"4b-27b-fid+4b-27b-fid-attn","full_until":32768,"benches":["contract_nli","longhealth"],"strategies":["target","transfer@auto"]}'
modal run lab.py --task gen_evals --gpu H200 --args '{"src":"2b","mapper_name":"2b-27b-fid+2b-27b-fid-attn","full_until":32768,"max_ctx":32768,"benches":["ruler","mrcr"],"strategies":["transfer@auto","transfer@auto+rep1024"]}'
modal run agent_lab.py --task tau2 --gpu H200 --args '{"strategies":["target"],"domains":["airline"],"full_until":32768,"log_requests":true,"run_name":"replay"}'
modal run agent_lab.py --task tau2 --gpu H200 --args '{"mapper_name":"4b-27b-fid+4b-27b-fid-attn","full_until":32768,"strategies":["transfer@auto+rep4096m"],"domains":["airline","retail","telecom"],"turn_tail_max":8192,"turn_tail_msgs":10,"native_tools":true,"local_user_and_judges":false}'
modal run agent_lab.py --task terminal_bench --gpu H200 --args '{"mapper_name":"4b-27b-fid+4b-27b-fid-attn","full_until":32768,"strategies":["target","transfer@auto+rep4096m"],"task_names":["gpt2-codegolf","break-filter-js-from-html","llm-inference-batching-scheduler","write-compressor","merge-diff-arc-agi-task","pytorch-model-cli","reshard-c4-data","largest-eigenval","password-recovery","log-summary-date-ranges"],"concurrency":10,"turn_tail_max":8192,"turn_tail_msgs":10,"agent_timeout_multiplier":2}'
modal run agent_lab.py --task tau2 --gpu H200 --args '{"full_until":32768,"strategies":["target-drop"],"domains":["airline","retail","telecom"],"turn_tail_max":8192,"turn_tail_msgs":10,"local_user_and_judges":false}'
modal run lab.py --task distill --gpu H200 --args '{"mapper_name":"4b-27b-fid+4b-27b-fid-attn","out_name":"4b-27b-fid-agent","steps":600,"prefix_choices":[2048,4096,8192,12288],"prefix":8192,"cont":128,"tail":64,"eval_tail":128,"chat_frac":0.15,"tool_frac":0.15,"retrieval_frac":0.3,"varied_frac":0.45,"copy_frac":0.35,"agent_logs":["tau2/replay/target/airline_requests.jsonl"],"agent_frac":0.4,"agent_tail_max":1024,"agent_tail_msgs":2,"eval_agent":24,"lr":1e-5,"warmup":20,"eval_every":200,"n_eval":4,"niah_len":16384,"niah_n":10}'
modal run agent_lab.py --task tau2 --gpu H200 --args '{"mapper_name":"4b-27b-fid-agent+4b-27b-fid-agent-attn","full_until":32768,"strategies":["transfer@auto"],"domains":["airline","retail","telecom"],"turn_tail_max":4096,"turn_tail_msgs":3,"native_tools":true,"local_user_and_judges":false}'
modal run agent_lab.py --task tau2_replay --gpu H200 --args '{"domains":["telecom","retail"],"max_prompts":160,"max_new":128,"ttft_n":16,"variants":[{"name":"native","strategy":"target"},{"name":"agent","strategy":"transfer@auto+rep4096m","native_tools":true,"turn_tail_max":8192,"turn_tail_msgs":10},{"name":"lighter_nograph","strategy":"transfer@auto","native_tools":true,"turn_tail_max":4096,"turn_tail_msgs":3,"mapper":"4b-27b-fid-agent+4b-27b-fid-agent-attn","env":{"XKV_NO_GRAPH_TAIL":"1"}},{"name":"deleted","strategy":"target","drop_mapped":true,"turn_tail_max":8192,"turn_tail_msgs":10}]}'
modal run lab.py --task gen_evals --gpu H200 --args '{"src":"2b","mapper_name":"2b-27b-fid+2b-27b-fid-attn","full_until":32768,"max_ctx":32768,"benches":["mrcr"],"strategies":["transfer@auto+rep3072m"],"limits":{"mrcr":120},"bench_kw":{"mrcr":{"min_ctx":4096,"batch":4,"offset":60}}}'
modal run lab.py --task quality --gpu H100 --args '{"mapper_name":"4b-27b-fid+4b-27b-fid-attn","n_seqs":32,"prefix":4096,"tails":[1,128],"mc_limit":300,"mc_tails":[1]}'
```

The official τ2 user simulator and Terminal-Bench need the Modal secrets `xkv-openai` (`OPENAI_API_KEY`) and
`xkv-harbor-modal` (a Modal token for the task sandboxes). Thinking-on rows use `"env":{"XKV_THINKING":"1"}`, the
QMSum judge `"env":{"XKV_QMSUM_JUDGE":"1"}`. Code benchmarks use `"env":{"XKV_CODE_PROMPT":"free"}`; BigCodeBench
executes in the official evaluation image (`modal deploy exec_app.py`, then `"exec_modal":true`).
