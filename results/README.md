# results/

Curated result files behind every number in the top-level README. Files with a `what` or `note` field describe
themselves; the rest are the raw output of one Modal task (`task`, `args`, `wall_s`, `result`), so `args` is the
exact configuration. Per-run raw outputs are written to `results/raw/` (not tracked).

Map names: `4b-27b-fid` / `2b-27b-fid` are the fidelity maps used for the current results (`+...-attn` is the
attention-KV map that goes with them); `4b-27b-fid-agent` is the fidelity map fine-tuned on agent traces (lighter τ2
setting only); `4b-27b-chat`, `4b-27b-w8192-distill-ret` and `4b-27b-r2-*` are earlier maps (phase 1–2).

## Quality

| file | what it holds |
|---|---|
| `quality_4b_fid.json` | 4B → 27B generation benchmarks with the fidelity map (ContractNLI, LongHealth, BAMBOO, QMSum, MATH-500, code) |
| `quality_4b_suite.json` | 4B → 27B suite with native and 4B-alone baselines (phase-2 map) |
| `quality_2b_fid.json` | 2B → 27B tab with the fidelity map: RULER, MRCR (dev and held-out), LongBench v2, QuALITY, LoCoMo, ZebraLogic, thinking-on runs, recall mode |
| `quality_2b_suite.json` | 2B → 27B native and 2B-alone baselines, ARC-Chat and MATH-500 (phase-2 map) |
| `quality_64k_ruler_mrcr.json` | RULER and MRCR at 64K (beyond the 32K source window) |
| `heldout_fidelity_gate.json` | Benchmarks never used for tuning: FineWeb-Edu held-out agreement/KL and five multiple-choice sets, per map |
| `oracle_diagnosis.json` | Native attention KV or native DeltaNet state swapped in, to see which part of the cache loses quality |
| `quality_2k_ridge.json`, `quality_2k_distilled.json`, `quality_2k_longctx_map.json` | Phase-1 teacher-forced quality of the early ridge / distilled maps (2K prefixes) |
| `longctx_32k_*.json` | Phase-1 long-context agreement and needle-in-a-haystack sweeps (tails, windows, calibration) |

## Agents

| file | what it holds |
|---|---|
| `tau2_fid.json` | Every τ2 run: native, settings, controls (policy deleted), agent map; official and native-27B user simulator |
| `tau2_replay.json` | Teacher-forced τ2 replay: next-action match per variant, native-token share, batch-1 TTFT |
| `tau2_telecom_failure_analysis.json` | First lookup tool per telecom conversation, native vs transfer |
| `terminal_bench_tb2.json` | Terminal-Bench 2.0, native vs agent setting, default and 2× agent time limit |
| `agent_map.json` | Training history of the agent-trained map (KL on web text and on held-out agent requests) |

## Speed

| file | what it holds |
|---|---|
| `ttft_h100_final_engine_two_runs.json`, `ttft_h100_8k-32k_final_engine.json` | Headline TTFT, 8K–32K, final engine |
| `ttft_h100_recall_mode.json` | TTFT with repair (recall mode), both pairs |
| `ttft_h100_8k-64k.json`, `ttft_h200_64k-128k_reps3.json`, `ttft_h200_128k-1m*.json` | TTFT 64K to 1M |
| `ttft_h100_untuned_lengths_batches.json` | TTFT at lengths and batch sizes the kernels were not tuned for |
| `tpot_and_batch_sweeps.json` | Decode speed (TPOT) and batch sweeps |
| `serve_h200_8k-32k*.json` | Online serving with Poisson arrivals: goodput and throughput, native vs transfer (`_fid_recall`: current map, fast and recall mode) |
| `profile_h100.json`, `profile_kernels_h100.json` | Where transfer time goes (stages, kernels) |

## Correctness and maps

| file | what it holds |
|---|---|
| `selftest_parity.json` | Engine vs Hugging Face parity (0.8B → 2B pair) |
| `decode_selftest.json`, `spec_decode_row_independence.json` | Decoder and batched speculative-decoding checks |
| `calibrate_4b-27b_*.json` | Ridge-map calibration runs (phase 1) |
| `distill_short.json`, `distill_long_retrieval.json` | Phase-1 map distillation runs |
