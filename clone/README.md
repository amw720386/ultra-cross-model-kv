# clone/: kvx, the earlier OneTriangle clone

Code for the open clone of OneTriangle's cross-model KV transfer for plain-attention models (Qwen3, Qwen2.5, Gemma,
Llama, Phi, OLMo). It is separate from the `xkv` engine in `src/xkv`, which handles the hybrid
Qwen3.5-4B → Qwen3.8-27B pair. Results are in [../docs/onetriangle-clone.md](../docs/onetriangle-clone.md).

## The clone in detail

This repo rebuilds, from public material only, the method that OneTriangle (YC S26) sells as "the fastest, cheapest inference powered by KV cache transfer." A small model prefills the prompt. Its KV cache, with RoPE removed, is mapped into a large model's KV space and the target's RoPE is applied again. The large model then decodes from the mapped cache.

## Status against the target's claims (2026-11-19)

| OneTriangle claim | Our clone, measured |
|---|---|
| ~8.4x TTFT for Qwen3 1.7B -> 32B at 32K (7.34 s -> 0.87 s, H100, HF native baseline) | HF+FA2 on B200: 5.12 s -> 0.83 s (6.2x). Inside vLLM 0.34 on H200 against native vLLM on the same GPU: 5.16 s -> 1.02 s (**5.1x**); 3.6x at 4K |
| 9.6x for Qwen2.5 3B -> 72B at 127K | Against native vLLM, one request at a time. One B200, 32K: 4.61 s -> 0.57 s (**8.1x**). At 128K with YaRN, tensor parallel 2 on 2x B200: 17.1 s -> 3.38 s (**5.05x**). Every rank runs the full 3B prefill, and that prefill is the remaining bottleneck. Quality at 128K: first-token KL 0.056 |
| Long context (claims at 127K) | Qwen3 1.7B -> 32B at 128K with YaRN on one B200: 21.0 s -> 3.46 s (**6.1x**). Quality on real 128K book prompts: first-token KL 0.047. The maps were fit on 1K-token sequences |
| "First token computed natively" | It is required on the 24x pair. Plain ridge on Qwen2.5 3B -> 72B gives KL 4.09, worse than no context, because the 72B parks its attention sink on token 0. A fixed calibration-mean K/V template at token 0 gives 0.390 with no target compute; with a 256-token tail, 0.088 |
| 12.6x for Ministral 3B -> Mistral Medium 128B | Not testable: Mistral Medium weights are not public |
| 20% lower prefill cost, 40% lower TTFT | $ per 1M prefill tokens, bs1 on H200: $0.199 -> $0.039 at 32K (5.1x) |
| Lower KL than NVIDIA's linear mapper (no number published) | Qwen3 14B -> 32B: paper ridge k8 KL 0.112 -> ours 0.045 (-60%). Qwen3 1.7B -> 32B: 0.339 -> 0.139 with a 64-token native tail (-59%) |
| Quality retention | 1.7B -> 32B with a 64-token native tail: HellaSwag 99%, MMLU 5-shot 99%. 14B -> 32B: 100%. Gemma-2, Mistral, Llama, Qwen2.5 and an MoE target: 97-103% / 88-99% |
| "Restore relative importances" | Our reading is a family-specific key restore: RMS to the target's `k_norm` for Qwen3, a global gain of 1.1 for Qwen2.5 and Gemma-2, none for Llama. Plus native sink and tail tokens |
| Stage-2 KL-distilled rank-16 residual | Diverged or overfit in 3 tries. Calibration domain matters more: mixed web+book calibration halves 8K book KL |
| No API change, vLLM integration | `kvx/vllm_connector.py`: a vLLM v1 KV connector. The prompt prefix is reported as an external cache hit, and mapped K/V are written into paged blocks |

## Scoreboard: every OneTriangle number vs ours

<!-- SCOREBOARD:BEGIN -->
**OneTriangle Tier 1: 24 / 45 beaten** (tied 3, behind 4, on quality hold 5, blocked 2, open 7): the 9 homepage peak speedups, their measured end-to-end TTFT points, and the 15 rows of their benchmark bundle. Only Tier 1 is the goal.

| id | pair | metric | context / benchmark | theirs | ours | owner | status |
|---|---|---|---|---|---|---|---|
| home-ministral-3-3b-mistral-medium-3-5-128b-8K-ttft | Ministral-3-3B -> Mistral-Medium-3.5-128B | TTFT s (mapped) | 8K | 0.36 | 0.1842 | speed | beaten |
| home-ministral-3-3b-mistral-medium-3-5-128b-16K-ttft | Ministral-3-3B -> Mistral-Medium-3.5-128B | TTFT s (mapped) | 16K | 0.64 | 0.2781 | speed | beaten |
| home-ministral-3-3b-mistral-medium-3-5-128b-32K-ttft | Ministral-3-3B -> Mistral-Medium-3.5-128B | TTFT s (mapped) | 32K | 1.45 | 0.5155 | speed | beaten |
| home-ministral-3-3b-mistral-medium-3-5-128b-64K-ttft | Ministral-3-3B -> Mistral-Medium-3.5-128B | TTFT s (mapped) | 64K | 4.01 | 1.139 | speed | beaten |
| home-ministral-3-3b-mistral-medium-3-5-128b-peak | Ministral-3-3B -> Mistral-Medium-3.5-128B | peak speedup x (headline) | 32K | 12.6 | 7.11 | speed | behind |
| home-qwen3-1-7b-qwen3-32b-8K-ttft | Qwen3-1.7B -> Qwen3-32B | TTFT s (mapped) | 8K | 0.21 | 0.1077 | speed | beaten |
| home-qwen3-1-7b-qwen3-32b-16K-ttft | Qwen3-1.7B -> Qwen3-32B | TTFT s (mapped) | 16K | 0.4 | 0.154 | speed | beaten |
| home-qwen3-1-7b-qwen3-32b-32K-ttft | Qwen3-1.7B -> Qwen3-32B | TTFT s (mapped) | 32K | 0.87 | 0.2952 | speed | beaten |
| home-qwen3-1-7b-qwen3-32b-peak | Qwen3-1.7B -> Qwen3-32B | peak speedup x (headline) | 1M | 11.0 | 8.913 | speed | hold |
| home-qwen2-5-3b-qwen2-5-72b-32K-ttft | Qwen2.5-3B -> Qwen2.5-72B | TTFT s (mapped) | 32K | 1.46 | 0.3041 | speed | beaten |
| home-qwen2-5-3b-qwen2-5-72b-64K-ttft | Qwen2.5-3B -> Qwen2.5-72B | TTFT s (mapped) | 64K | 3.46 | 0.569 | speed | beaten |
| home-qwen2-5-3b-qwen2-5-72b-127K-ttft | Qwen2.5-3B -> Qwen2.5-72B | TTFT s (mapped) | 127K | 9.74 | 1.074 | speed | beaten |
| home-qwen2-5-3b-qwen2-5-72b-262K-ttft | Qwen2.5-3B -> Qwen2.5-72B | TTFT s (mapped) | 262K | 36.45 | 2.1554 | speed | hold |
| home-qwen2-5-3b-qwen2-5-72b-524K-ttft | Qwen2.5-3B -> Qwen2.5-72B | TTFT s (mapped) | 524K | 136.7 | 4.2536 | speed | hold |
| home-qwen2-5-3b-qwen2-5-72b-peak | Qwen2.5-3B -> Qwen2.5-72B | peak speedup x (headline) | 127K | 9.6 | 10.2285 | speed | hold |
| home-gemma-4-e2b-gemma-4-31b-peak | Gemma-4-E2B -> Gemma-4-31B | peak speedup x (headline) | 1M | 9.4 |  | speed | open |
| home-qwen3-5-2b-qwen3-8-27b-peak | Qwen3.5-2B -> Qwen3.8-27B | peak speedup x (headline) | 1M | 7.7 |  | speed | open |
| home-gemma-4-e4b-gemma-4-31b-peak | Gemma-4-E4B -> Gemma-4-31B | peak speedup x (headline) | 1M | 4.7 |  | speed | open |
| home-qwen3-5-4b-qwen3-8-27b-8K-ttft | Qwen3.5-4B -> Qwen3.8-27B | TTFT s (mapped) | 8K | 0.63 | 0.3351 | speed | beaten |
| home-qwen3-5-4b-qwen3-8-27b-16K-ttft | Qwen3.5-4B -> Qwen3.8-27B | TTFT s (mapped) | 16K | 1.25 | 0.8017 | speed | beaten |
| home-qwen3-5-4b-qwen3-8-27b-32K-ttft | Qwen3.5-4B -> Qwen3.8-27B | TTFT s (mapped) | 32K | 2.81 | 1.2244 | speed | beaten |
| home-qwen3-5-4b-qwen3-8-27b-64K-ttft | Qwen3.5-4B -> Qwen3.8-27B | TTFT s (mapped) | 64K | 6.33 | 3.8213 | speed | beaten |
| home-qwen3-5-4b-qwen3-8-27b-peak | Qwen3.5-4B -> Qwen3.8-27B | peak speedup x (headline) | 1M | 3.9 |  | speed | open |
| home-glm-4-7-flash-glm-5-3-peak | GLM-4.7-Flash -> GLM-5.3 | peak speedup x (headline) | 1M | 3.9 | None | speed | blocked |
| home-qwen3-1-7b-qwen3-8b-8K-ttft | Qwen3-1.7B -> Qwen3-8B | TTFT s (mapped) | 8K | 0.17 | 0.1044 | speed | beaten |
| home-qwen3-1-7b-qwen3-8b-16K-ttft | Qwen3-1.7B -> Qwen3-8B | TTFT s (mapped) | 16K | 0.34 | 0.1847 | speed | beaten |
| home-qwen3-1-7b-qwen3-8b-32K-ttft | Qwen3-1.7B -> Qwen3-8B | TTFT s (mapped) | 32K | 0.79 | 0.3769 | speed | beaten |
| home-qwen3-1-7b-qwen3-8b-peak | Qwen3-1.7B -> Qwen3-8B | peak speedup x (headline) | 1M | 3.0 | 2.7794 | speed | hold |
| bundle-arc-chat-mapped | Qwen3.5-4B -> Qwen3.8-27B | score (mapped) | ARC-Chat | 0.97 | 0.9249 | quality | behind |
| bundle-contractnli-mapped | Qwen3.5-4B -> Qwen3.8-27B | score (mapped) | ContractNLI | 0.61 | 0.77 | quality | beaten |
| bundle-bamboo-paperqa-mapped | Qwen3.5-4B -> Qwen3.8-27B | score (mapped) | BAMBOO PaperQA | 0.893 | 0.881 | quality | tied |
| bundle-longhealth-mapped | Qwen3.5-4B -> Qwen3.8-27B | score (mapped) | LongHealth | 0.76 | 0.88 | quality | beaten |
| bundle-qmsum-mapped | Qwen3.5-4B -> Qwen3.8-27B | score (mapped) | QMSum | 0.4065 | 0.468 | quality | beaten |
| bundle-cowtext-mapped | Qwen3.5-4B -> Qwen3.8-27B | score (mapped) | CowText | 0.65 | None | quality | blocked |
| bundle-humaneval-mapped | Qwen3.5-4B -> Qwen3.8-27B | score (mapped) | HumanEval | 0.91 | 0.8963 | quality | tied |
| bundle-mbpp-mapped | Qwen3.5-4B -> Qwen3.8-27B | score (mapped) | MBPP | 0.88 | 0.8132 | quality | behind |
| bundle-bigcodebench-mapped | Qwen3.5-4B -> Qwen3.8-27B | score (mapped) | BigCodeBench | 0.55 | 0.5895 | quality | beaten |
| bundle-math-500-mapped | Qwen3.5-4B -> Qwen3.8-27B | score (mapped) | MATH-500 | 0.59 | 0.9 | quality | beaten |
| bundle-tau2-airline-mapped | Qwen3.5-4B -> Qwen3.8-27B | score (mapped) | tau2 Airline | 0.5882352941 | 0.4118 | quality | tied |
| bundle-tau2-retail-mapped | Qwen3.5-4B -> Qwen3.8-27B | score (mapped) | tau2 Retail | 0.4117647059 |  | quality | open |
| bundle-tau2-telecom-mapped | Qwen3.5-4B -> Qwen3.8-27B | score (mapped) | tau2 Telecom | 0.4705882353 |  | quality | open |
| bundle-tau2-overall-mapped | Qwen3.5-4B -> Qwen3.8-27B | score (mapped) | tau2 Overall | 0.4901960784 |  | quality | open |
| bundle-terminal-bench-2-0-mapped | Qwen3.5-4B -> Qwen3.8-27B | score (mapped) | Terminal-Bench 2.0 | 0.1 | 0.2603 | quality | beaten |
| x-ml-ttft-8k-speedup | Llama-3.1-Minitron-4B-Width-Base -> Llama-3.1-8B | TTFT speedup x [mapped_direct+rank8_residual] | 8K, resident source cache | 7.91 | 12.31646246586625 | speed | beaten |
| x-tb-qwen-72-sq-3b-72b-cache-transfer-x | Qwen2.5-3B -> Qwen2.5-72B | TTFT speedup relative to native 72B @32K [mapped] | quality vs TTFT @32K; pooled QuALITY + ContractNLI slice | 8.69 | 5.4135 | speed | behind |

All rows (Tier 2 and untracked included):

**Beaten 155 of 1143** published OneTriangle numbers (tied 26, behind 59, blocked 2, open 731). Full row table: [docs/SCOREBOARD.md](docs/SCOREBOARD.md). Rules: same pair, context/benchmark and N or more; TTFT on their GPU class (H100) and speedups against native vLLM/SGLang.

| pair | category | rows | beaten | tied | behind | blocked | open |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B -> Qwen3-8B | quality | 83 | 7 | 8 | 3 | 0 | 56 |
| Qwen3.5-4B -> Qwen3.8-27B | quality | 46 | 6 | 3 | 2 | 1 | 34 |
| Qwen3-1.7B -> Qwen3-4B | quality | 39 | 4 | 0 | 0 | 0 | 31 |
| Qwen3-4B-Instruct-2507 -> Qwen3-4B-Thinking-2507 | quality | 37 | 0 | 0 | 7 | 0 | 22 |
| GLM-4.7-Flash -> GLM-5.3 | TTFT component | 32 | 0 | 0 | 0 | 0 | 32 |
| Gemma-4-E2B -> Gemma-4-31B | TTFT component | 32 | 0 | 0 | 0 | 0 | 32 |
| Gemma-4-E4B -> Gemma-4-31B | TTFT component | 32 | 0 | 0 | 0 | 0 | 32 |
| Ministral-3-3B -> Mistral-Medium-3.5-128B | TTFT component | 32 | 12 | 0 | 4 | 0 | 16 |
| Qwen3-1.7B -> Qwen3-4B–32B | TTFT component | 32 | 14 | 0 | 4 | 0 | 0 |
| Qwen3-1.7B -> Qwen3-8B | TTFT component | 32 | 15 | 0 | 5 | 0 | 0 |
| Qwen3.5-2B -> Qwen3.8-27B | TTFT component | 32 | 0 | 0 | 0 | 0 | 32 |
| Qwen3.5-4B -> Qwen3.8-27B | TTFT component | 32 | 0 | 0 | 0 | 0 | 32 |
| Qwen3.5-2B -> Qwen3.8-27B | quality | 31 | 0 | 0 | 0 | 0 | 31 |
| Llama-3.1-Minitron-4B-Width-Base -> Llama-3.1-8B | quality | 27 | 0 | 0 | 0 | 0 | 21 |
| Qwen3-32B -> Qwen3-8B | quality | 26 | 0 | 0 | 0 | 0 | 18 |
| Qwen2.5-3B -> Qwen2.5-72B | TTFT component | 24 | 6 | 0 | 3 | 0 | 4 |
| Gemma-4-E4B -> Gemma-4-31B | quality | 20 | 0 | 0 | 0 | 0 | 20 |
| GLM-4.7-Flash -> GLM-5.3 | quality | 19 | 0 | 0 | 0 | 0 | 19 |
| Qwen3-0.6B -> Qwen3-1.7B | quality | 18 | 3 | 1 | 0 | 0 | 0 |
| Gemma-4-E2B -> Gemma-4-31B | quality | 17 | 0 | 0 | 0 | 0 | 17 |
| Ultravox-8B -> Ultravox-1B-(ridge-+-rank-16-residual) | quality | 16 | 0 | 0 | 0 | 0 | 16 |
| GLM-4.7-Flash -> GLM-5.3 | TTFT table | 15 | 0 | 0 | 0 | 1 | 14 |
| Gemma-4-E2B -> Gemma-4-31B | TTFT table | 15 | 0 | 0 | 0 | 0 | 15 |
| Gemma-4-E4B -> Gemma-4-31B | TTFT table | 15 | 1 | 0 | 1 | 0 | 13 |
| Ministral-3-3B -> Mistral-Medium-3.5-128B | TTFT table | 15 | 4 | 0 | 5 | 0 | 6 |
| Qwen3-1.7B -> Qwen3-32B | TTFT table | 15 | 4 | 0 | 5 | 0 | 0 |
| Qwen3-1.7B -> Qwen3-8B | TTFT table | 15 | 5 | 0 | 5 | 0 | 2 |
| Qwen3.5-2B -> Qwen3.8-27B | TTFT table | 15 | 0 | 0 | 0 | 0 | 15 |
| Qwen3.5-4B -> Qwen3.8-27B | TTFT table | 15 | 5 | 0 | 0 | 0 | 10 |
| Qwen3-4B -> Ministral-3-14B | quality | 14 | 0 | 0 | 0 | 0 | 10 |
| Qwen3-30B-A3B-Instruct-2507 -> Qwen3-30B-A3B-Thinking | quality | 13 | 0 | 0 | 0 | 0 | 6 |
| Qwen3-4B -> Ministral-3-14B | speed/cost | 13 | 0 | 0 | 0 | 0 | 13 |
| Qwen2.5-3B -> Qwen2.5-72B | quality | 12 | 0 | 1 | 3 | 0 | 8 |
| Qwen2.5-3B -> Qwen2.5-72B | TTFT table | 11 | 3 | 0 | 3 | 0 | 0 |
| Qwen3-1.7B -> Qwen3-0.6B | diagnostic | 9 | 3 | 2 | 1 | 0 | 0 |
| Qwen3-1.7B -> Qwen3-4B | diagnostic | 9 | 6 | 0 | 0 | 0 | 0 |
| Qwen3-14B -> Qwen3-32B | diagnostic | 9 | 0 | 0 | 0 | 0 | 6 |
| Qwen3-14B -> Qwen3-8B | diagnostic | 9 | 6 | 0 | 0 | 0 | 0 |
| Qwen3-32B -> Qwen3-14B | diagnostic | 9 | 0 | 0 | 0 | 0 | 6 |
| Qwen3-4B -> Qwen3-1.7B | diagnostic | 9 | 6 | 0 | 0 | 0 | 0 |
| Qwen3-4B -> Qwen3-32B-(direct-depth-map,-repair-32,-adaptive-speculation) | quality | 9 | 0 | 0 | 0 | 0 | 9 |
| Qwen3-4B -> Qwen3-8B | diagnostic | 9 | 4 | 2 | 0 | 0 | 0 |
| Qwen3-4B-Instruct-2507 -> Qwen3-4B-Thinking-2507 | speed/cost | 9 | 0 | 0 | 0 | 0 | 9 |
| Qwen3-8B -> Qwen3-14B | diagnostic | 9 | 6 | 0 | 0 | 0 | 0 |
| Qwen3-8B -> Qwen3-4B | diagnostic | 9 | 4 | 2 | 0 | 0 | 0 |
| Ministral-3-3B -> Mistral-Medium-3.5-128B | quality | 8 | 0 | 0 | 0 | 0 | 8 |
| Qwen3-1.7B -> Qwen3-14B | quality | 8 | 2 | 2 | 0 | 0 | 4 |
| Qwen3-1.7B -> Qwen3-32B | quality | 8 | 2 | 2 | 1 | 0 | 3 |
| Qwen3-4B -> Qwen3-32B-(direct-depth-map,-repair-32,-adaptive-speculation) | speed/cost | 8 | 0 | 0 | 0 | 0 | 8 |
| TinyLlama-1.1B -> Qwen3-4B-Instruct | quality | 8 | 0 | 0 | 0 | 0 | 8 |
| Llama-3.1-Minitron-4B-Width-Base -> Llama-3.1-8B | speed/cost | 7 | 2 | 0 | 0 | 0 | 5 |
| Qwen3-0.6B -> Qwen3-1.7B | diagnostic | 7 | 4 | 0 | 0 | 0 | 0 |
| Qwen3-32B -> Qwen3-8B-(32B-prefill,-ridge-map,-8B-decode) | quality | 7 | 0 | 0 | 0 | 0 | 7 |
| Llama-3.1-8B -> Qwen2.5-72B-Instruct | speed/cost | 6 | 0 | 0 | 0 | 0 | 6 |
| Qwen3-1.7B -> Qwen3-0.6B | quality | 6 | 1 | 0 | 3 | 0 | 0 |
| Qwen3-14B -> Qwen3-32B | quality | 6 | 0 | 0 | 0 | 0 | 4 |
| Qwen3-14B -> Qwen3-8B | quality | 6 | 4 | 0 | 0 | 0 | 0 |
| Qwen3-30B-A3B-Instruct -> Qwen3-30B-A3B-Thinking | quality | 6 | 2 | 0 | 0 | 0 | 2 |
| Qwen3-32B -> Qwen3-14B | quality | 6 | 0 | 0 | 0 | 0 | 4 |
| Qwen3-4B -> Qwen3-1.7B | quality | 6 | 4 | 0 | 0 | 0 | 0 |
| Qwen3-8B -> Qwen3-14B | quality | 6 | 4 | 0 | 0 | 0 | 0 |
| Qwen3-adjacent-sizes-0.6B..32B-(10-directions,-mean) | diagnostic | 6 | 0 | 0 | 0 | 0 | 6 |
| Llama-3.1-8B -> Qwen2.5-72B-Instruct | quality | 5 | 0 | 0 | 0 | 0 | 5 |
| Qwen3-1.7B -> Qwen3-8B | speed/cost | 5 | 0 | 0 | 0 | 0 | 1 |
| Ultravox-70B -> Ultravox-8B | TTFT component | 5 | 0 | 0 | 0 | 0 | 5 |
| Ultravox-70B -> Ultravox-8B | speed/cost | 5 | 0 | 0 | 0 | 0 | 5 |
| Ministral-3-3B -> Mistral-Medium-3.5-128B | speed/cost | 4 | 0 | 0 | 0 | 0 | 4 |
| Qwen2.5-3B -> Qwen2.5-72B | speed/cost | 4 | 0 | 0 | 1 | 0 | 3 |
| Qwen3-30B-A3B-Instruct-2507 -> Qwen3-30B-A3B-Thinking-2507 | quality | 4 | 0 | 0 | 0 | 0 | 3 |
| Qwen3-4B -> Qwen3-8B | quality | 4 | 2 | 2 | 0 | 0 | 0 |
| Qwen3-8B -> Qwen3-4B | quality | 4 | 3 | 1 | 0 | 0 | 0 |
| Ultravox-8B -> Ultravox-1B-(ridge-+-rank-16-residual) | speed/cost | 4 | 0 | 0 | 0 | 0 | 4 |
| unstated-(playground-default-pair;-selector-offers-Qwen3-1.7B -> 8B,-Qwen3.5-4B -> Qwen3.8-27B) | quality | 4 | 0 | 0 | 0 | 0 | 4 |
| unstated-(playground-default-pair;-selector-offers-Qwen3-1.7B -> 8B,-Qwen3.5-4B -> Qwen3.8-27B) | speed/cost | 4 | 1 | 0 | 3 | 0 | 0 |
| Qwen3-32B -> Qwen3-8B | speed/cost | 3 | 0 | 0 | 0 | 0 | 1 |
| Ultravox-70B -> Ultravox-8B | quality | 3 | 0 | 0 | 0 | 0 | 3 |
| Qwen-3.5-4b -> fine-tuned-Qwen-3.8-27b | quality | 2 | 0 | 0 | 0 | 0 | 2 |
| Qwen3-0.6B -> 1.7B | diagnostic | 2 | 0 | 0 | 0 | 0 | 2 |
| Qwen3-1.7B -> Qwen3-14B | speed/cost | 2 | 0 | 0 | 0 | 0 | 0 |
| Qwen3-1.7B -> Qwen3-32B | speed/cost | 2 | 0 | 0 | 0 | 0 | 0 |
| Qwen3-1.7B -> Qwen3-4B | speed/cost | 2 | 0 | 0 | 0 | 0 | 0 |
| Qwen3-4B -> Qwen3-32B-(repair-32-+-adaptive-speculation) | speed/cost | 2 | 0 | 0 | 0 | 0 | 2 |
| Qwen3.5-4B -> Qwen3.8-27B | speed/cost | 2 | 0 | 0 | 0 | 0 | 0 |
| TinyLlama-1.1B -> Qwen3-4B-Instruct | speed/cost | 2 | 0 | 0 | 0 | 0 | 2 |
| toy-4-layer-source -> perturbed-4-layer-target-(alpha=0.12) | quality | 2 | 0 | 0 | 0 | 0 | 2 |
| unstated | quality | 2 | 0 | 0 | 0 | 0 | 2 |
| unstated-(target-described-as-8b) | quality | 2 | 0 | 0 | 0 | 0 | 2 |
| Llama,-Qwen | quality | 1 | 0 | 0 | 0 | 0 | 1 |
| Qwen3-0.6B -> 1.7B | quality | 1 | 0 | 0 | 0 | 0 | 1 |
| Qwen3-4B-Base -> Qwen3-8B | quality | 1 | 0 | 0 | 0 | 0 | 1 |
| Qwen3-4B-Instruct -> Qwen3-8B | quality | 1 | 0 | 0 | 0 | 0 | 1 |
| Qwen3-adjacent-sizes-0.6B..32B-(10-directions) | diagnostic | 1 | 0 | 0 | 0 | 0 | 1 |
| Qwen3-adjacent-sizes-0.6B..32B-(10-directions,-mean) | quality | 1 | 0 | 0 | 0 | 0 | 1 |
| generic | quality | 1 | 0 | 0 | 0 | 0 | 1 |
<!-- SCOREBOARD:END -->

## What the target claims

| Claim | Number | Source |
|---|---|---|
| Prefill cost / TTFT | 20% lower / 40% lower | YC company page |
| TTFT speedup | ~10x | onetriangle.ai |
| Ministral 3B -> Mistral Medium 128B at 32K | 12.6x (18.22 s -> 1.45 s) | onetriangle.ai |
| Qwen2.5 3B -> 72B at 127K | 9.6x | onetriangle.ai |
| Qwen3 1.7B -> 32B at 32K | ~8.4x | onetriangle.ai |
| Quality | "lower KL than NVIDIA's KV-transfer paper" (no number published) | YC page |
| Method | ridge (lam 0.01) on K and V, then a rank-16 residual distilled with KL; first token and last k tokens computed natively | onetriangle.ai/blog/cache-transfer |

Closest paper: Heo et al. (NVIDIA), arXiv 2608.03893. It uses a per-head ridge from the top-k source layers. For Qwen3 14B -> 32B at k=8 it keeps 97.6% of benchmark accuracy. The paper reports no KL.

## What is built

- `kvx/core.py` contains the mapper pipeline:
  - **RoPE strip:** keys are read from the cache and un-rotated with each model's own cos/sin, so it works for any RoPE model.
  - **Gram statistics:** one pass accumulates statistics over all source and target layers.
  - **Layer choice:** an R² table picks the top-k source layers for each target layer.
  - **Fit:** closed-form ridge.
  - **Importance restore:** rescaling of the mapped keys.
  - **Eval:** next-token KL between the mapped and native caches.
- `modal_fit_eval.py` fits and evaluates one model pair in a single Modal job.


## Across families (KL to native, 960-token mapped prefix, mean over 64 continuation tokens)

| Pair | Head layout | Small model alone | Best mapped | + 64 native tail | Accuracy retention with tail (HellaSwag / MMLU5) |
|---|---|---|---|---|---|
| Qwen3 1.7B -> 32B | 8x128 -> 8x128 | 0.752 | 0.219 (k24, rms, 4000 seqs) | 0.144 | 99% / 98% (mixed calibration: 101% / 99%) |
| Qwen3 14B -> 32B | 8x128 -> 8x128 | 0.258 | 0.079 (k16, rms, 2000 seqs) | 0.052 (0.045 with first 4 native) | 100% / 103% |
| Qwen3 1.7B -> 8B | 8x128 -> 8x128 | 0.455 | 0.133 (kvcat k16, 4000 seqs) | 0.087 | 101% / 96% (500 seqs) |
| Llama-3.2 1B -> 3.1 8B | 8x64 -> 8x128 | 0.322 | 0.147 (kvcat k16, 2000 seqs) | 0.099 | 99% / 91% |
| Llama-3.1 8B -> 70B | 8x128 -> 8x128 | 0.304 | 0.178 (k20) | 0.124 | 99% / MMLU OOM on 2xH200 |
| Qwen2.5 3B -> 7B | 2x128 -> 4x128 | 0.207 | 0.142 (kvcat k16, gain 1.1) | 0.125 | 102% / 90% (500 seqs) |
| Qwen2.5 1.5B -> 7B | 2x128 -> 4x128 | 0.276 | 0.239 (k16) | 0.177 | 103% / 88% |
| Ministral-8B -> Mistral-Small-24B | 8x128 -> 8x128 | 0.252 | 0.117 (k16, gain 1.1, 1000 seqs) | 0.081 | 100% / 98% |
| Phi-3-mini -> Phi-3-medium | 32x96 (MHA) -> 10x128 | 0.206 | 0.121 (k8, gain 1.1, λ_rel 1e-3) | 0.095 | - |
| OLMo-2 1B -> 7B | 16x128 (MHA) -> 32x128 (MHA) | 0.297 | 0.229 (k8, gain 1.1) | 0.155 | 101% / 85% |
| Qwen2.5 3B -> 72B (OneTriangle's 9.6x pair) | 2x128 -> 8x128 | 0.457 | 0.390 (k16 + calibration-mean sink template at token 0; plain ridge 4.09) | 0.086 (4 native sink tokens + 256 tail) | - |
| Gemma-3 1B -> 4B (multimodal checkpoint, text model) | 1x256 -> 4x256 | 0.308 | 0.206 (k16, gain 1.1) | 0.160 | 99% / 96% |
| Gemma-3 4B -> 12B | 4x256 -> 8x256 | 0.190 | 0.271 with ridge (values fail); **0.145 with an MLP value mapper** (-47%; KV-concat made it worse: 1.46) | 0.212 -> **0.103** (MLP) | 93% / 92% (ridge) |
| Gemma-3 270M -> 1B (local/global RoPE, sliding window) | 1x256 -> 1x256 | 0.423 | 0.222 (k16, gain 1.1) | 0.149 | 101% / n.a. (MMLU near chance) |
| Gemma-2 2B -> 9B | 4x256 -> 8x256 | 0.187 | 0.078 (k16, gain 1.1) | 0.054 | 100% / 97% |
| Qwen3 1.7B -> 30B-A3B (MoE target) | 8x128 -> 4x128 | 0.538 | 0.200 (k24, rms, 2000 seqs) | 0.137 | 98% / 99% |
| DeepSeek-R1-Distill-Qwen 1.5B -> 7B | 2x128 -> 4x128 | 1.057 | 0.553 (k16, gain 1.1) | 0.400 | not run |

The MoE target maps as well as the dense ones. Its speed case is weak, though: the 30B-A3B uses about 3.3B active parameters per token, so prefill costs about 6.6 GF/token plus attention. The 1.7B source plus the mapper costs about 5.4 GF/token. The TTFT ceiling is therefore about 1x at short context and about 2x at 32K, where attention dominates.

### Source-size frontier into Qwen3-32B (KL, 2000-4000 calibration sequences)

| Source | 32K TTFT ceiling | small alone | mapped (k16/24 + rms) | + tail 64 | + 4 sink + tail 256 (or 64) |
|---|---|---|---|---|---|
| 0.6B | 8.0x | 1.051 | 0.946 | 0.729 | 0.216 |
| 1.7B | 6.8x | 0.752 | 0.219 | 0.144 | 0.062 |
| 4B | 3.7x | 0.540 | 0.179 | 0.126 | 0.055 |
| 8B | 2.8x | 0.368 | 0.150 | 0.105 | 0.044 |
| 14B | 1.9x | 0.257 | 0.079 | 0.052 | 0.045 |

The 32K ceiling uses the compute roofline: (source prefill + dense mapper) against native 32B prefill. 1.7B is the sweet spot: it has a 6.8x ceiling and keeps 98-99% of benchmark accuracy with a 64-token native tail. 4B and 8B lose most of the speed for a modest gain in quality. 0.6B maps poorly unless a large native tail and native sink tokens are added.

Long context (Qwen3-1.7B -> 32B, 8K-token PG19 books, 4000-seq k24 maps): the 1.7B alone gives KL 1.036. Mixed web+book calibration gives 0.278, 0.210 with 64 native tail tokens, and 0.152 with 4 sink + 256 tail tokens; FineWeb-only calibration gave 0.479 mapped. The full stack (mix, λ_rel 1e-3, KV-concat inputs, k16) gives 0.274 / 0.206 / 0.148.

At **32K tokens** (8 PG19 test books, B200) the full stack does not degrade:

| Setting | KL |
|---|---|
| 1.7B alone | 1.018 |
| Mapped | 0.249 |
| + 64 native tail tokens | 0.188 |
| + 1024 native tail tokens (3% of the prompt) | 0.079 |
| + 4 sink and 1024 tail tokens | 0.076 | Adding 64 native tail tokens gives 0.330, and 4 native sink tokens plus 256 native tail tokens give 0.210. The same settings at 1K give 0.219 / 0.144 / 0.062. Mapping quality halves from 1K to 8K, the main open gap. OneTriangle names exact recall at long context as its hardest case too.

### Calibration domain is the long-context lever (Qwen3-1.7B -> 8B, k24 + RMS)

| Calibration set | KL on 1K FineWeb-Edu | KL on 8K PG19 books |
|---|---|---|
| FineWeb-Edu, 4000 x 1K | **0.144** | 0.371 |
| PG19 train, 1000 x 4K | 0.307 (2000 x 1K) | 0.189 |
| Mix: 2000 FineWeb + 2000 PG19 (x 1K) | 0.158 | **0.186** |

Most of the apparent "long-context gap" came from domain mismatch: the maps were calibrated on web text and evaluated on books. One mixed calibration set halves the 8K book KL and costs about 10% on web text. The mapper should be calibrated on a mix that resembles serving traffic. OneTriangle's pitch to post-train on traffic points the same way.

### Recommended quality recipe (all levers stacked, Qwen3-1.7B -> 8B)

- 4000 calibration sequences, a mix of web and books;
- relative ridge λ 1e-3;
- K and V of each source layer as mapper input;
- top-k source layers;
- RMS key restore;
- 4 native sink tokens and a native tail.

This gives KL 0.142 (k24) mapped, 0.095 with a 64-token tail, and 0.049 with 4 sink + 256 tail tokens. The paper's ridge k=8 gives 0.244. Benchmark retention with the 64-token tail: HellaSwag 103%, MMLU 5-shot 98%.

What carries across families:

- **The native tail** is the largest single lever.
- **More calibration data** helps: 500 to 4000 sequences gives -10 to -25%.
- **Feeding both K and V of each source layer** helps most where keys are the bottleneck (Qwen2.5).
- **The key "importance restore" is family-specific:**
  - Qwen3 uses RMS restore to the target's `k_norm`.
  - Qwen2.5 and Gemma-2 use a global key gain of 1.1.
  - Llama uses none.

The paper's absolute ridge λ of 0.01 is fragile. On Phi-3, which has no qk-norm and large key magnitudes, the solve is ill-conditioned: R² > 1 and mapped KL 8.6. A relative λ (1e-3 times the mean Gram diagonal) fixes it, and is now the default.

**Attention sinks break the 24x pair, Qwen2.5-3B -> 72B.** Plain ridge gives KL 4.09, worse than no context at all (0.945).
Keys map fine and values fail, and recomputing the last 256 tokens natively does not help (3.72). Diagnosis: the 72B parks
its attention sink on token 0, the ridge mis-maps that token's value, and every later query reads it. The fix is one token.
Computing token 0 natively gives 0.391. A fixed calibration-mean K/V template at token 0, which needs no target compute at
serving time, gives 0.390. With 4 native sink tokens and a 256-token native tail it reaches 0.086. Importance-selected
recompute (RaReCache rank disagreement) plus 4 sinks gives 0.149 at a 30% budget, worse than the plain tail at a similar
budget. Without the sinks, every selector fails (rd 3.55, last-N 3.69, random 2.20 at 30%).

On the one failing pair, Gemma-3 4B -> 12B, ridge mapping does worse than running the 4B alone. Adding an MLP residual on the value maps (as the NVIDIA paper does for its failing pairs) halves KL, 0.275 -> 0.145, and beats the 4B. Held-out value R² did not change, 0.677 -> 0.672. R² is a poor proxy for output KL.

Losers, all measured:

- the KL-distilled rank-16 residual (OneTriangle stage 2 as described): it diverges or overfits;
- hidden-state features;
- native first m layers;
- layer-adaptive k;
- per-token key norm;
- low-rank dense mapper;
- grouped scatter mapper.
- fp8 source prefill (Qwen2.5-3B, H100): 32K 0.889 s in bf16 vs 1.059 s in fp8. The small GEMMs do not pay for per-call
  activation quantization;
- importance-selected recompute (RaReCache rank disagreement) on the 72B pair: 0.149 at a 30% budget, vs 0.086 for 4 sinks
  plus a 256-token tail at 27%.


## Reproduce

Commands run from this directory. Modal volume: `kvx` (override with KVX_VOLUME).

1. Fit maps and measure quality with the recommended recipe:

   ```bash
   KVX_GPU=B200 modal run modal_fit_eval.py::main --src Qwen/Qwen3-1.7B --tgt Qwen/Qwen3-32B \
     --ks 16 --modes rms --tails 64,256 --tail-k 16 --sinks 4 --n-calib 4000 --calib-src mix \
     --lam-rel 1e-3 --feat kvcat --bench-n 200 --bench-tail 64 --save-maps 1 --run-id my-run
   ```

   - Pairs with the same KV width also work with `--feat kv`. That halves the mapper cost.
   - Add `--eval-l 8192 --n-eval 16` (or 32768) to evaluate long contexts on PG19.

2. Measure serving TTFT and quality in vLLM 0.34, native first and then through the connector:

   ```bash
   KVX_GPU=H200 modal run modal_vllm.py --mode native --lens 4096,8192,16384,32768
   KVX_GPU=H200 modal run modal_vllm.py --mode mapped --maps /vol/runs/<run>/maps_k24.pt --fp8 1 --tag _mine
   python tools/vllm_quality.py out/vllm/native_Qwen3-32B.json out/vllm/mapped_Qwen3-32B_mine.json
   ```

3. Use the connector in your own vLLM:

   ```python
   KVTransferConfig(
       kv_connector="CrossModelConnector",
       kv_connector_module_path="kvx.vllm_connector",
       kv_role="kv_both",
       kv_connector_extra_config={
           "src": "Qwen/Qwen3-1.7B",
           "maps": "<maps.pt>",
           "tail": 64,
           "min_len": 1024,
           "restore": "rms",
           "fp8": True,
       },
   )
   ```

   Set `VLLM_ATTENTION_BACKEND=FLASH_ATTN`. The connector writes the packed `[blocks, H, block, 2D]` layout.

## Measured results

### Qwen2.5-3B -> Qwen2.5-72B in vLLM (OneTriangle's 9.6x pair; Modal B200, FLASH_ATTN on both sides, one request at a time)

| Prompt | Native vLLM | Mapped (fp8 k16 maps, sink template, 256-token native tail) | Speedup |
|---|---|---|---|
| 4K | 0.461 s | 0.099 s | 4.7x |
| 8K | 0.941 s | 0.156 s | 6.0x |
| 16K | 2.02 s | 0.278 s | 7.3x |
| 32K | 4.61 s | 0.570 s | 8.1x |

Quality against native vLLM on 16 FineWeb-Edu prefixes: approximate first-token KL 0.076, top-1 agreement 81%,
and greedy decoding matches for the first 7.4 tokens on average. The HF KL on 64 docs is 0.088. OneTriangle reports
9.6x at 127K. Our compute roofline at 32K is 13.4x. The source prefill runs with cuDNN SDPA (1.5x faster than torch's default pick) and the KV write copies whole blocks.
Before those two changes, 32K was 0.860 s (5.4x).

Serving bugs found on the way: vLLM 0.34 ignores `VLLM_ATTENTION_BACKEND` (pass `attention_backend`). FlashInfer reads
each KV block as `[2, H, N, D]`, so the connector refuses it. And `NewRequestData.num_computed_tokens` already counts the
connector's own tokens, so it is not the prefix-cache hit.

### Qwen2.5-3B -> Qwen2.5-72B at 128K in vLLM (YaRN x4, tensor parallel 2 on 2x B200, FLASH_ATTN, one request at a time)

| Prompt | Native vLLM TP=2 | Mapped (each rank maps its own KV heads; sink template, 1024-token native tail) | Speedup |
|---|---|---|---|
| 32K | 2.54 s | 0.542 s | 4.7x |
| 64K | 6.24 s | 1.25 s | 5.0x |
| 130,816 | 17.1 s | 3.38 s | 5.05x |

Quality at 128K on 4 real PG19 prompts: approximate first-token KL 0.056 and top-1 agreement 3/4. OneTriangle reports
9.6x at 127K on this pair. Tensor parallelism halves the native time, but every rank still runs the whole 3B source
prefill (3.1 s at 128K), so that prefill is now the bottleneck. Splitting the source across ranks is the next lever.

### Qwen3-1.7B -> Qwen3-32B at 128K in vLLM (YaRN x4 on both models; B200, FLASH_ATTN, one request at a time)

| Prompt | Native vLLM | Mapped (fp8 k24 mixed-calibration maps, 1024-token native tail) | Speedup |
|---|---|---|---|
| 32K | 2.46 s | 0.643 s | 3.8x |
| 64K | 6.82 s | 1.40 s | 4.9x |
| 130,816 | 21.0 s | 3.51 s | 6.0x |

Quality at 128K, on 4 real PG19 prompts of 130,816 tokens: approximate first-token KL 0.048 and top-1 agreement 3/4.
The maps were fit on 1K-token sequences without YaRN. At 128K the connector takes 3.1 s: about half is the 1.7B source
prefill and half is the dense k24 mapper.

### Qwen3-1.7B -> Qwen3-32B (OneTriangle's headline Qwen pair; Modal B200, run cost ~$2.10)

| Variant | KL mean | KL first | HellaSwag (200) | MMLU 5-shot (200) |
|---|---|---|---|---|
| Native 32B | 0 | 0 | 0.695 | 0.805 |
| Run only the 1.7B | 0.752 | 0.871 | 0.50 | 0.635 |
| Ridge k=8 | 0.339 | 0.689 | | |
| Ridge k=8 + RMS restore | 0.272 | 0.588 | | |
| k=16 + RMS restore | 0.276 | 0.560 | 0.61 | 0.65 |
| k=16 + RMS + last 64 tokens native | **0.183** | **0.180** | **0.675 (97%)** | **0.77 (96%)** |
| k=16 + RMS + last 256 tokens native | 0.118 | 0.110 | | |
| **4000-seq calibration**, k=24 + RMS | 0.219 | 0.512 | 0.605 | 0.61 |
| 4000-seq, k=24 + RMS + last 64 native | **0.144** | **0.160** | **0.685 (99%)** | **0.79 (98%)** |
| 4000-seq, k=24 + RMS + first 4 + last 256 native | 0.062 | 0.062 | | |

Measured TTFT (Modal H200, HF forwards, batch 1, dense k=16 mapper, last 64 tokens native):

| Prompt | Native 32B | 1.7B prefill | + mapper | Mapped TTFT | Speedup |
|---|---|---|---|---|---|
| 4K | 0.568 s | 0.047 s | 0.100 s | 0.157 s | 3.6x |
| 8K | 1.223 s | 0.099 s | 0.208 s | 0.288 s | 4.2x |
| 16K | 2.817 s | 0.237 s | 0.453 s | 0.579 s | 4.9x |
| 32K | 7.281 s | 0.641 s | 1.083 s | 1.310 s | **5.6x** |

The table above uses HF SDPA. Re-measured on Modal B200 with flash-attention-2 for both models and a block-accumulate mapper (addmm over source-layer views: no gather copies, FLOPs proportional to k):

| Prompt | Native 32B (HF+FA2) | Mapped TTFT | Speedup | Native vLLM 0.34 (B200) | Speedup vs vLLM |
|---|---|---|---|---|---|
| 4K | 0.345 s | 0.135 s | 2.5x | 0.222 s | 1.6x |
| 8K | 0.761 s | 0.202 s | 3.8x | 0.470 s | 2.3x |
| 16K | 1.833 s | 0.372 s | 4.9x | 1.025 s | 2.8x |
| 32K | 5.120 s | **0.829 s** | 6.2x | 2.484 s | 3.0x |

OneTriangle reports 7.34 s -> 0.87 s (~8.4x) on an H100 at 32K for this pair, against an HF native baseline. Our mapped time matches theirs: 0.83 s. A native baseline matters a lot. vLLM's native prefill is 2x faster than HF+FA2, and it is the fair baseline for a serving claim. ### Best serving result: vLLM 0.34 on an H200, fp8 dense mapper, Triton epilogue, k=24 maps from 4000 calibration sequences

| Prompt | Native vLLM | With connector | Speedup | $/1M prefill tok native -> mapped |
|---|---|---|---|---|
| 4K | 0.433 s | 0.120 s | 3.6x | $0.133 -> $0.037 |
| 8K | 0.940 s | 0.214 s | 4.4x | $0.145 -> $0.033 |
| 16K | 2.108 s | 0.432 s | 4.9x | $0.162 -> $0.033 |
| 32K | 5.159 s | **1.017 s** | **5.1x** | $0.199 -> $0.039 |

First-token quality against native vLLM on 16 FineWeb-Edu prompts: approximate KL 0.119, top-1 agreement 94%.

With a 1024-token native tail (quality-first), H200 TTFT is 0.296 s at 8K, 0.545 s at 16K and 1.188 s at 32K: 3.2x / 3.9x / 4.3x against native vLLM. At 32K the tail costs +0.16 s and cuts KL on books from 0.188 to 0.079.

The recommended default uses mixed web+book calibration maps. It serves at the same speed (32K: 1.029 s, 5.0x) with KL 0.125 and top-1 94% on the same prompts, and it is 42% better on 8K books.

### Inside vLLM 0.34 on an H200 at up to 32K (Modal; 1.7B source and dense mapper on the same GPU; k=16 maps from 500-sequence calibration)

| Prompt | Native vLLM | With connector | Speedup |
|---|---|---|---|
| 4K | 0.433 s | 0.122 s | 3.5x |
| 8K | 0.940 s | 0.230 s | 4.1x |
| 16K | 2.108 s | 0.471 s | 4.5x |
| 32K | 5.159 s | **1.095 s** | **4.7x** |

First-token quality on 16 prompts: approximate KL 0.133, top-1 agreement 87.5%. These maps use 500 calibration sequences; the 4000-sequence maps measured better in HF evaluation.

Prefill cost per 1M prompt tokens, at Modal's H200 price of $4.54/h, batch 1, from the TTFTs above:

| Prompt | Native vLLM | With connector | Cheaper by |
|---|---|---|---|
| 4K | $0.133 | $0.038 | 3.5x |
| 8K | $0.145 | $0.035 | 4.1x |
| 16K | $0.162 | $0.036 | 4.5x |
| 32K | $0.199 | $0.042 | 4.7x |

OneTriangle reports 8.4x at 32K for this pair. Their baseline is native HF forwards on an H100. Ours is native vLLM on the same H200, which is the harder bar.


### Against prefill/decode disaggregation (H200 numbers above, roofline transfer)

With native P/D disaggregation the target still prefills every prompt token, on a prefill GPU. It then ships 256 KB/token of 32B KV to the decode GPU. At 32K that is 8.4 GB: about 19 ms over NVLink (450 GB/s) or about 170 ms over 400 Gb/s InfiniBand. So disaggregation does not lower TTFT: 5.16 s + transfer. It only frees the decode GPU from prefill work. Cross-model transfer cuts the prefill itself: 1.02 s at 32K.

The two compose. The source prefill and mapper can run on a cheap prefill GPU and ship the 1.7B's KV, which is 112 KB/token, 2.3x less than the 32B's. The mapper then runs on the decode GPU. Our first connector version ran the source on a second GPU and copied mapped KV over PCIe; that copy cost 0.23 s at 16K. Shipping source KV instead, and mapping on the receiver, is the better split.

### Inside vLLM 0.34: `CrossModelConnector` (Vast 2x RTX PRO 6000, 1.7B + block mapper on GPU1, Qwen3-32B on GPU0)

Final configuration: the 1.7B, a dense one-GEMM mapper and Qwen3-32B all on one RTX PRO 6000. The connector writes mapped K/V straight into vLLM's paged blocks.

| Prompt | Native vLLM | With connector | Speedup |
|---|---|---|---|
| 4K | 0.804 s | **0.223 s** | **3.6x** |
| 8K | 1.607 s | **0.381 s** | **4.2x** |
| 16K | 3.683 s | **0.761 s** | **4.8x** |

The first-token quality check against native vLLM gives approximate KL 0.111 and 94% top-1 agreement.

Adding a fused Triton epilogue (k_norm RMS restore and RoPE in one pass) on top of the fp8 mapper gives 0.197 s / 0.339 s / 0.744 s at 4K / 8K / 16K. At that point the connector is GEMM-bound: the dense K+V map takes about 0.24 s at 16K and the 1.7B prefill about 0.17 s.

With an fp8 (e4m3, row-wise scaled) dense mapper, centered per request, TTFT is 0.204 s at 4K (3.9x), 0.350 s at 8K (4.6x) and 0.762 s at 16K (4.8x). Quality is unchanged: approximate KL 0.116, top-1 94%. The fp8 GEMM alone is 1.61x faster than bf16 on this card. At 16K, though, the per-layer RMS restore and RoPE passes, which are memory-bound, now limit the mapper. At 16K, the connector spends about 0.17 s on the source prefill and about 0.45 s on the dense mapper. vLLM spends about 0.08 s on the 64-token tail.

The first version ran the source on GPU1 with a per-layer mapper and copied over PCIe:

| Prompt | Native vLLM | With connector | Speedup |
|---|---|---|---|
| 4K | 0.804 s | 0.482 s | 1.67x |
| 8K | 1.607 s | 0.896 s | 1.79x |
| 16K | 3.683 s | 1.798 s | 2.05x |

Quality inside vLLM, on 16 FineWeb-Edu prompts of 961 tokens, against native vLLM:

- first-token approximate KL over the top-20 log-probabilities: 0.115;
- top-1 agreement: 94%.

The connector reports the mapped prefix to vLLM as an external KV hit. vLLM therefore prefills only the last 64 tokens natively, and the connector writes the mapped K/V into the paged blocks. Three costs are still large:

- the 1.7B prefill runs as an HF model;
- the mapper uses a Python loop over layers;
- every layer crosses PCIe from GPU1. At 32K the compute roofline caps the speedup at 10.9x. Two parts of our path run well above their floors:

- The mapper takes 0.44 s against a floor of about 0.25 s. The dense GEMM does 1.75x the needed FLOPs, and the RoPE handling adds overhead.
- The 64-token tail forward takes 0.23 s against a floor of about 20 ms.

### Qwen3-14B -> Qwen3-32B (NVIDIA paper's main pair; Vast H200 NVL, $3.27/h, run cost ~$0.70)

| Variant | KL mean | KL first | HellaSwag (200) | MMLU 5-shot (200) |
|---|---|---|---|---|
| Native 32B | 0 | 0 | 0.695 | 0.805 |
| Run only the 14B | 0.258 | 0.275 | 0.665 | 0.81 |
| Ridge k=8 (paper config) | 0.112 | 0.182 | | |
| Ridge k=8 + RMS key restore | 0.094 | 0.174 | | |
| Ridge k=16 + RMS restore | 0.096 | 0.214 | 0.69 | 0.81 |
| k=16 + RMS + last 64 tokens native | **0.063** | **0.069** | 0.70 | 0.85 |
| k=16 + RMS + last 256 tokens native | 0.037 | 0.041 | | |
| **2000-seq calibration**, k=16 + RMS | 0.079 | 0.142 | 0.67 | 0.815 |
| 2000-seq, k=16 + RMS + last 64 native | 0.052 | 0.052 | 0.69 (100%) | 0.835 |
| 2000-seq, k=16 + RMS + first 4 + last 64 native | **0.045** | **0.048** | | |

This matches the paper's retention: it reports 97.6% on average, and we measure 99% on HellaSwag and 100% on MMLU. MMLU on 200 items can't separate this pair, since the 14B alone scores as high as the 32B. KL gives the finer signal, and two additions cut it:

- RMS key restore cuts KL by 16%.
- Recomputing the last 64 prompt tokens natively cuts it by another third.

KL is KL(native || mapped) of the target's next-token distribution. The source model's cache maps a 960-token FineWeb-Edu prefix, and the target processes the 64-token continuation natively.

### Qwen3-1.7B -> Qwen3-8B (L40S, 500 x 1024 calibration, 64 eval sequences)

| Variant | KL mean | KL first token | Top-1 agree | Cost |
|---|---|---|---|---|
| Target sees no context | 1.049 | 6.66 | 0.622 | |
| Run only the 1.7B | 0.455 | 0.653 | 0.727 | |
| Ridge k=8 (NVIDIA paper) | 0.244 | 0.599 | 0.806 | $0.27 per run |
| Ridge k=8 + RMS key restore | **0.191** | 0.531 | 0.822 | same run |
| Exact K + mapped V (oracle) | 0.107 | 0.108 | 0.874 | |
| Mapped K + exact V (oracle) | 0.155 | 0.483 | 0.835 | |

- **RMS key restore:** puts back the target's per-token, per-head key RMS, which is the scale its `k_norm` would impose. This is our reading of "restore relative importances", and it cuts KL by 22%.
- **Values limit quality on this pair:** the exact-K + mapped-V and mapped-K + exact-V oracle rows show it. OneTriangle's blog reports the reverse for its pairs.
- **HellaSwag retention:** 92%.
- **MMLU 5-shot:** below the 1.7B alone when the target recomputes only the last token. The last-N native recompute run is next.
