# ultra-cross-model-kv

**Prefill a 27B model with a 4B model, and get its first token up to 21× sooner at 1M context.**

`ultra-cross-model-kv` turns a Qwen3.5-4B prefill into a ready-to-decode cache for Qwen3.8-27B. It covers the
full-attention KV *and* the Gated-DeltaNet recurrent states of these hybrid models. The 27B then answers without
ever prefilling the prompt itself. It is a drop-in, single-GPU path: no second GPU, no cache transfer over the wire.

## Time to first token (Qwen3.5-4B → Qwen3.8-27B)

One configuration throughout (distilled long-context maps, 8K source window, 128-token tail), measured end to end:

| prompt | native 27B | **ours** | speedup | OneTriangle native → transfer (speedup) |
|---:|---:|---:|---:|---:|
| 8K | 0.96 s | **0.53 s** | 1.8× | 1.61 → 0.63 s (2.6×) |
| 16K | 1.98 s | **0.81 s** | 2.4× | 3.37 → 1.25 s (2.7×) |
| 32K | 4.34 s | **1.44 s** | **3.0×** | 7.94 → 2.81 s (2.8×) |
| 64K | 10.0 s | **2.97 s** | **3.4×** | 18.74 → 6.33 s (3.0×) |
| 128K | 25.5 s | **5.07 s** | **5.0×** | 54.87 → 16.41 s (3.3×)¹ |
| 256K | 74.7 s | **10.2 s** | **7.4×** | 171.4 → 48.1 s (3.6×)¹ |
| 512K | 243.1 s | **20.3 s** | **12.0×** | 589.3 → 157.4 s (3.7×)¹ |
| **1M** | 873.2 s | **40.7 s** | **21.5×** | 2164.7 → 560.2 s (3.9×)¹ |

≤64K: 1×H100 80GB, median of 3. ≥128K: 1×H200, single run (128K median of 3). OneTriangle numbers are from
their published 4B → 27B chart (H100, HF forwards). ¹ Their values above 64K are projections; ours are measured.

**Our time to first token beats theirs at every length, by 1.2× at 8K and 14× at 1M.** Our speedup ratio is higher
from 32K on, even though our native baseline is 1.7–2.5× faster than theirs.

## Quality

Native Qwen3.8-27B is the reference. Agreement means the transferred model picks the same next token as native 27B.

**Short prompts** (2K-token held-out FineWeb-Edu prompts, 256-token continuations; MC = 250 questions each from HellaSwag,
PIQA, ARC-Challenge, MMLU and WinoGrande):

| | next-token agreement | KL to native | perplexity ratio | MC retention² |
|---|---:|---:|---:|---:|
| Qwen3.5-4B alone | 73.7% | 0.368 | 1.28 | 87.0% |
| ours: ridge maps, tail 1 | 88.8% | 0.088 | 1.06 | 82.9% |
| **ours: distilled maps, tail 1** | **92.2%** | **0.040** | **1.02** | **88.8%** |
| **ours: distilled maps, tail 32** | **93.2%** | **0.032** | **1.01** | **97.1%** |

**Long prompts** (32K tokens, the configuration in the TTFT table):

| | next-token agreement | KL to native | needle-in-a-haystack, 1 key / 4 keys |
|---|---:|---:|---:|
| native 27B | 100% | 0 | 100% / 100% |
| Qwen3.5-4B alone | 73.3% | 0.353 | 100% / 100% |
| **ours** | **91.7%** | **0.053** | **95% / 100%** |

² Chance-normalized retention, (ours − chance)/(native − chance), averaged over the five tasks. This is the metric
OneTriangle reports in their V0 study: 85.1% retention, 61% next-token agreement and a 1.14 perplexity ratio, on
the much closer 0.6B → 1.7B pair. They have not published quality numbers for 4B → 27B.

## How it works

Qwen3.5 and Qwen3.8 are hybrid models. Three of every four layers are Gated DeltaNet (a linear-attention RNN with a
128×128 state per head); the fourth is gated full attention. Most of their "cache" is recurrent state, so a KV-only
transfer cannot work. We synthesize every piece the 27B needs:

1. **Source prefill.** The 4B runs over the prompt. We read only its normalized residual stream.
2. **Per-layer maps.** For each 27B layer, a linear map from a depth-matched 4B layer (picked by fit) predicts what
   that layer caches:
   - *attention layers:* per-token `k` (post-norm, pre-RoPE) and `v`. The 27B's own RoPE is applied afterwards, so
     the map is position-free.
   - *DeltaNet layers:* the raw `k`, `v` projections and the gate inputs `a`, `b`. They then run through **the 27B's
     own** causal conv and gated delta-rule scan, which gives exactly the recurrent state the 27B expects.
3. **Tail repair.** The 27B natively runs the last *N* prompt tokens on top of the synthesized cache and emits the
   first token.
4. **Fit, then distill.** Maps start as closed-form ridge regressions on 1M calibration tokens. They are then fine-tuned
   end to end on KL(native 27B ‖ transferred 27B), with gradients through the frozen 27B's attention and DeltaNet
   scans. The long-context maps also see synthetic record-lookup prompts, which taught them to preserve retrievable
   facts. The needle eval uses a different template, filler and answer length.

The 27B never runs its MLPs or attention over the prompt. Its prefill reduces to fused GEMMs and linear-time scans
inside the 4B's forward pass.

**Sliding-window source.** At long context, the 4B's own attention becomes the bottleneck. The source runs its 8
attention layers with an 8K window while its DeltaNet layers still see the whole prompt. Source cost becomes linear
in prompt length, while the 27B still attends over the full synthesized KV. This is where the 5–21× speedups at
128K–1M come from.

## Layout

```
src/xkv/qwen35.py       lean hybrid forward: chunked prefill, capture hooks, sliding-window ring cache, fused conv,
                        state-only delta-rule scan
src/xkv/mapper.py       ridge statistics, per-layer maps, transfer prefill (differentiable for distillation)
src/xkv/evals.py        MC tasks (lm-eval formats), needle-in-a-haystack, held-out text splits
src/xkv/experiments.py  selftest, calibrate, distill, quality, longctx, bench, profile
modal_jobs/             Modal image + one entrypoint per GPU type
results/                outputs of every run quoted above
```

## Reproduce

All GPU work runs on Modal. From `modal_jobs/`:

```bash
PYTHONPATH=../src modal run prefetch.py
PYTHONPATH=../src modal run lab.py --task selftest --gpu L40S
PYTHONPATH=../src modal run lab.py --task calibrate --gpu H200 --minutes 60 --args '{"n_seqs":256,"seq_len":4096}'
PYTHONPATH=../src modal run lab.py --task calibrate --gpu H200 --minutes 60 --args '{"n_seqs":32,"seq_len":32768,"window":8192}'
PYTHONPATH=../src modal run lab.py --task distill --gpu H200 --minutes 100 --args '{"mapper_name":"4b-27b-r2-n32x32768-w8192","out_name":"4b-27b-w8192-distill-ret","steps":500,"prefix_choices":[4096,8192,12288],"prefix":12288,"tail":64,"eval_tail":128,"window":8192,"retrieval_frac":0.5}'
PYTHONPATH=../src modal run lab.py --task bench --gpu H200 --minutes 70 --args '{"mapper_name":"4b-27b-w8192-distill-ret","lengths":[131072,262144,524288,1048576],"reps":1,"tail":128,"window":8192}'
```

Short-prompt distillation: `--task distill` with `{"mapper_name":"4b-27b-r2-n256x4096","steps":1500,"prefix_choices":[512,1024,2048,4096],"tail":1}`.

## Notes on method

- **Baselines.** Our native column is the same 27B on our own engine: FlashAttention-2 + fla kernels, 8K chunked
  prefill, one GPU. It matches Hugging Face logits (99.6% top-1, `results/selftest_parity.json`). OneTriangle's
  native column uses HF forwards (2×H100 at 32K), so absolute TTFT is the apples-to-apples comparison.
- **Beyond 262K.** Qwen3.8-27B's native context is 262K. The 512K/1M rows measure speed only (plain RoPE, random
  tokens); quality claims stop at 32K.
- **Map choice.** The short-prompt quality rows use maps distilled on 0.5–4K prompts. The long-context map used for
  TTFT scores a little lower on short prompts (91.0% agreement, 85.4% MC retention at tail 1; 96.7% at tail 32;
  `results/quality_2k_longctx_map.json`).
- **MC with tail repair.** MC prompts are short, so a 32-token tail covers much of each prompt natively. Tail 1 is
  the pure-transfer number. On MMLU specifically, tail-1 transfer still trails the 4B alone.
- **Sample sizes.** LM agreement uses 16 prompts at 2K (4,096 scored tokens) and 6 at 32K. Needle results are n=20
  per setting at 32K.
- bf16 throughout.
