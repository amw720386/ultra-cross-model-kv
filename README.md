# ultra-cross-model-kv

**Prefill a 27B model with a 4B model.** `ultra-cross-model-kv` turns a Qwen3.5-4B prefill into a ready-to-decode
cache for Qwen3.8-27B. It covers both the full-attention KV *and* the Gated-DeltaNet recurrent states of these hybrid
models. The 27B then answers without ever prefilling the prompt itself.

On one GPU, measured end to end:

| prompt | native 27B TTFT | **ours (4B → 27B)** | speedup | OneTriangle (published) |
|---:|---:|---:|---:|---:|
| 8K | 0.94 s | **0.48 s** | 1.9× | 0.63 s (2.6×) |
| 32K | 4.27 s | **1.59 s** | 2.7× | 2.81 s (2.8×) |
| 64K | 10.5 s | **4.20 s** | 2.5× | 6.33 s (3.0×) |
| 256K | 77.3 s | **10.7 s** | **7.2×** | 48.1 s (3.6×)¹ |
| 512K | 236.7 s | **19.9 s** | **11.9×** | 157.4 s (3.7×)¹ |
| 1M | 833.2 s | **39.5 s** | **21.1×** | 560.2 s (3.9×)¹ |

¹ OneTriangle's values above 64K are projections; ours are measured. ≤64K rows: 1×H100 80GB; ≥256K rows: 1×H200.
Rows ≥256K use the sliding-window source mode (below).

**Quality** (4B → 27B, held-out FineWeb-Edu, native 27B as reference):

| | next-token agreement | KL to native | MC retention² |
|---|---:|---:|---:|
| Qwen3.5-4B alone | 74.0% | 0.368 | 86.3% |
| ours, ridge maps, tail 1 | 88.8% | 0.088 | 82.9% |
| ours, ridge maps, tail 32 | 89.9% | 0.073 | **96.6%** |
| ours, 32K prompt, long-context maps | **93.0%** | **0.044** | – |

² Mean chance-normalized accuracy retention, (ours − chance)/(native − chance), over HellaSwag, PIQA, ARC-Challenge,
MMLU and WinoGrande (250 questions each). This is the same retention metric OneTriangle reports (85.1% in their
V0 report, on an easier 0.6B → 1.7B pair).

## How it works

Qwen3.5 / Qwen3.8 are hybrid models. Three of every four layers are Gated DeltaNet (a linear-attention RNN with a
128×128 state per head); the fourth is gated full attention. A KV-only transfer cannot work for them, since most
of the "cache" is recurrent state. We synthesize every piece the target needs:

1. **Source prefill.** The 4B runs over the prompt. Its normalized residual stream at each layer is the only thing we
   read from it.
2. **Ridge maps, one per target layer.** Each map goes from a depth-matched 4B layer (chosen by fit) to:
   - *full-attention layers:* the 27B's per-token `k` (post-norm, pre-RoPE) and `v`. RoPE is applied with the
     target's own positions, so the map is position-free.
   - *DeltaNet layers:* the 27B's raw `k`, `v` projections and the gate inputs `a`, `b`. They then go through
     the **27B's own** causal conv and gated delta-rule scan, which yields exactly the recurrent state the 27B
     expects. Only the last 3 tokens of `q` are mapped, to fill the conv history.
3. **Tail repair.** The 27B natively processes the last *N* prompt tokens on top of the synthesized cache (N = 1…128)
   and emits the first token.
4. **Distillation (optional).** The maps are fine-tuned end to end. The loss is KL(native 27B ‖ transferred 27B) on
   continuation tokens, with gradients flowing through the frozen 27B's attention and DeltaNet scans.

The target never runs its MLPs or attention over the prompt. Its prefill reduces to GEMMs plus linear-time scans,
fused into the source's forward pass, so nothing moves between GPUs.

**Sliding-window source mode.** At long context the 4B's own attention becomes the bottleneck. The source can run
its 8 attention layers with a sliding window while its DeltaNet layers still see the whole prompt. Source cost is
then linear in prompt length, while the 27B still attends over the full synthesized KV at decode. Maps for this mode
are calibrated on 32K-token windowed prefills.

## Reproduce

All GPU work runs on Modal (`modal_jobs/lab.py`); models live in a Modal volume.

```bash
cd modal_jobs
PYTHONPATH=../src modal run prefetch.py
PYTHONPATH=../src modal run lab.py --task selftest --gpu L40S
PYTHONPATH=../src modal run lab.py --task calibrate --gpu H200 --minutes 60 --args '{"n_seqs":256,"seq_len":4096}'
PYTHONPATH=../src modal run lab.py --task quality --gpu H100 --minutes 90 --args '{"mapper_name":"4b-27b-r2-n256x4096","mc_limit":250}'
PYTHONPATH=../src modal run lab.py --task bench --gpu H100 --minutes 30 --args '{"mapper_name":"4b-27b-r2-n256x4096","lengths":[8192,32768,65536]}'
```

Raw outputs of every run quoted here are in `results/`.

## Notes on method

- **Baselines.** OneTriangle times native prefill with Hugging Face forwards on H100s (2×H100 at 32K). Our "native"
  column is the same 27B on our own engine (FlashAttention-2 + fla kernels, 8K chunked prefill, one GPU), which is
  1.5–2.6× faster than their native times. Our speedup ratios are therefore against a much stronger baseline.
  Absolute TTFT is the apples-to-apples column.
- **Beyond 262K.** Qwen3.8-27B's native context is 262K. The 512K/1M timings use plain RoPE and measure speed
  only; long-context *quality* claims here stop at 32K.
- **Retrieval.** Needle-in-a-haystack at 32K is 100% for native 27B. With a full-attention source we get 90–95%;
  with the sliding-window source, 65–85% depending on tail length. Window mode trades exact recall for speed.
- **MC retention with tail repair.** MC prompts are short (often under 100 tokens), so a 32-token tail covers a
  sizable fraction of them natively. Tail-1 numbers are the pure-transfer result.
- bf16 everywhere; medians of 3 runs ≤64K, a single run ≥256K.
