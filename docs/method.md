# Method

## How it works

Qwen3.5 and Qwen3.8 are hybrid: three of every four layers are Gated DeltaNet (a linear-attention RNN with a
128×128 state per head), the fourth is gated full attention. Most of the "cache" is recurrent state, so a KV-only
transfer cannot work. Every piece the 27B needs is synthesized:

1. **Source prefill.** The 4B runs over the prompt; only its normalized residual stream is read.
2. **Per-layer maps.** For each 27B layer, a linear map from a depth-matched 4B layer predicts what it caches:
   - *attention layers:* per-token `k` (post-norm, pre-RoPE) and `v`, from several 4B layers plus the token
     embedding (`src/xkv/attnmap.py`); the 27B's own RoPE is applied afterwards.
   - *DeltaNet layers:* the raw `k`, `v` projections and gate inputs `a`, `b`, run through **the 27B's own** causal
     conv and gated delta-rule scan to produce the recurrent state.
3. **Native tail.** The 27B runs the last prompt tokens natively on the synthesized cache and emits the first token:
   512 tokens for prompts ≤ 4K, tapering to 128 at ≥ 8K. Optional *native spans* compute selected regions in place
   (tool schemas, the newest agent turn).
4. **Recall mode (`+repN`).**
   1. The source reads the tail over its own cache.
   2. Each 64-token block of the mapped prefix is scored by the source's attention from the tail (max over tail
      queries, summed over heads and 4B attention layers up to the deepest mapped one; first 4 positions excluded).
   3. The 27B re-encodes the top-N tokens natively in one batched pass: absolute-position causal attention over the
      full cache; each span's DeltaNet scan starts from the mapped end-of-prefix state (variable-length kernels).
   4. Only the attention KV it writes is kept.

   `+repNm` uses whole chat messages instead of blocks: a message scores its best block; selecting one also takes
   the next message (request + answer); messages longer than N/2 fall back to blocks; units are taken greedily.
5. **Fit, then distill.** Maps start as closed-form ridge regressions on 1M calibration tokens, then are fine-tuned
   on KL(native 27B ‖ transferred 27B) with gradients through the frozen 27B. Data: FineWeb-Edu, synthetic record
   lookups, synthetic tool-use dialogues, LongAlign-10k, Tulu-3 SFT. Not decontaminated against the eval sets.

**Engine.** Fused Triton causal-conv kernel, state-only delta-rule scan, CUDA-graphed persistent tail (≤ 32K),
CUDA-graph batched decoding with per-sequence lengths, source-drafted speculative decoding with exact recurrent-state
rollback (rejected tokens re-scanned with β = 0, g = 0). fla's norm kernels re-autotune for every new sequence
length; freezing those keys made generation evals ~4× faster.

## Modes

| | **fast mode** (default) | **recall mode** (`+repN`) |
|---|---|---|
| 27B computes natively | last 128 prompt tokens (512 for ≤ 4K, tapering to 128 at 8K) | the same + N tokens chosen by the source's attention |
| TTFT speedup 8K / 16K / 32K, 4B → 27B | 2.4–2.8× / 2.8–3.0× / 2.95–3.1× | 0.97–2.55× depending on N |
| TTFT speedup 8K / 16K / 32K, 2B → 27B | 3.94× / 4.47× / 4.94× | `+rep1024`: 2.03× / 2.85× / 3.57× |
| use for | everything unless labelled | exact recall |

Agents use two further settings; see [agents.md](agents.md#settings).

## Glossary

| term | meaning |
|---|---|
| **transfer / native** | *Transfer*: the 4B reads the prompt, the 27B answers from the mapped cache. *Native*: the 27B reads the prompt itself. |
| **mapped** | A prompt token whose 27B cache entries are predicted from the 4B, not computed by the 27B. |
| **adaptive tail** | The fast-mode native tail: 512 tokens for prompts ≤ 4K, tapering to 128 at ≥ 8K. Some rows use a fixed 128-token tail and say so. |
| **repair, `+repN`, `+repNm`** | Recall mode: the 27B re-encodes N tokens picked by the 4B's attention, as 64-token blocks (`+repN`) or whole messages (`+repNm`). |
| **native span** | A region the 27B computes natively in place, e.g. the `<tools>` block. |
| **graphed tail** | The native tail as a CUDA graph, captured once per prompt length. Pays off when lengths repeat; costs time when they vary ([speed.md](speed.md#graphed-tail-with-varying-prompt-lengths)). |
| **Gated DeltaNet state** | The recurrent state three of every four layers keep instead of a KV cache. |
| **fidelity / phase-2 / agent map** | *Fidelity map* (`4b-27b-fid`, `2b-27b-fid`): current, used unless stated. *Phase-2 map* (`4b-27b-chat`, `2b-27b-chat`): earlier chat-distilled maps. *Agent map* (`4b-27b-fid-agent`): fidelity map fine-tuned with τ2 airline traces. |
| **lighter agent setting / agent setting** | τ2 / Terminal-Bench configurations ([agents.md](agents.md#settings)). |
| **policy-deleted control** (`target-drop`) | Native 27B sees only what the agent setting computes natively; the policy and older history are deleted. |

## Evaluation caveats

- **Configuration was chosen on evaluation items.** Treat these as development-set results:
  - full 4B attention up to 32K, the phase-2 map, the free-form code prompt → LongHealth, ContractNLI, HumanEval;
  - fidelity maps → RULER, MRCR, ZebraLogic subsets;
  - adaptive tail → ZebraLogic, MBPP;
  - recall mode and its size → 2B RULER / MRCR / LongBench v2 and earlier τ2 runs;
  - agent setting → τ2 telecom replay and official telecom;
  - lighter agent setting and its map → pre-registered, scored once; airline is in-sample for that map.

  Guards: the [held-out fidelity gate](benchmarks.md#held-out-fidelity-gate) and the held-out MRCR sample.
- **Mixed configurations.** No single configuration wins everywhere: recall mode buys RULER and MRCR, the agent
  setting buys τ2, both at a TTFT cost.
- **Harness.** Every benchmark was reimplemented (`src/xkv/genevals.py`); OneTriangle's prompts and scorers are not
  public. Code benchmarks use a free-form "answer in a code block" prompt (a forced code prefix cost native 6 points
  on HumanEval); MBPP shows its asserts; MATH-500 allows 3072 new tokens (8192 thinking; ZebraLogic 12288).
- **Greedy continuations diverge.** Teacher-forced next-token agreement is 90–93%, but free-running outputs drift
  apart early. Task scores are the meaningful measure.
- **Parity** against Hugging Face (99.6% top-1, `selftest_parity.json`) was checked on 0.8B → 2B, not on the 27B.
- **Timing.** The transfer reuses a preallocated graphed tail; native allocates its cache inside the timed region
  (< 20 ms). Same engine and GPU for both arms.
- bf16 throughout, no quantization. Differences of 2–3 points (13 on τ2) are within noise.

## Known limitations

- **Agents with long rule-heavy policies.** In the lighter setting the transferred τ2 telecom policy misleads the
  agent (0.22, below the 0.44 policy-deleted control). Matching native needs the agent setting, which is slower than
  native. Terminal-Bench: 0.10 vs native 0.70.
- **Exact recall beyond 32K** is lost (8K sliding window in the source).
- **Graphed tail with varying prompt lengths** costs ~0.5 s per request unless turned off or bucketed.
- **Some wins need recall mode** (2B pair): RULER and LongBench v2 use `+rep1024` (2.0–3.6×), MRCR `+rep3072m`
  (1.4–2.8×), instead of 3.9–4.9× in fast mode.

## Next steps

- **Keep a garbled policy from misleading:** leave out or down-weight attention to the mapped policy span (would
  have to be disclosed as "policy not transferred").
- **Prefix-cache the policy** per domain. Native serving gets the same cache, so measure before claiming a speedup.
- **Train the attention map on varied agent data** (public multi-turn tool-use sets, τ2 held out).
- **Thinking mode for τ2**, to close the gap between our native (0.68–0.70) and theirs (0.86).
- **Terminal-Bench turn cost:** reuse the previous turn's transferred cache and map only new tokens.
