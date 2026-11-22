# New pairs: tap maps for hybrid, MLA, Gemma 4 and cross-tokenizer targets (seat kvx-pairs)

`kvx/core.py` maps per-token K/V of plain attention models. The pairs OneTriangle publishes now include targets whose
cache isn't a per-token K/V:
- Qwen3.8-27B: 48 GatedDeltaNet layers that carry a conv state and a recurrent state, plus 16 attention layers.
- GLM-5.3: an MLA latent plus a DSA indexer key.
- Gemma 4: sliding and global layers with different head sizes, KV-shared layers, and K = V on global layers.

`kvx/hybrid.py` handles all of them the same way.

## Method

A target's cache is a function of a few per-token module outputs ("taps"):

| layer kind | taps (per token) | cache built by the model from them |
|---|---|---|
| attention | `k_norm` output (else `k_proj`), pre-RoPE; `v_norm` output (Gemma 4, else `v_proj`) | RoPE(k) incl. partial rotary and YaRN m, v |
| GatedDeltaNet | `in_proj_qkv` (pre-conv q,k,v), `in_proj_a`, `in_proj_b` | conv state (last 4 pre-conv rows), recurrent state (gated delta rule) |
| MLA (+DSA) | `kv_a_proj_with_mqa` (latent + rope key, pre-norm), `indexer.k_norm` | normalized latent, RoPE key, indexer key |

1. Calibration records the taps of both models: mix 2000 FineWeb-Edu + 2000 PG19-train × 1024 tokens, positions with stride 4.
2. A ridge map is fit per target layer (or per K/V unit with `split_kv`) from a window of source layers:
   - `w` nearest by relative depth (`w=0`: all), plus the nearest source layer of the same kind;
   - one Gram per distinct window, Cholesky in fp64, `lam_rel` 1e-3;
   - optional recolor: the per-128-channel target covariance is restored after the ridge shrinkage.
3. To build a mapped cache, the target runs over the prefix with hooks that replace its taps at positions >= A (the
   native sink tokens). The model's own code then builds the cache, and only the cache is kept. The continuation, or
   a native tail, runs on it.

   A serving kernel computes the same cache from the mapped taps without the target's MLPs. That's kvx-quality's
   vLLM state writer for the hybrid pair.

## Files (Modal volume `kvx`, `/vol/runs/<run_id>/`)

- `hybrid_maps.pt`, format `kvx-hybrid-v1` (agreed with kvx-quality 2026-11-22): per target layer `sel`, `W` bf16,
  `b`, `mu`, `rc` (recolor), `r2`, `lam_eff`; `tgt_taps`, `lin_dims`, `sinks`, `sink_template`.
  Loader: `hybrid.load_maps_file`. CPU fixture for writers: `tests/fixtures/hybrid_fixture.pt`.
- `split_kv` runs: `hybrid_units.pt` (unit-keyed). For attention-only targets also `maps_core.pt` in the
  `modal_fit_eval.py` `maps_k` format, so `kvx/vllm_connector.py` can serve them.
- Qwen3 ladder (`modal_fit_eval.py`, kv features, rms restore): `maps_k24.pt`, `maps_all/maps_k{4,8,12,16}.pt`, `sink_template.pt`.

## Maps that exist (KL(native || mapped), 64 FineWeb-Edu docs, P=960, 64-token continuation)

| run | pair | k24 | k8 | tmpl4 + tail256 | source alone | GPU, $ |
|---|---|---|---|---|---|---|
| pairs-q17-q4 | Qwen3-1.7B -> 4B | .130 | .156 | .047 | .369 | A100, .62 |
| pairs-q17-q8 | Qwen3-1.7B -> 8B | .153 | .185 | .051 | .456 | H100, .51 |
| pairs-q17-q14 | Qwen3-1.7B -> 14B | .192 | .231 | .056 | .567 | A100, .99 |
| pairs-q17-q32 | Qwen3-1.7B -> 32B | .228 | .264 | .073 | .752 | H200, 1.30 |

Hybrid, Gemma 4, Mistral-Medium and GLM fits are queued.

## Gotchas found

- YaRN: HF and vLLM multiply cos/sin by the attention factor m. `core.unrope` now divides by cos²+sin² (49c94fb).
- Qwen3.5 k_norm is `1 + w`, and some channels are exactly 0, so `restore_rms` masks them (the smoke was NaN before).
- HF's DSA (GLM-5.3) gives different results for a one-shot forward and a cached two-step forward, so evals report a `ref_native_twostep` floor.
- TF32 Gram accumulation isn't PSD-preserving: the Cholesky failed on 4B->27B. Grams are fp32.
- Gemma 4 VLM checkpoints need `attn_implementation` per sub-config, because the audio tower rejects sdpa.
- Ministral-3-3B and Mistral-Medium-3.5 tokenizers differ only in 4 unused special ids; Gemma 4 E2B vs 31B differ in 1 glitch token.
