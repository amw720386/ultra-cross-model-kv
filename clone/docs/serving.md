# Serving plan: cross-model KV transfer inside vLLM

## Goal

Measure time to first token (TTFT) and $ per 1M prefill tokens on H100 and H200. Compare three ways of serving the target model:

- native prefill;
- prefill/decode disaggregation (native prefill on a separate instance);
- mapped prefill: a small-model prefill, then the mapper, then the target recomputes only the last N prompt tokens.

## Shape: a vLLM v1 KV connector

vLLM v1 can load KV computed outside the engine into the paged KV buffer through a KV connector (`KVConnectorBase_V1`). LMCache and NIXL use this interface. A connector called `CrossModelConnector` would work like this:

1. **Scheduler side: `get_num_new_matched_tokens(request, num_computed)`**
   - It returns `P - N`, the prompt prefix the mapper will supply.
   - It does this when the prompt is at least `min_len` tokens long and a fitted map exists for this (source, target) pair.
   - vLLM then schedules only the last N tokens for native prefill.
2. **Scheduler side: `update_state_after_alloc` and `build_connector_meta`**
   - These record which block ids receive the mapped prefix.
3. **Worker side: `start_load_kv(forward_context)`**
   - The source model runs prefill on `prompt[:P-N]`. In the first version this is an HF model in the connector process; later it is a second vLLM engine.
   - The mapped K and V are written into the target's paged KV tensors at the block ids from step 2:
     - remove RoPE from the source keys;
     - run one GEMM per target layer;
     - apply the RMS restore;
     - apply the target's RoPE at absolute positions.
   - A fused Triton kernel does the gather, GEMM and RoPE per layer.
4. **Worker side: `wait_for_layer_load(layer)`**
   - Layer-wise pipelining: target layer l can start on the tail tokens once its mapped KV is written.

## Limits (H100 SXM: 989 TF dense bf16, 3.35 TB/s)

- **Mapper FLOPs:** `T × Lt × k·1024·1024 × 2 (K and V) × 2`. For Qwen3-32B at k=8 that is 2.1 GF per token, or 69 TF for a 32K prompt: 70 ms at peak, 6.7% of 1.7B→32B mapped TTFT.
- **Mapper bytes:**
  - Weights: k·1024·1024 per target layer, per K and per V, × 2 bytes × 64 layers = 2.1 GB at k=8, read once per request. Batching requests amortises this.
  - Source KV in plus target KV out: 32K × (28 + 64) × 2 KB × 2 = 12 GB. That is 3.6 ms per GB, about 43 ms at 32K.
- **Transfer under disaggregation:** the 32B KV is 256 KB per token, so 8.4 GB at 32K. That takes 19 ms over NVLink (450 GB/s) and 170 ms over 400 Gb/s InfiniBand. Transferring the source KV (28 layers × 2 KB × 2 = 112 KB/token) and mapping on the decode GPU cuts the link bytes by 2.3x.

## Order of work

1. HF-level TTFT benchmark (`kvx/ttft.py`): native vs source prefill + map + tail, at 4K–32K. This is the reference point; OneTriangle's benchmark rig also used HF forwards.
2. A Triton kernel per target layer that fuses gather, GEMM, RMS restore and RoPE, and writes into vLLM's paged layout.
3. The connector, with an HF source model in-process, on a single GPU.
4. Disaggregation: the source runs on a cheap GPU (L40S or L4), ships source KV over NIXL/LMCache, and the mapper runs on the target GPU.
