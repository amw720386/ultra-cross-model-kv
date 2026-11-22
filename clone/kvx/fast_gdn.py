"""Fast GatedDeltaNet for HF Qwen3.5 / Qwen3-Next SOURCE prefills inside the vLLM image.

transformers 5.22 routes the chunked gated delta rule through a kernels-hub decorator with a pure-torch fallback; in the
vLLM image neither the hub nor `fla` resolves, so HF runs torch_chunk_gated_delta_rule: 12K small bmm/triangular-solve
calls per 8K-token Qwen3.5-4B prefill (0.74 s on an H100, ~5x the FLOP floor). vLLM 0.34 ships the same FLA Triton
kernel under vllm.third_party; patch() points HF's module-level function at it (bf16 q/k/v as the kernel requires; the
final state, if requested, is returned in HF/fla orientation [B, H, dk, dv]: vLLM's kernel returns [B, H, dv, dk]).
"""
import torch

_PATCHED = []


def _vllm_kernel():
    for mod in ("vllm.third_party.flash_linear_attention.ops", "vllm.model_executor.layers.fla.ops", "fla.ops"):
        try:
            return __import__(mod, fromlist=["chunk_gated_delta_rule"]).chunk_gated_delta_rule
        except (ImportError, AttributeError):
            continue
    return None


def patch(modules=("transformers.models.qwen3_5.modeling_qwen3_5", "transformers.models.qwen3_next.modeling_qwen3_next",
                   "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")):
    """Replace torch_chunk_gated_delta_rule in the given HF modules. Returns the patched module names."""
    k = _vllm_kernel()
    if k is None or not torch.cuda.is_available():
        return []
    import importlib

    def fast(query, key, value, g, beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False,
             cu_seqlens=None, **kw):
        dt = torch.bfloat16
        o, S = k(q=query.to(dt).contiguous(), k=key.to(dt).contiguous(), v=value.to(dt).contiguous(),
                 g=g.contiguous(), beta=beta.contiguous(), initial_state=initial_state,
                 output_final_state=output_final_state, cu_seqlens=cu_seqlens,
                 use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
        if S is not None:
            S = S.transpose(-1, -2)
        return o.to(query.dtype), S

    out = []
    for name in modules:
        try:
            m = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(m, "torch_chunk_gated_delta_rule"):
            m.torch_chunk_gated_delta_rule = fast
            out.append(name)
    _PATCHED[:] = out
    return out
