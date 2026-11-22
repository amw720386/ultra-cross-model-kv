"""Source-model prefill through vLLM's own flash-attention varlen kernel (FA3 on Hopper, FA4 on Blackwell), registered as a
transformers attention implementation. Single sequence, causal, no padding (the connector's source prefill).
HF SDPA ran the 1.7B's 128K prefill at ~50% of H100 peak even with cuDNN; this is the kernel native vLLM uses."""
import torch

NAME = "vllm_fa"
_REGISTERED = False
# FP8 attention (FA3 on Hopper): q/k/v cast to e4m3 with per-KV-head scales, descaled in-kernel. Set by use_vllm_fa(fp8=).
FP8 = False
# sliding attention window for the SOURCE prefill (0 = full causal). Each source token attends to the last WINDOW tokens
# only: attention FLOPs drop from O(T^2) to O(T*W). The target still attends over the whole mapped prefix.
WINDOW = 0
# what the kernel was actually asked to do. Proof that a windowed arm ran windowed: windowed_calls == calls_over_window
# > 0 and window_sizes == [[WINDOW - 1, 0]] (a chunk with S <= WINDOW is full causal == window, so it passes no
# window_size). Default arm: windowed_calls == 0 unless the source has its own sliding layers (sliding_layer_calls > 0).
# Reset with reset_stats().
STATS = {"calls": 0, "windowed_calls": 0, "calls_over_window": 0, "sliding_layer_calls": 0, "window_sizes": [], "fp8_calls": 0}


def reset_stats():
    STATS.update(calls=0, windowed_calls=0, calls_over_window=0, sliding_layer_calls=0, window_sizes=[], fp8_calls=0)


def quant_e4m3(x, n_kv):
    """x [T, Hx, D] bf16 -> (e4m3 [T, Hx, D], descale [1, n_kv] fp32): one scale per KV head group (FA3's descale shape)."""
    T, Hx, D = x.shape
    g = x.view(T, n_kv, Hx // n_kv, D)
    amax = g.abs().amax(dim=(0, 2, 3)).float().clamp(min=1e-6)  # [n_kv]
    s = amax / 448.0
    q = (g / s.view(1, n_kv, 1, 1).to(g.dtype)).clamp(-448, 448).to(torch.float8_e4m3fn).view(T, Hx, D)
    return q, s.view(1, n_kv)


def _fa_func():
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    try:
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
    except Exception:
        get_flash_attn_version = None
    return flash_attn_varlen_func, get_flash_attn_version


def vllm_fa_attention(module, query, key, value, attention_mask=None, scaling=None, dropout=0.0, **kwargs):
    """query [B, Hq, T, D], key/value [B, Hk, S, D] -> (out [B, T, Hq, D], None). B must be 1; causal over S == T."""
    fn, ver = _fa_func()
    B, Hq, T, D = query.shape
    assert B == 1, "vllm_fa source attention supports one sequence"
    S = key.shape[2]
    q = query[0].transpose(0, 1).contiguous()  # [T, Hq, D]
    k = key[0].transpose(0, 1).contiguous()
    v = value[0].transpose(0, 1).contiguous()
    cu_q = torch.tensor([0, T], dtype=torch.int32, device=q.device)
    cu_k = torch.tensor([0, S], dtype=torch.int32, device=q.device)
    kw = {}
    if FP8:
        Hk = k.shape[1]
        q, kw["q_descale"] = quant_e4m3(q, Hk)
        k, kw["k_descale"] = quant_e4m3(k, Hk)
        v, kw["v_descale"] = quant_e4m3(v, Hk)
    # the layer's own attention options (Gemma: local sliding-window layers, logit softcap); HF passes them as kwargs
    sw = kwargs.get("sliding_window") or getattr(module, "sliding_window", None)
    sw = int(sw) if sw else 0
    win = min(w for w in (sw, WINDOW) if w) if (sw or WINDOW) else 0
    if win and S > win:
        kw["window_size"] = [win - 1, 0]
    STATS["calls"] += 1
    if WINDOW and S > WINDOW:  # calls the module window must cut (chunks with S <= WINDOW are full causal == window)
        STATS["calls_over_window"] += 1
    if sw:  # the layer's own sliding window (Gemma local layers); GLM-4.7-Flash has none
        STATS["sliding_layer_calls"] += 1
    if "window_size" in kw:
        STATS["windowed_calls"] += 1
        if kw["window_size"] not in STATS["window_sizes"]:
            STATS["window_sizes"].append(list(kw["window_size"]))
    if FP8:
        STATS["fp8_calls"] += 1
    sc = kwargs.get("softcap") or getattr(module, "attn_logit_softcapping", None)
    if sc:
        kw["softcap"] = float(sc)
    if ver is not None:
        v_ = ver(head_size=D)
        if v_ is not None:
            kw["fa_version"] = v_
    out = fn(q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=T, max_seqlen_k=S,
             softmax_scale=scaling if scaling is not None else D ** -0.5, causal=True, **kw)
    if isinstance(out, tuple):
        out = out[0]
    return out[None], None


def use_vllm_fa(model, fp8=False, window=0):
    """Register the kernel with transformers and switch `model` to it. Returns True on success.
    transformers builds no mask for a registered custom implementation (mask None), and FA aligns causal to the
    bottom right, so chunked prefill (q = a chunk, k = cache + chunk) is exact."""
    global _REGISTERED, FP8, WINDOW
    FP8 = bool(fp8)
    WINDOW = int(window or 0)
    from transformers import AttentionInterface
    if not _REGISTERED:
        AttentionInterface.register(NAME, vllm_fa_attention)
        _REGISTERED = True
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(NAME)
    else:
        model.config._attn_implementation = NAME
    return True
