"""Target cache from mapped taps for hybrid GatedDeltaNet targets (Qwen3.5 / Qwen3.8), without running the target.

kvx-pairs' maps (kvx-hybrid-v1, kvx/hybrid.py) predict per-token target "taps" for the mapped prefix:
  attention layer: k (pre-RoPE, post k_norm) and v;  GatedDeltaNet layer: qkv (pre-conv, q|k|v), a, b.
This module turns those taps into exactly what the target's own forward would have cached (HF Qwen3_5 semantics):
  attention: K = partial RoPE(k) at positions 0..T-1, V = v                         -> paged KV blocks
  lin:       conv_state = last kernel-1 pre-conv qkv rows (vLLM keeps kernel-1; HF keeps kernel)
             recurrent_state = gated delta rule over all T tokens on conv-ed, l2-normed k and v (fp32)
Only a few small target parameters are needed (conv1d weight/bias, A_log, dt_bias per lin layer; rotary config),
read from the target checkpoint (TargetParams.from_state_dict). The serving writer (vllm_connector) calls
attn_kv / lin_states and copies the results into vLLM's caches; tests/test_hybrid_state_cpu.py checks them
against HF's cache on kvx-pairs' fixture.
"""
import math

import torch
import torch.nn.functional as F


class TargetParams:
    """Per-layer small params of a Qwen3.5-style target. cfg: text config dict (to_dict()) or object."""

    def __init__(self, cfg, lin, attn_head_dim):
        g = (lambda k, d=None: cfg.get(k, d)) if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
        rp = g("rope_parameters") or g("rope_scaling") or {}
        self.theta = float(rp.get("rope_theta") or g("rope_theta") or 10000.0)
        self.partial = float(rp.get("partial_rotary_factor") or g("partial_rotary_factor") or 1.0)
        self.rope_type = rp.get("rope_type", rp.get("type", "default")) or "default"
        if self.rope_type not in ("default", "mrope"):
            raise NotImplementedError(f"rope_type {self.rope_type}: only plain RoPE (text mrope == RoPE) here")
        self.head_dim = int(attn_head_dim)
        self.Hk, self.Hv = int(g("linear_num_key_heads")), int(g("linear_num_value_heads"))
        self.dk, self.dv = int(g("linear_key_head_dim")), int(g("linear_value_head_dim"))
        self.kernel = int(g("linear_conv_kernel_dim"))
        self.layer_types = list(g("layer_types"))
        self.lin = lin  # {layer: dict(conv_w [C, K], conv_b [C] or None, A_log [Hv], dt_bias [Hv])}

    @classmethod
    def from_state_dict(cls, cfg, sd, prefix="model.layers."):
        """sd: target state dict (or a lazy mapping name -> tensor) with HF Qwen3_5 names."""
        g = (lambda k, d=None: cfg.get(k, d)) if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
        lin = {}
        for l, t in enumerate(g("layer_types")):
            if t != "linear_attention":
                continue
            p = f"{prefix}{l}.linear_attn."
            w = sd[p + "conv1d.weight"].float()
            lin[l] = dict(conv_w=w.reshape(w.shape[0], -1), conv_b=sd[p + "conv1d.bias"].float() if (p + "conv1d.bias") in sd else None,
                          A_log=sd[p + "A_log"].float(), dt_bias=sd[p + "dt_bias"].float())
        hd = g("head_dim") or g("hidden_size") // g("num_attention_heads")
        return cls(cfg, lin, hd)

    def to(self, device):
        for d in self.lin.values():
            for k, v in list(d.items()):
                if v is not None:
                    d[k] = v.to(device)
        return self


# ---------------------------------------------------------------- attention layers

def rope_cos_sin(p, T, device, dtype=torch.float32, offset=0):
    rd = int(p.head_dim * p.partial)
    inv = 1.0 / (p.theta ** (torch.arange(0, rd, 2, device=device, dtype=torch.float32) / rd))
    pos = torch.arange(offset, offset + T, device=device, dtype=torch.float32)
    f = torch.outer(pos, inv)
    emb = torch.cat([f, f], -1)
    return emb.cos().to(dtype), emb.sin().to(dtype), rd


def apply_partial_rope(k, cos, sin, rd):
    """k [T, H, D] pre-RoPE -> RoPE on the first rd dims (rotate_half within them), rest passed through."""
    kr, kp = k[..., :rd], k[..., rd:]
    h = rd // 2
    rot = torch.cat([-kr[..., h:], kr[..., :h]], -1)
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.cat([kr * c + rot * s, kp], -1)


def attn_kv(p, k_tap, v_tap, offset=0):
    """k_tap, v_tap [T, H*D] (mapped, k already rms-restored) -> K post-RoPE, V as [T, H, D] float32."""
    T = k_tap.shape[0]
    D = p.head_dim
    k = k_tap.float().view(T, -1, D)
    v = v_tap.float().view(T, -1, D)
    cos, sin, rd = rope_cos_sin(p, T, k.device, offset=offset)
    return apply_partial_rope(k, cos, sin, rd), v


# ---------------------------------------------------------------- GatedDeltaNet layers

def causal_conv_silu(x, w, b=None):
    """x [T, C] pre-conv, w [C, K] depthwise causal conv (HF Conv1d weight [C,1,K] reshaped) -> silu(conv) [T, C]."""
    T, C = x.shape
    K = w.shape[1]
    y = F.conv1d(F.pad(x.t()[None], (K - 1, 0)), w[:, None, :], b, groups=C)[0].t()
    return F.silu(y)


def _l2n(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def delta_rule_state_ref(k, v, beta, g, S0=None):
    """Sequential gated delta rule (fp32). k [T,H,dk] (l2-normed), v [T,H,dv], beta/g [T,H]. Returns S [H, dk, dv]."""
    T, H, dk = k.shape
    dv = v.shape[-1]
    S = torch.zeros(H, dk, dv, dtype=torch.float32, device=k.device) if S0 is None else S0.float().clone()
    for t in range(T):
        S = S * g[t].exp()[:, None, None]
        mem = (S * k[t][:, :, None]).sum(1)  # [H, dv]
        delta = (v[t] - mem) * beta[t][:, None]
        S = S + k[t][:, :, None] * delta[:, None, :]
    return S


def delta_rule_state(k, v, beta, g):
    """Final state [H, dk, dv] fp32: vLLM's fla chunk kernel on CUDA/ROCm (same recurrence, chunked), else the reference."""
    if k.is_cuda:
        try:
            from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule
        except ImportError:
            chunk_gated_delta_rule = None
        if chunk_gated_delta_rule is not None:
            # q is only needed for the kernel's outputs, which we drop: pass k (already l2-normed; no in-kernel norm)
            _, S = chunk_gated_delta_rule(q=k[None].contiguous(), k=k[None].contiguous(), v=v[None].contiguous(),
                                          g=g[None].contiguous(), beta=beta[None].contiguous(), initial_state=None,
                                          output_final_state=True, use_qk_l2norm_in_kernel=False)
            return S[0].float()  # fla returns [B, H, dk, dv]
    return delta_rule_state_ref(k.float(), v.float(), beta.float(), g.float())


def lin_states(p, l, qkv, a, b, conv_rows=None):
    """Mapped taps of lin layer l for T prefix tokens -> (conv_state [C, kernel-1] float32 (HF orientation, oldest
    first), recurrent_state [Hv, dk, dv] float32). qkv [T, C] pre-conv (q|k|v), a/b [T, Hv]."""
    prm = p.lin[l]
    T = qkv.shape[0]
    x = qkv.float()
    kd = p.Hk * p.dk
    y = causal_conv_silu(x, prm["conv_w"], prm["conv_b"])
    k = y[:, kd:2 * kd].view(T, p.Hk, p.dk)
    v = y[:, 2 * kd:].view(T, p.Hv, p.dv)
    k = _l2n(k)
    if p.Hv != p.Hk:
        k = k.repeat_interleave(p.Hv // p.Hk, dim=1)
    beta = torch.sigmoid(b.float())
    g = -prm["A_log"].exp() * F.softplus(a.float() + prm["dt_bias"])
    S = delta_rule_state(k, v, beta, g)
    n = (conv_rows or p.kernel - 1)
    last = x[-n:] if T >= n else torch.cat([x.new_zeros(n - T, x.shape[1]), x], 0)
    return last.t().contiguous(), S


# ---------------------------------------------------------------- serving glue (used by kvx.vllm_hybrid_connector)

def k_norm_weights(cfg, sd, prefix="model.layers."):
    """{attn layer: (1 + k_norm.weight) fp32}: Qwen3.5 RMSNorm scales by (1 + weight)."""
    g = (lambda k, d=None: cfg.get(k, d)) if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    return {l: 1.0 + sd[f"{prefix}{l}.self_attn.k_norm.weight"].float()
            for l, t in enumerate(g("layer_types")) if t == "full_attention"}


def taps_from_feats(mf, maps, feats, T, knw, sinks=None, recolor=True):
    """Source features (per source layer [T, d]) -> mapped target taps {t: {name: [T, w]}} (fp32), rms-restored keys,
    positions < sinks from the file's sink_template (calibration-mean target taps).
    recolor (kvx-pairs, kvx/hybrid.apply): per bd-channel block, Y = (Y - mu) @ rc + mu on the first nb*bd channels,
    before the rms restore; used when the map carries rc. GEMM in the map's dtype (bf16 as saved/deployed)."""
    A = int(mf.get("sinks", 0) if sinks is None else sinks)
    tmpl = mf.get("sink_template") or {}
    out = {}
    for t, m in maps.items():
        Wm = m["W"]
        X = torch.cat([feats[s].to(Wm.dtype) for s in m["sel"]], 1)
        Y = (X @ Wm).float() + m["b"].float()
        rc = m.get("rc")
        if recolor and rc is not None:
            nb, bd = rc.shape[0], rc.shape[1]
            mu = m["mu"].to(Y)
            Yc = (Y[:, :nb * bd] - mu[:nb * bd]).view(-1, nb, bd).transpose(0, 1)
            Y = torch.cat([torch.bmm(Yc, rc.to(Y)).transpose(0, 1).reshape(-1, nb * bd) + mu[:nb * bd],
                           Y[:, nb * bd:]], 1)
        d, o = {}, 0
        for name, wd in mf["tgt_taps"][t]["taps"]:
            d[name] = Y[:, o:o + wd]
            o += wd
        if mf["tgt_taps"][t]["kind"] == "attn" and mf.get("restore", "rms") == "rms":
            w = knw[t].to(Y.device)
            D = w.shape[0]
            u = d["k"].view(T, -1, D) / w
            d["k"] = (w * u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + 1e-6)).view(T, -1)
        if A and t in tmpl:
            n = min(A, T)
            for name in d:
                d[name] = torch.cat([tmpl[t][name][:n].float().to(Y.device), d[name][n:]], 0)
        out[t] = d
    return out


def caches_from_taps(params, kinds, given):
    """Mapped taps -> {t: ("attn", K [T,H,D], V) | ("lin", conv_state [C, kernel-1], recurrent [Hv, dk, dv])} fp32."""
    out = {}
    for t, d in given.items():
        if kinds[t] == "attn":
            K, V = attn_kv(params, d["k"], d["v"])
            out[t] = ("attn", K, V)
        elif kinds[t] == "lin":
            conv, S = lin_states(params, t, d["qkv"], d["a"], d["b"])
            out[t] = ("lin", conv, S)
    return out


def write_lin_state(conv_cache, ssm_cache, slot, conv, S, dim_first):
    """conv_cache [slots, C, k-1] (dim_first) or [slots, k-1, C]; ssm_cache [slots, Hv, dv, dk] (vLLM's GDN layout:
    (num_v_heads, head_v_dim, head_k_dim)). conv [C, k-1] and S [Hv, dk, dv] from caches_from_taps."""
    if dim_first:
        conv_cache[slot].copy_(conv.to(conv_cache.dtype))
    else:
        conv_cache[slot].copy_(conv.t().to(conv_cache.dtype))
    ssm_cache[slot].copy_(S.transpose(-1, -2).to(ssm_cache.dtype))
