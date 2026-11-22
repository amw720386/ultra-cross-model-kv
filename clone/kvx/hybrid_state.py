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
        if self.rope_type not in ("default", "mrope", "yarn"):
            raise NotImplementedError(f"rope_type {self.rope_type}: plain RoPE (text mrope == RoPE) or yarn only")
        self.yarn = None
        if self.rope_type == "yarn":
            self.yarn = dict(factor=float(rp["factor"]), orig=int(rp["original_max_position_embeddings"]),
                             beta_fast=float(rp.get("beta_fast") or 32), beta_slow=float(rp.get("beta_slow") or 1),
                             attention_factor=rp.get("attention_factor"), truncate=rp.get("truncate", True))
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

def _yarn_inv_freq(rd, base, y, device):
    """HF transformers _compute_yarn_parameters: blend interpolated / extrapolated frequencies with a linear ramp
    between the beta_fast / beta_slow correction dims; returns (inv_freq, attention_factor)."""
    pos_freqs = base ** (torch.arange(0, rd, 2, device=device, dtype=torch.float32) / rd)
    inv_extra, inv_inter = 1.0 / pos_freqs, 1.0 / (y["factor"] * pos_freqs)

    def corr_dim(n_rot):
        return (rd * math.log(y["orig"] / (n_rot * 2 * math.pi))) / (2 * math.log(base))

    low, high = corr_dim(y["beta_fast"]), corr_dim(y["beta_slow"])
    if y["truncate"]:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, rd - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(rd // 2, device=device, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    extra_w = 1 - ramp
    inv = inv_inter * (1 - extra_w) + inv_extra * extra_w
    af = y["attention_factor"]
    if af is None:
        af = 0.1 * math.log(y["factor"]) + 1.0 if y["factor"] > 1 else 1.0
    return inv, float(af)


def rope_cos_sin(p, T, device, dtype=torch.float32, offset=0):
    rd = int(p.head_dim * p.partial)
    scale = 1.0
    if getattr(p, "yarn", None):
        inv, scale = _yarn_inv_freq(rd, p.theta, p.yarn, device)
    else:
        inv = 1.0 / (p.theta ** (torch.arange(0, rd, 2, device=device, dtype=torch.float32) / rd))
    pos = torch.arange(offset, offset + T, device=device, dtype=torch.float32)
    f = torch.outer(pos, inv)
    emb = torch.cat([f, f], -1)
    return (emb.cos() * scale).to(dtype), (emb.sin() * scale).to(dtype), rd


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
        chunk_gated_delta_rule = None
        # vLLM 0.34 ships FLA under vllm.third_party (the old path failed silently -> the per-token Python loop below:
        # 25.8 s mapped TTFT at 8K on Qwen3.8-27B TP2, linear in T); standalone flash-linear-attention as a last resort
        for mod in ("vllm.third_party.flash_linear_attention.ops", "vllm.model_executor.layers.fla.ops", "fla.ops"):
            try:
                chunk_gated_delta_rule = __import__(mod, fromlist=["chunk_gated_delta_rule"]).chunk_gated_delta_rule
                break
            except (ImportError, AttributeError):
                continue
        if chunk_gated_delta_rule is not None:
            # q is only needed for the kernel's outputs, which we drop: pass k (already l2-normed; no in-kernel norm).
            # The kernel rejects fp32 q/k/v: bf16 inputs as vLLM's own GDN layer passes them (g fp32, state fp32)
            kb, vb = k[None].to(torch.bfloat16).contiguous(), v[None].to(torch.bfloat16).contiguous()
            _, S = chunk_gated_delta_rule(q=kb, k=kb, v=vb, g=g[None].float().contiguous(),
                                          beta=beta[None].to(torch.bfloat16).contiguous(), initial_state=None,
                                          output_final_state=True, use_qk_l2norm_in_kernel=False)
            S = S[0].float()  # fla returns [B, H, dk, dv]
            if _FLA_OK.get("ok") is None:  # one-time check of the bf16 kernel against the fp32 recurrence
                n = min(k.shape[0], 256)
                _, Sn = chunk_gated_delta_rule(q=kb[:, -n:], k=kb[:, -n:], v=vb[:, -n:],
                                               g=g[None, -n:].float().contiguous(),
                                               beta=beta[None, -n:].to(torch.bfloat16).contiguous(),
                                               initial_state=None, output_final_state=True,
                                               use_qk_l2norm_in_kernel=False)
                ref = delta_rule_state_ref(k[-n:].float(), v[-n:].float(), beta[-n:].float(), g[-n:].float())
                rel = ((Sn[0].float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()
                _FLA_OK.update(ok=rel < 0.05, rel=rel)
                print(f"[kvx] fla chunk_gated_delta_rule check: rel err {rel:.4g} over {n} tokens -> "
                      f"{'fast path' if rel < 0.05 else 'FALLBACK to the reference loop'}", flush=True)
            if _FLA_OK["ok"]:
                return S
    return delta_rule_state_ref(k.float(), v.float(), beta.float(), g.float())


_FLA_OK = {}


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


def _rank_heads(H, rank, tp):
    """Head range of a tensor-parallel rank; replicated heads when tp > H (vLLM's KV-head replication)."""
    if tp <= H:
        n = H // tp
        return rank * n, (rank + 1) * n
    h = rank // (tp // H)
    return h, h + 1


def slice_for_rank(p, st, rank, tp):
    """One rank's share of a caches_from_taps entry under vLLM tensor parallelism.
    attn: K/V heads of the rank. lin: conv_state rows = q|k|v sections, each split 1/tp (vLLM shards in_proj_qkv and
    conv1d per section: rank r holds q[r], k[r], v[r]); recurrent state = the rank's v heads."""
    if tp == 1:
        return st
    if st[0] == "attn":
        _, K, V = st
        a, b = _rank_heads(K.shape[1], rank, tp)
        return ("attn", K[:, a:b], V[:, a:b])
    _, conv, S = st
    kd, vd = p.Hk * p.dk, p.Hv * p.dv
    qk = kd // tp
    vv = vd // tp
    rows = torch.cat([conv[rank * qk:(rank + 1) * qk], conv[kd + rank * qk:kd + (rank + 1) * qk],
                      conv[2 * kd + rank * vv:2 * kd + (rank + 1) * vv]], 0)
    a, b = _rank_heads(S.shape[0], rank, tp)
    return ("lin", rows, S[a:b])


# ---------------------------------------------------------------- xkv maps (kvx-pairs' kvx/xkv_adapter.py), per layer

def tgt_shim(cfg):
    """Stand-in for a target decoder (kvx.hybrid.layer_taps / xkv_adapter only read layer kinds and cfg dims), so the
    vLLM connector needs no second HF copy of the target."""
    from types import SimpleNamespace as NS
    g = (lambda k, d=None: cfg.get(k, d)) if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    layers = []
    for t in g("layer_types"):
        if t == "linear_attention":
            layers.append(NS(linear_attn=NS(in_proj_qkv=None, in_proj_a=None, in_proj_b=None)))
        else:
            layers.append(NS(linear_attn=None, self_attn=NS(is_kv_shared_layer=False, k_norm=NS(), v_norm=None,
                                                             k_proj=None, v_proj=None)))
    return NS(cfg=cfg if not isinstance(cfg, dict) else NS(**cfg), layers=layers)


def xkv_layer_taps(xm, feats, emb, emb_prev, j, kind, c, out_dtype=torch.float32):
    """One target layer's kvx-hybrid-v1 taps from xkv maps ([B, T, w] each) == xkv_adapter.mapped_taps(...)[j], computing
    only this layer's GEMM slice (the connector streams layer by layer)."""
    from .xkv_adapter import _mm
    g = (lambda k: c[k]) if isinstance(c, dict) else (lambda k: getattr(c, k))
    attn = xm.get("attn")
    if kind == "attn" and attn is not None and j in attn["layers"]:
        d = attn["parts"][j]
        dt = emb.dtype
        y = d["b"].to(device=emb.device, dtype=dt).expand(*emb.shape[:-1], -1).clone()
        for i in attn["layers"][j]:
            y = y + feats[i].to(dt) @ d[i].to(device=emb.device, dtype=dt)
        if attn["embed"]:
            y = y + emb @ d["e"].to(device=emb.device, dtype=dt) + emb_prev @ d["p"].to(device=emb.device, dtype=dt)
        mlp = attn.get("mlp")
        if mlp and j in mlp:
            m = mlp[j]
            f = torch.cat([feats[i].to(dt) for i in attn["layers"][j]] + ([emb, emb_prev] if attn["embed"] else []), -1)
            y = y + torch.nn.functional.gelu(f @ m["w1"].to(f) + m["b1"].to(f)) @ m["w2"].to(f)
        h = y.shape[-1] // 2
        return {"k": y[..., :h].to(out_dtype), "v": y[..., h:].to(out_dtype)}
    i = xm["src_of"][j]
    sl = xm["slices"][i] if i in xm["slices"] else xm["slices"][str(i)]
    a0, a1 = sl[j] if j in sl else sl[str(j)]
    W = xm["W"][i]
    Wj = torch.cat([W[:-1, a0:a1], W[-1:, a0:a1]], 0)  # keep the bias row last
    y = _mm(feats[i], Wj)
    if kind == "lin":
        kd = g("linear_key_head_dim") * g("linear_num_key_heads")
        vd = g("linear_value_head_dim") * g("linear_num_value_heads")
        nv = g("linear_num_value_heads")
        q = _mm(feats[i], xm["Wq"][j])
        return {"qkv": torch.cat([q, y[..., :kd + vd]], -1).to(out_dtype),
                "a": y[..., kd + vd:kd + vd + nv].to(out_dtype), "b": y[..., kd + vd + nv:kd + vd + 2 * nv].to(out_dtype)}
    h = y.shape[-1] // 2
    return {"k": y[..., :h].to(out_dtype), "v": y[..., h:].to(out_dtype)}


# ---------------------------------------------------------------- native first layers (exact token identity)

def build_prefix_model(text_cfg, sd, m, device="cpu", dtype=torch.bfloat16):
    """The target's embeddings + first m decoder layers as an HF Qwen3_5TextModel (weights from a state-dict mapping with
    'model.[language_model.]layers.i.*' / '...embed_tokens.weight' keys). Running it over the whole prompt gives the
    first m layers' taps NATIVELY (arXiv 2611.12062: early layers carry exact token identity, which needle retrieval
    needs); cost ~ m / num_layers of a target prefill."""
    import re as _re
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    c = dict(text_cfg) if isinstance(text_cfg, dict) else text_cfg.to_dict()
    c["num_hidden_layers"] = m
    c["layer_types"] = list(c["layer_types"])[:m]
    for k in ("architectures", "transformers_version", "torch_dtype", "dtype"):
        c.pop(k, None)
    model = Qwen3_5TextModel(Qwen3_5TextConfig(**c))
    want = {}
    for k, v in sd.items():
        mm = _re.search(r"(?:^|\.)(embed_tokens\.weight|layers\.(\d+)\..*)$", k)
        if not mm:
            continue
        if mm.group(2) is not None and int(mm.group(2)) >= m:
            continue
        want[mm.group(1)] = v
    missing, _ = model.load_state_dict(want, strict=False)
    bad = [k for k in missing if not k.startswith("norm.") and "rotary" not in k]
    if bad:
        raise RuntimeError(f"build_prefix_model: missing weights {bad[:5]}")
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def native_prefix_taps(model, ids):
    """{layer: {tap: [T, w]}} float32 for the prefix model's layers (same taps as the maps predict)."""
    got, hs_ = {}, []
    for l, layer in enumerate(model.layers):
        la = getattr(layer, "linear_attn", None)
        mods = ([("qkv", la.in_proj_qkv), ("a", la.in_proj_a), ("b", la.in_proj_b)] if la is not None else
                [("k", layer.self_attn.k_norm), ("v", layer.self_attn.v_proj)])
        for name, mod in mods:
            hs_.append(mod.register_forward_hook(
                lambda m_, i_, o, l=l, name=name: got.setdefault(l, {}).__setitem__(name, o.reshape(o.shape[1], -1).float())))
    try:
        model(input_ids=ids, use_cache=False)
    finally:
        for h in hs_:
            h.remove()
    return got


# ---------------------------------------------------------------- tensor-parallel map sharding (memory)

def _rank_cols(kind, taps, p, rank, tp, kv_heads, head_dim):
    """Column indices of one target layer's map output that this TP rank needs, and the rank-local tap widths."""
    if kind == "attn":
        a, b = _rank_heads(kv_heads, rank, tp)
        kw = kv_heads * head_dim
        k = list(range(a * head_dim, b * head_dim))
        return k + [kw + i for i in k], [("k", len(k)), ("v", len(k))]
    kd, vd = p.Hk * p.dk, p.Hv * p.dv
    qk, vv = kd // tp, vd // tp
    q = list(range(rank * qk, (rank + 1) * qk))
    k = [kd + i for i in q]
    v = list(range(2 * kd + rank * vv, 2 * kd + (rank + 1) * vv))
    nv = p.Hv // tp
    off = 2 * kd + vd
    a = list(range(off + rank * nv, off + (rank + 1) * nv))
    bb = [p.Hv + i for i in a]
    return q + k + v + a + bb, [("qkv", len(q) + len(k) + len(v)), ("a", nv), ("b", nv)]


def shard_for_rank(mf, maps, p, rank, tp, kv_heads):
    """(maps, tgt_taps, params) holding only this rank's output columns (W, b, mu, recolor blocks) and its share of
    the GatedDeltaNet params, so taps_from_feats + caches_from_taps directly yield the rank's slice (== slice_for_rank
    of the full result) with 1/tp of the map memory."""
    import copy
    if tp == 1:
        return maps, mf["tgt_taps"], p
    out, taps_r = {}, [dict(x) for x in mf["tgt_taps"]]
    for t, m in maps.items():
        kind = mf["tgt_taps"][t]["kind"]
        cols, widths = _rank_cols(kind, mf["tgt_taps"][t]["taps"], p, rank, tp, kv_heads, p.head_dim)
        idx = torch.tensor(cols, device=m["W"].device)
        r = dict(sel=m["sel"], W=m["W"].index_select(1, idx).contiguous(), b=m["b"].index_select(0, idx.to(m["b"].device)))
        if m.get("mu") is not None:
            r["mu"] = m["mu"].index_select(0, idx.to(m["mu"].device))
        rc = m.get("rc")
        if rc is not None:
            bd = rc.shape[1]
            span = rc.shape[0] * bd
            inb = [c for c in cols if c < span]
            assert all(c % bd == 0 for c in inb[::bd]) and len(inb) % bd == 0, "TP slice not aligned to recolor blocks"
            blocks = torch.tensor(sorted({c // bd for c in inb}), device=rc.device)
            r["rc"] = rc.index_select(0, blocks)
        out[t] = r
        taps_r[t] = dict(kind=kind, taps=widths)
    pr = copy.copy(p)
    pr.Hk, pr.Hv = p.Hk // tp, p.Hv // tp
    kd, vd = p.Hk * p.dk, p.Hv * p.dv
    qk, vv = kd // tp, vd // tp
    ch = list(range(rank * qk, (rank + 1) * qk)) + list(range(kd + rank * qk, kd + (rank + 1) * qk)) + \
        list(range(2 * kd + rank * vv, 2 * kd + (rank + 1) * vv))
    hv = list(range(rank * pr.Hv, (rank + 1) * pr.Hv))
    pr.lin = {}
    for l, d in p.lin.items():
        dev = d["conv_w"].device
        ci, hi = torch.tensor(ch, device=dev), torch.tensor(hv, device=dev)
        pr.lin[l] = dict(conv_w=d["conv_w"].index_select(0, ci), conv_b=None if d["conv_b"] is None else d["conv_b"].index_select(0, ci),
                         A_log=d["A_log"].index_select(0, hi), dt_bias=d["dt_bias"].index_select(0, hi))
    return out, taps_r, pr
