"""Lean forward pass for Qwen3.5-family hybrid models (Gated DeltaNet + gated full attention).

The Hugging Face modules hold the weights; this file owns the control flow so that we can
chunk long prefills, capture per-layer features, and inject externally produced caches.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------------------

_FA = None
STATS: dict = {}


def _flash():
    """Return a flash_attn_func(q, k, v, causal, window_size) or None."""
    global _FA
    if _FA is not None:
        return _FA or None
    try:
        from flash_attn import flash_attn_func  # type: ignore

        _FA = flash_attn_func
        return _FA
    except Exception:
        pass
    for name in ("kernels-community/flash-attn2", "kernels-community/flash-attn"):
        try:
            from kernels import get_kernel  # type: ignore

            k = get_kernel(name)
            fn = getattr(k, "flash_attn_func", None)
            if fn is not None:
                _FA = fn
                return fn
        except Exception:
            continue
    _FA = False
    return None


def attention(q, k, v, window: int | None = None):
    """Causal GQA attention, bottom-right aligned. q (B,L,Hq,D), k/v (B,T,Hkv,D), T >= L."""
    fa = _flash()
    if fa is not None:
        ws = (window - 1, 0) if window else (-1, -1)
        out = fa(q, k, v, causal=True, window_size=ws)
        return out[0] if isinstance(out, tuple) else out
    # SDPA fallback with explicit mask (short sequences / tests only)
    L, T = q.shape[1], k.shape[1]
    i = torch.arange(L, device=q.device)[:, None] + (T - L)
    j = torch.arange(T, device=q.device)[None, :]
    mask = j <= i
    if window:
        mask &= j > i - window
    o = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask, enable_gqa=True
    )
    return o.transpose(1, 2)


def _gdr():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

    return chunk_gated_delta_rule, fused_recurrent_gated_delta_rule


def delta_scan(q, k, v, g, beta, state):
    """Gated delta rule over a chunk. q,k (B,L,Hv,Dk) raw (l2norm applied in kernel), v (B,L,Hv,Dv)."""
    chunk, recur = _gdr()
    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (q, k, v, g, beta) + ((state,) if state is not None else ()))
    fn = recur if q.shape[1] <= 16 and not needs_grad else chunk
    o, s = fn(q, k, v, g=g, beta=beta, initial_state=state, output_final_state=True,
              use_qk_l2norm_in_kernel=True)
    return o, s


def delta_state(k, v, g, beta, state):
    """Final gated-delta-rule state only (skips the per-token outputs). k raw (l2-normalized here).

    Uses the WY-representation pieces of fla's chunk forward; falls back to the full kernel if the
    internals are unavailable or autograd is needed.
    """
    if torch.is_grad_enabled() and (k.requires_grad or v.requires_grad or state.requires_grad
                                    or g.requires_grad or beta.requires_grad):
        return delta_scan(k, k, v, g, beta, state)[1]
    try:
        from fla.modules.l2norm import l2norm_fwd
        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
        from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
        from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
        from fla.ops.utils import chunk_local_cumsum, solve_tril
    except Exception as e:
        STATS["delta_state_fallback"] = repr(e)
        return delta_scan(k, k, v, g, beta, state)[1]
    STATS["delta_state_fast"] = STATS.get("delta_state_fast", 0) + 1
    k = l2norm_fwd(k.contiguous())[0]
    v, beta = v.contiguous(), beta.contiguous()
    g = chunk_local_cumsum(g.contiguous(), chunk_size=64)
    A = chunk_scaled_dot_kkt_fwd(k=k, g=g, beta=beta, output_dtype=torch.float32)
    A = solve_tril(A=A, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g)
    _, _, final = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=g, initial_state=state, output_final_state=True,
                                               save_new_value=False)
    return final


def _dwconv_silu(xp, w):
    """Depthwise causal conv + SiLU. xp (B, L+K-1, C) incl. K-1 history rows, w (K, C)."""
    K = w.shape[0]
    L = xp.shape[1] - K + 1
    out = xp[:, 0:L].float() * w[0].float()
    for i in range(1, K):
        out = out + xp[:, i:i + L].float() * w[i].float()
    return F.silu(out).to(xp.dtype)


try:
    dwconv_silu = torch.compile(_dwconv_silu, dynamic=True)
except Exception:  # pragma: no cover
    dwconv_silu = _dwconv_silu


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


# --------------------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------------------


@dataclass
class Spec:
    n_layers: int
    types: list
    full: list
    lin: list
    hidden: int
    nq: int
    nkv: int
    hd: int
    rot: int
    theta: float
    nk: int
    nv: int
    dk: int
    dv: int
    conv_k: int

    @property
    def key_dim(self):
        return self.nk * self.dk

    @property
    def value_dim(self):
        return self.nv * self.dv

    @property
    def conv_dim(self):
        return 2 * self.key_dim + self.value_dim


class Cache:
    """Preallocated hybrid cache. K is stored post-RoPE, layout (B, T, Hkv, D)."""

    def __init__(self, spec: Spec, batch: int, max_len: int, device, dtype=torch.bfloat16, functional=False):
        """functional=True grows K/V by concatenation and never writes in place (autograd-safe)."""
        self.spec, self.max_len, self.len, self.functional = spec, max_len, 0, functional
        self.ring, self.base = False, {j: 0 for j in spec.full}
        n = 0 if functional else max_len
        self.k = {j: torch.empty(batch, n, spec.nkv, spec.hd, device=device, dtype=dtype) for j in spec.full}
        self.v = {j: torch.empty(batch, n, spec.nkv, spec.hd, device=device, dtype=dtype) for j in spec.full}
        self.conv = {j: torch.zeros(batch, spec.conv_k - 1, spec.conv_dim, device=device, dtype=dtype) for j in spec.lin}
        self.S = {j: torch.zeros(batch, spec.nv, spec.dk, spec.dv, device=device, dtype=torch.float32) for j in spec.lin}

    def clone(self, upto: int | None = None) -> "Cache":
        n = self.len if upto is None else upto
        c = Cache.__new__(Cache)
        c.spec, c.max_len, c.len, c.functional = self.spec, self.max_len, n, self.functional
        c.ring, c.base = self.ring, dict(self.base)
        c.k = {j: t.clone() for j, t in self.k.items()}
        c.v = {j: t.clone() for j, t in self.v.items()}
        c.conv = {j: t.clone() for j, t in self.conv.items()}
        c.S = {j: t.clone() for j, t in self.S.items()}
        return c

    def write_kv(self, j, s, k, v):
        if self.functional:
            assert self.k[j].shape[1] == s
            self.k[j] = torch.cat((self.k[j], k), dim=1)
            self.v[j] = torch.cat((self.v[j], v), dim=1)
        else:
            self.k[j][:, s:s + k.shape[1]] = k
            self.v[j][:, s:s + v.shape[1]] = v

    def nbytes(self) -> int:
        tot = 0
        for d in (self.k, self.v, self.conv, self.S):
            for t in d.values():
                tot += t.numel() * t.element_size() * (self.len / self.max_len if d in (self.k, self.v) else 1)
        return int(tot)


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------


class Hybrid:
    def __init__(self, hf):
        self.hf = hf
        lm = hf.model.language_model
        c = lm.config
        self.embed, self.layers, self.norm, self.lm_head = lm.embed_tokens, lm.layers, lm.norm, hf.lm_head
        rp = c.rope_parameters
        types = list(c.layer_types)
        self.spec = Spec(
            n_layers=c.num_hidden_layers, types=types,
            full=[i for i, t in enumerate(types) if t == "full_attention"],
            lin=[i for i, t in enumerate(types) if t == "linear_attention"],
            hidden=c.hidden_size, nq=c.num_attention_heads, nkv=c.num_key_value_heads, hd=c.head_dim,
            rot=int(c.head_dim * rp.get("partial_rotary_factor", 1.0)), theta=float(rp["rope_theta"]),
            nk=c.linear_num_key_heads, nv=c.linear_num_value_heads, dk=c.linear_key_head_dim,
            dv=c.linear_value_head_dim, conv_k=c.linear_conv_kernel_dim,
        )
        self.device = self.embed.weight.device
        self.dtype = self.embed.weight.dtype
        s = self.spec
        self.inv_freq = 1.0 / (s.theta ** (torch.arange(0, s.rot, 2, device=self.device, dtype=torch.float32) / s.rot))

    # ---- rope -------------------------------------------------------------------------
    def rope_tables(self, pos0: int, L: int):
        pos = torch.arange(pos0, pos0 + L, device=self.device, dtype=torch.float32)
        f = pos[:, None] * self.inv_freq[None]
        emb = torch.cat((f, f), dim=-1)
        return emb.cos().to(self.dtype)[None, :, None, :], emb.sin().to(self.dtype)[None, :, None, :]

    def rope(self, x, cos, sin):
        r = self.spec.rot
        xr, xp = x[..., :r], x[..., r:]
        return torch.cat((xr * cos + rotate_half(xr) * sin, xp), dim=-1)

    def conv_w(self, j):
        if not hasattr(self, "_convw"):
            self._convw = {}
        if j not in self._convw:
            self._convw[j] = self.layers[j].linear_attn.conv1d.weight[:, 0, :].t().contiguous()  # (K, C)
        return self._convw[j]

    def new_cache(self, batch: int, max_len: int, functional: bool = False, window: int | None = None,
                  chunk: int = 8192) -> Cache:
        """window: allocate a sliding ring of window+chunk keys instead of max_len (prefill-only cache)."""
        if window and max_len > window + chunk:
            c = Cache(self.spec, batch, window + chunk, self.device, self.dtype, functional)
            c.ring = True
            return c
        return Cache(self.spec, batch, max_len, self.device, self.dtype, functional)

    # ---- mixers -----------------------------------------------------------------------
    def _full(self, j, x, cache: Cache, pos0, rope, window, cap):
        s, attn = self.spec, self.layers[j].self_attn
        B, L, _ = x.shape
        qg = attn.q_proj(x).view(B, L, s.nq, 2 * s.hd)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(B, L, s.nq * s.hd)
        q = attn.q_norm(q)
        k = attn.k_norm(attn.k_proj(x).view(B, L, s.nkv, s.hd))
        v = attn.v_proj(x).view(B, L, s.nkv, s.hd)
        if cap is not None:
            cap("tgt_full", j, x=x, k=k, v=v)
        cos, sin = rope
        q, kr = self.rope(q, cos, sin), self.rope(k, cos, sin)
        if cache.ring:
            # sliding-window ring: keep only the last `window` keys in front of the new chunk
            base = cache.base[j]
            idx = pos0 - base
            if idx + L > cache.max_len:
                keep = min(window - 1, idx)
                cache.k[j][:, :keep] = cache.k[j][:, idx - keep:idx].clone()
                cache.v[j][:, :keep] = cache.v[j][:, idx - keep:idx].clone()
                cache.base[j] = base = pos0 - keep
                idx = keep
            cache.write_kv(j, idx, kr, v)
            o = attention(q, cache.k[j][:, : idx + L], cache.v[j][:, : idx + L], window)
        else:
            cache.write_kv(j, pos0, kr, v)
            o = attention(q, cache.k[j][:, : pos0 + L], cache.v[j][:, : pos0 + L], window)
        o = o.reshape(B, L, -1) * torch.sigmoid(gate)
        return attn.o_proj(o)

    def _lin(self, j, x, cache: Cache, cap):
        s, la = self.spec, self.layers[j].linear_attn
        B, L, _ = x.shape
        qkv = la.in_proj_qkv(x)
        z = la.in_proj_z(x)
        b = la.in_proj_b(x)
        a = la.in_proj_a(x)
        if cap is not None:
            cap("tgt_lin", j, x=x, qkv=qkv, a=a, b=b)
        xx = torch.cat((cache.conv[j], qkv), dim=1)
        cache.conv[j] = xx[:, -(s.conv_k - 1):].contiguous()
        mixed = dwconv_silu(xx, self.conv_w(j))
        q, k, v = torch.split(mixed, [s.key_dim, s.key_dim, s.value_dim], dim=-1)
        q = q.reshape(B, L, s.nk, s.dk)
        k = k.reshape(B, L, s.nk, s.dk)
        v = v.reshape(B, L, s.nv, s.dv)
        rep = s.nv // s.nk
        if rep > 1:
            q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
        g = -la.A_log.float().exp() * F.softplus(a.float() + la.dt_bias)
        beta = b.sigmoid()
        o, S = delta_scan(q, k, v, g, beta, cache.S[j])
        cache.S[j] = S
        o = la.norm(o.reshape(-1, s.dv), z.reshape(-1, s.dv)).reshape(B, L, -1)
        return la.out_proj(o)

    # ---- forward ----------------------------------------------------------------------
    def forward(self, ids, cache: Cache, logits: str | None = "last", window: int | None = None,
                cap: Callable | None = None, stop_after: int | None = None, inputs_embeds=None):
        """Process ids (B,L) appended to `cache`. logits: 'last' | 'all' | None.

        cap(kind, layer, **tensors) is called with kind in {'src', 'tgt_full', 'tgt_lin'}.
        stop_after: run only layers [0, stop_after] (feature extraction; deeper caches are left untouched).
        """
        h = self.embed(ids) if inputs_embeds is None else inputs_embeds
        B, L, _ = h.shape
        pos0 = cache.len
        rope = self.rope_tables(pos0, L)
        last = self.spec.n_layers - 1 if stop_after is None else stop_after
        for i in range(last + 1):
            layer = self.layers[i]
            x = layer.input_layernorm(h)
            if cap is not None:
                cap("src", i, x=x)
            if self.spec.types[i] == "full_attention":
                m = self._full(i, x, cache, pos0, rope, window, cap)
            else:
                m = self._lin(i, x, cache, cap)
            h = h + m
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        cache.len = pos0 + L
        if stop_after is not None or logits is None:
            return None
        if logits == "last":
            h = h[:, -1:]
        return self.lm_head(self.norm(h)).float()

    def prefill(self, ids, cache: Cache, chunk: int = 8192, logits: str | None = "last", **kw):
        out = None
        for s in range(0, ids.shape[1], chunk):
            out = self.forward(ids[:, s:s + chunk], cache,
                               logits=logits if s + chunk >= ids.shape[1] else None, **kw)
        return out

    @torch.no_grad()
    def greedy(self, cache: Cache, first_logits, max_new: int, stop_ids=()) -> list[int]:
        tok = int(first_logits[0, -1].argmax())
        out = [tok]
        for _ in range(max_new - 1):
            if tok in stop_ids:
                break
            lg = self.forward(torch.tensor([[tok]], device=self.device), cache)
            tok = int(lg[0, -1].argmax())
            out.append(tok)
        return out


def load(path: str, device="cuda", dtype=torch.bfloat16):
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    hf = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype, device_map=device)
    hf.eval()
    hf.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(path)
    return Hybrid(hf), tok
