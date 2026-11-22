"""Lean source prefill for the connector: the source model's forward written as plain tensor code.

Why: the HF eager forward spent ~30-40% of the source prefill in small elementwise kernels (separate q/k/v GEMMs with
tiny N under TP, rotate_half cats, unfused norms) and returned post-RoPE keys that the mapper then had to unrope. Here:
one fused QKV GEMM and one fused gate/up GEMM per layer, norms/RoPE/SiLU fused by torch.compile (optional), attention
through vLLM's flash-attn varlen kernel (or SDPA on CPU), no LM head, and the PRE-RoPE keys and values come out
directly into preallocated per-layer buffers [Hkv, T, D] (chunked prefill writes into the same buffers: no cache
concatenation transients at 512K-1M).

Supports the Qwen3 (q/k norm), Qwen2 (qkv bias) and Llama layouts, after kvx.tp_source.shard_source (row-parallel
o/down all-reduce over the TP group) or unsharded.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F


def _rms(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def _rope(x, cos, sin):  # x [T, H, D]; cos/sin [T, D] (HF convention, attention factor included)
    d = x.shape[-1] // 2
    rh = torch.cat((-x[..., d:], x[..., :d]), -1)
    return (x * cos[:, None] + rh * sin[:, None]).to(x.dtype)


def _pre(h, ln, Wqkv, bqkv, qn, kn, cos, sin, eps, Hq, Hk, D):
    x = _rms(h, ln, eps)
    qkv = F.linear(x, Wqkv, bqkv)
    q, k, v = qkv.split([Hq * D, Hk * D, Hk * D], -1)
    q, k, v = q.view(-1, Hq, D), k.view(-1, Hk, D), v.view(-1, Hk, D)
    if qn is not None:
        q, k = _rms(q, qn, eps), _rms(k, kn, eps)
    return _rope(q, cos, sin), _rope(k, cos, sin), k, v


def _mlp(h, ln, Wgu, I, eps):
    x = _rms(h, ln, eps)
    g, u = F.linear(x, Wgu).split([I, I], -1)
    return F.silu(g) * u


class LeanSource:
    def __init__(self, model, compile=False):
        self.m = model
        cfg = model.config
        self.eps = cfg.rms_norm_eps
        tp = getattr(model, "_kvx_tp", None)
        self.group = tp["group"] if tp else None
        self.tp = tp is not None and tp["world"] > 1
        self.layers = []
        for L in model.model.layers:
            a, mlp = L.self_attn, L.mlp
            D = a.head_dim
            Hq, Hk = a.q_proj.out_features // D, a.k_proj.out_features // D
            Wqkv = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0).contiguous()
            b = None
            if a.q_proj.bias is not None:
                b = torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias], 0).contiguous()
            qn = a.q_norm.weight if hasattr(a, "q_norm") else None
            kn = a.k_norm.weight if hasattr(a, "k_norm") else None
            Wgu = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0).contiguous()
            self.layers.append(dict(ln1=L.input_layernorm.weight, ln2=L.post_attention_layernorm.weight, Wqkv=Wqkv, b=b,
                                    qn=qn, kn=kn, Wo=a.o_proj.weight, Wgu=Wgu, Wd=mlp.down_proj.weight,
                                    I=mlp.gate_proj.weight.shape[0], Hq=Hq, Hk=Hk, D=D, scale=a.scaling,
                                    bo=getattr(a.o_proj, "bias", None)))
        self.pre, self.mlp = _pre, _mlp
        if compile:
            self.pre = torch.compile(_pre, dynamic=True)
            self.mlp = torch.compile(_mlp, dynamic=True)

    def _ar(self, x):
        if self.tp:
            dist.all_reduce(x, group=self.group)
        return x

    @torch.no_grad()
    def forward(self, ids, cos, sin, attn, chunk=0):
        """ids LongTensor [T]; cos/sin [T, D] source RoPE tables; attn(q [n,Hq,D], k [S,Hk,D], v, scale) -> [n,Hq,D]
        with causal alignment to the bottom right. Returns [(K_pre [1,Hk,T,D], V [1,Hk,T,D])] per layer."""
        T = ids.shape[0]
        dev = ids.device
        emb = self.m.model.embed_tokens.weight
        dt = emb.dtype
        outs = [(torch.empty(1, l["Hk"], T, l["D"], dtype=dt, device=dev), torch.empty(1, l["Hk"], T, l["D"], dtype=dt, device=dev))
                for l in self.layers]
        chunked = bool(chunk) and T > chunk
        step = chunk if chunked else T
        for a in range(0, T, step):
            b_ = min(T, a + step)
            h = F.embedding(ids[a:b_], emb)
            c, s = cos[a:b_].to(dt), sin[a:b_].to(dt)
            for li, l in enumerate(self.layers):
                q, kr, k, v = self.pre(h, l["ln1"], l["Wqkv"], l["b"], l["qn"], l["kn"], c, s, self.eps, l["Hq"], l["Hk"], l["D"])
                K, V = outs[li]
                K[0, :, a:b_] = k.transpose(0, 1)
                V[0, :, a:b_] = v.transpose(0, 1)
                if chunked:  # earlier chunks' keys re-roped on the fly: no post-RoPE key buffer (7.5 GB/rank at 1M)
                    kk = torch.cat([_rope(K[0, :, :a].transpose(0, 1), cos[:a].to(dt), sin[:a].to(dt)), kr]) if a else kr
                    vv = V[0, :, :b_].transpose(0, 1)
                else:
                    kk, vv = kr, v
                fn = attn(li) if getattr(attn, "per_layer", False) else attn  # per-layer attention (window schedule)
                o = fn(q, kk, vv, l["scale"])
                ao = self._ar(F.linear(o.reshape(o.shape[0], -1), l["Wo"]))
                if l["bo"] is not None:
                    ao = ao + l["bo"]
                h = h + ao
                y = self.mlp(h, l["ln2"], l["Wgu"], l["I"], self.eps)
                h = h + self._ar(F.linear(y, l["Wd"]))
        return outs


def sdpa_attn(q, k, v, scale, window=0):
    """Reference / CPU attention with FA's bottom-right causal alignment (and an optional sliding window)."""
    n, Hq, D = q.shape
    S, Hk = k.shape[0], k.shape[1]
    kk = k.repeat_interleave(Hq // Hk, 1).transpose(0, 1)
    vv = v.repeat_interleave(Hq // Hk, 1).transpose(0, 1)
    qpos = torch.arange(n, device=q.device)[:, None] + S - n
    mask = torch.arange(S, device=q.device)[None] <= qpos
    if window:
        mask &= torch.arange(S, device=q.device)[None] > qpos - window
    o = F.scaled_dot_product_attention(q.transpose(0, 1), kk, vv, attn_mask=mask, scale=scale)
    return o.transpose(0, 1)


def fa_attn(window=0, fp8=False):
    """vLLM flash-attn varlen (kvx.fa) as attn(q, k, v, scale)."""
    from . import fa

    def f(q, k, v, scale):
        old = fa.FP8, fa.WINDOW
        fa.FP8, fa.WINDOW = fp8, window
        try:
            o, _ = fa.vllm_fa_attention(None, q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                                        scaling=scale)
        finally:
            fa.FP8, fa.WINDOW = old
        return o[0]
    return f
