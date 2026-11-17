"""Fused causal depthwise conv + SiLU that reads the conv history and new rows separately (no concat)."""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None


if triton is not None:
    @triton.jit
    def _conv_silu_kernel(H, X, W, O, L, C, sHb, sHt, sXb, sXt, sWk, sOb, sOt,
                          K: tl.constexpr, BL: tl.constexpr, BC: tl.constexpr):
        pl = tl.program_id(0)
        pc = tl.program_id(1)
        b = tl.program_id(2)
        t = pl * BL + tl.arange(0, BL)
        c = pc * BC + tl.arange(0, BC)
        cm = c < C
        tm = t < L
        acc = tl.zeros((BL, BC), dtype=tl.float32)
        for i in tl.static_range(K):
            src = t + i - (K - 1)
            w = tl.load(W + i * sWk + c, mask=cm, other=0.0).to(tl.float32)
            mx = (src >= 0) & tm
            xv = tl.load(X + b * sXb + src[:, None] * sXt + c[None, :], mask=mx[:, None] & cm[None, :], other=0.0)
            mh = (src < 0) & tm
            hv = tl.load(H + b * sHb + (src + K - 1)[:, None] * sHt + c[None, :], mask=mh[:, None] & cm[None, :],
                         other=0.0)
            acc += (xv.to(tl.float32) + hv.to(tl.float32)) * w[None, :]
        out = acc * tl.sigmoid(acc)
        tl.store(O + b * sOb + t[:, None] * sOt + c[None, :], out.to(O.dtype.element_ty),
                 mask=tm[:, None] & cm[None, :])


def conv_silu(hist, x, w):
    """hist (B, K-1, C), x (B, L, C), w (K, C); channel stride 1 for all. Returns silu(conv) (B, L, C)."""
    B, L, C = x.shape
    K = w.shape[0]
    assert hist.stride(-1) == 1 and x.stride(-1) == 1 and w.stride(-1) == 1
    out = torch.empty(B, L, C, device=x.device, dtype=x.dtype)
    BL, BC = 32, 128
    grid = (triton.cdiv(L, BL), triton.cdiv(C, BC), B)
    _conv_silu_kernel[grid](hist, x, w, out, L, C, hist.stride(0), hist.stride(1), x.stride(0), x.stride(1),
                            w.stride(0), out.stride(0), out.stride(1), K=K, BL=BL, BC=BC)
    return out


def next_conv_state(hist, x):
    """Last K-1 rows of concat(hist, x)."""
    n = hist.shape[1]
    if x.shape[1] >= n:
        return x[:, -n:].contiguous()
    return torch.cat((hist, x), dim=1)[:, -n:].contiguous()
