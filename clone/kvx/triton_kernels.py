"""Fused mapped-key epilogue: k_norm RMS restore + RoPE in one pass over [T, Lt, H, D] (bf16, in place).

Replaces ~6 elementwise kernels per target layer (64 layers -> ~400 launches, several fp32 temporaries).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _restore_rope_kernel(Y, W, COS, SIN, T, LH, H: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr,
                         DO_RMS: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)  # one program per (t, l, h) row; int64: T*Lt*H*D passes 2^31 at ~32K tokens
    t = pid // LH
    lh = pid % LH
    l = lh // H
    offs = tl.arange(0, HALF)
    base = Y + pid * D
    x1 = tl.load(base + offs).to(tl.float32)
    x2 = tl.load(base + HALF + offs).to(tl.float32)
    if DO_RMS:
        w1 = tl.load(W + l * D + offs).to(tl.float32)
        w2 = tl.load(W + l * D + HALF + offs).to(tl.float32)
        u1 = x1 / w1
        u2 = x2 / w2
        ms = (tl.sum(u1 * u1, 0) + tl.sum(u2 * u2, 0)) / D
        r = 1.0 / tl.sqrt(ms + 1e-6)
        x1 = w1 * u1 * r
        x2 = w2 * u2 * r
    c1 = tl.load(COS + t * D + offs).to(tl.float32)
    c2 = tl.load(COS + t * D + HALF + offs).to(tl.float32)
    s1 = tl.load(SIN + t * D + offs).to(tl.float32)
    s2 = tl.load(SIN + t * D + HALF + offs).to(tl.float32)
    # rope with rotate_half: [x1, x2] * cos + [-x2, x1] * sin
    o1 = x1 * c1 - x2 * s1
    o2 = x2 * c2 + x1 * s2
    tl.store(base + offs, o1.to(Y.dtype.element_ty))
    tl.store(base + HALF + offs, o2.to(Y.dtype.element_ty))


@triton.jit
def _restore_rope_rows_kernel(Y, W, COS, SIN, NROWS, LH, H: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr,
                              DO_RMS: tl.constexpr, ROWS: tl.constexpr):
    """ROWS (t, l, h) rows of D values per program: the one-row kernel launched T*Lt*H tiny programs (21M at 32K on 72B)."""
    pid = tl.program_id(0).to(tl.int64)
    r = pid * ROWS + tl.arange(0, ROWS).to(tl.int64)
    m = (r < NROWS)[:, None]
    t = r // LH
    l = (r % LH) // H
    offs = tl.arange(0, HALF)[None, :]
    base = r[:, None] * D + offs
    x1 = tl.load(Y + base, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(Y + base + HALF, mask=m, other=0.0).to(tl.float32)
    if DO_RMS:
        wb = l[:, None] * D + offs
        w1 = tl.load(W + wb, mask=m, other=1.0).to(tl.float32)
        w2 = tl.load(W + wb + HALF, mask=m, other=1.0).to(tl.float32)
        u1 = x1 / w1
        u2 = x2 / w2
        ms = (tl.sum(u1 * u1, 1) + tl.sum(u2 * u2, 1)) / D
        rr = (1.0 / tl.sqrt(ms + 1e-6))[:, None]
        x1 = w1 * u1 * rr
        x2 = w2 * u2 * rr
    cb = t[:, None] * D + offs
    c1 = tl.load(COS + cb, mask=m, other=0.0).to(tl.float32)
    c2 = tl.load(COS + cb + HALF, mask=m, other=0.0).to(tl.float32)
    s1 = tl.load(SIN + cb, mask=m, other=0.0).to(tl.float32)
    s2 = tl.load(SIN + cb + HALF, mask=m, other=0.0).to(tl.float32)
    o1 = x1 * c1 - x2 * s1
    o2 = x2 * c2 + x1 * s2
    tl.store(Y + base, o1.to(Y.dtype.element_ty), mask=m)
    tl.store(Y + base + HALF, o2.to(Y.dtype.element_ty), mask=m)


def restore_rope_(Y, kw, cos, sin, rows=32):
    """Y [T, Lt, H, D] contiguous (modified in place); kw [Lt, D] or None; cos/sin [>=T, D]."""
    T, Lt, H, D = Y.shape
    assert Y.is_contiguous()
    n = T * Lt * H
    grid = (triton.cdiv(n, rows),)
    _restore_rope_rows_kernel[grid](Y, kw if kw is not None else Y, cos, sin, n, Lt * H, H=H, D=D, HALF=D // 2,
                                    DO_RMS=kw is not None, ROWS=rows, num_warps=4)
    return Y


@triton.jit
def _build_x8_kernel(SRC, s_h, s_t, COS, SIN, MU, ROWMAX, X8, n, NCOL, col0,
                     HS: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr, ROWS: tl.constexpr,
                     UNROPE: tl.constexpr, PASS: tl.constexpr, F32: tl.constexpr):
    """One source segment ([HS, n, D] post-RoPE K or V of one layer, any head/token strides) -> mapper input columns
    [col0, col0 + HS*D) of every token row: unrope (exact for YaRN's attention factor: divide by cos^2 + sin^2),
    subtract the centering vector, then PASS 0: row amax (atomic max over segments), PASS 1: e4m3 with the row scale."""
    pid = tl.program_id(0).to(tl.int64)
    t = pid * ROWS + tl.arange(0, ROWS).to(tl.int64)          # [ROWS]
    h = tl.arange(0, HS).to(tl.int64)                          # [HS]
    d = tl.arange(0, HALF).to(tl.int64)                        # [HALF]
    m = (t < n)[:, None, None]
    src = SRC + t[:, None, None] * s_t + h[None, :, None] * s_h + d[None, None, :]
    x1 = tl.load(src, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(src + HALF, mask=m, other=0.0).to(tl.float32)
    if UNROPE:
        cb = t[:, None, None] * D + d[None, None, :]
        c1 = tl.load(COS + cb, mask=m, other=1.0)
        c2 = tl.load(COS + cb + HALF, mask=m, other=1.0)
        s1 = tl.load(SIN + cb, mask=m, other=0.0)
        s2 = tl.load(SIN + cb + HALF, mask=m, other=0.0)
        # unrope(k) = (k*cos - rotate_half(k)*sin) / (cos^2 + sin^2), rotate_half(k) = [-k2, k1]
        y1 = (x1 * c1 + x2 * s1) / (c1 * c1 + s1 * s1)
        y2 = (x2 * c2 - x1 * s2) / (c2 * c2 + s2 * s2)
    else:
        y1 = x1
        y2 = x2
    mb = col0 + h[None, :, None] * D + d[None, None, :]
    y1 = y1 - tl.load(MU + mb)
    y2 = y2 - tl.load(MU + mb + HALF)
    if PASS == 0:
        a = tl.maximum(tl.max(tl.max(tl.abs(y1), 2), 1), tl.max(tl.max(tl.abs(y2), 2), 1))
        tl.atomic_max(ROWMAX + t, a, mask=t < n)
    else:
        s = tl.maximum(tl.load(ROWMAX + t, mask=t < n, other=1.0), 1e-12) / 448.0
        s = s[:, None, None]
        dst = X8 + t[:, None, None] * NCOL + mb
        if F32:  # debug/CPU: scaled values unrounded (the Triton interpreter's e4m3 cast is wrong near +-128)
            tl.store(dst, y1 / s, mask=m)
            tl.store(dst + HALF, y2 / s, mask=m)
        else:
            tl.store(dst, (y1 / s).to(X8.dtype.element_ty), mask=m)
            tl.store(dst + HALF, (y2 / s).to(X8.dtype.element_ty), mask=m)


def build_x8(segments, cos, sin, mu, n, rows=4, f32=False):
    """Fused mapper-input build + centered row-wise e4m3 quantization.
    segments: list of (src [HS, n, D] post-RoPE view (token stride arbitrary, D contiguous), col0, unrope: bool),
    covering the columns of the mapper input; cos/sin [n, D] fp32 at the chunk's absolute positions; mu [NCOL] fp32.
    Returns X8 [n, NCOL] float8_e4m3fn and sx [n, 1] fp32 (X ~= X8 * sx + mu), the operands of quant8 in the connector."""
    NCOL = mu.numel()
    dev = mu.device
    rowmax = torch.zeros(n, dtype=torch.float32, device=dev)
    X8 = torch.empty(n, NCOL, dtype=torch.float32 if f32 else torch.float8_e4m3fn, device=dev)
    cos, sin = cos.float().contiguous(), sin.float().contiguous()
    grid = (triton.cdiv(n, rows),)
    for p in (0, 1):
        for src, col0, unrope in segments:
            HS, n_, D = src.shape
            assert n_ == n and src.stride(2) == 1, (src.shape, src.stride())
            _build_x8_kernel[grid](src, src.stride(0), src.stride(1), cos, sin, mu, rowmax, X8, n, NCOL, col0,
                                   HS=HS, D=D, HALF=D // 2, ROWS=rows, UNROPE=bool(unrope), PASS=p, F32=f32,
                                   num_warps=4)
    sx = (rowmax.clamp(min=1e-12) / 448.0)[:, None]
    return X8, sx
