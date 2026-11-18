"""Cross-model cache synthesis.

For every target layer j we learn a ridge map from the *source* model's normalized residual
stream at a depth-matched layer i(j) to the per-token quantities the target layer caches:

* full-attention layers:  k (post k_norm, pre-RoPE) and v      -> written to the target KV cache
* Gated DeltaNet layers:  raw q|k|v projections (pre-conv), a, b -> run through the target's own
  conv + gated delta rule scan to synthesize the recurrent state.

Because the target's own nonlinearities (RoPE, conv, gates, delta rule) are applied to the mapped
inputs, the map stays linear and position-free while the cache it produces is exactly the shape and
semantics the target expects. Prefill cost for the target is reduced to a set of GEMMs plus linear-time
scans; the target never runs its MLPs or attention over the prompt.
"""
from __future__ import annotations

import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from . import qwen35 as Q
from .kernels import conv_silu, next_conv_state
from .qwen35 import Cache, Hybrid, delta_scan, delta_state, dwconv_silu


def out_dims(spec, j):
    if spec.types[j] == "full_attention":
        return {"k": spec.nkv * spec.hd, "v": spec.nkv * spec.hd}
    return {"q": spec.key_dim, "k": spec.key_dim, "v": spec.value_dim, "a": spec.nv, "b": spec.nv}


PROFILE: dict = {}  # {"mode": "src" | "gemm_only" | "gemm" | None} -- timing ablations only


def candidates(n_src: int, n_tgt: int, j: int, radius: int = 2) -> list[int]:
    c = round(j * (n_src - 1) / (n_tgt - 1))
    return sorted({min(max(c + d, 0), n_src - 1) for d in range(-radius, radius + 1)})


class Stats:
    """Streaming sufficient statistics for ridge fits (all fp32 on GPU)."""

    def __init__(self, src: Hybrid, tgt: Hybrid, radius: int = 2):
        self.ss, self.ts = src.spec, tgt.spec
        dev = tgt.device
        d = self.ss.hidden + 1
        self.cand = {j: candidates(self.ss.n_layers, self.ts.n_layers, j, radius) for j in range(self.ts.n_layers)}
        used = sorted({i for c in self.cand.values() for i in c})
        self.XtX = {i: torch.zeros(d, d, device=dev) for i in used}
        self.XtY, self.YtY, self.Ys = {}, {}, {}
        for j, cs in self.cand.items():
            o = sum(out_dims(self.ts, j).values())
            for i in cs:
                self.XtY[(i, j)] = torch.zeros(d, o, device=dev)
            self.YtY[j] = torch.zeros(o, device=dev)
            self.Ys[j] = torch.zeros(o, device=dev)
        self.n = 0
        self._x = {}

    @staticmethod
    def _aug(x):
        x = x.reshape(-1, x.shape[-1]).float()
        return torch.cat((x, torch.ones(x.shape[0], 1, device=x.device)), dim=1)

    @torch.no_grad()
    def add(self, src: Hybrid, tgt: Hybrid, ids, window=None, chunk=8192):
        """Accumulate statistics over token ids (B, L)."""
        B, L = ids.shape
        assert B == 1, "calibration streams one sequence at a time"
        need = set(self.XtX)
        xs = {}

        def cap_src(kind, i, x=None, **_):
            if kind == "src" and i in need:
                xs.setdefault(i, []).append(x)

        sc = src.new_cache(B, L, window=window, chunk=chunk)
        src.prefill(ids, sc, chunk=chunk, logits=None, window=window, cap=cap_src, stop_after=max(need))
        del sc
        X = {i: self._aug(torch.cat(v, dim=1)) for i, v in xs.items()}
        del xs
        for i, x in X.items():
            self.XtX[i] += x.T @ x

        pos = [0]

        def cap_tgt(kind, j, x=None, **t):
            if kind == "tgt_full":
                y = torch.cat((t["k"].flatten(2), t["v"].flatten(2)), dim=-1)
            elif kind == "tgt_lin":
                y = torch.cat((t["qkv"], t["a"], t["b"]), dim=-1)
            else:
                return
            y = y.reshape(-1, y.shape[-1]).float()
            n = y.shape[0]
            s = pos[0]
            for i in self.cand[j]:
                self.XtY[(i, j)] += X[i][s:s + n].T @ y
            self.YtY[j] += (y * y).sum(0)
            self.Ys[j] += y.sum(0)
            if j == self.ts.n_layers - 1:
                pos[0] += n

        tc = tgt.new_cache(B, L)
        tgt.prefill(ids, tc, chunk=chunk, logits=None, cap=cap_tgt)
        self.n += B * L
        assert pos[0] == B * L, (pos[0], B * L)

    def solve(self, lam: float = 1e-3, select: str = "r2"):
        """Return a Mapper choosing, per target layer, the candidate source layer with the best fit R^2."""
        maps, report = {}, {}
        for j, cs in self.cand.items():
            dims = out_dims(self.ts, j)
            tss = self.YtY[j].double() - self.Ys[j].double() ** 2 / self.n
            best = None
            for i in cs:
                A = self.XtX[i].double()
                reg = lam * A.diagonal()[:-1].mean()
                A = A + torch.diag(torch.cat((torch.full((A.shape[0] - 1,), float(reg), device=A.device),
                                              torch.zeros(1, device=A.device, dtype=A.dtype))))
                XtY = self.XtY[(i, j)].double()
                W = torch.linalg.solve(A, XtY)
                sse = self.YtY[j].double() - 2 * (W * XtY).sum(0) + (W * (self.XtX[i].double() @ W)).sum(0)
                r2 = 1 - sse / tss.clamp_min(1e-12)
                groups, o = {}, 0
                for name, d in dims.items():
                    groups[name] = float(r2[o:o + d].mean())
                    o += d
                score = groups["k"] + groups["v"]
                if best is None or score > best[0]:
                    best = (score, i, W.float(), groups)
            _, i, W, groups = best
            maps[j] = (i, W)
            report[j] = {"src": i, **{k: round(v, 4) for k, v in groups.items()}}
        return Mapper(self.ss, self.ts, maps), report


class Mapper:
    def __init__(self, src_spec, tgt_spec, maps: dict, dtype=torch.bfloat16):
        self.ss, self.ts = src_spec, tgt_spec
        self.src_of = {j: i for j, (i, _) in maps.items()}
        self.by_src = defaultdict(list)
        for j, (i, _) in sorted(maps.items()):
            self.by_src[i].append(j)
        self.max_src = max(self.src_of.values())
        # Per source layer: one fused GEMM producing every target quantity needed per token
        # (q of linear layers is excluded; it is only needed for the last conv_k-1 tokens).
        self.W, self.Wq, self.slices = {}, {}, {}
        for i, js in self.by_src.items():
            cols, sl, o = [], {}, 0
            for j in js:
                W = maps[j][1]
                if tgt_spec.types[j] == "full_attention":
                    cols.append(W)
                    sl[j] = (o, o + W.shape[1])
                    o += W.shape[1]
                else:
                    kd = tgt_spec.key_dim
                    self.Wq[j] = W[:, :kd].to(dtype)
                    rest = W[:, kd:]
                    cols.append(rest)
                    sl[j] = (o, o + rest.shape[1])
                    o += rest.shape[1]
            self.W[i] = torch.cat(cols, dim=1).to(dtype)
            self.slices[i] = sl

    def to(self, device):
        self.W = {i: w.to(device) for i, w in self.W.items()}
        self.Wq = {j: w.to(device) for j, w in self.Wq.items()}
        return self

    def trainable(self):
        """Turn the maps into fp32 leaf tensors that require grad; return them."""
        self.W = {i: w.float().requires_grad_(True) for i, w in self.W.items()}
        self.Wq = {j: w.float().requires_grad_(True) for j, w in self.Wq.items()}
        return list(self.W.values()) + list(self.Wq.values())

    def n_params(self):
        return sum(w.numel() for w in self.W.values()) + sum(w.numel() for w in self.Wq.values())

    def state_dict(self):
        return {"W": self.W, "Wq": self.Wq, "slices": self.slices, "src_of": self.src_of}

    def save(self, path):
        bf = torch.bfloat16
        torch.save({"W": {i: w.detach().to(bf).cpu() for i, w in self.W.items()},
                    "Wq": {j: w.detach().to(bf).cpu() for j, w in self.Wq.items()},
                    "slices": self.slices, "src_of": self.src_of}, path)

    @classmethod
    def load(cls, path, src_spec, tgt_spec, device):
        d = torch.load(path, map_location=device)
        m = cls.__new__(cls)
        m.ss, m.ts = src_spec, tgt_spec
        m.W, m.Wq, m.slices, m.src_of = d["W"], d["Wq"], d["slices"], d["src_of"]
        m.by_src = defaultdict(list)
        for j, i in sorted(m.src_of.items()):
            m.by_src[i].append(j)
        m.max_src = max(m.src_of.values())
        return m


def _mm(x, W):
    """x (..., d) @ W (d+1, o) with the bias row last."""
    W = W.to(x.dtype)
    return torch.addmm(W[-1], x.reshape(-1, x.shape[-1]), W[:-1]).view(*x.shape[:-1], W.shape[1])


def adaptive_tail(n: int, lo: int = 128, hi: int = 512, short: int = 4096, long: int = 8192) -> int:
    """Length-dependent native repair tail: `hi` tokens for prompts up to `short`, tapering linearly to `lo` at
    `long` and beyond. Short prompts gain little TTFT from transfer, while exact constraint recall (puzzles, code
    specs) depends on the mapped attention KV; >=8K prompts keep the fixed 128-token tail (TTFT unchanged)."""
    if n <= short:
        t = hi
    elif n >= long:
        t = lo
    else:
        t = round(hi - (hi - lo) * (n - short) / (long - short))
    return min(t, n)


def transfer_prefill(src: Hybrid, tgt: Hybrid, mapper: Mapper, ids, tail: int = 1, chunk: int = 8192,
                     window: int | None = None, max_len: int | None = None, timings: dict | None = None,
                     grad: bool = False, keep_src: bool = False, full_until: int | None = None,
                     graph_tail: bool | None = None, sink: int = 0, between=None, native_spans=None,
                     oracle: str | None = None):
    """keep_src=True also runs every source layer and the tail through the source and returns
    (cache, logits, source_cache) so the source can draft for speculative decoding."""
    if full_until is not None and ids.shape[1] <= full_until:
        window = None  # short enough: the source uses full attention; the sliding window only kicks in beyond
    with torch.set_grad_enabled(grad):
        return _transfer_prefill(src, tgt, mapper, ids, tail, chunk, window, max_len, timings, grad, keep_src,
                                 graph_tail, sink, between, native_spans, oracle)


GRAPH_TAIL_MAX = 32768  # persistent graphed tail for prompts up to this length (memory-bounded; None disables)


def _transfer_prefill(src, tgt, mapper, ids, tail, chunk, window, max_len, timings, grad, keep_src=False,
                      graph_tail=None, sink=0, between=None, native_spans=None, oracle=None):
    """Build the target cache for prompt `ids` (1, T) from a source prefill; return (cache, first_logits).

    The source prefills ids[:, :T-tail]; the target natively processes the last `tail` tokens
    ("tail repair") on top of the synthesized cache, which also yields the first-token logits.
    native_spans: optional [(a, b), ...] inside the mapped region that the target computes natively in place
    (e.g. tool schemas, where exact strings matter); the source still reads them for its own state.
    """
    assert tail >= 1
    sync = torch.cuda.synchronize
    T = ids.shape[1]
    Tm = T - tail
    ts = tgt.spec
    if graph_tail is None:
        graph_tail = (not grad) and GRAPH_TAIL_MAX is not None and T <= GRAPH_TAIL_MAX and T - tail > 0
    eng = None
    if graph_tail:
        from .decode import TailEngine

        q = 8192
        eng = TailEngine.get(tgt, ids.shape[0], ((max(max_len or 0, T + 512) + q - 1) // q) * q, tail)
        tc = eng.new_cache()
    else:
        if not grad:
            from .decode import TailEngine

            TailEngine.release()  # long prompts need the memory more than the graphed tail
        tc = tgt.new_cache(ids.shape[0], max_len or T + 512, functional=grad)
    sync()
    t0 = time.perf_counter()
    sc = None
    if Tm > 0:
        sc = src.new_cache(ids.shape[0], T if keep_src else Tm, window=window, chunk=chunk)
        kd = ts.key_dim
        cur = {"s": 0, "L": 0, "rope": None}

        sink = sink if Tm > sink else 0
        if sink:
            # the first `sink` tokens (attention sink) are computed natively by the target
            tgt.forward(ids[:, :sink], tc, logits=None)

        attn = getattr(mapper, "attn", None)

        def cap(kind, i, x=None, **_):
            if kind != "src" or PROFILE.get("mode") == "src" or cur.get("skip"):
                return
            off = max(0, sink - cur["s"])
            if off >= x.shape[1]:
                return
            if attn is not None and i in attn.need:
                xo = x[:, off:] if off else x
                cur["buf"][i] = xo
                for j, r in attn.ready_at.items():
                    if r != i:
                        continue
                    B_, L_ = xo.shape[:2]
                    y = attn.predict(j, cur["buf"], cur["emb"][:, off:], cur["emb_prev"][:, off:])
                    h = ts.nkv * ts.hd
                    k = y[..., :h].reshape(B_, L_, ts.nkv, ts.hd)
                    v = y[..., h:].reshape(B_, L_, ts.nkv, ts.hd)
                    cos, sin = cur["rope"]
                    if off:
                        cos, sin = cos[:, off:], sin[:, off:]
                    tc.write_kv(j, cur["s"] + off, tgt.rope(k, cos, sin), v)
            if i not in mapper.by_src:
                return
            Y = _mm(x[:, off:] if off else x, mapper.W[i])
            if PROFILE.get("mode") == "gemm_only":
                return
            s, L = cur["s"] + off, cur["L"] - off
            B = x.shape[0]
            for j in mapper.by_src[i]:
                a0, a1 = mapper.slices[i][j]
                y = Y[..., a0:a1]
                if ts.types[j] == "full_attention":
                    if attn is not None:
                        continue  # written by the richer attention map above
                    h = ts.nkv * ts.hd
                    k = y[..., :h].reshape(B, L, ts.nkv, ts.hd)
                    v = y[..., h:].reshape(B, L, ts.nkv, ts.hd)
                    cos, sin = cur["rope"]
                    if off:
                        cos, sin = cos[:, off:], sin[:, off:]
                    tc.write_kv(j, s, tgt.rope(k, cos, sin), v)
                else:
                    la = tgt.layers[j].linear_attn
                    kv = y[..., : kd + ts.value_dim]
                    a = y[..., kd + ts.value_dim: kd + ts.value_dim + ts.nv]
                    b = y[..., kd + ts.value_dim + ts.nv:]
                    if Q.FUSED_CONV and not grad:
                        mixed = conv_silu(tc.conv[j][..., kd:], kv, tgt.conv_w(j)[:, kd:])
                        tc.conv[j] = torch.cat((tc.conv[j][..., :kd], next_conv_state(tc.conv[j][..., kd:], kv)),
                                               dim=-1)
                    else:
                        xx = torch.cat((tc.conv[j][..., kd:], kv), dim=1)
                        tc.conv[j] = torch.cat((tc.conv[j][..., :kd], xx[:, -(ts.conv_k - 1):]), dim=-1)
                        mixed = dwconv_silu(xx, tgt.conv_w(j)[:, kd:])
                    k = mixed[..., :kd].reshape(B, L, ts.nk, ts.dk)
                    v = mixed[..., kd:].reshape(B, L, ts.nv, ts.dv)
                    rep = ts.nv // ts.nk
                    if rep > 1:
                        k = k.repeat_interleave(rep, dim=2)
                    g = -la.A_log.float().exp() * F.softplus(a.float() + la.dt_bias)
                    if PROFILE.get("mode") == "gemm":
                        continue
                    tc.S[j] = delta_state(k, v, g, b.sigmoid(), tc.S[j])
                    if s + L == cur["seg_end"]:  # q channels of the conv state before native tokens
                        n = ts.conv_k - 1
                        qraw = _mm(x[:, -n:], mapper.Wq[j])
                        if qraw.shape[1] < n:
                            qraw = torch.cat((tc.conv[j][:, qraw.shape[1]:, :kd], qraw), dim=1)
                        tc.conv[j] = torch.cat((qraw, tc.conv[j][..., kd:]), dim=-1)

        spans = sorted((max(a, sink), min(b, Tm)) for a, b in (native_spans or []) if min(b, Tm) > max(a, sink))
        segs, pos = [], 0
        for a, b in spans:
            if a > pos:
                segs.append((pos, a, False))
            segs.append((max(a, pos), b, True))
            pos = max(pos, b)
        if pos < Tm:
            segs.append((pos, Tm, False))
        for a, b, native in segs:
            for s in range(a, b, chunk):
                e = min(s + chunk, b)
                cur.update(s=s, L=e - s, rope=tgt.rope_tables(s, e - s), seg_end=b, skip=native, buf={})
                if attn is not None and not native:
                    emb = src.embed(ids[:, max(s - 1, 0):e])
                    if s == 0:
                        emb = torch.cat((torch.zeros_like(emb[:, :1]), emb), dim=1)
                    cur.update(emb=emb[:, 1:], emb_prev=emb[:, :-1])
                src.forward(ids[:, s:e], sc, logits=None, window=window, cap=cap,
                            stop_after=None if keep_src else mapper.max_src)
                if between is not None:
                    between()
            if native:
                tc.len = a
                tgt.forward(ids[:, a:b], tc, logits=None)
        tc.len = Tm
        if oracle:
            # diagnostic only: replace one component of the mapped cache with the native one
            cn = tgt.new_cache(ids.shape[0], Tm + 8)
            tgt.prefill(ids[:, :Tm], cn, chunk=chunk, logits=None)
            if oracle in ("attn", "both"):
                for j in cn.k:
                    tc.k[j][:, :Tm] = cn.k[j][:, :Tm]
                    tc.v[j][:, :Tm] = cn.v[j][:, :Tm]
            if oracle in ("lin", "both"):
                for j in cn.S:
                    tc.S[j] = cn.S[j].clone()
                    tc.conv[j] = cn.conv[j].clone()
            del cn
    sync()
    t1 = time.perf_counter()
    if eng is not None:
        logits = eng.run_tail(tc, ids[:, Tm:])
    else:
        logits = tgt.forward(ids[:, Tm:], tc, logits="last")
    sync()
    t2 = time.perf_counter()
    if timings is not None:
        timings.update(src_and_map=t1 - t0, tail=t2 - t1, total=t2 - t0)
    if keep_src:
        if sc is None:
            sc = src.new_cache(ids.shape[0], T, window=window, chunk=chunk)
        src.prefill(ids[:, Tm:], sc, chunk=chunk, logits=None, window=window)
        return tc, logits, sc
    return tc, logits


@torch.no_grad()
def native_prefill(model: Hybrid, ids, chunk: int = 8192, max_len: int | None = None, timings: dict | None = None,
                   window: int | None = None, between=None):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c = model.new_cache(ids.shape[0], max_len or ids.shape[1] + 512)
    logits = model.prefill(ids, c, chunk=chunk, window=window, between=between)
    torch.cuda.synchronize()
    if timings is not None:
        timings.update(total=time.perf_counter() - t0)
    return c, logits
