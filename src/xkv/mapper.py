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

from .qwen35 import Cache, Hybrid, delta_scan


def out_dims(spec, j):
    if spec.types[j] == "full_attention":
        return {"k": spec.nkv * spec.hd, "v": spec.nkv * spec.hd}
    return {"q": spec.key_dim, "k": spec.key_dim, "v": spec.value_dim, "a": spec.nv, "b": spec.nv}


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

        sc = src.new_cache(B, L)
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


def transfer_prefill(src: Hybrid, tgt: Hybrid, mapper: Mapper, ids, tail: int = 1, chunk: int = 8192,
                     window: int | None = None, max_len: int | None = None, timings: dict | None = None,
                     grad: bool = False):
    with torch.set_grad_enabled(grad):
        return _transfer_prefill(src, tgt, mapper, ids, tail, chunk, window, max_len, timings, grad)


def _transfer_prefill(src, tgt, mapper, ids, tail, chunk, window, max_len, timings, grad):
    """Build the target cache for prompt `ids` (1, T) from a source prefill; return (cache, first_logits).

    The source prefills ids[:, :T-tail]; the target natively processes the last `tail` tokens
    ("tail repair") on top of the synthesized cache, which also yields the first-token logits.
    """
    assert tail >= 1
    sync = torch.cuda.synchronize
    T = ids.shape[1]
    Tm = T - tail
    ts = tgt.spec
    tc = tgt.new_cache(ids.shape[0], max_len or T + 512, functional=grad)
    sync()
    t0 = time.perf_counter()
    if Tm > 0:
        sc = src.new_cache(ids.shape[0], Tm)
        kd = ts.key_dim
        cur = {"s": 0, "L": 0, "rope": None}

        def cap(kind, i, x=None, **_):
            if kind != "src" or i not in mapper.by_src:
                return
            Y = _mm(x, mapper.W[i])
            s, L = cur["s"], cur["L"]
            B = x.shape[0]
            for j in mapper.by_src[i]:
                a0, a1 = mapper.slices[i][j]
                y = Y[..., a0:a1]
                if ts.types[j] == "full_attention":
                    h = ts.nkv * ts.hd
                    k = y[..., :h].reshape(B, L, ts.nkv, ts.hd)
                    v = y[..., h:].reshape(B, L, ts.nkv, ts.hd)
                    cos, sin = cur["rope"]
                    tc.write_kv(j, s, tgt.rope(k, cos, sin), v)
                else:
                    la = tgt.layers[j].linear_attn
                    kv = y[..., : kd + ts.value_dim]
                    a = y[..., kd + ts.value_dim: kd + ts.value_dim + ts.nv]
                    b = y[..., kd + ts.value_dim + ts.nv:]
                    xx = torch.cat((tc.conv[j][:, kd:], kv.transpose(1, 2)), dim=-1)
                    tc.conv[j] = torch.cat((tc.conv[j][:, :kd], xx[..., -(ts.conv_k - 1):]), dim=1)
                    mixed = F.silu(F.conv1d(xx, la.conv1d.weight[kd:], None, groups=kd + ts.value_dim)).transpose(1, 2)
                    k = mixed[..., :kd].reshape(B, L, ts.nk, ts.dk)
                    v = mixed[..., kd:].reshape(B, L, ts.nv, ts.dv)
                    rep = ts.nv // ts.nk
                    if rep > 1:
                        k = k.repeat_interleave(rep, dim=2)
                    g = -la.A_log.float().exp() * F.softplus(a.float() + la.dt_bias)
                    _, S = delta_scan(k, k, v, g, b.sigmoid(), tc.S[j])
                    tc.S[j] = S
                    if s + L == Tm:  # q channels of the conv state for the final tokens
                        n = ts.conv_k - 1
                        qraw = _mm(x[:, -n:], mapper.Wq[j]).transpose(1, 2)
                        if qraw.shape[-1] < n:
                            qraw = torch.cat((tc.conv[j][:, :kd, qraw.shape[-1]:], qraw), dim=-1)
                        tc.conv[j] = torch.cat((qraw, tc.conv[j][:, kd:]), dim=1)

        for s in range(0, Tm, chunk):
            e = min(s + chunk, Tm)
            cur.update(s=s, L=e - s, rope=tgt.rope_tables(s, e - s))
            src.forward(ids[:, s:e], sc, logits=None, window=window, cap=cap, stop_after=mapper.max_src)
        del sc
        tc.len = Tm
    sync()
    t1 = time.perf_counter()
    logits = tgt.forward(ids[:, Tm:], tc, logits="last")
    sync()
    t2 = time.perf_counter()
    if timings is not None:
        timings.update(src_and_map=t1 - t0, tail=t2 - t1, total=t2 - t0)
    return tc, logits


@torch.no_grad()
def native_prefill(model: Hybrid, ids, chunk: int = 8192, max_len: int | None = None, timings: dict | None = None,
                   window: int | None = None):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c = model.new_cache(ids.shape[0], max_len or ids.shape[1] + 512)
    logits = model.prefill(ids, c, chunk=chunk, window=window)
    torch.cuda.synchronize()
    if timings is not None:
        timings.update(total=time.perf_counter() - t0)
    return c, logits
