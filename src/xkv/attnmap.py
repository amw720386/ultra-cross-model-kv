"""Richer maps for the target's full-attention KV.

Oracle experiments show that recall failures come from the mapped attention KV, not the recurrent state:
swapping in the native attention KV restores RULER at 16K from 0.72 to 1.00, swapping in the native DeltaNet
state does not help. The attention layers are few (16 of 64) and narrow (k|v = 2048 values per token), so they
can afford much richer inputs than one source layer:

    [k|v]_j(t) = sum_{i in L_j} x_i(t) W_ji  +  e(t) U_j  +  e(t-1) P_j  +  b_j

where x_i are the source's normalized residual streams at the candidate layers L_j around the depth-matched
layer and e is the source's input embedding (token identity, which exact copying needs). Fitted in closed form
(ridge) and then fine-tuned end to end with the rest of the maps.
"""
from __future__ import annotations

import torch

from .mapper import candidates
from .qwen35 import Hybrid


class AttnStats:
    def __init__(self, src: Hybrid, tgt: Hybrid, radius: int = 2, embed: bool = True):
        self.ss, self.ts, self.embed = src.spec, tgt.spec, embed
        self.layers = {j: candidates(self.ss.n_layers, self.ts.n_layers, j, radius) for j in self.ts.full}
        h = self.ss.hidden
        dev = tgt.device
        self.dim = {j: h * len(ls) + (2 * h if embed else 0) + 1 for j, ls in self.layers.items()}
        o = 2 * self.ts.nkv * self.ts.hd
        self.FtF = {j: torch.zeros(d, d, device=dev) for j, d in self.dim.items()}
        self.FtY = {j: torch.zeros(d, o, device=dev) for j, d in self.dim.items()}
        self.YtY = {j: torch.zeros(o, device=dev) for j in self.dim}
        self.Ys = {j: torch.zeros(o, device=dev) for j in self.dim}
        self.n = 0

    def features(self, xs: dict, emb, emb_prev, j):
        parts = [xs[i] for i in self.layers[j]]
        if self.embed:
            parts += [emb, emb_prev]
        f = torch.cat([p.reshape(-1, p.shape[-1]).float() for p in parts], dim=1)
        return torch.cat((f, torch.ones(f.shape[0], 1, device=f.device)), dim=1)

    @torch.no_grad()
    def add(self, src: Hybrid, tgt: Hybrid, ids, chunk: int = 8192):
        B, L = ids.shape
        assert B == 1
        need = sorted({i for ls in self.layers.values() for i in ls})
        xs = {}

        def cap_src(kind, i, x=None, **_):
            if kind == "src" and i in need:
                xs.setdefault(i, []).append(x)

        sc = src.new_cache(B, L)
        src.prefill(ids, sc, chunk=chunk, logits=None, cap=cap_src, stop_after=max(need))
        del sc
        X = {i: torch.cat(v, dim=1) for i, v in xs.items()}
        emb = src.embed(ids)
        emb_prev = torch.cat((torch.zeros_like(emb[:, :1]), emb[:, :-1]), dim=1)
        ys = {}

        def cap_tgt(kind, j, x=None, **t):
            if kind == "tgt_full":
                ys.setdefault(j, []).append(torch.cat((t["k"].flatten(2), t["v"].flatten(2)), dim=-1))

        tc = tgt.new_cache(B, L)
        tgt.prefill(ids, tc, chunk=chunk, logits=None, cap=cap_tgt)
        del tc
        for j in self.dim:
            y = torch.cat(ys.pop(j), dim=1).reshape(-1, 2 * self.ts.nkv * self.ts.hd).float()
            for s in range(0, L, 4096):  # bound the feature matrix size
                f = self.features({i: X[i][:, s:s + 4096] for i in self.layers[j]}, emb[:, s:s + 4096],
                                  emb_prev[:, s:s + 4096], j)
                yy = y[s:s + 4096]
                self.FtF[j] += f.T @ f
                self.FtY[j] += f.T @ yy
            self.YtY[j] += (y * y).sum(0)
            self.Ys[j] += y.sum(0)
        self.n += L

    def solve(self, lam: float = 1e-3):
        W, report = {}, {}
        for j in self.dim:
            A = self.FtF[j].double()
            reg = lam * A.diagonal()[:-1].mean()
            A.diagonal()[:-1] += reg
            XtY = self.FtY[j].double()
            w = torch.linalg.solve(A, XtY)
            tss = self.YtY[j].double() - self.Ys[j].double() ** 2 / self.n
            sse = self.YtY[j].double() - 2 * (w * XtY).sum(0) + (w * (self.FtF[j].double() @ w)).sum(0)
            r2 = 1 - sse / tss.clamp_min(1e-12)
            h = r2.shape[0] // 2
            report[j] = {"k": round(float(r2[:h].mean()), 4), "v": round(float(r2[h:].mean()), 4)}
            W[j] = w.float()
        return AttnMap(self.ss, self.ts, self.layers, W, self.embed), report


class AttnMap:
    """Per attention layer j: weights split by input part so no feature concat is needed at inference."""

    def __init__(self, ss, ts, layers: dict, W: dict, embed: bool, dtype=torch.bfloat16):
        self.ss, self.ts, self.layers, self.embed = ss, ts, layers, embed
        h = ss.hidden
        self.parts = {}
        for j, w in W.items():
            d, o = {}, 0
            for i in layers[j]:
                d[i] = w[o:o + h].to(dtype)
                o += h
            if embed:
                d["e"] = w[o:o + h].to(dtype)
                d["p"] = w[o + h:o + 2 * h].to(dtype)
                o += 2 * h
            d["b"] = w[o].to(dtype)
            self.parts[j] = d
        self.ready_at = {j: max(ls) for j, ls in layers.items()}  # source layer after which j can be computed
        self.need = sorted({i for ls in layers.values() for i in ls})

    def trainable(self):
        ps = []
        for d in self.parts.values():
            for k in list(d):
                d[k] = d[k].float().requires_grad_(True)
                ps.append(d[k])
        return ps

    def predict(self, j, xs: dict, emb, emb_prev):
        d = self.parts[j]
        y = d["b"].to(emb.dtype)
        for i in self.layers[j]:
            y = y + xs[i] @ d[i].to(xs[i].dtype)
        if self.embed:
            y = y + emb @ d["e"].to(emb.dtype) + emb_prev @ d["p"].to(emb.dtype)
        return y

    def state_dict(self):
        return {"layers": self.layers, "embed": self.embed,
                "parts": {j: {str(k): v.detach().to(torch.bfloat16).cpu() for k, v in d.items()}
                          for j, d in self.parts.items()}}

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, ss, ts, device):
        sd = torch.load(path, map_location=device)
        m = cls.__new__(cls)
        m.ss, m.ts, m.layers, m.embed = ss, ts, sd["layers"], sd["embed"]
        m.parts = {j: {(int(k) if k.lstrip("-").isdigit() else k): v for k, v in d.items()}
                   for j, d in sd["parts"].items()}
        m.ready_at = {j: max(ls) for j, ls in m.layers.items()}
        m.need = sorted({i for ls in m.layers.values() for i in ls})
        return m


def task_calibrate_attn(src="4b", tgt="27b", n_seqs=64, seq_len=16384, radius=2, lam=1e-3, embed=True, name=None,
                        source="fineweb"):
    import time

    from . import evals
    from .experiments import ART, get

    sm, _ = get(src)
    tm, tok = get(tgt)
    seqs = evals.load_heldout_text(tok, n_seqs, seq_len, seed=0, split="calib", source=source)
    st = AttnStats(sm, tm, radius=radius, embed=embed)
    t = time.time()
    for k, s in enumerate(seqs):
        st.add(sm, tm, torch.tensor([s], device="cuda"))
        if k % 8 == 0:
            print(f"[attn-calib] {k}/{len(seqs)} {time.time() - t:.0f}s mem={torch.cuda.max_memory_allocated() / 1e9:.1f}GB",
                  flush=True)
    m, rep = st.solve(lam=lam)
    name = name or f"{src}-{tgt}-attn-r{radius}{'-emb' if embed else ''}-n{n_seqs}x{seq_len}"
    (ART / "maps").mkdir(parents=True, exist_ok=True)
    m.save(ART / "maps" / f"{name}.pt")
    return {"name": name, "tokens": st.n, "report": rep, "seconds": time.time() - t}
