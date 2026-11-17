"""Batched decoding for the hybrid models: per-sequence lengths, CUDA graphs, speculative decoding.

A ``BatchCache`` holds B independent sequences of different lengths. Full-attention layers use
``flash_attn_with_kvcache`` (which appends new keys at each row's own length); Gated-DeltaNet layers
keep a per-row conv window and fp32 state. All buffers are static, so a decode step or a whole
speculative round can be captured in one CUDA graph.

Speculative decoding uses the source model (whose cache already exists after a transfer prefill)
as the draft and the target as the verifier. Greedy acceptance keeps outputs identical to plain
greedy target decoding (up to bf16 batch-shape numerics). Rolling back the recurrent layers is exact:
we re-run the delta rule from the round's starting state with ``beta = 0, g = 0`` on rejected
tokens, which leaves the state unchanged at those positions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .qwen35 import Cache, Hybrid


def _fa_kv():
    from flash_attn import flash_attn_with_kvcache  # type: ignore

    return flash_attn_with_kvcache


def _recurrent():
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    return fused_recurrent_gated_delta_rule


class BatchCache:
    def __init__(self, model: Hybrid, B: int, cap: int, window: int | None = None):
        s, dev, dt = model.spec, model.device, model.dtype
        self.model, self.B, self.cap, self.window = model, B, cap, window
        self.k = {j: torch.zeros(B, cap, s.nkv, s.hd, device=dev, dtype=dt) for j in s.full}
        self.v = {j: torch.zeros(B, cap, s.nkv, s.hd, device=dev, dtype=dt) for j in s.full}
        self.conv = {j: torch.zeros(B, s.conv_k - 1, s.conv_dim, device=dev, dtype=dt) for j in s.lin}
        self.S = {j: torch.zeros(B, s.nv, s.dk, s.dv, device=dev, dtype=torch.float32) for j in s.lin}
        self.lens = torch.zeros(B, dtype=torch.int32, device=dev)
        self.pos_off = torch.zeros(B, dtype=torch.int64, device=dev)

    def load(self, b: int, c: Cache):
        """Copy a batch-1 prefill cache into row b. Sliding-ring caches keep their last window-1 keys."""
        n = c.len
        m = n
        for j in self.k:
            if c.ring:
                avail = n - c.base[j]
                m = min(avail, (self.window - 1) if self.window else avail)
                src_k, src_v = c.k[j][0, avail - m:avail], c.v[j][0, avail - m:avail]
            else:
                src_k, src_v = c.k[j][0, :n], c.v[j][0, :n]
            assert m <= self.cap, (m, self.cap)
            self.k[j][b, :m] = src_k
            self.v[j][b, :m] = src_v
        for j in self.conv:
            self.conv[j][b] = c.conv[j][0]
            self.S[j][b] = c.S[j][0]
        self.lens[b] = m
        self.pos_off[b] = n - m

    def snapshot(self):
        return {"S": {j: t.clone() for j, t in self.S.items()},
                "conv": {j: t.clone() for j, t in self.conv.items()}, "lens": self.lens.clone()}


def step(model: Hybrid, bc: BatchCache, tokens, stash: dict | None = None):
    """Append tokens (B, L) to every row of bc (in place, lens advanced). Returns logits (B, L, V) fp32."""
    s = model.spec
    B, L = tokens.shape
    fa = _fa_kv()
    rec = _recurrent()
    h = model.embed(tokens)
    pos = (bc.lens.long() + bc.pos_off)[:, None] + torch.arange(L, device=tokens.device)[None]
    f = pos[..., None].float() * model.inv_freq
    emb = torch.cat((f, f), dim=-1)
    cos, sin = emb.cos().to(model.dtype)[:, :, None, :], emb.sin().to(model.dtype)[:, :, None, :]
    ws = (bc.window - 1, 0) if bc.window else (-1, -1)
    K = s.conv_k
    for i, layer in enumerate(model.layers):
        x = layer.input_layernorm(h)
        if s.types[i] == "full_attention":
            attn = layer.self_attn
            q, gate = attn.q_proj(x).view(B, L, s.nq, 2 * s.hd).chunk(2, dim=-1)
            gate = gate.reshape(B, L, s.nq * s.hd)
            q = model.rope(attn.q_norm(q), cos, sin)
            k = model.rope(attn.k_norm(attn.k_proj(x).view(B, L, s.nkv, s.hd)), cos, sin)
            v = attn.v_proj(x).view(B, L, s.nkv, s.hd)
            o = fa(q, bc.k[i], bc.v[i], k=k, v=v, cache_seqlens=bc.lens, causal=True, window_size=ws)
            m = attn.o_proj(o.reshape(B, L, -1) * torch.sigmoid(gate))
        else:
            la = layer.linear_attn
            qkv = la.in_proj_qkv(x)
            z, b, a = la.in_proj_z(x), la.in_proj_b(x), la.in_proj_a(x)
            xx = torch.cat((bc.conv[i], qkv), dim=1)
            w = model.conv_w(i)
            acc = xx[:, 0:L].float() * w[0].float()
            for t in range(1, K):
                acc = acc + xx[:, t:t + L].float() * w[t].float()
            mixed = F.silu(acc).to(model.dtype)
            q, k, v = torch.split(mixed, [s.key_dim, s.key_dim, s.value_dim], dim=-1)
            q, k = q.reshape(B, L, s.nk, s.dk), k.reshape(B, L, s.nk, s.dk)
            v = v.reshape(B, L, s.nv, s.dv)
            rep = s.nv // s.nk
            if rep > 1:
                q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
            g = -la.A_log.float().exp() * F.softplus(a.float() + la.dt_bias)
            beta = b.sigmoid()
            o, S = rec(q, k, v, g=g, beta=beta, initial_state=bc.S[i], output_final_state=True,
                       use_qk_l2norm_in_kernel=True)
            if stash is not None:
                stash.setdefault(i, []).append((k, v, g, beta, qkv))
            bc.S[i].copy_(S)
            bc.conv[i].copy_(xx[:, -(K - 1):])
            o = la.norm(o.reshape(-1, s.dv), z.reshape(-1, s.dv)).reshape(B, L, -1)
            m = la.out_proj(o)
        h = h + m
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    bc.lens.add_(L)
    return model.lm_head(model.norm(h)).float()


def rollback(model: Hybrid, bc: BatchCache, snap: dict, stash: dict, keep):
    """Keep only the first keep[b] tokens appended since `snap` (exact for the recurrent layers)."""
    rec = _recurrent()
    K = model.spec.conv_k
    for j, items in stash.items():
        k = torch.cat([it[0] for it in items], 1)
        v = torch.cat([it[1] for it in items], 1)
        g = torch.cat([it[2] for it in items], 1)
        beta = torch.cat([it[3] for it in items], 1)
        raw = torch.cat([it[4] for it in items], 1)
        L = k.shape[1]
        m = torch.arange(L, device=k.device)[None] < keep[:, None]
        _, S = rec(k, k, v, g=g * m[..., None], beta=beta * m[..., None].to(beta.dtype),
                   initial_state=snap["S"][j], output_final_state=True, use_qk_l2norm_in_kernel=True)
        bc.S[j].copy_(S)
        xx = torch.cat((snap["conv"][j], raw), dim=1)
        idx = keep[:, None].long() + torch.arange(K - 1, device=k.device)[None]
        bc.conv[j].copy_(xx.gather(1, idx[..., None].expand(-1, -1, xx.shape[-1])))
    bc.lens.copy_(snap["lens"] + keep.to(bc.lens.dtype))


class Graphed:
    """Capture fn() once (after warm-up runs) and replay it. fn must only touch static buffers."""

    def __init__(self, fn, warm: int = 2, enabled: bool = True):
        self.fn, self.enabled = fn, enabled
        if not enabled:
            return
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(warm):
                fn()
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            self.out = fn()
        torch.cuda.synchronize()

    def __call__(self):
        if not self.enabled:
            return self.fn()
        self.g.replay()
        return self.out


class Decoder:
    """Greedy batch decoder: plain target decoding or source-drafted speculative decoding."""

    def __init__(self, tgt: Hybrid, B: int, cap: int, src: Hybrid | None = None, src_cap: int | None = None,
                 src_window: int | None = None, gamma: int = 4, graphs: bool = True):
        self.tgt, self.src, self.B, self.gamma = tgt, src, B, gamma
        self.tc = BatchCache(tgt, B, cap)
        self.dc = BatchCache(src, B, src_cap or cap, window=src_window) if src is not None else None
        self.last = torch.zeros(B, dtype=torch.long, device=tgt.device)
        self.emit = torch.zeros(B, gamma + 1, dtype=torch.long, device=tgt.device)
        self.keep = torch.ones(B, dtype=torch.int32, device=tgt.device)
        with torch.no_grad():
            self.g_plain = Graphed(self._plain, enabled=graphs)
            self.g_spec = Graphed(self._spec, enabled=graphs) if src is not None else None

    # captured bodies -------------------------------------------------------------------
    def _plain(self):
        lg = step(self.tgt, self.tc, self.last[:, None])
        self.last.copy_(lg[:, -1].argmax(-1))
        return self.last

    def _spec(self):
        G = self.gamma
        snap_t, snap_d = self.tc.snapshot(), self.dc.snapshot()
        st_d, st_t = {}, {}
        toks, cur = [self.last], self.last
        for i in range(G + 1):  # feed last, d1..dG to the draft
            lg = step(self.src, self.dc, cur[:, None], stash=st_d)
            cur = lg[:, -1].argmax(-1)
            if i < G:
                toks.append(cur)
        X = torch.stack(toks, 1)  # (B, G+1)
        p = step(self.tgt, self.tc, X, stash=st_t).argmax(-1)  # target prediction after each prefix
        a = (p[:, :G] == X[:, 1:]).int().cumprod(1).sum(1)  # accepted drafts
        keep = (a + 1).int()
        bonus = p.gather(1, a[:, None].long())[:, 0]
        rollback(self.tgt, self.tc, snap_t, st_t, keep)
        rollback(self.src, self.dc, snap_d, st_d, keep)
        E = torch.cat((X[:, 1:], torch.zeros_like(X[:, :1])), 1)
        E.scatter_(1, a[:, None].long(), bonus[:, None])
        self.emit.copy_(E)
        self.keep.copy_(keep)
        self.last.copy_(bonus)
        return self.emit

    # public ----------------------------------------------------------------------------
    def load(self, b: int, tgt_cache: Cache, first_token: int, src_cache: Cache | None = None):
        self.tc.load(b, tgt_cache)
        if src_cache is not None:
            self.dc.load(b, src_cache)
        self.last[b] = first_token

    @torch.no_grad()
    def generate(self, max_new: int, stop_ids=(), spec: bool = False, n_active: int | None = None):
        """Return (list of token lists incl. the first token, stats). Rows >= n_active are padding."""
        n_active = n_active or self.B
        first = self.last.tolist()
        outs = [[t] for t in first[:n_active]]
        done = [t in stop_ids for t in first[:n_active]]
        rounds = 0
        accepted = 0
        acc_row = [0] * n_active
        rounds_row = [0] * n_active
        while not all(done) and min(len(o) for o, d in zip(outs, done) if not d) < max_new:
            rounds += 1
            if spec:
                E = self.g_spec().tolist()
                K = self.keep.tolist()
                for b in range(n_active):
                    if done[b]:
                        continue
                    accepted += K[b] - 1
                    acc_row[b] += K[b] - 1
                    rounds_row[b] += 1
                    for t in E[b][:K[b]]:
                        if len(outs[b]) >= max_new:
                            done[b] = True
                            break
                        outs[b].append(t)
                        if t in stop_ids:
                            done[b] = True
                            break
            else:
                nxt = self.g_plain().tolist()
                for b in range(n_active):
                    if done[b]:
                        continue
                    outs[b].append(nxt[b])
                    if nxt[b] in stop_ids or len(outs[b]) >= max_new:
                        done[b] = True
            for b in range(n_active):
                if len(outs[b]) >= max_new:
                    done[b] = True
        stats = {"rounds": rounds, "mean_accepted_per_round": accepted / max(1, sum(rounds_row)) if spec else 0.0}
        if spec:
            stats["accepted_per_round_by_row"] = [a / max(1, r) for a, r in zip(acc_row, rounds_row)]
        return outs, stats


class TailEngine:
    """Persistent target cache + CUDA graph for the native tail of a transfer prefill.

    The transfer writes mapped K/V straight into the persistent buffers; the recurrent state and conv window are
    copied into static buffers and the tail runs as one graph replay (removes per-kernel launch overhead).
    The returned cache aliases the persistent buffers, so it must be consumed before the next transfer.
    """

    _inst: dict = {}

    @classmethod
    def get(cls, model: Hybrid, B: int, cap: int, L: int):
        key = (id(model), B, cap, L)
        if key not in cls._inst:
            for k in [k for k in cls._inst if k[0] == key[0]]:
                del cls._inst[k]  # one engine per model at a time: bounded memory with variable caps/tails
            torch.cuda.empty_cache()
            cls._inst[key] = cls(model, B, cap, L)
        return cls._inst[key]

    def __init__(self, model: Hybrid, B: int, cap: int, L: int):
        self.model, self.B, self.cap, self.L = model, B, cap, L
        self.bc = BatchCache(model, B, cap)
        self.ids = torch.zeros(B, L, dtype=torch.long, device=model.device)
        self.out = torch.zeros(B, 1, model.lm_head.weight.shape[0], dtype=torch.float32, device=model.device)
        with torch.no_grad():
            self.g = Graphed(self._body)

    def _body(self):
        lg = step(self.model, self.bc, self.ids)
        self.out.copy_(lg[:, -1:])
        return self.out

    def new_cache(self) -> Cache:
        s = self.model.spec
        c = Cache.__new__(Cache)
        c.spec, c.max_len, c.len, c.functional = s, self.cap, 0, False
        c.ring, c.base = False, {j: 0 for j in s.full}
        c.k, c.v = dict(self.bc.k), dict(self.bc.v)
        dev, dt = self.model.device, self.model.dtype
        c.conv = {j: torch.zeros(self.B, s.conv_k - 1, s.conv_dim, device=dev, dtype=dt) for j in s.lin}
        c.S = {j: torch.zeros(self.B, s.nv, s.dk, s.dv, device=dev, dtype=torch.float32) for j in s.lin}
        return c

    @torch.no_grad()
    def run_tail(self, c: Cache, ids_tail):
        for j in c.conv:
            self.bc.conv[j].copy_(c.conv[j])
            self.bc.S[j].copy_(c.S[j])
        self.bc.lens.fill_(c.len)
        self.ids.copy_(ids_tail)
        out = self.g()
        c.conv, c.S = dict(self.bc.conv), dict(self.bc.S)
        c.len += ids_tail.shape[1]
        return out
