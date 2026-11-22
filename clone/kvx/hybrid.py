"""Cross-model cache transfer for any HF decoder through module taps (kvx-pairs).

Works for pairs the KV-only path (core.py) cannot handle: hybrid linear-attention models (Qwen3.5 / Qwen3.8
GatedDeltaNet layers carry a conv state and a recurrent state, not a per-token KV), partial RoPE, YaRN, VLM
wrappers.

Idea: a target's cache is a function of a few per-token module outputs ("taps"):
  attention layer:   pre-RoPE key (k_norm output, else k_proj output) and v_proj output
  GatedDeltaNet:     in_proj_qkv (pre-conv q,k,v), in_proj_a, in_proj_b
Calibration records the taps of both models and fits a ridge map per target layer from a window of source layers.
At eval/serve time the target runs over the prefix with forward hooks that REPLACE its taps by the mapped values
(positions >= A, the native sink tokens). The model's own code then builds the cache (RoPE, conv state, chunked
gated delta rule), so the cache is exactly "mapped quantities in, target cache semantics out". Only the cache is
kept; the continuation (or a native tail) runs on it. The prefix pass through the target is an eval device: a
serving kernel computes the same cache from the mapped taps without the target's MLPs.
"""
import math

import torch


# ---------------------------------------------------------------- model access

class Dec:
    """Text decoder view of a CausalLM or *ForConditionalGeneration: .text (layers), .lm_head, .cfg."""

    def __init__(self, model):
        inner = model.model
        self.model = model
        self.text = getattr(inner, "language_model", inner)
        self.lm_head = model.lm_head
        cfg = model.config
        self.cfg = getattr(cfg, "text_config", None) or cfg
        self.layers = list(self.text.layers)

    @property
    def device0(self):  # device of the input embeddings (multi-GPU device_map: where features are gathered)
        return self.text.get_input_embeddings().weight.device

    @torch.no_grad()
    def forward(self, ids, cache=None, keep=0, use_cache=True):
        out = self.text(input_ids=ids, past_key_values=cache, use_cache=use_cache)
        h = out.last_hidden_state
        if keep:
            h = h[:, -keep:]
        logits = self.lm_head(h)
        cap = getattr(self.cfg, "final_logit_softcapping", None)
        if cap:
            logits = torch.tanh(logits / cap) * cap
        return logits, out.past_key_values


def load(name, dtype=torch.bfloat16, attn="sdpa", path=None, device_map=None, max_memory=None):
    from transformers import AutoConfig, AutoModelForCausalLM
    p = path or name
    cfg = AutoConfig.from_pretrained(p)
    mt = getattr(getattr(cfg, "text_config", cfg), "model_type", "")
    if mt.startswith("gemma4") and attn == "sdpa":
        # eager: in bf16 its prefill and cached-decode paths are bit-identical (two-step == one-shot), so the native
        # two-step floor measures only the mapping; sdpa's kernels differ between the two paths by bf16 noise
        # (fp32: sdpa == eager to 1e-6, so this is consistency, not correctness)
        attn = "eager"
    if cfg.architectures and "ConditionalGeneration" in cfg.architectures[0]:
        from transformers import AutoModelForImageTextToText
        # per sub-config attention: the text model gets `attn`; vision/audio towers (unused here) get eager, since
        # Gemma4AudioModel rejects sdpa
        subs = list(getattr(cfg, "sub_configs", {}) or {})
        impl = {k: (attn if k == "text_config" else "eager") for k in subs} if subs else attn
        if isinstance(impl, dict):
            impl[""] = attn
        m = AutoModelForImageTextToText.from_pretrained(p, dtype=dtype, attn_implementation=impl,
                                                        device_map=device_map, max_memory=max_memory)
    else:
        m = AutoModelForCausalLM.from_pretrained(p, dtype=dtype, attn_implementation=attn, device_map=device_map,
                                                 max_memory=max_memory)
    return Dec(m.eval())


def layer_taps(dec):
    """Per layer: (kind, [(tap name, module, heads-flag)]); kind in attn | lin | none (KV-shared layer: no own cache).
    attn taps: k = k_norm output if present else k_proj output (pre-RoPE); v = v_norm output if present (Gemma 4:
    what the cache stores) else v_proj output."""
    out = []
    for l in dec.layers:
        if getattr(l, "linear_attn", None) is not None:
            la = l.linear_attn
            out.append(("lin", [("qkv", la.in_proj_qkv, 0), ("a", la.in_proj_a, 0), ("b", la.in_proj_b, 0)]))
            continue
        sa = l.self_attn
        if hasattr(sa, "kv_a_proj_with_mqa"):  # MLA (GLM-4.7-Flash, GLM-5.3, DeepSeek): latent + rope key, pre-norm
            taps = [("ckv", sa.kv_a_proj_with_mqa, 0)]
            ix = getattr(sa, "indexer", None)
            if ix is not None:  # DSA indexer key cache (pre-RoPE, post k_norm)
                taps.append(("ik", ix.k_norm, 0))
            out.append(("mla", taps))
            continue
        if getattr(sa, "is_kv_shared_layer", False):
            out.append(("none", []))
            continue
        kmod = getattr(sa, "k_norm", None)
        vmod = getattr(sa, "v_norm", None)
        out.append(("attn", [("k", kmod if kmod is not None else sa.k_proj, int(kmod is not None)),
                             ("v", vmod if vmod is not None else sa.v_proj, int(vmod is not None))]))
    return out


def tap_widths(dec):
    """Per layer: [(tap name, width)] of the flattened [B,T,width] outputs, from a 2-token probe forward (per-layer
    head dims differ in Gemma 4). Cached on dec."""
    if getattr(dec, "_tap_widths", None) is None:
        got = {}
        p = next(dec.text.parameters())
        ids = torch.ones(1, 2, dtype=torch.long, device=p.device)
        with Taps(dec).record(lambda l, d: got.__setitem__(l, [(k, v.shape[-1]) for k, v in d.items()])):
            dec.forward(ids, keep=1, use_cache=False)
        order = {k: [n for n, _, _ in taps] for k, taps in [(i, t[1]) for i, t in enumerate(layer_taps(dec))]}
        dec._tap_widths = [sorted(got.get(l, []), key=lambda x: order[l].index(x[0])) for l in range(len(dec.layers))]
    return dec._tap_widths


def lin_dims(dec):
    """GatedDeltaNet geometry (same for every lin layer of a model), stored in kvx-hybrid-v1 files."""
    c = dec.cfg
    if not hasattr(c, "linear_key_head_dim"):
        return None
    return dict(key_dim=c.linear_key_head_dim * c.linear_num_key_heads,
                value_dim=c.linear_value_head_dim * c.linear_num_value_heads,
                num_k_heads=c.linear_num_key_heads, num_v_heads=c.linear_num_value_heads,
                head_k_dim=c.linear_key_head_dim, head_v_dim=c.linear_value_head_dim,
                conv_kernel=c.linear_conv_kernel_dim, qkv_order="q|k|v")


def lin_q_width(dec):
    c = dec.cfg
    return c.linear_key_head_dim * c.linear_num_key_heads


class Taps:
    """Forward hooks on every tap. record: outputs (flattened [B,T,w]) go to on_layer(l, {name: tensor}) once all
    taps of layer l fired. override: tap outputs at positions >= start are replaced by given[l][name][:, :T]."""

    def __init__(self, dec, layers=None):
        self.dec, self.hs = dec, []
        self.taps = layer_taps(dec)
        self.layers = set(range(len(self.taps)) if layers is None else layers)
        self.mode, self.buf = None, {}

    def __enter__(self):
        for l, (kind, taps) in enumerate(self.taps):
            if l not in self.layers:
                continue
            for name, mod, _ in taps:
                self.hs.append(mod.register_forward_hook(self._hook(l, name, len(taps))))
        return self

    def __exit__(self, *a):
        for h in self.hs:
            h.remove()
        self.hs = []

    def _hook(self, l, name, ntaps):
        def f(mod, args, out):
            B, T = out.shape[0], out.shape[1]
            if self.mode == "record":
                d = self.buf.setdefault(l, {})
                d[name] = out.reshape(B, T, -1)
                if len(d) == ntaps:
                    self.on_layer(l, self.buf.pop(l))
                return None
            if self.mode == "override":
                g = self.given.get(l, {}).get(name)
                if g is None:
                    return None
                s = self.start
                if s >= T:
                    return None
                new = out.clone()
                flat = new.view(B, T, -1)
                flat[:, s:] = g[:, s:T].to(device=out.device, dtype=out.dtype)
                return new
            return None
        return f

    def record(self, on_layer):
        self.mode, self.on_layer = "record", on_layer
        return self

    def override(self, given, start=0):
        self.mode, self.given, self.start = "override", given, start
        return self


# ---------------------------------------------------------------- features

def source_features(src, ids, drop_q=True):
    """Per source layer: [B*T, d_l] float32 features (concatenated taps; GatedDeltaNet q dropped by default)."""
    feats = {}
    qw = lin_q_width(src) if any(k == "lin" for k, _ in layer_taps(src)) else 0
    taps_s = layer_taps(src)
    kinds = [k for k, _ in taps_s]

    def on_layer(l, d):
        if kinds[l] == "lin":
            qkv = d["qkv"][..., qw:] if drop_q else d["qkv"]
            x = torch.cat([qkv, d["a"], d["b"]], -1)
        else:
            x = torch.cat([d[n] for n, _, _ in taps_s[l][1]], -1)
        feats[l] = x.reshape(-1, x.shape[-1]).to(device=src.device0, dtype=torch.bfloat16)

    with Taps(src).record(on_layer):
        src.forward(ids, keep=1, use_cache=False)
    return [feats.get(l) for l in range(len(kinds))]  # None for KV-shared layers


def windows(Ls, Lt, w, kinds_s, kinds_t, same_kind=True):
    """Source layers per target layer: the w nearest by relative depth, plus the nearest source layer of the same
    kind (an attention target always sees a source attention layer)."""
    res = []
    for t in range(Lt):
        if kinds_t[t] == "none":
            res.append(())
            continue
        c = (t + 0.5) * Ls / Lt - 0.5
        order = sorted([s for s in range(Ls) if kinds_s[s] != "none"], key=lambda s: (abs(s - c), s))
        sel = set(order if w <= 0 else order[:w])  # w <= 0: every source layer (attention-only pairs: k ~ Ls was best)
        if same_kind and kinds_t[t] in kinds_s:
            sel.add(next(s for s in order if kinds_s[s] == kinds_t[t]))
        res.append(tuple(sorted(sel)))
    return res


# ---------------------------------------------------------------- ridge over windows

class WindowRidge:
    """Centered ridge Y_t ~ X_{window(t)} W_t + b_t, with one X Gram per distinct window."""

    def __init__(self, ds, dt, win, targets, device, bd=128):
        self.ds, self.dt, self.win, self.targets, self.dev = ds, dt, win, list(targets), device
        self.bd = bd  # block size for recolor covariances (one head); widths that are not multiples get one tail block
        self.wins = sorted({win[t] for t in self.targets})
        self.n = 0
        self.shx = self.shy = None
        z = lambda *s, dtype=torch.float32: torch.zeros(*s, device=device, dtype=dtype)
        self.xx = {w: z(self.din(w), self.din(w)) for w in self.wins}
        self.sx = {w: z(self.din(w), dtype=torch.float64) for w in self.wins}
        self.xy = {t: z(self.din(win[t]), dt[t]) for t in self.targets}
        self.sy = {t: z(dt[t], dtype=torch.float64) for t in self.targets}
        self.yy = {t: z(dt[t], dtype=torch.float64) for t in self.targets}
        self.yyb = {t: z(dt[t] // bd, bd, bd) for t in self.targets}  # sum of y_b y_b^T per full block

    def din(self, w):
        return sum(self.ds[s] for s in w)

    def X(self, feats, w):
        return torch.cat([feats[s].float() - self.shx[s] for s in w], 1)

    def add_x(self, feats):
        if self.shx is None:
            self.shx = [None if f is None else f.float().mean(0) for f in feats]
        N = feats[0].shape[0]
        self.n += N
        self._feats, self._xw = feats, {}
        for w in self.wins:
            X = self.X(feats, w)
            self.xx[w].addmm_(X.T, X)
            self.sx[w] += X.sum(0).double()
            del X
        return N

    def _x(self, w):  # one cached window: consecutive target layers usually share it
        if w not in self._xw:
            self._xw = {w: self.X(self._feats, w)}
        return self._xw[w]

    def add_y(self, t, Y):  # Y [N, dt] float32, rows aligned with the last add_x
        if self.shy is None:
            self.shy = {}
        Y = Y.to(self.dev)
        if t not in self.shy:
            self.shy[t] = Y.mean(0)
        Y = Y - self.shy[t]
        X = self._x(self.win[t])
        Y = Y.to(X.device)
        self.xy[t].addmm_(X.T, Y)
        self.sy[t] += Y.sum(0).double()
        self.yy[t] += (Y.double() ** 2).sum(0)
        nb = self.dt[t] // self.bd
        if nb:
            Yb = Y[:, :nb * self.bd].view(-1, nb, self.bd).transpose(0, 1)  # [nb, N, bd]
            self.yyb[t].baddbmm_(Yb.transpose(1, 2), Yb)

    def done_batch(self):
        self._xw, self._feats = {}, None

    def _recolor(self, t, W, A, lam, my, n):
        """Per block: R = C_hat^-1/2 C_y^1/2 (symmetric roots) so that (Y_hat - mu) R has the target covariance.
        A is the regularized centered X Gram; C_hat uses the unregularized one."""
        nb, bd = self.dt[t] // self.bd, self.bd
        if not nb:
            return None
        Wb = W[:, :nb * bd].view(W.shape[0], nb, bd).permute(1, 0, 2)  # [nb, din, bd]
        A.diagonal().sub_(lam)
        Ch = (torch.einsum("bid,ij,bje->bde", Wb, A, Wb) / n).double()
        A.diagonal().add_(lam)
        mb = my[:nb * bd].double().view(nb, bd)
        Cy = self.yyb[t].to(W.device).double() / n - mb[:, :, None] * mb[:, None, :]

        def root(C, p):
            e, U = torch.linalg.eigh((C + C.transpose(1, 2)) / 2)
            e = e.clamp(min=e.amax(-1, keepdim=True) * 1e-6)
            return (U * e.pow(p)[:, None, :]) @ U.transpose(1, 2)
        return (root(Ch, -0.5) @ root(Cy, 0.5)).float()

    def offload(self):
        """Move the accumulated statistics to host RAM (solve then streams one window at a time to the GPU)."""
        for d in (self.xx, self.sx, self.xy, self.sy, self.yy, self.yyb):
            for k in list(d):
                d[k] = d[k].cpu()
        if self.shy:
            self.shy = {k: v.cpu() for k, v in self.shy.items()}
        self.shx = [None if v is None else v.cpu() for v in self.shx]
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def solve(self, lam_rel=1e-3, groups=None, recolor=False, device=None, free=True):
        """Returns {t: dict(sel, W [din, dt] fp32, b [dt], r2 {group: R^2})}; groups[t] = [(name, width)].
        device: where to solve (stats may live on the CPU after offload()); free: drop each window's stats once solved."""
        dev = device or self.dev
        n = self.n
        maps = {}
        base_dev = dev
        for w in self.wins:
            dev = base_dev
            din = self.xx[w].shape[0]
            sdt = torch.float64
            if torch.device(dev).type == "cuda":  # fp64 solve needs ~3 din^2 doubles (A, factor, temporaries)
                torch.cuda.empty_cache()
                free_b = torch.cuda.mem_get_info(torch.device(dev))[0]
                bw = 8 * din * max(self.dt[t] for t in self.targets if self.win[t] == w)
                if free_b < 3.5 * 8 * din * din + bw:
                    sdt = torch.float32  # fp32 Cholesky with the ridge (as modal_fit_eval for big systems)
                    if free_b < 3.5 * 4 * din * din + bw / 2:  # GPU full (e.g. a 128B target): solve on the host
                        dev = "cpu"
            mx = (self.sx[w] / n).to(dev, sdt)
            A = self.xx[w].to(dev).to(sdt)
            A.addr_(mx, mx, alpha=-n)  # centered in place (no din x din temporary)
            lam = lam_rel * A.diagonal().mean().item()
            A.diagonal().add_(lam)
            Lc, info = torch.linalg.cholesky_ex(A)
            for _ in range(3):  # not PD (e.g. TF32-accumulated Gram): raise the ridge x10, logged in the maps
                if info.item() == 0:
                    break
                A.diagonal().add_(9 * lam)
                lam *= 10
                Lc, info = torch.linalg.cholesky_ex(A)
            assert info.item() == 0, f"cholesky failed for window {w} even at lam {lam:g}"
            if free:
                del self.xx[w]
            for t in [t for t in self.targets if self.win[t] == w]:
                my = (self.sy[t] / n).to(dev, sdt)
                B = self.xy[t].to(dev).to(sdt)
                B.addr_(mx, my, alpha=-n)
                if free:
                    del self.xy[t]
                W = torch.cholesky_solve(B, Lc)
                yyc = self.yy[t].to(dev).double() - n * my.double() ** 2
                sse = yyc - (B * W).sum(0) - lam * (W * W).sum(0)
                r2 = {}
                o = 0
                for name, wd in (groups[t] if groups else [("all", self.dt[t])]):
                    r2[name] = float(1 - sse[o:o + wd].sum() / yyc[o:o + wd].sum().clamp(min=1e-30))
                    o += wd
                xm = torch.cat([self.shx[s].to(dev).double() for s in w]) + mx
                b = (self.shy[t].to(dev).double() + my.double()) - xm @ W.double()
                rc = self._recolor(t, W, A, lam, my, n) if recolor else None
                maps[t] = dict(sel=list(w), W=W.float(), b=b.float(), r2=r2, mu=(self.shy[t].to(dev).double() + my.double()).float(), rc=rc,
                               lam_eff=lam)
                del B, W
            del A, Lc
        return maps


# ---------------------------------------------------------------- calibration

def _rows(x, B, stride):  # [B*T, d] -> every stride-th position of each sequence, [B*ceil(T/stride), d]
    if stride <= 1 or x is None:
        return x
    return x.view(B, -1, x.shape[-1])[:, ::stride].reshape(-1, x.shape[-1])


def units(dec, split_kv):
    """Fit units: one per layer, or (split_kv) two per attention layer: K taps and V taps separately.
    Returns [(layer, kind, [tap names])]; kind attn_k / attn_v for split attention units."""
    out = []
    for l, ((kind, taps), widths) in enumerate(zip(layer_taps(dec), tap_widths(dec))):
        names = [n for n, _ in widths]
        if split_kv and kind == "attn":
            out += [(l, "attn_k", ["k"]), (l, "attn_v", ["v"])]
        else:
            out.append((l, kind, names))
    return out


def unit_windows(us, ut, nls, nlt, w):
    """Per target unit: source units by relative layer depth (w nearest layers; w <= 0: all); split K/V units only
    see source units of the same kind; other kinds also get the nearest source unit of their own kind."""
    res = []
    for (t, kt, _) in ut:
        if kt == "none":
            res.append(())
            continue
        c = (t + 0.5) * nls / nlt - 0.5
        cand = [i for i, (s, ks, _) in enumerate(us) if ks != "none" and (not kt.startswith("attn_") or ks == kt)]
        layers = sorted({us[i][0] for i in cand}, key=lambda s: (abs(s - c), s))
        keep = set(layers if w <= 0 else layers[:w])
        sel = {i for i in cand if us[i][0] in keep}
        same = [i for i in cand if us[i][1] == kt]
        if same and not any(us[i][1] == kt for i in sel):
            sel.add(min(same, key=lambda i: (abs(us[i][0] - c), us[i][0])))
        res.append(tuple(sorted(sel)))
    return res


def unit_feats(src, feats, us, drop_q=True):
    """Per-layer source features -> per-unit features (column slices of the layer's [k | v] block)."""
    if all(len(n) and kind not in ("attn_k", "attn_v") for _, kind, n in us):
        return feats
    widths = tap_widths(src)
    out = []
    for (l, kind, names) in us:
        f = feats[l]
        if f is None or kind not in ("attn_k", "attn_v"):
            out.append(f)
            continue
        wk = dict(widths[l])["k"]
        out.append(f[:, :wk] if kind == "attn_k" else f[:, wk:])
    return out


def calibrate(src, tgt, batches, w=3, targets=None, drop_q=True, device="cuda", stride=1, split_kv=False):
    """batches: iterable of id tensors [B, T] (on device), or (ids_s, ids_t, idx) for cross-tokenizer pairs.
    stride: use every stride-th position for the statistics (paper / modal_fit_eval: 4).
    split_kv: separate K and V maps per attention layer (K from source K, V from source V, the modal_fit_eval "kv"
    recipe; half the input width). Maps are then keyed by unit index; rr.units = (source units, target units).
    targets: target LAYER ids. Returns WindowRidge."""
    us, ut = units(src, split_kv), units(tgt, split_kv)
    nls, nlt = len(src.layers), len(tgt.layers)
    win = unit_windows(us, ut, nls, nlt, w)
    tl = set(range(nlt) if targets is None else targets)
    targets = [j for j, (t, k, _) in enumerate(ut) if t in tl and k != "none"]
    ds, rr = None, None
    tw = tap_widths(tgt)
    dt = [sum(dict(tw[t])[n] for n in names) for (t, _, names) in ut]
    by_layer = {}
    for j, (t, _, names) in enumerate(ut):
        by_layer.setdefault(t, []).append((j, names))
    for ids in batches:
        if isinstance(ids, tuple):  # cross-tokenizer: (ids_s [B,Ts], ids_t [B,Tt], idx [B,Tt])
            ids_s, ids, idx = ids
            feats = source_features_aligned(src, ids_s, idx, drop_q=drop_q)
        else:
            feats = source_features(src, ids, drop_q=drop_q)
        B = ids.shape[0]
        feats = unit_feats(src, [_rows(f, B, stride) for f in feats], us, drop_q)
        if rr is None:
            ds = [0 if f is None else f.shape[1] for f in feats]
            rr = WindowRidge(ds, dt, win, targets, device)
            rr.units = (us, ut)
        rr.add_x(feats)
        del feats
        tset = set(targets)

        def on_layer(l, d):
            for j, names in by_layer.get(l, []):
                if j in tset:
                    Y = torch.cat([d[n] for n in names], -1)
                    rr.add_y(j, _rows(Y.reshape(-1, Y.shape[-1]), B, stride).float())

        with Taps(tgt, layers=sorted({ut[j][0] for j in targets})).record(on_layer):
            tgt.forward(ids, keep=1, use_cache=False)
        rr.done_batch()
    return rr


def unit_groups(tgt, ut):
    """solve(groups=...) per target unit: [(tap, width)]."""
    tw = tap_widths(tgt)
    return [[(n, dict(tw[t])[n]) for n in names] for (t, _, names) in ut]


def apply_units(maps, feats, B, T, tgt, us, ut, recolor=False, drop_q=True, src=None, out_dtype=torch.float32):
    """Like apply() for unit-keyed maps (split_kv fits): returns {layer: {tap: [B, T, w]}}."""
    uf = unit_feats(src, feats, us, drop_q) if src is not None else feats
    groups = unit_groups(tgt, ut)
    per_unit = apply(maps, uf, B, T, groups, recolor=recolor, out_dtype=out_dtype)
    out = {}
    for j, d in per_unit.items():
        out.setdefault(ut[j][0], {}).update(d)
    return out


def to_core_maps(maps, us, ut, tgt):
    """Export split_kv attention maps to modal_fit_eval's maps_k format {"k": [per target layer], "v": [...]}
    (sel = source LAYER ids; W rows = their K (or V) blocks in sel order), for kvx/vllm_connector.py."""
    out = {"k": [], "v": []}
    for j, (t, kind, _) in enumerate(ut):
        if kind not in ("attn_k", "attn_v") or j not in maps:
            continue
        m = maps[j]
        D = tgt.layers[t].self_attn.head_dim  # per layer (Gemma 4: sliding 256, global 512; config is heterogeneous)
        H = m["W"].shape[1] // D
        out["k" if kind == "attn_k" else "v"].append(dict(sel=[us[i][0] for i in m["sel"]], W=m["W"].float().cpu(),
                                                         b=m["b"].float().cpu(), r2=m.get("r2"), mu=m.get("mu"),
                                                         H=H, D=D, layer=t))
    return out


# ---------------------------------------------------------------- apply

def apply(maps, feats, B, T, widths, recolor=False, out_dtype=torch.float32):
    """feats: per source layer [B*T, d]; returns {t: {tap name: [B, T, w]}} mapped (float32).
    recolor: per 128-channel block, (Y - mu) R + mu (target covariance instead of ridge-shrunk predictions)."""
    out = {}
    for t, m in maps.items():
        Wm = m["W"]
        X = torch.cat([feats[s].to(Wm.dtype) for s in m["sel"]], 1)  # bf16 maps (as saved / deployed): bf16 GEMM
        Y = (X @ Wm).float() + m["b"].float()
        rc = m.get("rc")
        if recolor and rc is not None:
            nb, bd = rc.shape[0], rc.shape[1]
            mu = m["mu"].to(Y)
            Yc = (Y[:, :nb * bd] - mu[:nb * bd]).view(-1, nb, bd).transpose(0, 1)
            Y = torch.cat([(torch.bmm(Yc, rc.to(Y))).transpose(0, 1).reshape(-1, nb * bd) + mu[:nb * bd],
                           Y[:, nb * bd:]], 1)
        d, o = {}, 0
        Y = Y.to(out_dtype)
        for name, wd in widths[t]:
            d[name] = Y[:, o:o + wd].view(B, T, wd)
            o += wd
        out[t] = d
    return out


def restore_rms(given, tgt):
    """Re-impose the target k_norm on mapped pre-RoPE keys: k = w * u / rms(u), u = k / w (per head); values too
    when the target's v tap is a scale-less norm (Gemma 4 v_norm: unit RMS per head)."""
    for t, (kind, taps) in enumerate(layer_taps(tgt)):
        if kind != "attn" or t not in given:
            continue
        vm = taps[1][1]
        if taps[1][2] and getattr(vm, "weight", None) is None:
            v = given[t]["v"]
            B, T, W = v.shape
            D = tgt.layers[t].self_attn.head_dim
            u = v.float().view(B, T, W // D, D)
            given[t]["v"] = (u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + 1e-6)).view(B, T, W).to(v.dtype)
        if taps[0][2] == 0:
            continue
        kn = taps[0][1]
        wt = kn.weight.float()
        if type(kn).__name__.startswith(("Qwen3_5", "Qwen3Next", "Gemma3", "Gemma2")):
            wt = 1.0 + wt  # (1 + weight) RMSNorm convention; Gemma4RMSNorm multiplies by weight itself
        k = given[t]["k"]
        kdt = k.dtype
        k = k.float()
        B, T, W = k.shape
        D = wt.shape[0]
        # channels whose norm weight is ~0 carry no information (native output ~0 there; 1+w can be exactly 0 in
        # Qwen3.5): divide only where |w| is not tiny, estimate the RMS on those channels, leave the rest at w*0
        keep = wt.abs() > 1e-3 * wt.abs().amax()
        safe = torch.where(keep, wt, torch.ones_like(wt))
        u = torch.where(keep, k.view(B, T, W // D, D) / safe, torch.zeros_like(wt))
        ms = u.pow(2).sum(-1, keepdim=True) / keep.sum().clamp(min=1)
        k = wt * u * torch.rsqrt(ms + 1e-6)
        given[t]["k"] = k.view(B, T, W).to(kdt)
    return given


@torch.no_grad()
def mapped_cache(tgt, ids_prefix, given, sinks=0):
    """Target cache for ids_prefix with taps at positions >= sinks replaced by `given`."""
    with Taps(tgt, layers=list(given)).override(given, start=sinks):
        _, cache = tgt.forward(ids_prefix, keep=1, use_cache=True)
    return cache


def maps_file(src_dec, tgt_dec, maps, src, tgt, lam_rel, w, drop_q, sinks=0, sink_template=None, **extra):
    """The kvx-hybrid-v1 dict (format agreed with kvx-quality 2026-11-22). W bf16, b fp32."""
    kinds_s = [k for k, _ in layer_taps(src_dec)]
    kinds_t = [k for k, _ in layer_taps(tgt_dec)]
    widths = tap_widths(tgt_dec)
    mp = {t: dict(sel=list(m["sel"]), W=m["W"].to(torch.bfloat16).cpu(), b=m["b"].float().cpu(), r2=m.get("r2"),
                  mu=None if m.get("mu") is None else m["mu"].float().cpu(),
                  rc=None if m.get("rc") is None else m["rc"].float().cpu())
          for t, m in maps.items()}
    return dict(format="kvx-hybrid-v1", src=src, tgt=tgt, lam_rel=lam_rel, w=w, drop_q=bool(drop_q),
                src_feats=[dict(kind=k) for k in kinds_s], src_widths=src_widths(src_dec, drop_q),
                tgt_taps=[dict(kind=kinds_t[t], taps=widths[t]) for t in range(len(kinds_t))],
                lin_dims=lin_dims(tgt_dec), src_lin_dims=lin_dims(src_dec), maps=mp, restore="rms",
                sinks=int(sinks), sink_template=sink_template or {}, **extra)


def src_widths(dec, drop_q=True):
    """Feature width per source layer (matches source_features)."""
    out = []
    qw = lin_q_width(dec) if lin_dims(dec) else 0
    for g, (kind, _) in zip(tap_widths(dec), layer_taps(dec)):
        wd = sum(x for _, x in g)
        out.append(wd - qw if (kind == "lin" and drop_q) else wd)
    return out


def load_maps_file(path, device="cpu"):
    d = torch.load(path, map_location=device)
    assert d.get("format") == "kvx-hybrid-v1", d.get("format")
    maps = {int(t): dict(sel=m["sel"], W=m["W"].float(), b=m["b"].float(), r2=m.get("r2"), mu=m.get("mu"), rc=m.get("rc"))
            for t, m in d["maps"].items()}
    widths = {t: [tuple(x) for x in d["tgt_taps"][t]["taps"]] for t in range(len(d["tgt_taps"]))}
    return d, maps, widths


# ---------------------------------------------------------------- cross-tokenizer pairs

def align(text, tok_s, tok_t, n_t, bos_s=False, bos_t=False, mode="cover"):
    """Tokenize `text` with both tokenizers; return (ids_s [Ts], ids_t [n_t], idx [n_t]) where idx[j] = the source
    token holding the last character of target token j (special / empty tokens map to the previous index, BOS -> 0).
    The source sequence covers exactly the text of the n_t target tokens. None if the text is too short.
    mode="boundary" (HeteroFold / kvx-research): the latest source token ending at or before the target token's end
    (no look-ahead past the target boundary); falls back to 0 before the first source boundary."""
    et = tok_t(text, add_special_tokens=False, return_offsets_mapping=True)
    off_t = et["offset_mapping"]
    ids_t = et["input_ids"]
    if bos_t:
        ids_t, off_t = [tok_t.bos_token_id] + ids_t, [(0, 0)] + off_t
    if len(ids_t) < n_t:
        return None
    ids_t, off_t = ids_t[:n_t], off_t[:n_t]
    end = max(e for _, e in off_t)
    es = tok_s(text[:end], add_special_tokens=False, return_offsets_mapping=True)
    ids_s, off_s = es["input_ids"], es["offset_mapping"]
    if bos_s:
        ids_s, off_s = [tok_s.bos_token_id] + ids_s, [(0, 0)] + off_s
    ends_s = [e for _, e in off_s]
    idx, j = [], 0
    for (a, e) in off_t:
        if e > a:
            if mode == "boundary":  # last source token whose end <= e
                while j < len(ends_s) - 1 and ends_s[j + 1] <= e:
                    j += 1
            else:  # first source token whose end >= the target token's end char
                while j < len(ends_s) - 1 and ends_s[j] < e:
                    j += 1
        idx.append(j)
    return torch.tensor(ids_s), torch.tensor(ids_t), torch.tensor(idx)


def source_features_aligned(src, ids_s, idx, drop_q=True):
    """ids_s [B, Ts] (right-padded ok), idx [B, Tt]: source features gathered to target positions, [B*Tt, d]."""
    B, Ts = ids_s.shape
    Tt = idx.shape[1]
    feats = source_features(src, ids_s, drop_q=drop_q)
    out = []
    for f in feats:
        if f is None:
            out.append(None)
            continue
        f = f.view(B, Ts, -1)
        out.append(torch.gather(f, 1, idx.to(f.device)[..., None].expand(B, Tt, f.shape[-1])).reshape(B * Tt, -1))
    return out


def pad_batch(seqs, pad_id):
    n = max(len(s) for s in seqs)
    return torch.stack([torch.cat([s, torch.full((n - len(s),), pad_id, dtype=s.dtype)]) for s in seqs])
