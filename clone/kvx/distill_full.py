"""Full-map KL distillation of a ridge KV map (KV-Lingo recipe, arXiv 2609.32610; kvx-research LEVERS 2026-11-22).

Differs from kvx/distill.py (rank-16 residual, lost 3 times): the WHOLE map (W, b per target layer, K and V) is
trainable and initialized AT the closed-form ridge solution, so step 0 equals the ridge map exactly (checked:
held-out KL at step 0 must equal the ridge KL). Loss = KL(target on the native sequence || target on the mapped
prefix cache) on continuation positions only. AdamW (wd 0), grad clip 1, cosine with 5% warmup, map in fp32,
models frozen in bf16. lr is picked by a short sweep on held-out KL (KV-Lingo: 5 points in 3e-7..3e-2).

Maps format in/out = modal_fit_eval's maps_k*.pt ({"k": [per target layer dict(sel, W, b, mu, H, D, ...)], "v": ...}),
so the distilled file drops into kvx.vllm_connector unchanged.
"""
import math

import torch
import torch.nn.functional as F

from . import core


class TrainableMaps(torch.nn.Module):
    def __init__(self, maps_k, maps_v):
        super().__init__()
        self.meta_k = [{k: v for k, v in m.items() if k not in ("W", "b")} for m in maps_k]
        self.meta_v = [{k: v for k, v in m.items() if k not in ("W", "b")} for m in maps_v]
        self.Wk = torch.nn.ParameterList([torch.nn.Parameter(m["W"].float().clone()) for m in maps_k])
        self.bk = torch.nn.ParameterList([torch.nn.Parameter(m["b"].float().clone()) for m in maps_k])
        self.Wv = torch.nn.ParameterList([torch.nn.Parameter(m["W"].float().clone()) for m in maps_v])
        self.bv = torch.nn.ParameterList([torch.nn.Parameter(m["b"].float().clone()) for m in maps_v])

    def maps(self):
        mk = [dict(m, W=w, b=b) for m, w, b in zip(self.meta_k, self.Wk, self.bk)]
        mv = [dict(m, W=w, b=b) for m, w, b in zip(self.meta_v, self.Wv, self.bv)]
        return mk, mv

    def export(self):
        mk, mv = self.maps()
        cpu = lambda ms: [{k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in m.items()} for m in ms]
        return cpu(mk), cpu(mv)


def _build_cache_grad(K, V, cos, sin):
    from transformers import DynamicCache
    c = DynamicCache()
    T = K.shape[3]
    for l in range(K.shape[0]):
        cc, ss_ = core._cs(cos, sin, l, T)
        c.update(core.rope(K[l], cc, ss_).to(torch.bfloat16), V[l].to(torch.bfloat16), l)
    return c


def _map_layer(W, b, X, sel, H, D, B, T, kwl, cos, sin):
    """One target layer: gather the selected source layers, GEMM, optional rms restore (kwl = target k_norm weight),
    RoPE (cos/sin given) -> bf16 [B, H, T, D]. Checkpointed by the caller: backward recomputes it from X (shared)."""
    Y = X[:, sel].reshape(X.shape[0], -1).float() @ W + b
    k = Y.view(B, T, H, D).permute(0, 2, 1, 3)
    if kwl is not None:
        w = kwl.view(1, 1, 1, -1)
        u = k / w
        k = w * u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + 1e-6)
    if cos is not None:
        k = core.rope(k, cos, sin)
    return k.to(torch.bfloat16)


def kl_step(S, T, tm, ids, P, cs, ss, ct, st, kw, restore="rms", grad=True):
    """Mean KL over continuation positions ids[:, P:] (teacher = target on the native sequence).
    Each target layer's map runs under activation checkpointing: autograd keeps only the bf16 cache rows, not the
    gathered [tokens, k * d] inputs (22 GB at P=3072 for 1.7B->8B k24 otherwise)."""
    from torch.utils.checkpoint import checkpoint
    from transformers import DynamicCache
    with torch.no_grad():
        Ks, Vs, _ = core.capture_kv(S, ids[:, :P], cs, ss)
        lp_t = F.log_softmax(T(input_ids=ids, logits_to_keep=ids.shape[1] - P).logits.float(), -1)
        Xk, Xv = core.flat_tokens(Ks), core.flat_tokens(Vs)
        del Ks, Vs
    B = ids.shape[0]
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        mk, mv = tm.maps()
        cache = DynamicCache()
        for l, (a, v) in enumerate(zip(mk, mv)):
            cc, ss_ = core._cs(ct, st, l, P)
            kwl = kw[l].float() if restore == "rms" else None
            if grad:
                K = checkpoint(_map_layer, a["W"], a["b"], Xk, a["sel"], a["H"], a["D"], B, P, kwl, cc, ss_, use_reentrant=False)
                V = checkpoint(_map_layer, v["W"], v["b"], Xv, v["sel"], v["H"], v["D"], B, P, None, None, None, use_reentrant=False)
            else:
                K = _map_layer(a["W"], a["b"], Xk, a["sel"], a["H"], a["D"], B, P, kwl, cc, ss_)
                V = _map_layer(v["W"], v["b"], Xv, v["sel"], v["H"], v["D"], B, P, None, None, None)
            cache.update(K, V, l)
        lg = T(input_ids=ids[:, P:], past_key_values=cache).logits.float()
        kl = (lp_t.exp() * (lp_t - F.log_softmax(lg, -1))).sum(-1)  # [B, L-P]
    return kl.mean()


def heldout_kl(S, T, tm, docs, P, cs, ss, ct, st, kw, restore="rms", bs=2):
    tot, n = 0.0, 0
    for i in range(0, len(docs), bs):
        tot += kl_step(S, T, tm, docs[i:i + bs], P, cs, ss, ct, st, kw, restore, grad=False).item() * len(docs[i:i + bs])
        n += len(docs[i:i + bs])
    return tot / max(1, n)


def train(S, T, maps_k, maps_v, train_docs, val_docs, P, cs, ss, ct, st, kw, lr, steps, bs=2, accum=4,
          restore="rms", eval_every=0, log=print, deadline=None):
    """Returns (TrainableMaps, history dict). train/val docs: LongTensor [N, L] on the GPU; P = mapped prefix length."""
    import time
    tm = TrainableMaps(maps_k, maps_v).to(train_docs.device)
    params = list(tm.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.999))
    warm = max(1, int(0.05 * steps))
    hist = dict(lr=lr, steps=steps, train=[], val=[])
    v0 = heldout_kl(S, T, tm, val_docs, P, cs, ss, ct, st, kw, restore)
    hist["val"].append((0, v0))
    log(f"distill lr {lr:g}: step 0 held-out KL {v0:.4f}")
    g = torch.Generator(device="cpu").manual_seed(0)
    N = len(train_docs)
    for step in range(steps):
        if deadline and time.time() > deadline:
            hist["stopped"] = step
            break
        f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
        for gr in opt.param_groups:
            gr["lr"] = lr * f
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(accum):
            idx = torch.randint(0, N, (bs,), generator=g)
            loss = kl_step(S, T, tm, train_docs[idx], P, cs, ss, ct, st, kw, restore) / accum
            loss.backward()
            tot += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(gn):
            hist["diverged"] = step
            log(f"distill lr {lr:g}: non-finite grad at step {step}")
            break
        opt.step()
        hist["train"].append(tot)
        if eval_every and (step + 1) % eval_every == 0:
            v = heldout_kl(S, T, tm, val_docs, P, cs, ss, ct, st, kw, restore)
            hist["val"].append((step + 1, v))
            log(f"distill lr {lr:g}: step {step + 1} train {sum(hist['train'][-eval_every:]) / eval_every:.4f} val {v:.4f}")
    if not eval_every or hist["val"][-1][0] != len(hist["train"]):
        hist["val"].append((len(hist["train"]), heldout_kl(S, T, tm, val_docs, P, cs, ss, ct, st, kw, restore)))
    return tm, hist
