"""KV-Lingo-style distillation of the ridge maps (arXiv 2609.32610 settings; research/LEVERS-papers-2026-11-22.md).

The full linear map of the chosen k (W, b per target layer, K and V) is initialized AT the ridge solution and trained
with AdamW (wd 0, clip 1.0, cosine with 5% warmup) on KL(target with native cache || target with translated cache)
over CONTINUATION positions only; models frozen (bf16), map in fp32. lr picked from a small sweep by held-out KL.
Step 0 is checked: held-out KL of the untrained copy must equal the ridge KL (prints both).
Calibration docs are FineWeb-Edu / PG19 train text (never an eval set).
"""
import copy, math, time

import torch

from kvx import core


def _params(maps_k, maps_v):
    ps = []
    for ms in (maps_k, maps_v):
        for m in ms:
            m["W"] = m["W"].float().clone().requires_grad_(True)
            m["b"] = m["b"].float().clone().requires_grad_(True)
            ps += [m["W"], m["b"]]
    return ps


def _apply(maps, Xs):
    Ls, B, H, T, D = Xs.shape
    X = core.flat_tokens(Xs).float()
    outs = []
    for m in maps:
        Y = X[:, m["sel"]].reshape(B * T, -1) @ m["W"] + m["b"]
        outs.append(Y.view(B, T, m["H"], m["D"]).permute(0, 2, 1, 3))
    return torch.stack(outs)


def _kl(P, maps_k, maps_v, ids, pre_len, restore, grad):
    """Mean KL over continuation positions; ids [B, pre_len + C]; target runs ids[pre_len-1:-1] on the mapped cache."""
    with torch.no_grad():
        Ks, Vs, _ = core.capture_kv(P.S, ids[:, :pre_len - 1], P.cs, P.ss)
        lp_ref = torch.log_softmax(P.T(input_ids=ids).logits[:, pre_len - 1:-1].float(), -1)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        Km = _apply(maps_k, Ks)
        if restore != "none" and not (restore == "rms" and P.kw is None):
            Km = core.restore_scale(Km, maps_k, restore, P.Ht, P.kw)
        Vm = _apply(maps_v, Vs)
        cache = core.build_cache(Km, Vm, P.ct, P.st)
        lg = P.T(input_ids=ids[:, pre_len - 1:-1], past_key_values=cache).logits.float()
        lp = torch.log_softmax(lg, -1)
        return (lp_ref.exp() * (lp_ref - lp)).sum(-1).mean()


def distill(P, k, train_ids, val_ids, pre_len, lrs=(1e-5, 3e-5, 1e-4), steps=400, bs=4, restore="rms", log=print):
    """Returns (best_maps_k, best_maps_v, report). train_ids/val_ids: LongTensor [N, pre_len + C]."""
    P.T.requires_grad_(False); P.S.requires_grad_(False)
    report = dict(k=k, steps=steps, bs=bs, lrs=list(lrs), pre_len=pre_len, cont=train_ids.shape[1] - pre_len, runs={})

    def val(mk, mv):
        tot = 0.0
        for i in range(0, len(val_ids), bs):
            tot += _kl(P, mk, mv, val_ids[i:i + bs].to(P.dev), pre_len, restore, False).item() * len(val_ids[i:i + bs])
        return tot / len(val_ids)

    base_k, base_v = P.maps[("k", k)], P.maps[("v", k)]
    best = None
    for lr in lrs:
        mk, mv = copy.deepcopy(base_k), copy.deepcopy(base_v)
        ps = _params(mk, mv)
        v0 = val(mk, mv)
        opt = torch.optim.AdamW(ps, lr=lr, weight_decay=0.0)
        warm = max(1, steps // 20)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                                  0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
        g = torch.Generator().manual_seed(0)
        t0 = time.time()
        hist = []
        for s in range(steps):
            idx = torch.randint(0, len(train_ids), (bs,), generator=g)
            loss = _kl(P, mk, mv, train_ids[idx].to(P.dev), pre_len, restore, True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ps, 1.0)
            opt.step(); sched.step()
            hist.append(loss.item())
            if not math.isfinite(hist[-1]):
                break
        v1 = val(mk, mv)
        report["runs"][str(lr)] = dict(val_step0=v0, val_end=v1, train_first20=sum(hist[:20]) / max(1, len(hist[:20])),
                                       train_last20=sum(hist[-20:]) / max(1, len(hist[-20:])), t=time.time() - t0)
        log(f"distill k{k} lr {lr}: val {v0:.5f} -> {v1:.5f} (train {report['runs'][str(lr)]['train_first20']:.4f} -> "
            f"{report['runs'][str(lr)]['train_last20']:.4f}) {time.time()-t0:.0f}s")
        for m in mk + mv:
            m["W"] = m["W"].detach(); m["b"] = m["b"].detach()
        if math.isfinite(v1) and (best is None or v1 < best[0]):
            best = (v1, lr, mk, mv)
        del opt, ps
        torch.cuda.empty_cache()
    report["val_ridge"] = report["runs"][str(lrs[0])]["val_step0"]
    if best is None or best[0] >= report["val_ridge"]:
        report["chosen"] = None  # no lr beat the ridge on held-out: keep ridge
        return base_k, base_v, report
    report["chosen"] = best[1]
    return best[2], best[3], report
