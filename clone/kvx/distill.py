"""Stage 2 (OneTriangle blog): rank-r residual on top of the ridge map, trained by KL distillation.

W = W0 + A B per target layer (A ~ N(0, sigma^2), B = 0), both models frozen.
Teacher: target with the full native context. Student: target on the continuation over the mapped prefix cache.
"""
import torch
from . import core


class Residual(torch.nn.Module):
    def __init__(self, maps, rank, sigma=None, xscale=None):
        """xscale: per-target-layer (mu, sd) of the selected inputs; the residual sees standardized x
        so Adam's per-entry steps do not blow up through an 8K fan-in with outlier key channels."""
        super().__init__()
        self.A = torch.nn.ParameterList()
        self.B = torch.nn.ParameterList()
        self.xscale = xscale
        for m in maps:
            din, dout = m["W"].shape
            sg = sigma if sigma is not None else din ** -0.5
            self.A.append(torch.nn.Parameter(torch.randn(din, rank, device=m["W"].device) * sg))
            self.B.append(torch.nn.Parameter(torch.zeros(rank, dout, device=m["W"].device)))

    def apply(self, maps, Xs):
        Ls, Bsz, H, T, D = Xs.shape
        X = core.flat_tokens(Xs).float()
        outs = []
        for i, m in enumerate(maps):
            x = X[:, m["sel"]].reshape(Bsz * T, -1)
            xr = x if self.xscale is None else (x - self.xscale[i][0]) / self.xscale[i][1]
            Y = x @ m["W"] + m["b"] + (xr @ self.A[i]) @ self.B[i]
            outs.append(Y.view(Bsz, T, m["H"], m["D"]).permute(0, 2, 1, 3))
        return torch.stack(outs)


def build_cache_grad(K, V, cos, sin):
    """Like core.build_cache but keeps autograd (bf16 cast is differentiable)."""
    from transformers import DynamicCache
    c = DynamicCache()
    T = K.shape[3]
    for l in range(K.shape[0]):
        c.update(core.rope(K[l], cos[:T], sin[:T]).to(torch.bfloat16), V[l].to(torch.bfloat16), l)
    return c


def train(S, T, mapsK, mapsV, docs, P, cs, ss, ct, st, rank=16, steps=200, lr=1e-3, bs=4,
          restore=None, log=print):
    """docs: LongTensor [N, L]. Returns (resK, resV, history)."""
    import torch.nn.functional as F
    dev = docs.device
    with torch.no_grad():  # input standardization from a sample of the training docs
        Ks0, Vs0, _ = core.capture_kv(S, docs[:8, :P], cs, ss)
        def scales(maps, Xs):
            X = core.flat_tokens(Xs).float()
            out = []
            for m in maps:
                x = X[:, m["sel"]].reshape(X.shape[0], -1)
                out.append((x.mean(0), x.std(0) + 1e-3))
            return out
        skK, skV = scales(mapsK, Ks0), scales(mapsV, Vs0)
        del Ks0, Vs0
    rk, rv = Residual(mapsK, rank, xscale=skK), Residual(mapsV, rank, xscale=skV)
    params = list(rk.parameters()) + list(rv.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    hist = []
    N, L = docs.shape
    for step in range(steps):
        warm = min(1.0, (step + 1) / 20)
        for g in opt.param_groups:
            g["lr"] = lr * warm
        idx = torch.randint(0, N, (bs,), device=dev)
        ids = docs[idx]
        with torch.no_grad():
            Ks, Vs, _ = core.capture_kv(S, ids[:, :P], cs, ss)
            lp_t = F.log_softmax(T(input_ids=ids, logits_to_keep=L - P).logits.float(), -1)
        Km = rk.apply(mapsK, Ks)
        if restore is not None:
            Km = restore(Km)
        Vm = rv.apply(mapsV, Vs)
        c = build_cache_grad(Km, Vm, ct, st)
        lg = T(input_ids=ids[:, P:], past_key_values=c).logits.float()
        loss = (lp_t.exp() * (lp_t - F.log_softmax(lg, -1))).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist.append(loss.item())
        if step % 20 == 0:
            log(f"distill step {step} kl {sum(hist[-20:]) / len(hist[-20:]):.4f}")
    return rk, rv, hist
