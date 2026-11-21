"""Nonlinear mapper: per target layer, an MLP on the selected source layers trained on the ridge residual.

Y = ridge(x) + MLP(x),  MLP = Linear(din, h) - GELU - Linear(h, h) - GELU - Linear(h, dout), last layer zero-init.
Paper (2608.03893) MLP: hidden 1024, Adam 1e-3, 20 epochs, MSE, batch 4096, per (layer, head); we use one MLP per
target layer over all heads (one GEMM chain per layer at serve time).
"""
import torch
from . import core


class LayerMLP(torch.nn.Module):
    def __init__(self, din, dout, h):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(din, h), torch.nn.GELU(), torch.nn.Linear(h, h),
                                       torch.nn.GELU(), torch.nn.Linear(h, dout))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def train_mlps(X, Y, maps, hidden=1024, epochs=20, lr=1e-3, bs=4096, val_frac=0.05, log=print):
    """X [N, Ls, ds] bf16 (CPU or GPU), Y [N, Lt, dt] bf16. Returns list of LayerMLP and per-layer val R2 (ridge, mlp)."""
    dev = maps[0]["W"].device
    N = X.shape[0]
    nv = int(N * val_frac)
    perm = torch.randperm(N)
    vi, ti = perm[:nv], perm[nv:]
    mlps, stats = [], []
    for t, m in enumerate(maps):
        sel = m["sel"]
        xs = X[:, sel].reshape(N, -1).to(dev)
        ys = Y[:, t].to(dev)
        xv = xs[vi].to(dev).float(); yv = ys[vi].to(dev).float()
        net = LayerMLP(xs.shape[1], ys.shape[1], hidden).to(dev)
        # input scale: standardize by per-feature std from ridge stats proxy (computed on a sample)
        samp = xs[ti[:8192]].to(dev).float()
        mu, sd = samp.mean(0), samp.std(0) + 1e-3
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        steps = epochs * (len(ti) // bs)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for ep in range(epochs):
            order = ti[torch.randperm(len(ti))]
            for j in range(0, len(order) - bs + 1, bs):
                b = order[j:j + bs]
                x = xs[b].to(dev, non_blocking=True).float()
                y = ys[b].to(dev, non_blocking=True).float()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = net((x - mu) / sd).float()
                base = x @ m["W"] + m["b"]
                loss = ((base + pred - y) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step(); sched.step()
        with torch.no_grad():
            base = xv @ m["W"] + m["b"]
            pred = net((xv - mu) / sd).float()
            var = ((yv - yv.mean(0)) ** 2).sum()
            r2_r = 1 - ((base - yv) ** 2).sum() / var
            r2_m = 1 - ((base + pred - yv) ** 2).sum() / var
        net.mu, net.sd = mu, sd
        net.eval()
        mlps.append(net)
        stats.append((r2_r.item(), r2_m.item()))
        if t % 8 == 0:
            log(f"mlp layer {t}: val R2 ridge {r2_r.item():.3f} -> mlp {r2_m.item():.3f}")
    return mlps, stats


@torch.no_grad()
def apply_mlp_maps(maps, mlps, Xs):
    Ls, B, H, T, D = Xs.shape
    X = core.flat_tokens(Xs).float()
    outs = []
    for m, net in zip(maps, mlps):
        x = X[:, m["sel"]].reshape(B * T, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            r = net((x - net.mu) / net.sd).float()
        Y = x @ m["W"] + m["b"] + r
        outs.append(Y.view(B, T, m["H"], m["D"]).permute(0, 2, 1, 3))
    return torch.stack(outs)
