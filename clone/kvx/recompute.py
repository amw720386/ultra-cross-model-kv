"""Selective native recompute of an arbitrary token subset on top of a mapped KV cache (RaReCache, arXiv 2610.11358).

rank-disagreement score (their eq.; bias cancels, app. A): s_t = sum_{layer, K/V} ||z_t||^2 - ||z_t V_r||^2 with
z_t = mapped prediction minus its calibration mean and V_r the top-r eigenvectors of the calibration prediction
covariance. One shared position set (top ceil(rho*T) by s) is recomputed by the target in one forward pass; each selected
query attends to all earlier prompt positions (mapped KV for unselected ones, recomputed KV for selected ones).
"""
import torch

from . import core


class PredBasis:
    """Running mean/covariance of mapped predictions per (kind, target layer) -> top-r basis."""

    def __init__(self):
        self.n, self.s, self.ss = 0, {}, {}

    def add(self, kind, Y):  # Y [L, B, H, T, D] mapped (pre-RoPE for keys)
        L, B, H, T, D = Y.shape
        Z = Y.permute(0, 1, 3, 2, 4).reshape(L, B * T, H * D).float()
        if kind not in self.s:
            self.s[kind] = torch.zeros(L, H * D, device=Y.device, dtype=torch.float64)
            self.ss[kind] = torch.zeros(L, H * D, H * D, device=Y.device, dtype=torch.float32)
        self.s[kind] += Z.sum(1).double()
        self.ss[kind] += torch.bmm(Z.transpose(1, 2), Z)
        if kind == "k":
            self.n += B * T

    def finalize(self, r=128):
        self.mu, self.V = {}, {}
        for kind in self.s:
            mu = (self.s[kind] / self.n).float()
            C = self.ss[kind] / self.n - mu[:, :, None] * mu[:, None, :]
            _, ev = torch.linalg.eigh(C.double())
            self.mu[kind], self.V[kind] = mu, ev[:, :, -r:].float()  # eigh is ascending: last r = top r
        return self


def rank_disagreement(basis, maps_out):
    """maps_out {kind: Y [L,B,H,T,D]} -> score [B, T] (energy outside the calibration span, summed over layers/kinds)."""
    score = None
    for kind, Y in maps_out.items():
        L, B, H, T, D = Y.shape
        Z = Y.permute(0, 1, 3, 2, 4).reshape(L, B * T, H * D).float() - basis.mu[kind][:, None]
        s = (Z.pow(2).sum(-1) - torch.bmm(Z, basis.V[kind]).pow(2).sum(-1)).sum(0).view(B, T)
        score = s if score is None else score + s
    return score


def select_top(score, rho, keep_first=0):
    """Top ceil(rho*T) positions per row (sorted); positions < keep_first are always included (native sinks)."""
    B, T = score.shape
    n = max(1, int(-(-rho * T // 1)))
    sc = score.clone()
    if keep_first:
        sc[:, :keep_first] = float("inf")
    return sc.topk(min(n, T), dim=1).indices.sort(1).values


@torch.no_grad()
def recompute_cache(model, ids, K, V, sel, cos, sin):
    """Mapped pre-RoPE K, V [L,B,H,P,D] for prompt ids [B,P]; natively recompute positions sel [B,n] (sorted).

    Returns a DynamicCache of length P where selected positions hold the target's own KV computed in one pass in which
    each selected query sees every earlier prompt position (mapped for unselected, recomputed for selected)."""
    from transformers import DynamicCache
    B, P = ids.shape
    n = sel.shape[1]
    c = core.build_cache(K, V, cos, sin)
    pos = torch.arange(P, device=ids.device)
    is_sel = torch.zeros(B, P, dtype=torch.bool, device=ids.device).scatter_(1, sel, True)
    q = sel[:, :, None]  # [B, n, 1]
    m_old = (pos[None, None, :] <= q) & ~is_sel[:, None, :]          # mapped keys: earlier and not replaced
    m_new = sel[:, None, :] <= q                                       # recomputed keys: earlier or self
    mask = torch.cat([m_old, m_new], -1)[:, None]                      # [B, 1, n, P + n], True = attend
    model(input_ids=ids.gather(1, sel), position_ids=sel, past_key_values=c, attention_mask=mask,
          cache_position=torch.arange(P, P + n, device=ids.device), use_cache=True, logits_to_keep=1)
    out = DynamicCache()
    idx = sel[:, None, :, None]
    for l, (k, v) in enumerate(core.cache_layers(c)):
        H, D = k.shape[1], k.shape[3]
        ix = idx.expand(B, H, n, D)
        k2 = k[:, :, :P].scatter(2, ix, k[:, :, P:])
        v2 = v[:, :, :P].scatter(2, ix, v[:, :, P:])
        out.update(k2.contiguous(), v2.contiguous(), l)
    return out
