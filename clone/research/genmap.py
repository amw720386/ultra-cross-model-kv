"""Batched HF generation on top of a mapped (or any externally built) prompt cache.

Left padding: prompt i of n_i cached tokens sits in slots [Tc - n_i, Tc) of a [L, B, H, Tc, D] cache, its keys roped at
positions 0..n_i-1; pad slots are zero and masked. HF generate derives position_ids from the attention-mask cumsum, so
every later token gets its own position n_i + t. The last `tail` prompt tokens are run natively by the target.
"""
import torch

from kvx import core


def batched_cache(P, prompts, tail, kv_fn):
    """prompts: list of token lists. kv_fn(ids[1, n]) -> (K, V) pre-RoPE [L, 1, H, n, D] for the cached part.
    Returns (input_ids [B, Tc+tail], attention_mask, DynamicCache)."""
    from transformers import DynamicCache
    B = len(prompts)
    ncache = [len(p) - tail for p in prompts]
    Tc = max(ncache)
    pad = P.tok.pad_token_id if P.tok.pad_token_id is not None else 0
    Kb = Vb = None
    ids = torch.full((B, Tc + tail), pad, dtype=torch.long)
    mask = torch.zeros((B, Tc + tail), dtype=torch.long)
    for i, p in enumerate(prompts):
        n = ncache[i]
        K, V = kv_fn(torch.tensor([p[:n]], device=P.dev))
        L_, _, H, _, D = K.shape
        if Kb is None:
            Kb = torch.zeros((L_, B, H, Tc, D), device=P.dev, dtype=P.T.dtype)
            Vb = torch.zeros_like(Kb)
        for l in range(L_):
            c, s = core._cs(P.ct, P.st, l, n)
            Kb[l, i, :, Tc - n:] = core.rope(K[l].float(), c, s)[0].to(Kb.dtype)
            Vb[l, i, :, Tc - n:] = V[l, 0].to(Kb.dtype)
        ids[i, Tc - n:] = torch.tensor(p)
        mask[i, Tc - n:] = 1
    cache = DynamicCache()
    for l in range(Kb.shape[0]):
        cache.update(Kb[l], Vb[l], l)
    return ids.to(P.dev), mask.to(P.dev), cache


@torch.no_grad()
def generate(P, prompts, kv_fn, tail=1, bs=16, **gen_kw):
    """Returns list of generated token lists (prompt excluded). kv_fn None = native generation (same batching)."""
    outs = []
    for b0 in range(0, len(prompts), bs):
        chunk = prompts[b0:b0 + bs]
        if kv_fn is None:
            T = max(map(len, chunk))
            pad = P.tok.pad_token_id if P.tok.pad_token_id is not None else 0
            ids = torch.tensor([[pad] * (T - len(p)) + p for p in chunk], device=P.dev)
            mask = torch.tensor([[0] * (T - len(p)) + [1] * len(p) for p in chunk], device=P.dev)
            g = P.T.generate(input_ids=ids, attention_mask=mask, **gen_kw)
        else:
            ids, mask, cache = batched_cache(P, chunk, tail, kv_fn)
            g = P.T.generate(input_ids=ids, attention_mask=mask, past_key_values=cache, **gen_kw)
        for row in g[:, ids.shape[1]:].tolist():
            outs.append(row)
    return outs
