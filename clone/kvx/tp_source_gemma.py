"""Tensor-parallel source prefill for the Gemma-4 connector (E2B / E4B sources).

Megatron split per decoder layer (HF transformers 5 Gemma4Text): q_proj rows by query head (head_dim differs by layer
type: sliding vs global), k/v rows by KV head (replicated on n/Hkv ranks when Hkv < n; v_proj may be absent when
K == V), o_proj row-parallel, MLP gate/up rows + down row-parallel (each layer's own width: KV-shared layers may have a
double-wide MLP). KV-shared layers have no k/v weights: they reuse the stored K/V of an earlier layer of their type,
which is sharded the same way. The per-layer-input gate/projection (hidden -> 256 -> hidden) stays replicated.

The connector maps from each source layer's k_norm / v_norm taps, which are rank-local after the split:
gather_taps() all-gathers them back to full width (no-op when every rank holds all KV heads, e.g. E2B's 1 KV head).
"""
import torch

from .tp_source import RowParallelLinear, _rows, gather_heads


def _layout(H, Hkv, world):
    if H % world:
        raise ValueError(f"source has {H} query heads, not divisible by TP={world}")
    if Hkv % world == 0:
        return Hkv // world, 1
    if world % Hkv == 0:
        return 1, world // Hkv
    raise ValueError(f"source KV heads {Hkv} vs TP={world}: neither divides the other")


def shard_gemma_source(dec, rank, world, group):
    """Shard a hybrid.Dec over a Gemma-4 text decoder in place. Stores per-layer (hkv_local, rep, D) in dec._kvx_tp."""
    cfg = dec.cfg
    H = cfg.num_attention_heads
    Hl = H // world
    per = {}
    for i, layer in enumerate(dec.layers):
        a, m = layer.self_attn, layer.mlp
        D = a.head_dim
        Hkv = H // a.num_key_value_groups
        hkv_l, rep = _layout(H, Hkv, world)
        kv0 = rank * hkv_l if rep == 1 else rank // rep
        _rows(a.q_proj, rank * Hl * D, (rank + 1) * Hl * D)
        if not a.is_kv_shared_layer:
            _rows(a.k_proj, kv0 * D, (kv0 + hkv_l) * D)
            if a.v_proj is not None:
                _rows(a.v_proj, kv0 * D, (kv0 + hkv_l) * D)
            per[i] = (hkv_l, rep, D)
        a.o_proj = RowParallelLinear(a.o_proj, rank, world, group)
        a.num_key_value_groups = Hl // hkv_l
        I = m.gate_proj.out_features
        if I % world:
            raise ValueError(f"source layer {i} MLP width {I} not divisible by TP={world}")
        Il = I // world
        _rows(m.gate_proj, rank * Il, (rank + 1) * Il)
        _rows(m.up_proj, rank * Il, (rank + 1) * Il)
        m.down_proj = RowParallelLinear(m.down_proj, rank, world, group)
    dec._kvx_tp = dict(rank=rank, world=world, group=group, layers=per)
    return dec


def gather_taps(dec, feats):
    """feats: per source layer [T, k_local | v_local] (None for KV-shared layers) -> full-width [T, k | v]."""
    tp = dec._kvx_tp
    world, group = tp["world"], tp["group"]
    out = []
    for i, x in enumerate(feats):
        if x is None or i not in tp["layers"]:
            out.append(x)
            continue
        hkv_l, rep, D = tp["layers"][i]
        if rep == world:  # every rank holds every KV head (Hkv == 1)
            out.append(x)
            continue
        T = x.shape[0]
        w = hkv_l * D
        k, v = x[:, :w].reshape(T, hkv_l, D), x[:, w:].reshape(T, hkv_l, D)
        k = gather_heads(k.contiguous(), world, group, rep).reshape(T, -1)
        v = gather_heads(v.contiguous(), world, group, rep).reshape(T, -1)
        out.append(torch.cat([k, v], 1))
    return out
