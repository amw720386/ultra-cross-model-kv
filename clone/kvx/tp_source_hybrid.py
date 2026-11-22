"""Tensor-parallel source prefill for the hybrid (Qwen3.5 GatedDeltaNet) connector.

Same Megatron split as kvx/tp_source.py, for a Qwen3.5 text decoder (HF transformers 5):
  - gated full attention: q_proj rows hold [q_h | gate_h] per query head (2*D rows per head), k/v by KV head (a KV
    head is replicated on n/Hkv ranks when Hkv < n), o_proj row-parallel;
  - GatedDeltaNet: split by key heads, each with its Hv/Hk value heads (HF repeat_interleaves k-head j onto v-heads
    j*r..j*r+r-1, so a contiguous k-head range owns a contiguous v-head range): in_proj_qkv rows [q | k | v], the
    depthwise conv1d channels in the same order, z/b/a rows, dt_bias, A_log; out_proj row-parallel;
  - MLP: gate/up rows, down row-parallel.
After every row-parallel all-reduce the residual stream is whole on each rank, so the xkv path (which reads
input_layernorm outputs, i.e. the residual stream) needs no gather. Attention K/V taps (hybrid-v1 maps) would be
rank-local: gather them with tp_source.gather_heads (rep from _kvx_tp).
"""
import torch
import torch.nn as nn

from .tp_source import RowParallelLinear, _rows


def _keep_rows(lin, idx):
    lin.weight = nn.Parameter(lin.weight.data[idx].contiguous(), requires_grad=False)
    if lin.bias is not None:
        lin.bias = nn.Parameter(lin.bias.data[idx].contiguous(), requires_grad=False)
    lin.out_features = len(idx)


def _gdn(g, rank, world, group):
    Hk, Hv, dk, dv = g.num_k_heads, g.num_v_heads, g.head_k_dim, g.head_v_dim
    if Hk % world:
        raise ValueError(f"GatedDeltaNet has {Hk} key heads, not divisible by TP={world}")
    r = Hv // Hk
    hk = Hk // world
    k0, v0 = rank * hk, rank * hk * r
    ar = torch.arange
    q_idx = ar(k0 * dk, (k0 + hk) * dk)
    k_idx = g.key_dim + q_idx
    v_idx = 2 * g.key_dim + ar(v0 * dv, (v0 + hk * r) * dv)
    ch = torch.cat([q_idx, k_idx, v_idx])
    _keep_rows(g.in_proj_qkv, ch)
    g.conv1d.weight = nn.Parameter(g.conv1d.weight.data[ch].contiguous(), requires_grad=False)
    if g.conv1d.bias is not None:
        g.conv1d.bias = nn.Parameter(g.conv1d.bias.data[ch].contiguous(), requires_grad=False)
    g.conv1d.in_channels = g.conv1d.out_channels = g.conv1d.groups = len(ch)
    _rows(g.in_proj_z, v0 * dv, (v0 + hk * r) * dv)
    _rows(g.in_proj_b, v0, v0 + hk * r)
    _rows(g.in_proj_a, v0, v0 + hk * r)
    g.dt_bias = nn.Parameter(g.dt_bias.data[v0:v0 + hk * r].contiguous(), requires_grad=False)
    g.A_log = nn.Parameter(g.A_log.data[v0:v0 + hk * r].contiguous(), requires_grad=False)
    g.out_proj = RowParallelLinear(g.out_proj, rank, world, group)
    g.num_k_heads, g.num_v_heads = hk, hk * r
    g.key_dim, g.value_dim = hk * dk, hk * r * dv
    g.conv_dim = 2 * g.key_dim + g.value_dim


def _attn(a, cfg, rank, world, group):
    from .tp_source import kv_layout
    H = cfg.num_attention_heads
    D = a.head_dim
    Hl = H // world
    hkv_l, rep = kv_layout(cfg, world)
    kv0 = rank * hkv_l if rep == 1 else rank // rep
    _rows(a.q_proj, rank * Hl * 2 * D, (rank + 1) * Hl * 2 * D)  # [q_h | gate_h] per head
    _rows(a.k_proj, kv0 * D, (kv0 + hkv_l) * D)
    _rows(a.v_proj, kv0 * D, (kv0 + hkv_l) * D)
    a.o_proj = RowParallelLinear(a.o_proj, rank, world, group)
    a.num_key_value_groups = Hl // hkv_l
    return rep


def shard_hybrid_source(dec, rank, world, group):
    """Shard a hybrid.Dec over a Qwen3.5 text decoder in place for TP rank `rank` of `world`. Returns dec."""
    cfg = dec.cfg
    I = cfg.intermediate_size
    if I % world:
        raise ValueError(f"source intermediate_size {I} not divisible by TP={world}")
    Il = I // world
    rep = 1
    for layer in dec.layers:
        if getattr(layer, "linear_attn", None) is not None:
            _gdn(layer.linear_attn, rank, world, group)
        if getattr(layer, "self_attn", None) is not None:
            rep = _attn(layer.self_attn, cfg, rank, world, group)
        m = layer.mlp
        _rows(m.gate_proj, rank * Il, (rank + 1) * Il)
        _rows(m.up_proj, rank * Il, (rank + 1) * Il)
        m.down_proj = RowParallelLinear(m.down_proj, rank, world, group)
    dec._kvx_tp = dict(rank=rank, world=world, rep=rep, group=group)
    return dec
