"""Tensor-parallel source prefill for the vLLM connector.

Under vLLM TP=n every worker used to run the WHOLE source model on the prompt (the 128K bottleneck: 3.1 s of the
3.4 s mapped TTFT on Qwen2.5-3B->72B TP=2). Here each rank keeps 1/n of the source (Megatron split: q/k/v/gate/up
by output rows, o/down by input columns + all-reduce), so a rank computes 1/n of the source FLOPs. The mapper reads
every source KV head, so each layer's local K/V heads are all-gathered afterwards (one collective per K and V).

Supported head layouts: Hkv % n == 0 (each rank owns Hkv/n KV heads) and n % Hkv == 0 (Qwen2.5-3B has 2 KV heads:
each KV head is replicated on n/Hkv ranks, each of which owns a disjoint group of its query heads).
"""
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class RowParallelLinear(nn.Module):
    """y = all_reduce(x_local @ W[:, local]^T) + b: the input features of this rank only."""

    def __init__(self, lin, rank, world, group):
        super().__init__()
        n = lin.in_features // world
        self.weight = nn.Parameter(lin.weight.data[:, rank * n:(rank + 1) * n].contiguous(), requires_grad=False)
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.data.clone(), requires_grad=False)
        self.group = group
        self.in_features, self.out_features = n, lin.out_features

    def forward(self, x):
        y = F.linear(x, self.weight)
        dist.all_reduce(y, group=self.group)
        return y if self.bias is None else y + self.bias


def _rows(lin, r0, r1):
    lin.weight = nn.Parameter(lin.weight.data[r0:r1].contiguous(), requires_grad=False)
    if lin.bias is not None:
        lin.bias = nn.Parameter(lin.bias.data[r0:r1].contiguous(), requires_grad=False)
    lin.out_features = r1 - r0


def kv_layout(cfg, world):
    """(local KV heads, replication factor): rep > 1 means n/Hkv ranks share each KV head."""
    H, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
    if H % world:
        raise ValueError(f"source has {H} query heads, not divisible by TP={world}")
    if Hkv % world == 0:
        return Hkv // world, 1
    if world % Hkv == 0:
        return 1, world // Hkv
    raise ValueError(f"source KV heads {Hkv} vs TP={world}: neither divides the other")


def shard_source(model, rank, world, group):
    """Shard a HF decoder (Qwen2/Qwen3/Llama/Mistral layout) in place for TP rank `rank` of `world`."""
    cfg = model.config
    H, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    Hl = H // world
    hkv_l, rep = kv_layout(cfg, world)
    kv0 = rank * hkv_l if rep == 1 else rank // rep  # first KV head of this rank
    I = cfg.intermediate_size
    if I % world:
        raise ValueError(f"source intermediate_size {I} not divisible by TP={world}")
    Il = I // world
    for layer in model.model.layers:
        a, m = layer.self_attn, layer.mlp
        _rows(a.q_proj, rank * Hl * D, (rank + 1) * Hl * D)
        _rows(a.k_proj, kv0 * D, (kv0 + hkv_l) * D)
        _rows(a.v_proj, kv0 * D, (kv0 + hkv_l) * D)
        a.o_proj = RowParallelLinear(a.o_proj, rank, world, group)
        _rows(m.gate_proj, rank * Il, (rank + 1) * Il)
        _rows(m.up_proj, rank * Il, (rank + 1) * Il)
        m.down_proj = RowParallelLinear(m.down_proj, rank, world, group)
        # HF attention reshapes with -1 heads and repeats KV by num_key_value_groups = Hl / hkv_l (unchanged ratio
        # when rep == 1; with rep > 1 it is Hl / 1)
        a.num_key_value_groups = Hl // hkv_l
        for attr, v in (("num_heads", Hl), ("num_attention_heads", Hl), ("num_key_value_heads", hkv_l)):
            if hasattr(a, attr):
                setattr(a, attr, v)
    model._kvx_tp = dict(rank=rank, world=world, rep=rep, group=group)
    return model


def gather_heads(x, world, group, rep=1, head_dim=-2):
    """x: this rank's heads, shape [..., Hl, D] with the head axis at `head_dim` (default -2).
    Returns the full tensor with ranks' heads in rank order (ranks sharing a KV head contribute it once)."""
    x = x.movedim(head_dim, 0).contiguous()  # [Hl, ...]
    Hl = x.shape[0]
    full = x.new_empty((world * Hl,) + tuple(x.shape[1:]))  # concatenated output form (gloo and NCCL both take it)
    dist.all_gather_into_tensor(full, x, group=group)  # [H, ...] rank-major = global head order
    if rep > 1:
        full = full.view((world, Hl) + tuple(x.shape[1:]))[::rep].reshape((-1,) + tuple(x.shape[1:]))
    return full.movedim(0, head_dim)
