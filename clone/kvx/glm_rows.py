"""Cache rows for a GLM-MoE-DSA target (MLA + DSA) from per-token tap outputs: the writer oracle for a serving connector.

Inputs per target layer (what kvx maps produce, hybrid.layer_taps kind "mla"):
  ckv: kv_a_proj_with_mqa output [T, kv_lora_rank + rope_dim] (pre-norm latent | pre-RoPE rope key)
  ik:  indexer.k_norm output     [T, index_head_dim]           ([rope part | pass part], pre-RoPE)
Outputs per layer:
  mla: [T, kv_lora_rank + rope_dim] = [kv_a_layernorm(latent) | RoPE(rope key)]   (576 wide on GLM-5.3)
  idx: [T, index_head_dim]          = [RoPE(rope part) | pass part]               (unquantized)

RoPE layout (GLM uses DeepSeek's interleaved pairs (x0, x1), (x2, x3), ...):
  layout="pairs": rotated values written back in place, pairs kept adjacent: the layout of an in-place
                  non-neox rotary kernel (vLLM is_neox_style=False).
  layout="hf":    HF apply_rotary_pos_emb_interleave output: [rotated evens | rotated odds] (what HF caches).
Both give the same q.k when the query uses the same layout; mixing them is the bug this module exists to catch.
cos/sin come from the target's own rotary_emb (YaRN attention factor included, as HF and vLLM both bake it in).
"""
import torch


def cos_sin(dec, positions, dtype=torch.float32):
    """positions [T] -> (cos, sin) [T, rope_dim] in the HF cat(freqs, freqs) form."""
    rot = dec.text.rotary_emb
    dev = rot.inv_freq.device  # sharded targets: the rotary module may sit on another GPU than the caller
    x = torch.zeros(1, dtype=dtype, device=dev)
    c, s = rot(x, positions.to(dev)[None])
    return c[0].to(positions.device), s[0].to(positions.device)


def rope_pairs(x, cos, sin, layout="pairs"):
    """Interleaved-pair RoPE on x [..., T, d] with cos/sin [T, d] (HF cat form)."""
    h = cos.shape[-1] // 2
    c, s = cos[..., :h].to(x.dtype), sin[..., :h].to(x.dtype)
    a, b = x[..., 0::2], x[..., 1::2]
    ra, rb = a * c - b * s, b * c + a * s
    if layout == "hf":
        return torch.cat([ra, rb], -1)
    return torch.stack([ra, rb], -1).flatten(-2)


def hf_to_pairs(x):
    """[evens | odds] -> interleaved pairs (last dim)."""
    h = x.shape[-1] // 2
    return torch.stack([x[..., :h], x[..., h:]], -1).flatten(-2)


def pairs_to_hf(x):
    return torch.cat([x[..., 0::2], x[..., 1::2]], -1)


def layer_rows(attn, ckv, ik, cos, sin, layout="pairs"):
    """One MLA(+DSA) layer: tap outputs [T, *] -> {"mla": [T, R + r], "idx": [T, Di] or None}."""
    R, r = attn.kv_lora_rank, attn.qk_rope_head_dim
    lat, kr = ckv[..., :R], ckv[..., R:R + r]
    out = {"mla": torch.cat([attn.kv_a_layernorm(lat), rope_pairs(kr, cos, sin, layout)], -1), "idx": None}
    ix = getattr(attn, "indexer", None)
    if ik is not None and ix is not None:
        ri = ix.qk_rope_head_dim
        out["idx"] = torch.cat([rope_pairs(ik[..., :ri], cos, sin, layout), ik[..., ri:]], -1)
    return out


@torch.no_grad()
def rows(dec, given, positions, layout="pairs"):
    """given {layer: {"ckv": [B*T or T, w], "ik": ...}} (hybrid tap dict for one sequence) -> {layer: rows}."""
    cos, sin = cos_sin(dec, positions)
    out = {}
    for l, d in given.items():
        attn = dec.layers[l].self_attn
        out[l] = layer_rows(attn, d["ckv"].reshape(len(positions), -1), None if d.get("ik") is None
                            else d["ik"].reshape(len(positions), -1), cos, sin, layout)
    return out


def to_blocks(x, block_size):
    """[T, D] -> [ceil(T / block_size), block_size, D], zero-padded (paged-cache view, block table = arange)."""
    T, D = x.shape
    nb = -(-T // block_size)
    pad = x.new_zeros(nb * block_size - T, D)
    return torch.cat([x, pad]).view(nb, block_size, D)
