"""Tensor-parallel Qwen3.5 source (kvx/tp_source_hybrid.py) on CPU with gloo: the sharded GatedDeltaNet + gated
attention + MLP decoder gives the same logits and the same xkv source features (input_layernorm outputs = the
residual stream, what the xkv mapper reads) as the whole model. TP=2 (KV heads split) and TP=4 (2 KV heads replicated
on 2 ranks each). Needs transformers >= 5 (qwen3_5): run in the vLLM image via tests/modal_unit_vllm_cpu.py."""
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("transformers.models.qwen3_5")


def _tiny(vh):
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from kvx import hybrid
    torch.manual_seed(3)
    cfg = Qwen3_5TextConfig(vocab_size=101, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                            linear_num_key_heads=4, linear_num_value_heads=vh, linear_key_head_dim=8,
                            linear_value_head_dim=8, linear_conv_kernel_dim=4,
                            layer_types=["linear_attention", "linear_attention", "full_attention", "linear_attention"],
                            partial_rotary_factor=0.25, max_position_embeddings=256)
    m = Qwen3_5ForCausalLM(cfg).eval().float()
    with torch.no_grad():  # zero-init norms / A would hide head-order bugs
        for n, p in m.named_parameters():
            if "norm" in n or "A_log" in n or "dt_bias" in n:
                p.add_(0.3 * torch.randn_like(p))
    m.config._attn_implementation = "sdpa"
    return hybrid.Dec(m)


def _port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _feats(dec, ids):
    from kvx.xkv_adapter import source_inputs
    feats, emb, _ = source_inputs(dec, ids, list(range(len(dec.layers))))
    logits, _ = dec.forward(ids, keep=ids.shape[1], use_cache=False)
    return {"logits": logits, "emb": emb, **{f"f{i}": v for i, v in feats.items()}}


def _worker(rank, world, port, vh, d):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from kvx.tp_source_hybrid import shard_hybrid_source
    dec = _tiny(vh)
    ids = torch.randint(0, 101, (1, 23), generator=torch.Generator().manual_seed(7))
    shard_hybrid_source(dec, rank, world, None)
    out = _feats(dec, ids)
    torch.save(out, os.path.join(d, f"r{rank}.pt"))
    dist.destroy_process_group()


@pytest.mark.parametrize("world,vh", [(2, 8), (4, 4), (4, 8)])
def test_sharded_hybrid_source_matches_whole(tmp_path, world, vh):
    ref = _feats(_tiny(vh), torch.randint(0, 101, (1, 23), generator=torch.Generator().manual_seed(7)))
    mp.spawn(_worker, args=(world, _port(), vh, str(tmp_path)), nprocs=world, join=True)
    for r in range(world):
        got = torch.load(os.path.join(tmp_path, f"r{r}.pt"))
        assert set(got) == set(ref)
        for k, v in ref.items():
            rel = ((got[k] - v).norm() / v.norm()).item()
            assert got[k].shape == v.shape and rel < 1e-4, (world, vh, r, k, rel)


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 64])
def test_lin_states_chunked_equal_whole(chunk):
    """Token-chunked GDN state (conv tail + recurrent state carried) == lin_states over all tokens (CPU reference),
    on a tiny Qwen3.5's own GDN params with random taps."""
    from kvx import hybrid_state as hs
    dec = _tiny(8)
    p = hs.TargetParams.from_state_dict(dec.cfg, dec.model.state_dict())
    g = torch.Generator().manual_seed(11)
    T = 21
    C = 2 * p.Hk * p.dk + p.Hv * p.dv
    n = 0
    for l in p.lin:
        qkv, a, b = torch.randn(T, C, generator=g), torch.randn(T, p.Hv, generator=g), torch.randn(T, p.Hv, generator=g)
        conv, S = hs.lin_states(p, l, qkv, a, b)
        carry = None
        for c0 in range(0, T, chunk):
            carry = hs.lin_states_chunk(p, l, qkv[c0:c0 + chunk], a[c0:c0 + chunk], b[c0:c0 + chunk], carry)
        assert torch.allclose(carry[0].t(), conv, atol=1e-6), l
        assert torch.allclose(carry[1], S, atol=1e-5, rtol=1e-4), (l, (carry[1] - S).abs().max())
        n += 1
    assert n == 3


@pytest.mark.parametrize("chunk", [5, 8, 23])
def test_source_chunked_cached_forward_features_equal_whole(chunk):
    """The hybrid source run in token chunks with its own cache (GDN conv/recurrent + attention KV carried) gives the
    same input_layernorm features (what xkv maps read) as one forward over all tokens."""
    dec = _tiny(8)
    ids = torch.randint(0, 101, (1, 23), generator=torch.Generator().manual_seed(7))
    whole = _feats(dec, ids)
    from kvx.vllm_hybrid_connector import chunked_source_features
    got = {i: [] for i in range(len(dec.layers))}
    for c0, feats in chunked_source_features(dec, ids, list(range(len(dec.layers))), chunk):
        for i, x in feats.items():
            got[i].append(x)
    for i in range(len(dec.layers)):
        x = torch.cat(got[i], 1)
        y = whole[f"f{i}"]
        rel = ((x - y).norm() / y.norm()).item()
        assert x.shape == y.shape and rel < 1e-4, (chunk, i, rel)
