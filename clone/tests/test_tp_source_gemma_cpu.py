"""Tensor-parallel Gemma-4 source (kvx/tp_source_gemma.py) on CPU with gloo: a tiny E-style Gemma-4 (per-layer
inputs, KV-shared layers with a double-wide MLP, sliding + global layers with different head dims) sharded over TP
ranks gives the same logits and, after gather_taps, the same per-layer k_norm/v_norm features (what the Gemma
connector maps from) as the whole model. TP=2 (2 KV heads split) and TP=4 (KV heads replicated), K==V on and off.
Needs transformers with Gemma 4: run in the vLLM image via tests/modal_unit_vllm_cpu.py."""
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _tiny(keqv):
    import transformers as tf
    if not hasattr(tf, "Gemma4TextConfig"):
        pytest.skip("transformers has no Gemma 4")
    from kvx import hybrid
    torch.manual_seed(2)
    cfg = tf.Gemma4TextConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                              num_attention_heads=4, num_key_value_heads=2, num_global_key_value_heads=1, head_dim=16,
                              global_head_dim=32, sliding_window=8, max_position_embeddings=256,
                              attention_k_eq_v=keqv, num_kv_shared_layers=2, use_double_wide_mlp=True,
                              hidden_size_per_layer_input=8, vocab_size_per_layer_input=97,
                              layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2,
                              rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1e6,
                                                                  "rope_type": "proportional"},
                                               "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"}})
    m = tf.Gemma4ForCausalLM(cfg).eval().float()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.add_(0.3 * torch.randn_like(p))
    m.config._attn_implementation = "sdpa"
    return hybrid.Dec(m)


def _port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _ids():
    return torch.randint(0, 97, (1, 21), generator=torch.Generator().manual_seed(4))


def _out(dec, feats):
    logits, _ = dec.forward(_ids(), keep=21, use_cache=False)
    return {"logits": logits, **{f"f{i}": x for i, x in enumerate(feats) if x is not None}}


def _worker(rank, world, port, keqv, d):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from kvx import hybrid
    from kvx.tp_source_gemma import gather_taps, shard_gemma_source
    dec = _tiny(keqv)
    hybrid.tap_widths(dec)  # full widths, before the split (as the connector does)
    shard_gemma_source(dec, rank, world, None)
    feats = gather_taps(dec, hybrid.source_features(dec, _ids()))
    torch.save(_out(dec, feats), os.path.join(d, f"r{rank}.pt"))
    dist.destroy_process_group()


@pytest.mark.parametrize("world,keqv", [(2, False), (4, False), (2, True), (4, True)])
def test_sharded_gemma_source_matches_whole(tmp_path, world, keqv):
    from kvx import hybrid
    dec = _tiny(keqv)
    ref = _out(dec, hybrid.source_features(dec, _ids()))
    assert sum(k.startswith("f") for k in ref) == 4  # 6 layers, 2 KV-shared
    mp.spawn(_worker, args=(world, _port(), keqv, str(tmp_path)), nprocs=world, join=True)
    for r in range(world):
        got = torch.load(os.path.join(tmp_path, f"r{r}.pt"))
        assert set(got) == set(ref)
        for k, v in ref.items():
            rel = ((got[k].float() - v.float()).norm() / v.float().norm()).item()
            assert got[k].shape == v.shape and rel < 1e-2, (world, keqv, r, k, rel)  # features are bf16


def _connector(d, mp_path, rank, world, src_tp):
    from kvx import hybrid
    from kvx.vllm_gemma_connector import GemmaCrossModelConnector
    if not getattr(hybrid.load, "_fp32", False):  # fp32 source in both arms: bf16 all-reduce order is not the test
        _load = hybrid.load
        def load32(*a, **k):
            k["dtype"] = torch.float32
            return _load(*a, **k)
        load32._fp32 = True
        hybrid.load = load32
    c = GemmaCrossModelConnector.__new__(GemmaCrossModelConnector)
    c.src_name, c.tgt_name, c.maps_path, c.sink_path = d, d, mp_path, None
    c.src_sdpa, c.max_len, c.tp_override, c.mem_fraction = "sdpa", 64, (rank, world), 0
    c.src_tp, c.tp_group_override = src_tp, (dist.group.WORLD if world > 1 else None)
    c._init_worker(torch.device("cpu"))
    return c


def _identity_maps(m, d, tmp):
    from kvx import hybrid
    dec = hybrid.load(d, attn="eager", dtype=torch.float32)
    tw = [dict(w) for w in hybrid.tap_widths(dec)]
    maps = {"k": [], "v": []}
    for t, layer in enumerate(m.model.layers):
        if layer.self_attn.is_kv_shared_layer:
            continue
        D = layer.self_attn.head_dim
        for kind in ("k", "v"):
            w = tw[t][kind]
            maps[kind].append(dict(sel=[t], W=torch.eye(w), b=torch.zeros(w), H=w // D, D=D, layer=t))
    p = os.path.join(tmp, "maps.pt")
    torch.save(maps, p)
    return p


def _cworker(rank, world, port, d, mp_path, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    c = _connector(d, mp_path, rank, world, 1)
    assert c.src_tp_on
    feats = c._features(_ids())
    res = {t: c._map_layer(t, feats, 21) for t in c.maps}
    torch.save({"kv": res, "h0": {t: c.maps[t]["k"]["h0"] for t in c.maps}}, os.path.join(out, f"c{rank}.pt"))
    dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_gemma_connector_src_tp_equals_tp1(tmp_path, world):
    """Connector under TP=world with the source split: each rank maps exactly its heads of the TP=1 connector's K/V."""
    dec = _tiny(False)
    d = str(tmp_path / "g4")
    dec.model.save_pretrained(d, safe_serialization=True)
    mp_path = _identity_maps(dec.model, d, str(tmp_path))
    c1 = _connector(d, mp_path, 0, 1, 0)
    f1 = c1._features(_ids())
    ref = {t: c1._map_layer(t, f1, 21) for t in c1.maps}
    mp.spawn(_cworker, args=(world, _port(), d, mp_path, str(tmp_path)), nprocs=world, join=True)
    for r in range(world):
        got = torch.load(os.path.join(tmp_path, f"c{r}.pt"))
        for t, (K, V) in got["kv"].items():
            h0, H = got["h0"][t], K.shape[1]
            for X, Y in ((K, ref[t][0][:, h0:h0 + H]), (V, ref[t][1][:, h0:h0 + H])):
                rel = ((X - Y).norm() / Y.norm()).item()
                assert X.shape == Y.shape and rel < 5e-3, (world, r, t, rel)  # maps are bf16 GEMMs
