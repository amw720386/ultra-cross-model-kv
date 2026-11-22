"""Gemma-4 connector identity oracle (CPU, needs a transformers with Gemma 4: run in the vLLM image via
tests/modal_unit_vllm_cpu.py): a tiny Gemma-4 mapped onto ITSELF with identity maps (target layer t reads source layer
t's k_norm / v_norm taps) must give exactly the target's own cache: post-RoPE K (per-type RoPE: default for sliding,
proportional partial for global) and v_norm'ed V, for both layer kinds, with K=V global layers."""
import os

import pytest
import torch


def _tiny(tmp):
    import transformers as tf
    if not hasattr(tf, "Gemma4TextConfig"):
        pytest.skip("transformers has no Gemma 4")
    cfg = tf.Gemma4TextConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                              num_attention_heads=4, num_key_value_heads=2, num_global_key_value_heads=1, head_dim=16,
                              global_head_dim=32, sliding_window=8, max_position_embeddings=256,
                              attention_k_eq_v=True, num_kv_shared_layers=0,
                              layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2,
                              rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1e6,
                                                                  "rope_type": "proportional"},
                                               "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"}})
    torch.manual_seed(0)
    m = tf.Gemma4ForCausalLM(cfg).eval().float()
    d = os.path.join(tmp, "g4")
    m.save_pretrained(d, safe_serialization=True)
    return m, d


def test_gemma_connector_identity_maps_reproduce_target_cache(tmp_path):
    from kvx import hybrid, core
    from kvx.vllm_gemma_connector import GemmaCrossModelConnector
    m, d = _tiny(str(tmp_path))
    dec = hybrid.load(d, attn="eager", dtype=torch.float32)
    tw = [dict(w) for w in hybrid.tap_widths(dec)]
    core_maps = {"k": [], "v": []}
    for t, layer in enumerate(m.model.layers):
        D = layer.self_attn.head_dim
        for kind in ("k", "v"):
            w = tw[t][kind]
            core_maps[kind].append(dict(sel=[t], W=torch.eye(w), b=torch.zeros(w), H=w // D, D=D, layer=t))
    mp = str(tmp_path / "maps_core.pt")
    torch.save(core_maps, mp)
    c = GemmaCrossModelConnector.__new__(GemmaCrossModelConnector)
    c.src_name, c.tgt_name, c.maps_path, c.sink_path = d, d, mp, None
    c.src_sdpa, c.max_len, c.tp_override, c.mem_fraction = "eager", 64, (0, 1), 0
    c._init_worker(torch.device("cpu"))
    c.Sdec = dec  # fp32 source for an exact comparison
    ids = torch.randint(0, 97, (1, 20))
    with torch.no_grad():
        o = m(input_ids=ids, use_cache=True)
    ref = core.cache_layers(o.past_key_values)
    feats = c._features(ids)
    for t in range(len(m.model.layers)):
        K, V = c._map_layer(t, feats, 20)
        k_ref, v_ref = ref[t][0][0].transpose(0, 1).float(), ref[t][1][0].transpose(0, 1).float()  # [T', H, D]
        n = k_ref.shape[0]  # a sliding cache may hold only the last window
        for X, Y in ((K[-n:], k_ref), (V[-n:], v_ref)):  # the deployed maps are bf16: identity GEMM in bf16
            rel = ((X - Y).norm() / Y.norm()).item()
            assert X.shape == Y.shape and rel < 1e-2, (t, rel)


def test_gemma_connector_paged_write_groups_and_window(tmp_path):
    """_write_requests: per-group block tables; sliding layers get only the last `window` mapped tokens (earlier
    blocks are vLLM's null block 0 and stay untouched); global layers get every token."""
    from types import SimpleNamespace as NS
    from kvx import hybrid
    from kvx.vllm_gemma_connector import GemmaCrossModelConnector, GemmaMeta, _GReq
    m, d = _tiny(str(tmp_path))
    dec = hybrid.load(d, attn="eager", dtype=torch.float32)
    tw = [dict(w) for w in hybrid.tap_widths(dec)]
    core_maps = {"k": [], "v": []}
    for t, layer in enumerate(m.model.layers):
        D = layer.self_attn.head_dim
        for kind in ("k", "v"):
            w = tw[t][kind]
            core_maps[kind].append(dict(sel=[t], W=torch.eye(w), b=torch.zeros(w), H=w // D, D=D, layer=t))
    mp = str(tmp_path / "maps_core.pt")
    torch.save(core_maps, mp)
    c = GemmaCrossModelConnector.__new__(GemmaCrossModelConnector)
    c.src_name, c.tgt_name, c.maps_path, c.sink_path = d, d, mp, None
    c.src_sdpa, c.max_len, c.tp_override, c.mem_fraction, c._layouts = "eager", 64, (0, 1), 0, {}
    c._backend = "FLASH_ATTN"
    bs, T = 4, 20
    sl = [t for t, ty in enumerate(m.config.layer_types) if ty == "sliding_attention"]
    gl = [t for t in range(len(m.config.layer_types)) if t not in sl]
    c.groups = [(bs, sl), (bs, gl)]
    c.layer_group = {**{t: 0 for t in sl}, **{t: 1 for t in gl}}
    c._init_worker(torch.device("cpu"))
    c.Sdec = dec
    nb = 12
    views = {}
    for t, layer in enumerate(m.model.layers):
        H, D = c.maps[t]["k"]["H"], c.maps[t]["k"]["D"]
        views[t] = torch.full((nb, bs, H, 2 * D), 7.0).permute(0, 2, 1, 3)  # FA layout view [blocks, H, block, 2D]
    gblocks = [1, 2, 3, 4, 5]          # 5 blocks of 4 = 20 tokens
    sblocks = [0, 0, 0, 6, 7]          # window 8: only the last 2 blocks are real, the rest are vLLM's null block
    ids = torch.randint(0, 97, (T,)).tolist()
    c._write_requests(GemmaMeta(requests=[_GReq(token_ids=ids, block_ids=(sblocks, gblocks))]), views, torch.device("cpu"))
    feats = c._features(torch.tensor([ids]))
    for t in range(len(m.model.layers)):
        K, V = c._map_layer(t, feats, T)
        D = K.shape[-1]
        phys = views[t].permute(0, 2, 1, 3)
        blocks = sblocks if t in sl else gblocks
        for p in range(T):
            b, o = blocks[p // bs], p % bs
            written = (t in gl) or p >= T - m.config.sliding_window
            if written:
                assert torch.allclose(phys[b, o, :, :D].float(), K[p], atol=2e-2), (t, p)
                assert torch.allclose(phys[b, o, :, D:].float(), V[p], atol=2e-2), (t, p)
        assert (views[t].permute(0, 2, 1, 3)[0] == 7.0).all()  # the null block is never written
