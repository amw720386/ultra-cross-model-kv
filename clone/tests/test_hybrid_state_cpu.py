"""kvx.hybrid_state reproduces HF's hybrid cache from mapped taps (kvx-pairs' fixture, no transformers needed)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kvx import hybrid_state as hs  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "hybrid_fixture.pt")


@pytest.fixture(scope="module")
def fx():
    if not os.path.exists(FIX):
        pytest.skip("fixture missing")
    d = torch.load(FIX, weights_only=False)
    p = hs.TargetParams.from_state_dict(d["tgt_config"], d["tgt_state"])
    return d, p


def test_attention_layers_match_hf(fx):
    d, p = fx
    for l, ref in d["ref_cache"].items():
        if ref["kind"] != "attn":
            continue
        t = d["mapped_taps"][l]
        K, V = hs.attn_kv(p, t["k"][0], t["v"][0])
        Kr = ref["K_post_rope"][0].permute(1, 0, 2)  # [H,T,D] -> [T,H,D]
        Vr = ref["V"][0].permute(1, 0, 2)
        assert torch.allclose(K, Kr, atol=1e-4, rtol=1e-4), (l, (K - Kr).abs().max())
        assert torch.allclose(V, Vr, atol=1e-5), l


def test_lin_layers_match_hf(fx):
    d, p = fx
    n_lin = 0
    for l, ref in d["ref_cache"].items():
        if ref["kind"] != "lin":
            continue
        t = d["mapped_taps"][l]
        conv, S = hs.lin_states(p, l, t["qkv"][0], t["a"][0], t["b"][0])
        hf_conv = ref["conv_state"][0]  # [C, kernel]: HF keeps kernel rows; vLLM keeps the last kernel-1
        assert torch.allclose(conv, hf_conv[:, -(p.kernel - 1):], atol=1e-5), l
        Sr = ref["recurrent_state"][0]
        assert torch.allclose(S, Sr, atol=1e-4, rtol=1e-3), (l, (S - Sr).abs().max())
        n_lin += 1
    assert n_lin == 6


def test_short_prefix_conv_state_pads(fx):
    d, p = fx
    l = next(l for l, r in d["ref_cache"].items() if r["kind"] == "lin")
    t = d["mapped_taps"][l]
    conv, _ = hs.lin_states(p, l, t["qkv"][0, :2], t["a"][0, :2], t["b"][0, :2])
    assert conv.shape == (t["qkv"].shape[-1], p.kernel - 1)
    assert torch.equal(conv[:, 0], torch.zeros_like(conv[:, 0]))
    assert torch.equal(conv[:, 1:], t["qkv"][0, :2].float().t())


def _fmaps(d):
    return {int(t): dict(sel=m["sel"], W=m["W"].float(), b=m["b"].float()) for t, m in d["maps_file"]["maps"].items()}


def test_taps_from_feats_reproduces_fixture_taps(fx):
    d, p = fx
    knw = hs.k_norm_weights(d["tgt_config"], d["tgt_state"])
    feats = [f.float() for f in d["source_feats"]]
    got = hs.taps_from_feats(d["maps_file"], _fmaps(d), feats, d["P"], knw)
    for t, ref in d["mapped_taps"].items():
        for name, v in ref.items():
            assert torch.allclose(got[t][name], v[0].float(), atol=2e-3, rtol=2e-3), (t, name, (got[t][name] - v[0]).abs().max())


@pytest.mark.parametrize("dim_first", [True, False])
def test_end_to_end_into_vllm_shaped_caches(fx, dim_first):
    """feats -> taps -> states -> vLLM-shaped caches; read back == HF reference cache."""
    d, p = fx
    knw = hs.k_norm_weights(d["tgt_config"], d["tgt_state"])
    feats = [f.float() for f in d["source_feats"]]
    kinds = {t: x["kind"] for t, x in enumerate(d["maps_file"]["tgt_taps"])}
    states = hs.caches_from_taps(p, kinds, hs.taps_from_feats(d["maps_file"], _fmaps(d), feats, d["P"], knw))
    slots, slot = 5, 3
    for t, ref in d["ref_cache"].items():
        if ref["kind"] != "lin":
            K, V = states[t][1], states[t][2]
            assert torch.allclose(K, ref["K_post_rope"][0].permute(1, 0, 2), atol=5e-3, rtol=5e-3)
            continue
        _, conv, S = states[t]
        C, k1 = conv.shape
        conv_c = torch.zeros(slots, C, k1) if dim_first else torch.zeros(slots, k1, C)
        ssm_c = torch.zeros(slots, p.Hv, p.dv, p.dk)
        hs.write_lin_state(conv_c, ssm_c, slot, conv, S, dim_first)
        back_conv = conv_c[slot] if dim_first else conv_c[slot].t()
        assert torch.allclose(back_conv, ref["conv_state"][0][:, -k1:], atol=5e-3)
        assert torch.allclose(ssm_c[slot].transpose(-1, -2), ref["recurrent_state"][0], atol=5e-3, rtol=5e-3)
        assert float(conv_c[slot - 1].abs().sum()) == 0 and float(ssm_c[slot + 1].abs().sum()) == 0


def test_recolor_matches_kvx_pairs_apply(fx):
    """taps_from_feats(recolor=True) == kvx.hybrid.apply(recolor=True) on GatedDeltaNet layers (no rms restore there)."""
    from kvx import hybrid
    d, p = fx
    mf = d["maps_file"]
    g = torch.Generator().manual_seed(3)
    maps = {}
    for t, m in _fmaps(d).items():
        if mf["tgt_taps"][t]["kind"] != "lin":
            continue
        w = m["W"].shape[1]
        bd = 32
        nb = w // bd
        maps[t] = dict(m, W=m["W"].to(torch.bfloat16), rc=torch.eye(bd).repeat(nb, 1, 1) + 0.1 * torch.randn(nb, bd, bd, generator=g),
                       mu=torch.randn(w, generator=g))
    feats = [f.float() for f in d["source_feats"]]
    widths = {t: [tuple(x) for x in mf["tgt_taps"][t]["taps"]] for t in maps}
    ref = hybrid.apply(maps, feats, 1, d["P"], widths, recolor=True)
    got = hs.taps_from_feats(dict(mf, sinks=0, sink_template={}), maps, feats, d["P"], knw={}, recolor=True)
    plain = hs.taps_from_feats(dict(mf, sinks=0, sink_template={}), maps, feats, d["P"], knw={}, recolor=False)
    for t in maps:
        for name, v in ref[t].items():
            assert torch.allclose(got[t][name], v[0], atol=1e-4, rtol=1e-4), (t, name)
        assert not torch.allclose(plain[t]["qkv"], got[t]["qkv"])


def test_tp_slices_reassemble(fx):
    d, p = fx
    knw = hs.k_norm_weights(d["tgt_config"], d["tgt_state"])
    feats = [f.float() for f in d["source_feats"]]
    kinds = {t: x["kind"] for t, x in enumerate(d["maps_file"]["tgt_taps"])}
    states = hs.caches_from_taps(p, kinds, hs.taps_from_feats(d["maps_file"], _fmaps(d), feats, d["P"], knw))
    kd = p.Hk * p.dk
    for t, st in states.items():
        parts = [hs.slice_for_rank(p, st, r, 2) for r in range(2)]
        if st[0] == "attn":
            assert torch.equal(torch.cat([x[1] for x in parts], 1), st[1]) and torch.equal(torch.cat([x[2] for x in parts], 1), st[2])
        else:
            q = torch.cat([x[1][:kd // 2] for x in parts], 0)
            k = torch.cat([x[1][kd // 2:kd] for x in parts], 0)
            v = torch.cat([x[1][kd:] for x in parts], 0)
            assert torch.equal(torch.cat([q, k, v], 0), st[1])
            assert torch.equal(torch.cat([x[2] for x in parts], 0), st[2])
    # tp > kv heads: replicated heads (fixture attention has 2 kv heads, tp=4 -> heads 0,0,1,1)
    t_attn = next(t for t, s in states.items() if s[0] == "attn")
    got = [hs.slice_for_rank(p, states[t_attn], r, 4)[1] for r in range(4)]
    assert torch.equal(got[0], got[1]) and torch.equal(got[2], states[t_attn][1][:, 1:2])


def test_yarn_matches_transformers():
    """Mapped-key RoPE with YaRN (Qwen3.8 to 1M: factor 4 over 262144) == transformers' yarn init (partial rotary)."""
    from transformers import PretrainedConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    rp = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144, "rope_theta": 1e7,
          "partial_rotary_factor": 0.25}
    cfg = dict(rope_parameters=rp, head_dim=256, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
               linear_value_head_dim=16, linear_conv_kernel_dim=4, layer_types=["full_attention"])
    p = hs.TargetParams(cfg, {}, 256)
    cos, sin, rd = hs.rope_cos_sin(p, 64, "cpu", offset=300000)
    hc = PretrainedConfig(rope_theta=1e7, head_dim=256, partial_rotary_factor=0.25, rope_scaling=dict(rp),
                          max_position_embeddings=1048576, hidden_size=256 * 8, num_attention_heads=8)
    try:
        inv, af = ROPE_INIT_FUNCTIONS["yarn"](hc, "cpu")
    except Exception as e:  # transformers API differences across versions
        pytest.skip(f"transformers yarn init unavailable: {e!r}")
    assert inv.shape[0] == rd // 2
    pos = torch.arange(300000, 300064, dtype=torch.float32)
    f = torch.outer(pos, inv.float())
    emb = torch.cat([f, f], -1)
    assert torch.allclose(cos, emb.cos() * af, atol=1e-5) and torch.allclose(sin, emb.sin() * af, atol=1e-5)
