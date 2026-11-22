"""CPU oracle: GLM-MoE-DSA cache rows built from tap outputs (kvx.glm_rows) == what HF caches natively."""
import pytest
import torch

from kvx import glm_rows as G
from test_hybrid_cpu import _record_taps, _tiny_glm

pytest.importorskip("transformers.models.glm_moe_dsa")


def _native(dec, ids):
    _, cache = dec.forward(ids, keep=1)
    return cache


def test_rows_match_hf_latent_and_indexer_cache():
    dec = _tiny_glm("dsa", 0)
    T = 27
    ids = torch.randint(0, 101, (1, T))
    cache = _native(dec, ids)
    given = _record_taps(dec, ids)
    pos = torch.arange(T)
    got = G.rows(dec, given, pos, layout="hf")
    pairs = G.rows(dec, given, pos, layout="pairs")
    for l, r in got.items():
        lay = cache.layers[l]
        k_pass, k_rot = lay.keys[0, 0], lay.values[0, 0]  # [T, R], [T, r]
        R = dec.layers[l].self_attn.kv_lora_rank
        assert torch.allclose(r["mla"][:, :R], k_pass, atol=1e-5), l
        assert torch.allclose(r["mla"][:, R:], k_rot, atol=1e-5), l
        assert torch.allclose(r["idx"], lay.indexer_keys[0], atol=1e-5), l
        # pairs layout is the in-place interleaved rotation of the same values
        assert torch.allclose(G.hf_to_pairs(r["mla"][:, R:]), pairs[l]["mla"][:, R:], atol=1e-6)
        assert torch.allclose(G.pairs_to_hf(pairs[l]["mla"][:, R:]), r["mla"][:, R:], atol=1e-6)
        assert not torch.allclose(pairs[l]["mla"][:, R:], k_rot, atol=1e-3)  # layouts really differ


def test_pairs_layout_scores_match_with_pairs_query():
    """q.k is layout-invariant only when q and k share the layout (the mixing bug gives different scores)."""
    torch.manual_seed(1)
    T, d = 9, 8
    q, k = torch.randn(T, d), torch.randn(T, d)
    dec = _tiny_glm("dsa", 0)
    c, s = G.cos_sin(dec, torch.arange(T))
    s_hf = G.rope_pairs(q, c, s, "hf") @ G.rope_pairs(k, c, s, "hf").T
    s_pp = G.rope_pairs(q, c, s, "pairs") @ G.rope_pairs(k, c, s, "pairs").T
    s_mix = G.rope_pairs(q, c, s, "pairs") @ G.rope_pairs(k, c, s, "hf").T
    assert torch.allclose(s_hf, s_pp, atol=1e-5)
    assert not torch.allclose(s_hf, s_mix, atol=1e-2)


def test_blocks_view():
    x = torch.arange(10 * 3, dtype=torch.float32).view(10, 3)
    b = G.to_blocks(x, 4)
    assert b.shape == (3, 4, 3)
    assert torch.equal(b.reshape(-1, 3)[:10], x) and (b.reshape(-1, 3)[10:] == 0).all()
