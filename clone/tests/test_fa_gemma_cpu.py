"""vllm_fa source attention (kvx/fa.py) == eager for Gemma-family sources (local sliding-window layers, logit softcap),
with a CPU reference of vLLM's varlen kernel. Needs a transformers with the model class (run in the vLLM image:
modal run tests/modal_unit_vllm_cpu.py). Gemma 2/3 always; Gemma 4 when transformers has it."""
import pytest
import torch

from kvx import core, fa


def _fa_ref(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal,
            window_size=None, softcap=0.0, **kw):
    T, Hq, D = q.shape
    S, Hk = k.shape[0], k.shape[1]
    kf = k.float().repeat_interleave(Hq // Hk, 1)
    vf = v.float().repeat_interleave(Hq // Hk, 1)
    sc = torch.einsum("thd,shd->hts", q.float(), kf) * softmax_scale
    if softcap:
        sc = softcap * torch.tanh(sc / softcap)
    qpos = torch.arange(T)[:, None] + S - T
    mask = torch.arange(S)[None] <= qpos
    if window_size is not None and window_size[0] >= 0:
        mask &= torch.arange(S)[None] >= qpos - window_size[0]
    sc = sc.masked_fill(~mask, float("-inf"))
    return torch.einsum("hts,shd->thd", sc.softmax(-1), vf).to(q.dtype)


def _tiny(kind):
    import transformers as tf
    common = dict(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=16, max_position_embeddings=256, sliding_window=8)
    if kind == "gemma2":
        cfg = tf.Gemma2Config(attn_logit_softcapping=30.0, final_logit_softcapping=None, **common)
        return tf.Gemma2ForCausalLM(cfg)
    if kind == "gemma3":
        cfg = tf.Gemma3TextConfig(**common)
        return tf.Gemma3ForCausalLM(cfg)
    if not hasattr(tf, "Gemma4TextConfig"):
        pytest.skip("transformers has no Gemma 4")
    cfg = tf.Gemma4TextConfig(**common)
    return tf.Gemma4ForCausalLM(cfg)


@pytest.mark.parametrize("kind", ["gemma2", "gemma3", "gemma4"])
def test_vllm_fa_gemma_source_matches_eager(kind, monkeypatch):
    torch.manual_seed(0)
    m = _tiny(kind).eval().float()
    ids = torch.randint(0, 97, (1, 24))
    m.config._attn_implementation = "eager"
    with torch.no_grad():
        ref = m(input_ids=ids, use_cache=True)
    monkeypatch.setattr(fa, "_fa_func", lambda: (_fa_ref, None))
    fa.use_vllm_fa(m)
    with torch.no_grad():
        out = m(input_ids=ids, use_cache=True)
    assert torch.allclose(out.logits, ref.logits, atol=1e-4), (out.logits - ref.logits).abs().max()
    for (k1, v1), (k2, v2) in zip(core.cache_layers(out.past_key_values), core.cache_layers(ref.past_key_values)):
        assert torch.allclose(k1, k2, atol=1e-4) and torch.allclose(v1, v2, atol=1e-4)
