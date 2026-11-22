"""kvx.distill_full on tiny random Qwen3 models (CPU): step 0 == the given maps, training lowers KL."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kvx import core, distill_full  # noqa: E402


def _tiny(seed, layers, kv_heads):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=kv_heads, head_dim=16, max_position_embeddings=256)
    return Qwen3ForCausalLM(cfg).eval()


def _maps(S, T):
    Ls, Hs, D = S.config.num_hidden_layers, S.config.num_key_value_heads, 16
    Lt, Ht = T.config.num_hidden_layers, T.config.num_key_value_heads
    g = torch.Generator().manual_seed(1)
    mk, mv = [], []
    for l in range(Lt):
        sel = [l % Ls, (l + 1) % Ls]
        for out in (mk, mv):
            out.append(dict(sel=sel, W=torch.randn(len(sel) * Hs * D, Ht * D, generator=g) * 0.2,
                            b=torch.zeros(Ht * D), mu=torch.zeros(Ht * D), H=Ht, D=D))
    return mk, mv


def test_step0_equals_maps_and_training_lowers_kl():
    S, T = _tiny(0, 2, 2), _tiny(1, 3, 2)
    for p in list(S.parameters()) + list(T.parameters()):
        p.requires_grad_(False)
    mk, mv = _maps(S, T)
    P, L = 24, 40
    docs = torch.randint(0, 97, (4, L), generator=torch.Generator().manual_seed(2))
    cs, ss = core.rope_cos_sin(S, L, "cpu")
    ct, st = core.rope_cos_sin(T, L, "cpu")
    kw = torch.stack([l.self_attn.k_norm.weight for l in T.model.layers]).float()
    # reference: KL with the untouched maps through the eval path (no autograd)
    with torch.no_grad():
        Ks, Vs, _ = core.capture_kv(S, docs[:, :P], cs, ss)
        K = core.restore_scale(core.apply_maps(mk, Ks).float(), mk, "rms", 2, kw)
        V = core.apply_maps(mv, Vs)
        c = core.build_cache(K, V, ct, st)
        lg = T(input_ids=docs[:, P:], past_key_values=c).logits.float()
        lp = torch.log_softmax(T(input_ids=docs).logits[:, P:].float(), -1)
        ref = core.kl_rows(lp, lg).mean().item()
    tm, hist = distill_full.train(S, T, mk, mv, docs, docs, P, cs, ss, ct, st, kw, lr=3e-3, steps=30, bs=4, accum=1)
    v0 = hist["val"][0][1]
    assert abs(v0 - ref) < 2e-3 * max(1.0, ref), (v0, ref)
    assert hist["val"][-1][1] < 0.8 * v0, hist["val"]
    ek, ev = tm.export()
    assert ek[0]["W"].shape == mk[0]["W"].shape and ek[0]["sel"] == mk[0]["sel"] and not ek[0]["W"].requires_grad
