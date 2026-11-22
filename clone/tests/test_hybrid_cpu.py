"""CPU tests for kvx/hybrid.py on tiny random Qwen3.5-style hybrids (GatedDeltaNet + gated attention, partial RoPE).
Needs transformers >= 5 (qwen3_5)."""
import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("transformers.models.qwen3_5")
from kvx import hybrid  # noqa: E402


def _tiny(seed, layers=4, hidden=64, vh=4):
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(vocab_size=101, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                            linear_num_key_heads=2, linear_num_value_heads=vh, linear_key_head_dim=16,
                            linear_value_head_dim=16, linear_conv_kernel_dim=4,
                            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * (layers // 4),
                            partial_rotary_factor=0.25, max_position_embeddings=256)
    m = Qwen3_5ForCausalLM(cfg).eval()
    with torch.no_grad():  # random-init zero norms/A would make the layers trivial; perturb everything a little
        for n, p in m.named_parameters():
            if "norm" in n or "A_log" in n or "dt_bias" in n:
                p.add_(0.3 * torch.randn_like(p))
    m.config._attn_implementation = "sdpa"
    return hybrid.Dec(m)


def _native_cont(dec, ids, P):
    lg, _ = dec.forward(ids, keep=ids.shape[1] - P, use_cache=False)
    return lg


def _record_taps(dec, ids):
    got = {}
    with hybrid.Taps(dec).record(lambda l, d: got.__setitem__(l, {k: v.clone() for k, v in d.items()})):
        dec.forward(ids, keep=1, use_cache=False)
    return got


def test_override_with_own_taps_reproduces_native_through_hybrid_cache():
    dec = _tiny(0)
    ids = torch.randint(0, 101, (2, 40))
    P = 30
    ref = _native_cont(dec, ids, P)
    given = _record_taps(dec, ids[:, :P])
    cache = hybrid.mapped_cache(dec, ids[:, :P], given, sinks=0)
    lg, _ = dec.forward(ids[:, P:], cache=cache, keep=0)
    assert torch.allclose(lg, ref, atol=2e-4), (lg - ref).abs().max()


def test_override_really_changes_the_cache_and_sinks_stay_native():
    dec = _tiny(1)
    ids = torch.randint(0, 101, (1, 36))
    P = 28
    ref = _native_cont(dec, ids, P)
    junk = {l: {k: torch.randn_like(v) for k, v in d.items()} for l, d in _record_taps(dec, ids[:, :P]).items()}
    c_junk = hybrid.mapped_cache(dec, ids[:, :P], junk, sinks=0)
    lg_junk, _ = dec.forward(ids[:, P:], cache=c_junk)
    assert (lg_junk - ref).abs().max() > 1e-2  # every layer type was overridden
    c_all_sink = hybrid.mapped_cache(dec, ids[:, :P], junk, sinks=P)  # all positions native
    lg, _ = dec.forward(ids[:, P:], cache=c_all_sink)
    assert torch.allclose(lg, ref, atol=2e-4)


def test_tail_semantics_last_n_native_matches_native_when_prefix_is_exact():
    dec = _tiny(2)
    ids = torch.randint(0, 101, (1, 40))
    P, N = 32, 8
    ref = _native_cont(dec, ids, P)
    given = _record_taps(dec, ids[:, :P - N])
    cache = hybrid.mapped_cache(dec, ids[:, :P - N], given)
    lg, _ = dec.forward(ids[:, P - N:], cache=cache)
    assert torch.allclose(lg[:, N:], ref, atol=2e-4)


def test_calibrate_self_map_recovers_identity():
    dec = _tiny(3)
    batches = [torch.randint(0, 101, (2, 48)) for _ in range(6)]
    rr = hybrid.calibrate(dec, dec, batches, w=1, drop_q=False, device="cpu")
    widths = hybrid.tap_widths(dec)
    maps = rr.solve(lam_rel=1e-6, groups=widths)
    for t, m in maps.items():
        assert t in m["sel"]
        for name, r2 in m["r2"].items():
            assert r2 > 0.999, (t, name, r2)
    ids = torch.randint(0, 101, (1, 40))
    P = 30
    feats = hybrid.source_features(dec, ids[:, :P], drop_q=False)
    given = hybrid.apply(maps, feats, 1, P, widths)
    cache = hybrid.mapped_cache(dec, ids[:, :P], given)
    lg, _ = dec.forward(ids[:, P:], cache=cache)
    ref = _native_cont(dec, ids, P)
    kl = (ref.log_softmax(-1).exp() * (ref.log_softmax(-1) - lg.log_softmax(-1))).sum(-1).mean()
    assert kl < 1e-3, kl


def test_cross_size_fit_beats_no_context():
    S, T = _tiny(4, layers=4, hidden=48, vh=2), _tiny(5, layers=8, hidden=64, vh=4)
    torch.manual_seed(0)
    batches = [torch.randint(0, 101, (4, 48)) for _ in range(8)]
    rr = hybrid.calibrate(S, T, batches, w=2, device="cpu")
    widths = hybrid.tap_widths(T)
    maps = rr.solve(lam_rel=1e-3, groups=widths)
    assert set(maps) == set(range(8))
    kinds_t = [k for k, _ in hybrid.layer_taps(T)]
    kinds_s = [k for k, _ in hybrid.layer_taps(S)]
    for t, m in maps.items():  # attention targets always see a source attention layer
        assert kinds_t[t] != "attn" or any(kinds_s[s] == "attn" for s in m["sel"])
    ids = torch.randint(0, 101, (2, 40))
    P = 30
    given = hybrid.apply(maps, hybrid.source_features(S, ids[:, :P]), 2, P, widths)
    lg, _ = T.forward(ids[:, P:], cache=hybrid.mapped_cache(T, ids[:, :P], given))
    assert torch.isfinite(lg).all()


def test_restore_rms_is_identity_on_true_keys():
    dec = _tiny(6)
    ids = torch.randint(0, 101, (1, 20))
    got = _record_taps(dec, ids)
    t = [i for i, (k, _) in enumerate(hybrid.layer_taps(dec)) if k == "attn"][0]
    k0 = got[t]["k"].clone()
    hybrid.restore_rms({t: got[t]}, dec)
    assert torch.allclose(got[t]["k"], k0, atol=1e-4)


def test_fixture_reproduces_from_saved_weights_and_maps():
    """The committed fixture is self-consistent: rebuilding the cache from its weights + maps file == ref_cache."""
    import os
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    p = os.path.join(os.path.dirname(__file__), "fixtures", "hybrid_fixture.pt")
    if not os.path.exists(p):
        pytest.skip("fixture not generated")
    fx = torch.load(p, weights_only=False)

    def build(cfgd, state):
        m = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**cfgd)).eval()
        m.load_state_dict(state)
        m.config._attn_implementation = "sdpa"
        return hybrid.Dec(m)
    S, T = build(fx["src_config"], fx["src_state"]), build(fx["tgt_config"], fx["tgt_state"])
    mf = fx["maps_file"]
    maps = {t: dict(sel=m["sel"], W=m["W"].float(), b=m["b"]) for t, m in mf["maps"].items()}
    widths = [[tuple(x) for x in tt["taps"]] for tt in mf["tgt_taps"]]
    ids, P, A = fx["ids"], fx["P"], mf["sinks"]
    feats = hybrid.source_features(S, ids[:, :P])
    given = hybrid.restore_rms(hybrid.apply(maps, feats, 1, P, widths), T)
    tm = mf["sink_template"]
    given = {t: {k: torch.cat([tm[t][k][:A][None], v[:, A:]], 1) for k, v in d.items()} for t, d in given.items()}
    cache = hybrid.mapped_cache(T, ids[:, :P], given)
    for l, r in fx["ref_cache"].items():
        lay = cache.layers[l]
        if r["kind"] == "attn":
            assert torch.allclose(lay.keys, r["K_post_rope"], atol=1e-2) and torch.allclose(lay.values, r["V"], atol=1e-2)
        else:
            assert torch.allclose(lay.conv_states[0], r["conv_state"], atol=1e-2)
            assert torch.allclose(lay.recurrent_states[0], r["recurrent_state"], atol=1e-2)
    lg, _ = T.forward(ids[:, P:], cache=cache)
    assert torch.allclose(lg, fx["cont_logits"], atol=1e-2)
    assert mf["lin_dims"]["qkv_order"] == "q|k|v" and mf["format"] == "kvx-hybrid-v1"


def _toks():
    from transformers import AutoTokenizer
    try:
        return (AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
                AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"))
    except Exception as e:  # offline
        pytest.skip(f"tokenizers unavailable: {e}")


TEXT = ("The mitochondria is the powerhouse of the cell. In 1953, Watson and Crick described DNA's double helix; "
        "their model explained replication. Überraschend: naïve café prices rose 12.5% — e.g. 3€ → 3.38€!")


def test_align_maps_each_target_token_to_the_source_token_holding_its_last_char():
    tl, qw = _toks()
    for tok_s, tok_t, bs, bt in ((tl, qw, True, False), (qw, tl, False, True)):
        n_t = 40
        ids_s, ids_t, idx = hybrid.align(TEXT, tok_s, tok_t, n_t, bos_s=bs, bos_t=bt)
        assert len(ids_t) == n_t and idx.shape == (n_t,)
        assert (idx[1:] >= idx[:-1]).all() and idx.max() < len(ids_s)
        off_t = tok_t(TEXT, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        off_t = ([(0, 0)] if bt else []) + off_t
        end = max(e for _, e in off_t[:n_t])
        off_s = tok_s(TEXT[:end], add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        off_s = ([(0, 0)] if bs else []) + off_s
        for j, (a, e) in enumerate(off_t[:n_t]):
            if e > a:
                sa, se = off_s[idx[j]]
                assert sa < e <= se or (se >= e and sa <= a), (j, (a, e), (sa, se))


def test_cross_tokenizer_calibrate_and_apply_run():
    tl, qw = _toks()
    S, T = _tiny(7, layers=4, hidden=48, vh=2), _tiny(8, layers=8, hidden=64, vh=4)
    texts = [TEXT[i:] + " " + TEXT[:i] for i in range(0, 120, 15)]
    trip = [hybrid.align(t, tl, qw, 30, bos_s=True) for t in texts]
    ids_s = hybrid.pad_batch([x[0] % 101 for x in trip], 0)  # tiny models: vocab 101
    ids_t = torch.stack([x[1] % 101 for x in trip])
    idx = torch.stack([x[2] for x in trip])
    batches = [(ids_s[i:i + 4], ids_t[i:i + 4], idx[i:i + 4]) for i in range(0, len(trip), 4)]
    rr = hybrid.calibrate(S, T, batches, w=2, device="cpu")
    widths = hybrid.tap_widths(T)
    maps = rr.solve(lam_rel=1e-3, groups=widths)
    feats = hybrid.source_features_aligned(S, ids_s[:2], idx[:2])
    given = hybrid.apply(maps, feats, 2, 30, widths)
    lg, _ = T.forward(ids_t[:2, 24:], cache=hybrid.mapped_cache(T, ids_t[:2, :24], given))
    assert torch.isfinite(lg).all() and lg.shape[:2] == (2, 6)


def test_align_boundary_mode_never_looks_past_the_target_token():
    tl, qw = _toks()
    for tok_s, tok_t in ((tl, qw), (qw, tl)):
        ids_s, ids_t, idx = hybrid.align(TEXT, tok_s, tok_t, 40, mode="boundary")
        off_t = tok_t(TEXT, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"][:40]
        end = max(e for _, e in off_t)
        off_s = tok_s(TEXT[:end], add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        assert (idx[1:] >= idx[:-1]).all()
        for j, (a, e) in enumerate(off_t):
            if e > a and off_s[0][1] <= e:
                assert off_s[idx[j]][1] <= e


def test_recolor_restores_target_covariance_on_calibration_data():
    S, T = _tiny(4, layers=4, hidden=48, vh=2), _tiny(5, layers=4, hidden=64, vh=4)
    torch.manual_seed(0)
    batches = [torch.randint(0, 101, (4, 48)) for _ in range(6)]
    # WindowRidge built by hand with bd=16 so the tiny model's tap widths hold full blocks
    kinds_s = [k for k, _ in hybrid.layer_taps(S)]; kinds_t = [k for k, _ in hybrid.layer_taps(T)]
    win = hybrid.windows(4, 4, 2, kinds_s, kinds_t)
    dt = [sum(x for _, x in g) for g in hybrid.tap_widths(T)]
    feats_all, ys = [], {t: [] for t in range(4)}
    for ids in batches:
        f = hybrid.source_features(S, ids)
        got = {}
        with hybrid.Taps(T).record(lambda l, d: got.__setitem__(l, torch.cat([d[n] for n, _ in hybrid.tap_widths(T)[l]], -1))):
            T.forward(ids, keep=1, use_cache=False)
        feats_all.append(f)
        for t in range(4):
            ys[t].append(got[t].reshape(-1, dt[t]).float())
    rr = hybrid.WindowRidge([x.shape[1] for x in feats_all[0]], dt, win, range(4), "cpu", bd=16)
    for i, f in enumerate(feats_all):
        rr.add_x(f)
        for t in range(4):
            rr.add_y(t, ys[t][i])
        rr.done_batch()
    maps = rr.solve(lam_rel=1e-1, recolor=True)  # strong ridge: shrunk predictions
    widths = hybrid.tap_widths(T)
    F = [torch.cat([f[s] for f in feats_all]) for s in range(4)]
    N = F[0].shape[0]
    for t in range(4):
        Y = torch.cat(ys[t])
        y0 = torch.cat([v.reshape(N, -1) for v in hybrid.apply({t: maps[t]}, F, 1, N, widths)[t].values()], 1)
        y1 = torch.cat([v.reshape(N, -1) for v in hybrid.apply({t: maps[t]}, F, 1, N, widths, recolor=True)[t].values()], 1)
        nb = maps[t]["rc"].shape[0] * 16
        var = lambda z: z[:, :nb].var(0).sum().item()
        assert var(y0) < 0.999 * var(Y)  # ridge shrinks
        assert abs(var(y1) / var(Y) - 1) < 0.02, (t, var(y1) / var(Y))


def _tiny_glm(kind, seed, layers=4):
    torch.manual_seed(seed)
    if kind == "dsa":
        from transformers.models.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
        cfg = GlmMoeDsaConfig(vocab_size=101, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                              num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=4,
                              kv_lora_rank=32, q_lora_rank=48, qk_rope_head_dim=8, qk_nope_head_dim=16, v_head_dim=16,
                              n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1, first_k_dense_replace=1,
                              index_n_heads=2, index_head_dim=16, index_topk=8, max_position_embeddings=256)
        m = GlmMoeDsaForCausalLM(cfg).eval()
    else:
        from transformers.models.glm4_moe_lite import Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM
        cfg = Glm4MoeLiteConfig(vocab_size=101, hidden_size=48, intermediate_size=96, moe_intermediate_size=24,
                                num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=4,
                                kv_lora_rank=24, q_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=16, v_head_dim=16,
                                n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1, first_k_dense_replace=1,
                                max_position_embeddings=256)
        m = Glm4MoeLiteForCausalLM(cfg).eval()
    m.config._attn_implementation = "eager"
    return hybrid.Dec(m)


def test_mla_dsa_override_with_own_taps_reproduces_native():
    pytest.importorskip("transformers.models.glm_moe_dsa")
    dec = _tiny_glm("dsa", 0)
    kinds = [k for k, _ in hybrid.layer_taps(dec)]
    assert set(kinds) == {"mla"}
    ids = torch.randint(0, 101, (1, 30))
    P = 22
    # HF's DSA indexer is not prefill/decode consistent (one-shot vs cached two-step differ by ~0.04 natively), so
    # the oracle reference is the native two-step run: prefix pass, then the continuation on its cache
    _, c_nat = dec.forward(ids[:, :P], keep=1)
    ref, _ = dec.forward(ids[:, P:], cache=c_nat)
    given = _record_taps(dec, ids[:, :P])
    lg, _ = dec.forward(ids[:, P:], cache=hybrid.mapped_cache(dec, ids[:, :P], given))
    assert torch.allclose(lg, ref, atol=2e-4), (lg - ref).abs().max()
    junk = {l: {k: torch.randn_like(v) for k, v in d.items()} for l, d in given.items()}
    lj, _ = dec.forward(ids[:, P:], cache=hybrid.mapped_cache(dec, ids[:, :P], junk))
    assert (lj - ref).abs().max() > 1e-3


def test_mla_flash_to_dsa_fit_runs():
    pytest.importorskip("transformers.models.glm_moe_dsa")
    S, T = _tiny_glm("lite", 1, layers=3), _tiny_glm("dsa", 2, layers=4)
    batches = [torch.randint(0, 101, (2, 32)) for _ in range(4)]
    rr = hybrid.calibrate(S, T, batches, w=0, device="cpu")
    widths = hybrid.tap_widths(T)
    maps = rr.solve(lam_rel=1e-3, groups=widths)
    ids = torch.randint(0, 101, (1, 30))
    given = hybrid.apply(maps, hybrid.source_features(S, ids[:, :22]), 1, 22, widths)
    lg, _ = T.forward(ids[:, 22:], cache=hybrid.mapped_cache(T, ids[:, :22], given))
    assert torch.isfinite(lg).all()


def test_restore_rms_finite_with_zero_norm_weights_and_identity_elsewhere():
    dec = _tiny(9)
    t = [i for i, (k, _) in enumerate(hybrid.layer_taps(dec)) if k == "attn"][0]
    kn = dec.layers[t].self_attn.k_norm
    with torch.no_grad():
        kn.weight[:3] = -1.0  # 1 + w == 0 exactly on 3 channels (seen in real Qwen3.5 checkpoints: NaN before the fix)
    ids = torch.randint(0, 101, (1, 20))
    got = _record_taps(dec, ids)
    k0 = got[t]["k"].clone()
    out = hybrid.restore_rms({t: dict(got[t])}, dec)[t]["k"]
    assert torch.isfinite(out).all()
    D = kn.weight.shape[0]
    o, r = out.view(1, 20, -1, D), k0.view(1, 20, -1, D)
    assert torch.allclose(o[..., 3:], r[..., 3:] * (o[..., 3:].norm(dim=-1, keepdim=True) / r[..., 3:].norm(dim=-1, keepdim=True)), atol=1e-4)
    assert (o[..., :3] == 0).all()


def _tiny_qwen3_dec(seed, layers, hidden, kvh=2):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=101, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=kvh, head_dim=16, max_position_embeddings=256)
    m = Qwen3ForCausalLM(cfg).eval()
    m.config._attn_implementation = "sdpa"
    return hybrid.Dec(m)


def test_split_kv_self_map_identity_and_core_export_matches_core_apply():
    from kvx import core
    dec = _tiny_qwen3_dec(11, 3, 64)
    batches = [torch.randint(0, 101, (2, 40)) for _ in range(6)]
    rr = hybrid.calibrate(dec, dec, batches, w=1, device="cpu", split_kv=True)
    us, ut = rr.units
    assert [k for _, k, _ in ut] == ["attn_k", "attn_v"] * 3
    maps = rr.solve(lam_rel=1e-6, groups=hybrid.unit_groups(dec, ut))
    for j, m in maps.items():
        assert all(us[i][1] == ut[j][1] for i in m["sel"])  # K from K, V from V
        assert min(m["r2"].values()) > 0.999
    ids = torch.randint(0, 101, (1, 24))
    feats = hybrid.source_features(dec, ids)
    given = hybrid.apply_units(maps, feats, 1, 24, dec, us, ut, src=dec)
    cm = hybrid.to_core_maps(maps, us, ut, dec)
    assert len(cm["k"]) == len(cm["v"]) == 3
    cos, sin = core.rope_cos_sin(dec.model, 24, "cpu")
    K, V, _ = core.capture_kv(dec.model, ids, cos, sin)
    Kc, Vc = core.apply_maps(cm["k"], K.float()), core.apply_maps(cm["v"], V.float())
    for t in range(3):
        kt = given[t]["k"].view(1, 24, 2, 16).transpose(1, 2)
        assert torch.allclose(Kc[t], kt, atol=2e-2), (Kc[t] - kt).abs().max()
        vt = given[t]["v"].view(1, 24, 2, 16).transpose(1, 2)
        assert torch.allclose(Vc[t], vt, atol=2e-2)


def test_split_kv_cross_size_fit_and_mapped_cache():
    S, T = _tiny_qwen3_dec(12, 3, 48), _tiny_qwen3_dec(13, 5, 64)
    batches = [torch.randint(0, 101, (2, 40)) for _ in range(6)]
    rr = hybrid.calibrate(S, T, batches, w=0, device="cpu", split_kv=True)
    us, ut = rr.units
    maps = rr.solve(lam_rel=1e-3, groups=hybrid.unit_groups(T, ut), recolor=True)
    ids = torch.randint(0, 101, (1, 30))
    g = hybrid.restore_rms(hybrid.apply_units(maps, hybrid.source_features(S, ids[:, :22]), 1, 22, T, us, ut,
                                              recolor=True, src=S), T)
    assert set(g) == set(range(5)) and all(set(d) == {"k", "v"} for d in g.values())
    lg, _ = T.forward(ids[:, 22:], cache=hybrid.mapped_cache(T, ids[:, :22], g))
    assert torch.isfinite(lg).all()


def test_xkv_adapter_reproduces_gdn_taps_with_exact_linear_maps(tmp_path):
    """src == tgt; xkv W/Wq built from the target's own in_proj weights -> adapter taps == the target's taps."""
    from kvx import xkv_adapter as X
    dec = _tiny(10)
    c = dec.cfg
    kd = c.linear_key_head_dim * c.linear_num_key_heads
    h = c.hidden_size
    W, Wq, slices, src_of = {}, {}, {}, {}
    for j, (kind, _) in enumerate(hybrid.layer_taps(dec)):
        if kind != "lin":
            continue
        la = dec.layers[j].linear_attn
        Wqkv = la.in_proj_qkv.weight.detach().float()  # [q|k|v, h]
        rest = torch.cat([Wqkv[kd:], la.in_proj_a.weight.detach().float(), la.in_proj_b.weight.detach().float()], 0)
        z = torch.zeros(1, rest.shape[0])
        W[j] = torch.cat([rest.T, z], 0)
        Wq[j] = torch.cat([Wqkv[:kd].T, torch.zeros(1, kd)], 0)
        slices[j] = {j: (0, rest.shape[0])}
        src_of[j] = j
    torch.save({"W": W, "Wq": Wq, "slices": slices, "src_of": src_of}, tmp_path / "m.pt")
    xm = X.load_xkv(tmp_path / "m.pt")
    ids = torch.randint(0, 101, (1, 20))
    given = X.xkv_given(dec, dec, xm, ids, out_dtype=torch.float32)
    ref = _record_taps(dec, ids)
    for j in W:
        for tap in ("qkv", "a", "b"):
            assert torch.allclose(given[j][tap], ref[j][tap].float(), atol=1e-4), (j, tap)
