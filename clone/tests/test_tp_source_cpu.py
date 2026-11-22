"""Tensor-parallel source prefill (kvx/tp_source.py) on CPU with gloo, 2 ranks:
(1) the sharded source gives the same logits and (all-gathered) KV as the whole model, for Hkv % n == 0 (Qwen3 tiny)
and n % Hkv == 0 (KV heads replicated across ranks, the Qwen2.5-3B case);
(2) the connector with src_tp=1 under TP=2 maps exactly what TP=1 maps (rank outputs joined along heads), kv and kvcat.
Run: pytest -q tests/test_tp_source_cpu.py"""
import os
os.environ.setdefault("TRITON_INTERPRET", "1")  # before triton is imported
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kvx import core


def _cfg(kind, hkv):
    from transformers import Qwen2Config, Qwen3Config
    kw = dict(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
              num_key_value_heads=hkv, max_position_embeddings=256)
    return Qwen3Config(head_dim=16, **kw) if kind == "qwen3" else Qwen2Config(**kw)


def _model(cfg):
    from transformers import AutoModelForCausalLM
    torch.manual_seed(1)
    m = AutoModelForCausalLM.from_config(cfg).eval()
    if hasattr(m.model.layers[0].self_attn.q_proj, "bias") and m.model.layers[0].self_attn.q_proj.bias is not None:
        with torch.no_grad():
            for l in m.model.layers:  # random biases so the bias slicing is exercised (init is zero)
                for p in (l.self_attn.q_proj, l.self_attn.k_proj, l.self_attn.v_proj):
                    p.bias.normal_()
    m.config._attn_implementation = "sdpa"
    return m


def _port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _worker(rank, world, port, kind, hkv, d, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)
    from kvx.tp_source import shard_source, gather_heads
    m = _model(_cfg(kind, hkv))
    ids = torch.randint(0, 97, (1, 21))
    shard_source(m, rank, world, None)
    with torch.no_grad():
        o = m(input_ids=ids, use_cache=True)
    tp = m._kvx_tp
    ks = [gather_heads(k[0], world, None, tp["rep"], 0) for k, _ in core.cache_layers(o.past_key_values)]
    vs = [gather_heads(v[0], world, None, tp["rep"], 0) for _, v in core.cache_layers(o.past_key_values)]
    if rank == 0:
        torch.save({"logits": o.logits, "k": ks, "v": vs, "ids": ids}, os.path.join(d, out))
    dist.destroy_process_group()


@pytest.mark.parametrize("kind,hkv", [("qwen3", 2), ("qwen2", 1)])
def test_sharded_source_matches_whole(tmp_path, kind, hkv):
    mp.spawn(_worker, args=(2, _port(), kind, hkv, str(tmp_path), "r.pt"), nprocs=2, join=True)
    r = torch.load(tmp_path / "r.pt")
    m = _model(_cfg(kind, hkv))
    with torch.no_grad():
        o = m(input_ids=r["ids"], use_cache=True)
    assert torch.allclose(r["logits"], o.logits, atol=1e-4), (r["logits"] - o.logits).abs().max()
    for (k, v), k2, v2 in zip(core.cache_layers(o.past_key_values), r["k"], r["v"]):
        assert k2.shape == k[0].shape
        assert torch.allclose(k2, k[0], atol=1e-5) and torch.allclose(v2, v[0], atol=1e-5)


def _conn(vc, d, maps, tp, src_tp):
    c = vc.CrossModelConnector.__new__(vc.CrossModelConnector)
    c.src_name, c.tgt_name, c.maps_path = d, d, maps
    c.restore, c.src_device, c.fp8, c.sink_path, c.chunk, c.max_len = "rms", None, False, None, 2, 64
    c.yarn, c.orig_ctx, c.fp8_source, c.src_sdpa, c.tp_override = 0.0, 32768, False, "cudnn", tp
    c.src_tp = src_tp
    return c


def _conn_worker(rank, world, port, d, maps, ids, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), TRITON_INTERPRET="1")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from tests.test_core_cpu import _import_connector_with_stub_vllm
    vc = _import_connector_with_stub_vllm()
    c = _conn(vc, d, maps, (rank, world), True)
    c.tp_group_override = dist.group.WORLD
    c._init_worker(torch.device("cpu"))
    assert c.src_tp_on and c.S.model.layers[0].self_attn.q_proj.out_features == 2 * 16
    res = {l: (K.clone(), V.clone()) for l, K, V in c._map_layers(ids)}
    torch.save(res, os.path.join(out, f"rank{rank}.pt"))
    dist.destroy_process_group()


@pytest.mark.parametrize("feat", ["kv", "kvcat"])
def test_connector_src_tp2_equals_tp1(tmp_path, feat):
    from tests.test_core_cpu import _import_connector_with_stub_vllm
    vc = _import_connector_with_stub_vllm()
    m = _model(_cfg("qwen3", 2))
    d = str(tmp_path / "tiny")
    m.save_pretrained(d, safe_serialization=True)
    Ls, H, D = 3, 2, 16
    ds = H * D * (2 if feat == "kvcat" else 1)
    g = torch.Generator().manual_seed(9)
    mk = lambda: [dict(sel=[0, 1, 2], W=torch.randn(3 * ds, H * D, generator=g) * 0.1, b=torch.randn(H * D, generator=g),
                       mu=torch.zeros(H * D), r2=torch.ones(H), vnat=torch.ones(H), H=H, D=D) for _ in range(Ls)]
    maps = str(tmp_path / "maps.pt")
    torch.save({"k": mk(), "v": mk(), "feat": feat}, maps)
    os.environ.setdefault("TRITON_INTERPRET", "1")
    ids = torch.randint(0, 97, (24,)).tolist()
    c = _conn(vc, d, maps, (0, 1), False)
    c._init_worker(torch.device("cpu"))
    full = {l: (K.clone(), V.clone()) for l, K, V in c._map_layers(ids)}
    mp.spawn(_conn_worker, args=(2, _port(), d, maps, ids, str(tmp_path)), nprocs=2, join=True)
    r0, r1 = torch.load(tmp_path / "rank0.pt"), torch.load(tmp_path / "rank1.pt")
    for l in range(Ls):
        K = torch.cat([r0[l][0], r1[l][0]], 1)
        V = torch.cat([r0[l][1], r1[l][1]], 1)
        # the connector runs the source in bf16: the all-reduce sums partial products in another order -> bf16 rounding
        for X, Y in ((K, full[l][0]), (V, full[l][1])):
            rel = ((X.float() - Y.float()).norm() / Y.float().norm()).item()
            assert X.shape == Y.shape and rel < 1e-2, rel


def _scaled_mm_emu(a, b, scale_a, scale_b, out_dtype, bias=None):  # CPU torch has per-tensor _scaled_mm only
    y = (a.float() * scale_a) @ (b.float() * scale_b)
    return (y if bias is None else y + bias.float()).to(out_dtype)


@pytest.mark.parametrize("feat", ["kv", "kvcat"])
def test_connector_fp8_chunked_quant_close_to_bf16(tmp_path, feat, monkeypatch):
    """fp8 mapper path (row-chunked centered e4m3 quantization of the mapper input) stays within e4m3 error of bf16."""
    from tests.test_core_cpu import _import_connector_with_stub_vllm
    vc = _import_connector_with_stub_vllm()
    m = _model(_cfg("qwen3", 2))
    d = str(tmp_path / "tiny")
    m.save_pretrained(d, safe_serialization=True)
    Ls, H, D = 3, 2, 16
    ds = H * D * (2 if feat == "kvcat" else 1)
    g = torch.Generator().manual_seed(3)
    mk = lambda: [dict(sel=[0, 1, 2], W=torch.randn(3 * ds, H * D, generator=g) * 0.1, b=torch.randn(H * D, generator=g),
                       mu=torch.zeros(H * D), r2=torch.ones(H), vnat=torch.ones(H), H=H, D=D) for _ in range(Ls)]
    maps = str(tmp_path / "maps.pt")
    torch.save({"k": mk(), "v": mk(), "feat": feat}, maps)
    ids = torch.randint(0, 97, (40,)).tolist()
    ref = _conn(vc, d, maps, (0, 1), False)
    ref._init_worker(torch.device("cpu"))
    full = {l: (K.float().clone(), V.float().clone()) for l, K, V in ref._map_layers(ids)}
    monkeypatch.setattr(torch, "_scaled_mm", _scaled_mm_emu)
    c = _conn(vc, d, maps, (0, 1), False)
    c.fp8 = True
    c._init_worker(torch.device("cpu"))
    for l, K, V in c._map_layers(ids):
        for X, Y in ((K, full[l][0]), (V, full[l][1])):
            rel = ((X.float() - Y).norm() / Y.norm()).item()
            assert rel < 0.06, (l, rel)


def _tiny_maps(tmp_path, feat, seed=5, sink=False):
    m = _model(_cfg("qwen3", 2))
    d = str(tmp_path / "tiny")
    m.save_pretrained(d, safe_serialization=True)
    Ls, H, D = 3, 2, 16
    ds = H * D * (2 if feat == "kvcat" else 1)
    g = torch.Generator().manual_seed(seed)
    mk = lambda: [dict(sel=[0, 1, 2], W=torch.randn(3 * ds, H * D, generator=g) * 0.1, b=torch.randn(H * D, generator=g),
                       mu=torch.zeros(H * D), r2=torch.ones(H), vnat=torch.ones(H), H=H, D=D) for _ in range(Ls)]
    maps = str(tmp_path / "maps.pt")
    torch.save({"k": mk(), "v": mk(), "feat": feat}, maps)
    sp = None
    if sink:
        sp = str(tmp_path / "sink.pt")
        torch.save({"K": torch.randn(Ls, H, 4, D, generator=g), "V": torch.randn(Ls, H, 4, D, generator=g)}, sp)
    return d, maps, sp


@pytest.mark.parametrize("feat,fp8", [("kv", False), ("kvcat", False), ("kv", True)])
def test_token_chunked_map_equals_whole(tmp_path, feat, fp8, monkeypatch):
    """map_chunk_tokens: chunks of 16 rows (40 tokens -> 16+16+8) give the same K/V as one pass, incl. the sink rows."""
    from tests.test_core_cpu import _import_connector_with_stub_vllm
    vc = _import_connector_with_stub_vllm()
    d, maps, sp = _tiny_maps(tmp_path, feat, sink=True)
    if fp8:
        monkeypatch.setattr(torch, "_scaled_mm", _scaled_mm_emu)
    ids = torch.randint(0, 97, (40,)).tolist()
    outs = []
    for tc, f8 in ((0, False), (0, fp8), (16, fp8)):
        c = _conn(vc, d, maps, (0, 1), False)
        c.fp8, c.sink_path, c.block, c.tok_chunk = f8, sp, 16, tc
        c._init_worker(torch.device("cpu"))
        outs.append({l: (K.float(), V.float()) for l, K, V in c._map_layers(ids)})
    for l in outs[0]:
        for i in (1, 2):
            for X, Y in zip(outs[i][l], outs[0][l]):
                assert X.shape == Y.shape == (40, 2, 16)
                # fp8: the chunked pass centers on the first chunk's mean (exact math, other rounding): both passes
                # stay within e4m3 error of bf16
                assert ((X - Y).norm() / Y.norm()).item() < (0.06 if fp8 else 1e-6)


@pytest.mark.parametrize("tok_chunk,start", [(0, 0), (16, 0), (16, 16), (32, 16)])
def test_start_load_kv_chunked_writes_paged_cache(tmp_path, tok_chunk, start):
    """start_load_kv end to end on CPU (stub forward context, token-major FA cache): token chunks and a prefix-cache
    hit write exactly the reference rows; blocks of the local hit stay untouched."""
    from types import SimpleNamespace as NS
    from tests.test_core_cpu import _import_connector_with_stub_vllm
    vc = _import_connector_with_stub_vllm()
    d, maps, _ = _tiny_maps(tmp_path, "kv")
    ids = torch.randint(0, 97, (48,)).tolist()
    ref = _conn(vc, d, maps, (0, 1), False)
    ref._init_worker(torch.device("cpu"))
    full = {l: (K.clone(), V.clone()) for l, K, V in ref._map_layers(ids)}
    c = _conn(vc, d, maps, (0, 1), False)
    c.block, c.tok_chunk, c.stats, c._layout, c._backend, c._worker_ready = 16, tok_chunk, dict(requests=0, mapped_tokens=0), None, "", False
    nb, H, D = 6, 2, 16
    caches = {l: torch.full((nb, 16, H, 2 * D), 7.0).permute(0, 2, 1, 3) for l in range(3)}
    be = NS(get_name=lambda: "FLASH_ATTN")
    fc = NS(no_compile_layers={f"model.layers.{l}.self_attn.attn": NS(kv_cache=[caches[l]], attn_backend=be) for l in range(3)})
    blocks = torch.tensor([4, 1, 3])
    slots = (torch.arange(16)[None] + blocks[:, None] * 16).flatten()
    meta = vc.CrossModelMeta(requests=[vc._Req(token_ids=ids, slot_mapping=slots, start=start)])
    c._get_connector_metadata = lambda: meta
    c.start_load_kv(fc)
    for l in range(3):
        phys = caches[l].permute(0, 2, 1, 3)  # [blocks, block, H, 2D]
        for t in range(48):
            b, o = blocks[t // 16].item(), t % 16
            if t < start:
                assert (phys[b, o] == 7.0).all()
            else:
                assert torch.allclose(phys[b, o, :, :D].float(), full[l][0][t].float(), atol=1e-2)
                assert torch.allclose(phys[b, o, :, D:].float(), full[l][1][t].float(), atol=1e-2)


def _fa_ref(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal,
            q_descale=None, k_descale=None, v_descale=None, window_size=None, **kw):
    """CPU stand-in for vLLM's flash_attn_varlen_func: one sequence, GQA, causal aligned bottom-right, fp8 descale."""
    def deq(x, s):
        if s is None:
            return x.float()
        Hx = x.shape[1]
        return x.float() * s.view(-1).repeat_interleave(Hx // s.numel()).view(1, Hx, 1)
    qf, kf, vf = deq(q, q_descale), deq(k, k_descale), deq(v, v_descale)
    T, Hq, D = qf.shape
    S, Hk = kf.shape[0], kf.shape[1]
    kf = kf.repeat_interleave(Hq // Hk, 1)
    vf = vf.repeat_interleave(Hq // Hk, 1)
    sc = torch.einsum("thd,shd->hts", qf, kf) * softmax_scale
    qpos = torch.arange(T)[:, None] + S - T
    mask = torch.arange(S)[None, :] <= qpos
    if window_size is not None and window_size[0] >= 0:
        mask &= torch.arange(S)[None, :] >= qpos - window_size[0]
    sc = sc.masked_fill(~mask, float("-inf"))
    return torch.einsum("hts,shd->thd", sc.softmax(-1), vf).to(torch.bfloat16)


@pytest.mark.parametrize("fp8", [False, True])
def test_vllm_fa_chunked_source_prefill_matches_sdpa(tmp_path, fp8, monkeypatch):
    """Connector source prefill with src_sdpa=vllm_fa and src_chunk=16 (3 chunks for 40 tokens) == one SDPA forward
    (mapped K/V); fp8 attention within e4m3 error."""
    from tests.test_core_cpu import _import_connector_with_stub_vllm
    from kvx import fa
    vc = _import_connector_with_stub_vllm()
    monkeypatch.setattr(fa, "_fa_func", lambda: (_fa_ref, None))
    d, maps, _ = _tiny_maps(tmp_path, "kv")
    ids = torch.randint(0, 97, (40,)).tolist()
    ref = _conn(vc, d, maps, (0, 1), False)
    ref._init_worker(torch.device("cpu"))
    full = {l: (K.float(), V.float()) for l, K, V in ref._map_layers(ids)}
    c = _conn(vc, d, maps, (0, 1), False)
    c.src_sdpa, c.src_chunk, c.src_fp8_attn = "vllm_fa", 16, fp8
    c._init_worker(torch.device("cpu"))
    for l, K, V in c._map_layers(ids):
        for X, Y in ((K, full[l][0]), (V, full[l][1])):
            rel = ((X.float() - Y).norm() / Y.norm()).item()
            assert rel < (0.08 if fp8 else 0.02), (l, rel)
    fa.FP8 = False


@pytest.mark.parametrize("fp8,tp,feat", [(False, 1, "kv"), (True, 1, "kv"), (False, 2, "kv"), (True, 2, "kvcat")])
def test_sparse_and_group_mapper_equal_dense(tmp_path, fp8, tp, feat, monkeypatch):
    """mapper=sparse (per target layer: gather its k selected source blocks, one GEMM) == the dense GEMM with zero
    blocks, for bf16 and e4m3 weights, under the TP column split, kv and kvcat inputs; also token chunks."""
    from tests.test_core_cpu import _import_connector_with_stub_vllm
    vc = _import_connector_with_stub_vllm()
    m = _model(_cfg("qwen3", 2))
    d = str(tmp_path / "tiny")
    m.save_pretrained(d, safe_serialization=True)
    Ls, H, D = 3, 2, 16
    ds = H * D * (2 if feat == "kvcat" else 1)
    g = torch.Generator().manual_seed(11)
    sels = [[2, 0], [1], [0, 1]]  # unsorted on purpose: rows follow the map's own sel order
    mk = lambda: [dict(sel=sels[l], W=torch.randn(len(sels[l]) * ds, H * D, generator=g) * 0.1, b=torch.randn(H * D, generator=g),
                       mu=torch.zeros(H * D), r2=torch.ones(H), vnat=torch.ones(H), H=H, D=D) for l in range(Ls)]
    maps = str(tmp_path / "maps.pt")
    torch.save({"k": mk(), "v": mk(), "feat": feat}, maps)
    if fp8:
        monkeypatch.setattr(torch, "_scaled_mm", _scaled_mm_emu)
    ids = torch.randint(0, 97, (40,)).tolist()
    for rank in range(tp):
        res = []
        for mapper, tc in (("dense", 0), ("sparse", 0), ("sparse", 16), ("group", 0), ("group", 16)):
            c = _conn(vc, d, maps, (rank, tp), False)
            c.fp8, c.mapper, c.block, c.tok_chunk, c.group = fp8, mapper, 16, tc, 2
            c._init_worker(torch.device("cpu"))
            assert (c.sp is not None) == (mapper != "dense")
            res.append({l: (K.float(), V.float()) for l, K, V in c._map_layers(ids)})
        for i in (1, 2, 3, 4):
            for l in range(Ls):
                for X, Y in zip(res[i][l], res[0][l]):
                    rel = ((X - Y).norm() / Y.norm()).item()
                    assert X.shape == Y.shape and rel < (0.06 if (fp8 and i in (2, 4)) else 1e-2), (i, l, rel)


@pytest.mark.parametrize("kvcat", [False, True])
def test_build_x8_matches_torch_build_and_quant(kvcat):
    """Fused Triton mapper-input build (unrope with a YaRN-like attention factor, layout, centering, row amax) ==
    the torch path (core.unrope + cat + quant8 math). f32 mode: the interpreter's e4m3 cast is wrong near 128."""
    from kvx import triton_kernels as tk
    g = torch.Generator().manual_seed(0)
    Hs, n, D, Ls = 2, 13, 16, 3
    cos = torch.rand(n, D, generator=g) + 0.5
    sin = torch.rand(n, D, generator=g)
    cos[:, D // 2:], sin[:, D // 2:] = cos[:, :D // 2], sin[:, :D // 2]
    ks = [torch.randn(1, Hs, n + 3, D, generator=g)[:, :, :n] for _ in range(Ls)]  # token-strided cache views
    vs = [torch.randn(1, Hs, n, D, generator=g) for _ in range(Ls)]
    if kvcat:
        X = torch.cat([torch.cat([core.unrope(k, cos, sin)[0], v[0]], 0).transpose(0, 1).reshape(n, -1) for k, v in zip(ks, vs)], 1)
        segs = [s for l, (k, v) in enumerate(zip(ks, vs)) for s in ((k[0], l * 2 * Hs * D, True), (v[0], (2 * l + 1) * Hs * D, False))]
    else:
        X = torch.cat([core.unrope(k, cos, sin)[0].transpose(0, 1).reshape(n, -1) for k in ks], 1)
        segs = [(k[0], l * Hs * D, True) for l, k in enumerate(ks)]
    mu = X[:4].mean(0)
    Y, sx = tk.build_x8(segs, cos, sin, mu, n, f32=True)
    s = (X - mu).abs().amax(1, keepdim=True) / 448
    assert torch.allclose(sx, s, rtol=1e-5)
    assert torch.allclose(Y, (X - mu) / s, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("feat,tc", [("kv", 0), ("kv", 16), ("kvcat", 16)])
def test_connector_fused_build_close_to_bf16(tmp_path, feat, tc, monkeypatch):
    """fp8 connector with the fused Triton build (CPU: f32 mode, interpreter) stays within e4m3 error of bf16, incl. chunks."""
    from tests.test_core_cpu import _import_connector_with_stub_vllm
    vc = _import_connector_with_stub_vllm()
    m = _model(_cfg("qwen3", 2))
    d = str(tmp_path / "tiny")
    m.save_pretrained(d, safe_serialization=True)
    Ls, H, D = 3, 2, 16
    ds = H * D * (2 if feat == "kvcat" else 1)
    g = torch.Generator().manual_seed(4)
    mk = lambda: [dict(sel=[0, 1, 2], W=torch.randn(3 * ds, H * D, generator=g) * 0.1, b=torch.randn(H * D, generator=g),
                       mu=torch.zeros(H * D), r2=torch.ones(H), vnat=torch.ones(H), H=H, D=D) for _ in range(Ls)]
    maps = str(tmp_path / "maps.pt")
    torch.save({"k": mk(), "v": mk(), "feat": feat}, maps)
    ids = torch.randint(0, 97, (40,)).tolist()
    ref = _conn(vc, d, maps, (0, 1), False)
    ref._init_worker(torch.device("cpu"))
    full = {l: (K.float(), V.float()) for l, K, V in ref._map_layers(ids)}
    monkeypatch.setattr(torch, "_scaled_mm", _scaled_mm_emu)
    c = _conn(vc, d, maps, (0, 1), False)
    c.fp8, c._fused_cpu, c.block, c.tok_chunk = True, True, 16, tc
    c._init_worker(torch.device("cpu"))
    for l, K, V in c._map_layers(ids):
        for X, Y in ((K, full[l][0]), (V, full[l][1])):
            rel = ((X.float() - Y).norm() / Y.norm()).item()
            assert rel < 0.04, (l, rel)  # f32 mode skips e4m3 rounding of the input: weight e4m3 error ~2-3%


def _hf_local(name):
    import glob as _g
    p = _g.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{name.replace('/', '--')}/snapshots/*/tokenizer*"))
    return os.path.dirname(p[0]) if p else None


@pytest.mark.parametrize("cross", [False, True])
def test_connector_cross_tokenizer_path(tmp_path, cross):
    """maps with 'align' (research/xfit.py): the source runs on its own tokens and target position j takes the aligned
    source token. Same tokenizer on both sides -> identity alignment -> exactly the plain path. Llama source tokenizer
    -> runs, one K/V row per target token."""
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM
    from tests.test_core_cpu import _import_connector_with_stub_vllm
    qd, ld = _hf_local("Qwen/Qwen3-0.6B"), _hf_local("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if qd is None or (cross and ld is None):
        pytest.skip("tokenizers not cached")
    vc = _import_connector_with_stub_vllm()
    tq = AutoTokenizer.from_pretrained(qd)
    torch.manual_seed(1)
    tgt = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tq), hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                                       num_attention_heads=4, num_key_value_heads=2, head_dim=16)).eval()
    dt = str(tmp_path / "tgt"); tgt.save_pretrained(dt); tq.save_pretrained(dt)
    if cross:
        tl = AutoTokenizer.from_pretrained(ld)
        src = LlamaForCausalLM(LlamaConfig(vocab_size=len(tl), hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                                           num_attention_heads=4, num_key_value_heads=2, head_dim=16)).eval()
        ds_ = str(tmp_path / "src"); src.save_pretrained(ds_); tl.save_pretrained(ds_)
    else:
        ds_ = dt
    Ls, H, D = 3, 2, 16
    g = torch.Generator().manual_seed(9)
    mk = lambda: [dict(sel=[0, 1, 2], W=torch.randn(3 * H * D, H * D, generator=g) * 0.1, b=torch.randn(H * D, generator=g),
                       mu=torch.zeros(H * D), r2=torch.ones(H), vnat=torch.ones(H), H=H, D=D) for _ in range(Ls)]
    mk_, mv_ = mk(), mk()
    plain, xm = str(tmp_path / "plain.pt"), str(tmp_path / "x.pt")
    torch.save({"k": mk_, "v": mv_, "feat": "kv"}, plain)
    torch.save({"k": mk_, "v": mv_, "feat": "kv", "align": "last_char", "bos_s": (1 if cross else None)}, xm)
    ids = tq("The quick brown fox jumps over the lazy dog near the river bank today.", add_special_tokens=False)["input_ids"]

    def run(maps):
        c = _conn(vc, ds_, maps, (0, 1), False)
        c.tgt_name, c.restore = dt, "none"
        c._init_worker(torch.device("cpu"))
        return c, {l: (K.float(), V.float()) for l, K, V in c._map_layers(ids)}
    cx, ox = run(xm)
    assert cx.xtok is not None
    for l in ox:
        assert ox[l][0].shape == (len(ids), H, D) and torch.isfinite(ox[l][0]).all()
    if not cross:
        s_ids, idx = cx._xalign(ids)
        assert s_ids == ids and torch.equal(idx, torch.arange(len(ids)))
        _, op = run(plain)
        for l in op:
            for X, Y in zip(ox[l], op[l]):
                assert torch.allclose(X, Y, atol=3e-2, rtol=2e-2), (X - Y).abs().max()
    else:
        s_ids, idx = cx._xalign(ids)
        assert s_ids[0] == 1 and idx.min() >= 1 and bool((idx[1:] >= idx[:-1]).all()) and idx.max() < len(s_ids)


def test_source_window_matches_masked_reference(tmp_path, monkeypatch):
    """src_window=W: the source prefill attends to the last W tokens only (chunked prefill included); == a dense
    reference with the same banded mask; and differs from full attention once T > W."""
    from tests.test_core_cpu import _import_connector_with_stub_vllm
    from kvx import fa
    vc = _import_connector_with_stub_vllm()
    monkeypatch.setattr(fa, "_fa_func", lambda: (_fa_ref, None))
    d, maps, _ = _tiny_maps(tmp_path, "kv")
    ids = torch.randint(0, 97, (40,)).tolist()
    outs = []
    for win, sc in ((0, 0), (8, 0), (8, 16)):
        c = _conn(vc, d, maps, (0, 1), False)
        c.src_sdpa, c.src_window, c.src_chunk = "vllm_fa", win, sc
        c._init_worker(torch.device("cpu"))
        outs.append({l: K.float() for l, K, V in c._map_layers(ids)})
    fa.WINDOW = 0
    for l in outs[0]:
        assert torch.allclose(outs[1][l], outs[2][l], atol=2e-2)        # chunked == one pass under the window
        assert torch.allclose(outs[1][l][:8], outs[0][l][:8], atol=2e-2)  # first W tokens see the same keys
        assert not torch.allclose(outs[1][l][-8:], outs[0][l][-8:], atol=2e-2) or l == 0


def _llama_cfg(hkv=2):
    from transformers import LlamaConfig
    return LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                       num_key_value_heads=hkv, head_dim=16, max_position_embeddings=256)


@pytest.mark.parametrize("kind,chunk", [("qwen3", 0), ("qwen3", 16), ("qwen2", 0), ("llama", 16)])
def test_lean_source_matches_hf(kind, chunk):
    """kvx.lean_source: pre-RoPE K and V of every layer == unrope(HF cache K) and HF V (incl. chunked prefill)."""
    from kvx.lean_source import LeanSource, sdpa_attn
    cfg = _llama_cfg() if kind == "llama" else _cfg(kind, 2)
    m = _model(cfg)
    ids = torch.randint(0, 97, (40,))
    cos, sin = core.rope_cos_sin(m, 40, "cpu")
    with torch.no_grad():
        o = m(input_ids=ids[None], use_cache=True)
    ref = core.cache_layers(o.past_key_values)
    out = LeanSource(m).forward(ids, cos, sin, sdpa_attn, chunk=chunk)
    for (K, V), (k, v) in zip(out, ref):
        assert torch.allclose(K, core.unrope(k, cos, sin), atol=1e-4), (K - core.unrope(k, cos, sin)).abs().max()
        assert torch.allclose(V, v, atol=1e-4)


def _lean_tp_worker(rank, world, port, d):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from kvx.tp_source import shard_source, gather_heads
    from kvx.lean_source import LeanSource, sdpa_attn
    m = _model(_cfg("qwen3", 2))
    ids = torch.load(os.path.join(d, "ids.pt"))
    cos, sin = core.rope_cos_sin(m, len(ids), "cpu")
    shard_source(m, rank, world, None)
    out = LeanSource(m).forward(ids, cos, sin, sdpa_attn)
    ks = [gather_heads(K[0], world, None, 1, 0) for K, _ in out]
    if rank == 0:
        torch.save(ks, os.path.join(d, "lean_tp.pt"))
    dist.destroy_process_group()


def test_lean_source_tp2_matches_whole(tmp_path):
    from kvx.lean_source import LeanSource, sdpa_attn
    m = _model(_cfg("qwen3", 2))
    ids = torch.randint(0, 97, (24,))
    torch.save(ids, tmp_path / "ids.pt")
    cos, sin = core.rope_cos_sin(m, 24, "cpu")
    whole = LeanSource(m).forward(ids, cos, sin, sdpa_attn)
    mp.spawn(_lean_tp_worker, args=(2, _port(), str(tmp_path)), nprocs=2, join=True)
    ks = torch.load(tmp_path / "lean_tp.pt")
    for (K, _), k2 in zip(whole, ks):
        assert torch.allclose(K[0], k2, atol=1e-4)


@pytest.mark.parametrize("fp8,tc", [(False, 0), (True, 16)])
def test_connector_lean_source_equals_hf_source(tmp_path, fp8, tc, monkeypatch):
    """src_lean=1 (pre-RoPE K straight from the lean forward, no unrope) maps the same K/V as the HF source path."""
    from tests.test_core_cpu import _import_connector_with_stub_vllm
    vc = _import_connector_with_stub_vllm()
    d, maps, _ = _tiny_maps(tmp_path, "kv")
    if fp8:
        monkeypatch.setattr(torch, "_scaled_mm", _scaled_mm_emu)
    ids = torch.randint(0, 97, (40,)).tolist()
    res = []
    for lean in (0, 1):
        c = _conn(vc, d, maps, (0, 1), False)
        c.src_lean, c.fp8, c.block, c.tok_chunk, c._fused_cpu = lean, fp8, 16, tc, fp8
        c._init_worker(torch.device("cpu"))
        res.append({l: (K.float(), V.float()) for l, K, V in c._map_layers(ids)})
    for l in res[0]:
        for X, Y in zip(res[1][l], res[0][l]):
            assert ((X - Y).norm() / Y.norm()).item() < 2e-2


def test_sink_template_pairs_format_equals_stacked(tmp_path):
    """kvx-pairs' sink template {layer: {"k": [A, H*D], "v": ...}} loads like the stacked {"K","V": [Lt, H, A, D]}."""
    from tests.test_core_cpu import _import_connector_with_stub_vllm
    vc = _import_connector_with_stub_vllm()
    d, maps, sp = _tiny_maps(tmp_path, "kv", sink=True)  # stacked format, Ls=3, H=2, A=4, D=16
    st = torch.load(sp)
    pairs = {t: {"k": st["K"][t].permute(1, 0, 2).reshape(4, -1), "v": st["V"][t].permute(1, 0, 2).reshape(4, -1)} for t in range(3)}
    sp2 = str(tmp_path / "sink_pairs.pt")
    torch.save(pairs, sp2)
    out = []
    for path in (sp, sp2):
        c = _conn(vc, d, maps, (0, 1), False)
        c.sink_path = path
        c._init_worker(torch.device("cpu"))
        out.append((c.sinkK.clone(), c.sinkV.clone()))
    assert torch.allclose(out[0][0], out[1][0]) and torch.allclose(out[0][1], out[1][1])


def test_lean_source_window_schedule_last_layers_full():
    """Per-layer attention schedule: window below, full attention in the last N layers. N = all layers == full;
    the last layers' K equal the full-attention ones only when N covers every layer (attention errors propagate)."""
    from kvx.lean_source import LeanSource, sdpa_attn
    m = _model(_cfg("qwen3", 2))
    ids = torch.randint(0, 97, (40,))
    cos, sin = core.rope_cos_sin(m, 40, "cpu")
    lean = LeanSource(m)
    full = lean.forward(ids, cos, sin, sdpa_attn)
    win = lambda q, k, v, s: sdpa_attn(q, k, v, s, window=8)
    def sched(n):
        f = lambda li: (sdpa_attn if li >= 3 - n else win)
        f.per_layer = True
        return f
    allfull = lean.forward(ids, cos, sin, sched(3))
    for (a, _), (b, _) in zip(full, allfull):
        assert torch.allclose(a, b, atol=1e-5)
    # a layer's own K/V come BEFORE its attention: full attention in layer 1 changes layer 2's K/V, never layer 1's
    part = lean.forward(ids, cos, sin, sched(2))
    allwin = lean.forward(ids, cos, sin, win)
    assert torch.allclose(part[1][0], allwin[1][0], atol=1e-5) and torch.allclose(part[1][1], allwin[1][1], atol=1e-5)
    assert not torch.allclose(part[2][1], allwin[2][1], atol=1e-4)
