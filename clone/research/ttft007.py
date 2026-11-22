"""Note 007 Tier-1 TTFT rows: Llama-3.1-Minitron-4B-Width-Base -> Llama-3.1-8B target TTFT at 8K with the source cache
resident (theirs: 38.25 ms vs 302.69 ms native prefill = 7.91x, H100, CUDA-synchronized p50, 20 warmups / 50 reps, real
FineWeb tokens, rank-8 residual on direct injection).

Both models share the llama3 RoPE (theta 5e5, factor 8) and 32 layers x 8 KV heads x 128, so direct injection is the
source cache itself (no transform). Arms timed: native target prefill of all N tokens (logits of the last one);
handoff-direct (target forward of token N on the resident source cache); handoff-distilled (unrope -> distilled linear
K/V maps -> rope -> target forward of token N). Quality of each arm: KL32 / 32-step top-1 on 64 WikiText-103 windows,
256-token prefix (their quality protocol), from research/kl32.
"""
import json, os, time

import torch

from kvx import core
from research import mc128, kl32, kvdistill


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _p50(fn, warm=20, reps=50):
    for _ in range(warm):
        fn()
    _sync()
    ts = []
    for _ in range(reps):
        _sync(); t0 = time.perf_counter()
        fn()
        _sync(); ts.append(1000 * (time.perf_counter() - t0))
    ts.sort()
    return dict(p50_ms=ts[len(ts) // 2], p10_ms=ts[len(ts) // 10], p90_ms=ts[9 * len(ts) // 10], reps=reps, warmups=warm)


@torch.no_grad()
def time_arms(P, ids, maps_key=None, warm=20, reps=50):
    from transformers import DynamicCache
    T, S = P.T, P.S
    N = ids.shape[1]
    res = {}
    res["native_prefill"] = _p50(lambda: T(input_ids=ids, logits_to_keep=1), warm, reps)
    out = S(input_ids=ids[:, :N - 1], past_key_values=DynamicCache(), use_cache=True, logits_to_keep=1)
    src_layers = core.cache_layers(out.past_key_values)  # resident source cache (post-RoPE, bf16)

    def direct():
        c = DynamicCache()
        for l, (k, v) in enumerate(src_layers):
            c.update(k, v, l)
        return T(input_ids=ids[:, N - 1:], past_key_values=c, logits_to_keep=1)
    res["handoff_direct"] = _p50(direct, warm, reps)
    if maps_key is not None:
        def mapped():
            ks, vs = [], []
            for l, (k, v) in enumerate(src_layers):
                c_, s_ = core._cs(P.cs, P.ss, l, N - 1)
                ks.append(core.unrope(k.float(), c_, s_)); vs.append(v)
            Km, Vm = P.mapped(torch.stack(ks), torch.stack(vs), maps_key, "none")
            c = core.build_cache(Km, Vm, P.ct, P.st)
            return T(input_ids=ids[:, N - 1:], past_key_values=c, logits_to_keep=1)
        res["handoff_distilled"] = _p50(mapped, warm, reps)
    res["tokens"] = N
    for k in ("handoff_direct", "handoff_distilled"):
        if k in res:
            res[k]["speedup_vs_native_hf"] = res["native_prefill"]["p50_ms"] / res[k]["p50_ms"]
    return res


def run_pair(src="nvidia/Llama-3.1-Minitron-4B-Width-Base", tgt="unsloth/Meta-Llama-3.1-8B", N=8192, out="", models=None,
             distill_steps=400, distill_lrs="1e-6,1e-5,3e-5", windows=64, prefix=256, warm=20, reps=50, **_):
    from transformers import AutoTokenizer
    from kvx.models import model_path
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_path(tgt))
    S, T = models if models else (mc128.load(src, dev, torch.bfloat16), mc128.load(tgt, dev, torch.bfloat16))
    P = mc128.Pair(S, T, tok, dev, Lmax=N + 64)
    bos = tok.bos_token_id
    res = dict(src=src, tgt=tgt, N=N, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu", torch=str(torch.__version__))
    dm = kl32.direct_maps(P)
    P.maps[("k", "dir")], P.maps[("v", "dir")] = dm["k"], dm["v"]
    P.norestore = {"dir", "d"}
    C = 64
    tr = mc128.fineweb(tok, 1024, prefix + C, skip=0, stop=mc128.EVAL_SKIP); tr[:, 0] = bos
    va = mc128.fineweb(tok, 16, prefix + C, skip=mc128.EVAL_SKIP + 5000); va[:, 0] = bos
    mk, mv, rep = kvdistill.distill(P, "dir", tr, va, prefix, [float(x) for x in distill_lrs.split(",")], distill_steps, 4,
                                     "none", log=lambda m: print(m, flush=True))
    res["distill"] = rep
    keys = ["dir"]
    if rep["chosen"] is not None:
        P.maps[("k", "d")], P.maps[("v", "d")] = mk, mv
        keys.append("d")
    wins = kl32.wikitext_windows(tok, windows, prefix + 32); wins[:, 0] = bos
    res["kl32_greedy"] = kl32.kl32(P, wins, prefix, 32, keys, [], "none", "greedy")
    print(f"[{time.time()-t0:.0f}s] kl32 " + json.dumps({a: (round(v['kl32'], 5), round(v['agree'], 4)) for a, v in res['kl32_greedy'].items()}), flush=True)
    ids = mc128.fineweb(tok, 1, N, skip=mc128.EVAL_SKIP + 7000).to(dev); ids[:, 0] = bos
    res["timing"] = time_arms(P, ids, "d" if "d" in keys else None, warm, reps)
    print(f"[{time.time()-t0:.0f}s] timing " + json.dumps(res["timing"]), flush=True)
    res["t_total"] = time.time() - t0
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        json.dump(res, open(out, "w"), indent=1)
    return res
