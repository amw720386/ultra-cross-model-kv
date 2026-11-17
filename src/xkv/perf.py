"""Performance tasks: kernel-level TTFT profiling and decode (TPOT) benchmarks."""
from __future__ import annotations

import json
import time
from collections import defaultdict

import torch


def _cat(name: str) -> str:
    n = name.lower()
    if "flash" in n and ("fwd" in n or "attn" in n):
        return "flash_attn"
    if any(s in n for s in ("chunk_", "solve_tril", "recompute_w_u", "fused_recurrent", "l2norm", "kkt", "cumsum")):
        return "fla_scan"
    if any(s in n for s in ("gemm", "cutlass", "sm90", "nvjet", "ampere_", "cublas", "xmma", "gemv")):
        return "gemm"
    if "triton" in n:
        return "triton_other"
    if "rms" in n or "norm" in n:
        return "norm"
    if "copy" in n or "cat" in n or "memcpy" in n or "memset" in n:
        return "copy"
    return "elementwise"


def kernel_table(fn, reps=2):
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    by_cat, by_name = defaultdict(float), defaultdict(float)
    for e in prof.key_averages():
        t = getattr(e, "device_time_total", None)
        if t is None:
            t = getattr(e, "cuda_time_total", 0)
        if t <= 0 or e.key.startswith("aten::") or e.key.startswith("cuda"):
            continue
        by_cat[_cat(e.key)] += t / reps / 1e6
        by_name[e.key[:90]] += t / reps / 1e6
    top = sorted(by_name.items(), key=lambda kv: -kv[1])[:25]
    return {"by_category_s": dict(sorted(by_cat.items(), key=lambda kv: -kv[1])),
            "gpu_busy_s": sum(by_cat.values()), "top_kernels_s": top}


def timeit(fn, reps=3):
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2]


def task_kprofile(src="4b", tgt="27b", mapper_name="4b-27b-w8192-distill-ret", lengths=(8192, 16384), tail=128,
                  window=8192, chunk=8192):
    from .experiments import get, load_mapper
    from .mapper import native_prefill, transfer_prefill

    sm, _ = get(src)
    tm, _ = get(tgt)
    mapper = load_mapper(mapper_name, sm, tm)
    res = {"gpu": torch.cuda.get_device_name(), "max_src": mapper.max_src, "rows": []}
    for L in lengths:
        ids = torch.randint(1000, 100000, (1, L), device="cuda")
        row = {"len": L}
        f_nat = lambda: native_prefill(tm, ids, chunk=chunk)
        f_src = lambda: native_prefill(sm, ids, chunk=chunk)
        f_src_stop = lambda: sm.prefill(ids, sm.new_cache(1, L, window=window, chunk=chunk), chunk=chunk,
                                        logits=None, window=window, stop_after=mapper.max_src)
        tim = {}
        f_tr = lambda: transfer_prefill(sm, tm, mapper, ids, tail=tail, window=window, chunk=chunk, timings=tim)
        f_tail = lambda: tm.forward(ids[:, :tail], tm.new_cache(1, L + 512), logits="last")
        for name, f in (("native_tgt", f_nat), ("native_src", f_src), ("src_stop_window", f_src_stop),
                        ("transfer", f_tr), ("tgt_tail_only", f_tail)):
            f()
            row[name + "_s"] = timeit(f)
        row["transfer_parts"] = dict(tim)
        row["transfer_kernels"] = kernel_table(f_tr)
        row["src_kernels"] = kernel_table(f_src_stop)
        print(json.dumps(row, indent=1), flush=True)
        res["rows"].append(row)
        torch.cuda.empty_cache()
    return res


# --------------------------------------------------------------------------------------
# decode
# --------------------------------------------------------------------------------------


def _bucket(n, q=1024):
    return ((n + q - 1) // q) * q


def task_decode_selftest(src="0.8b", tgt="2b", lens=(37, 300, 1500), new=48, gamma=4, window=256, chunk=512):
    """Batched/graphed/speculative decoding reproduces plain greedy decoding token for token."""
    from .decode import Decoder
    from .experiments import get
    from .mapper import native_prefill
    from .qwen35 import STATS

    sm, _ = get(src)
    tm, tok = get(tgt)
    stop = ()
    g = torch.Generator().manual_seed(0)
    prompts = [torch.randint(1000, 50000, (1, n), generator=g).cuda() for n in lens]
    # natural text makes speculative acceptance meaningful
    text = ("The history of science is the study of the development of science, including both the natural and "
            "social sciences. Science is a body of empirical, theoretical, and practical knowledge about the "
            "natural world, produced by scientists who emphasize the observation, explanation, and prediction of "
            "real-world phenomena. ") * 40
    tids = tok(text, return_tensors="pt").input_ids[:, :1200].cuda()
    prompts.append(tids)
    refs = []
    for p in prompts:
        c, lg = native_prefill(tm, p, max_len=p.shape[1] + new + 8)
        refs.append(tm.greedy(c, lg, new, stop))
    res = {"ref_lens": [len(r) for r in refs]}
    cap = _bucket(max(p.shape[1] for p in prompts) + new + gamma + 8)
    for graphs in (False, True):
        for B in (len(prompts), len(prompts) + 1):
            for spec, win in ((False, None), (True, None), (True, window)):
                t0 = time.time()
                dec = Decoder(tm, B, cap, src=sm if spec else None, src_window=win, gamma=gamma, graphs=graphs)
                for b in range(B):
                    p = prompts[min(b, len(prompts) - 1)]
                    c, lg = native_prefill(tm, p)
                    sc = None
                    if spec:
                        sc = sm.new_cache(1, p.shape[1], window=win, chunk=chunk)
                        sm.prefill(p, sc, chunk=chunk, logits=None, window=win)
                    dec.load(b, c, int(lg[0, -1].argmax()), sc)
                outs, st = dec.generate(new, stop, spec=spec, n_active=len(prompts))
                exact = [o == r for o, r in zip(outs, refs)]
                tokacc = [sum(a == b for a, b in zip(o, r)) / len(r) for o, r in zip(outs, refs)]
                key = f"graphs{int(graphs)}_B{B}_spec{int(spec)}_win{win}"
                res[key] = {"exact": exact, "tok_agree": tokacc, "stats": st, "s": time.time() - t0}
                print(key, json.dumps(res[key]), flush=True)
                del dec
                torch.cuda.empty_cache()
    res["kernel_stats"] = dict(STATS)
    return res


def task_decode_bench(src="4b", tgt="27b", mapper_name="4b-27b-w8192-distill-ret", lengths=(2048, 32768),
                      batches=(1, 2, 3, 4, 8, 12, 16), new=128, gamma=4, window=8192, tail=128, chunk=8192,
                      mem_budget_gb=None, text_seed=11, arms=("native", "transfer", "transfer_spec")):
    """TPOT / decode throughput. Equal-length real-text prompts per batch; every row decodes `new` tokens.

    native: native 27B prefill + graphed 27B decode. transfer: transferred cache + the same decode.
    transfer_spec: transferred cache + 4B-drafted speculative decode (the 4B cache comes from the transfer).
    TPOT = decode wall time / (new - 1), first token excluded (it belongs to TTFT).
    """
    from . import evals
    from .decode import Decoder
    from .experiments import get, load_mapper
    from .mapper import native_prefill, transfer_prefill

    sm, _ = get(src)
    tm, tok = get(tgt)
    mapper = load_mapper(mapper_name, sm, tm)
    free = torch.cuda.mem_get_info()[0] / 1e9
    budget = mem_budget_gb or (free - 12)
    s = tm.spec
    kv_per_tok = len(s.full) * 2 * s.nkv * s.hd * 2
    state = len(s.lin) * s.nv * s.dk * s.dv * 4
    res = {"gpu": torch.cuda.get_device_name(), "free_gb": free, "rows": []}
    for L in lengths:
        texts = evals.load_heldout_text(tok, max(batches), L, seed=text_seed, split="eval", source="fineweb")
        for B in batches:
            cap = _bucket(L + new + gamma + 8)
            need = B * (cap * kv_per_tok + state) / 1e9 + (L * kv_per_tok + state) / 1e9 * 2
            if need > budget or len(texts) < B:
                print(f"skip L={L} B={B} need={need:.1f}GB", flush=True)
                continue
            row = {"len": L, "batch": B}
            outs_by_arm = {}
            for arm in arms:
                spec = arm == "transfer_spec"
                t0 = time.time()
                dec = Decoder(tm, B, cap, src=sm if spec else None, src_window=window, gamma=gamma,
                              src_cap=_bucket(min(L, window) + new + gamma + 8))
                cap_s = time.time() - t0
                for b in range(B):
                    ids = torch.tensor([texts[b]], device="cuda")
                    sc = None
                    if arm == "native":
                        c, lg = native_prefill(tm, ids, chunk=chunk)
                    elif spec:
                        c, lg, sc = transfer_prefill(sm, tm, mapper, ids, tail=tail, window=window, chunk=chunk,
                                                     keep_src=True)
                    else:
                        c, lg = transfer_prefill(sm, tm, mapper, ids, tail=tail, window=window, chunk=chunk)
                    dec.load(b, c, int(lg[0, -1].argmax()), sc)
                    del c, sc
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                t = time.perf_counter()
                outs, st = dec.generate(new, (), spec=spec)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t
                n = min(len(o) for o in outs)
                row[arm] = {"tpot_ms": 1000 * dt / (n - 1), "tok_per_s": B * (n - 1) / dt, "graph_capture_s": cap_s,
                            "tokens": n, **st}
                outs_by_arm[arm] = outs
                del dec
                torch.cuda.empty_cache()
            if "transfer" in outs_by_arm and "transfer_spec" in outs_by_arm:
                a, b_ = outs_by_arm["transfer"], outs_by_arm["transfer_spec"]
                row["spec_exact_rows"] = sum(x == y for x, y in zip(a, b_)) / B
                row["spec_token_agree"] = sum(sum(p == q for p, q in zip(x, y)) / len(x) for x, y in zip(a, b_)) / B
            if "native" in outs_by_arm and "transfer" in outs_by_arm:
                a, b_ = outs_by_arm["native"], outs_by_arm["transfer"]
                row["transfer_vs_native_token_agree"] = sum(
                    sum(p == q for p, q in zip(x, y)) / len(x) for x, y in zip(a, b_)) / B
            row["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            print(json.dumps(row), flush=True)
            res["rows"].append(row)
    return res


def task_ttft_ab(src="4b", tgt="27b", mapper_name="4b-27b-w8192-distill-ret", lengths=(8192, 16384), tail=128,
                 window=8192, chunk=8192, reps=5, configs=None):
    """A/B engine options on TTFT (native and transfer) and check numerics of the fused conv."""
    from . import qwen35 as Q
    from .experiments import get, load_mapper
    from .kernels import conv_silu
    from .mapper import native_prefill, transfer_prefill

    sm, _ = get(src)
    tm, _ = get(tgt)
    mapper = load_mapper(mapper_name, sm, tm)
    res = {"gpu": torch.cuda.get_device_name()}
    # numerics: fused conv vs reference
    g = torch.Generator(device="cuda").manual_seed(0)
    h = torch.randn(2, 3, 10240, device="cuda", generator=g).bfloat16()
    x = torch.randn(2, 777, 12000, device="cuda", generator=g).bfloat16()[..., 100:10340]
    w = torch.randn(4, 10240, device="cuda", generator=g).bfloat16()
    ref = Q._dwconv_silu(torch.cat((h, x), 1), w)
    got = conv_silu(h, x, w)
    res["conv_maxabs_err"] = float((ref.float() - got.float()).abs().max())
    configs = configs or [{"name": "old", "FUSED_CONV": False, "RECUR_MAX": 16},
                          {"name": "fused_conv", "FUSED_CONV": True, "RECUR_MAX": 16},
                          {"name": "fused_conv+recur256", "FUSED_CONV": True, "RECUR_MAX": 256}]
    ids_by_len = {L: torch.randint(1000, 100000, (1, L), device="cuda") for L in lengths}
    rows = []
    for cfg in configs:
        for k, v in cfg.items():
            if k != "name":
                setattr(Q, k, v)
        for L, ids in ids_by_len.items():
            row = {"cfg": cfg["name"], "len": L}
            f_nat = lambda: native_prefill(tm, ids, chunk=chunk)
            f_tr = lambda: transfer_prefill(sm, tm, mapper, ids, tail=tail, window=window, chunk=chunk)
            f_tail = lambda: tm.forward(ids[:, :tail], tm.new_cache(1, L + 512), logits="last")
            for name, f in (("native", f_nat), ("transfer", f_tr), ("tail_only", f_tail)):
                f()
                row[name] = timeit(f, reps)
            row["speedup"] = row["native"] / row["transfer"]
            # logits parity of transfer vs old config is checked through agreement of first-token argmax
            _, lg = f_tr()
            row["first_tok"] = int(lg[0, -1].argmax())
            print(json.dumps(row), flush=True)
            rows.append(row)
    res["rows"] = rows
    return res


def task_bench_matrix(cells, **kw):
    """Run experiments.task_bench for several (lengths, batch) cells in one container."""
    from .experiments import task_bench

    out = []
    for c in cells:
        r = task_bench(**{**kw, "lengths": c["lengths"], "batch": c.get("batch", 1), "warmup": not out})
        out.append({"cell": c, **r})
        torch.cuda.empty_cache()
    return out


def task_spec_debug(src="4b", tgt="27b", mapper_name="4b-27b-w8192-distill-ret", L=2048, B=6, new=64, gamma=4,
                    window=8192, tail=128):
    """Is batched speculative decoding row-independent? Compare each row run alone (B=1) vs together."""
    from . import evals
    from .decode import Decoder
    from .experiments import get, load_mapper
    from .mapper import transfer_prefill

    sm, _ = get(src)
    tm, tok = get(tgt)
    mapper = load_mapper(mapper_name, sm, tm)
    texts = evals.load_heldout_text(tok, B, L, seed=11, split="eval", source="fineweb")
    cap = _bucket(L + new + gamma + 8)
    pre = []
    for t in texts:
        c, lg, sc = transfer_prefill(sm, tm, mapper, torch.tensor([t], device="cuda"), tail=tail, window=window,
                                     keep_src=True)
        pre.append((c, int(lg[0, -1].argmax()), sc))

    def run(rows, spec, graphs=True):
        dec = Decoder(tm, len(rows), cap, src=sm if spec else None, src_window=window, gamma=gamma,
                      src_cap=_bucket(min(L, window) + new + gamma + 8), graphs=graphs)
        for b, r in enumerate(rows):
            c, f, sc = pre[r]
            dec.load(b, c, f, sc if spec else None)
        o, st = dec.generate(new, (), spec=spec)
        del dec
        torch.cuda.empty_cache()
        return o, st

    res = {}
    solo_plain = [run([r], False)[0][0] for r in range(B)]
    solo_spec = [run([r], True) for r in range(B)]
    res["solo_spec_acc"] = [s[1]["mean_accepted_per_round"] for s in solo_spec]
    res["solo_spec_exact"] = [s[0][0] == p for s, p in zip(solo_spec, solo_plain)]
    bp, _ = run(list(range(B)), False)
    res["batch_plain_vs_solo_plain"] = [a == b for a, b in zip(bp, solo_plain)]
    bs, st = run(list(range(B)), True)
    res["batch_spec_acc_rows"] = st.get("accepted_per_round_by_row")
    res["batch_spec_vs_solo_spec"] = [a == s[0][0] for a, s in zip(bs, solo_spec)]
    res["batch_spec_vs_solo_plain"] = [a == b for a, b in zip(bs, solo_plain)]
    same, st2 = run([0] * B, True)
    res["same_row_spec_acc_rows"] = st2.get("accepted_per_round_by_row")
    res["same_row_spec_exact"] = [a == solo_plain[0] for a in same]
    bse, st3 = run(list(range(B)), True, graphs=False)
    res["eager_batch_spec_acc_rows"] = st3.get("accepted_per_round_by_row")
    res["eager_batch_spec_vs_solo_plain"] = [a == b for a, b in zip(bse, solo_plain)]
    print(json.dumps(res, indent=1), flush=True)
    return res


def task_tail_check(src="4b", tgt="27b", mapper_name="4b-27b-w8192-distill-ret", lengths=(8192, 16384, 32768),
                    tail=128, window=8192, full_until=32768, reps=5):
    """Graphed persistent tail vs eager tail: logits agreement and TTFT."""
    from . import evals
    from .experiments import get, load_mapper, teacher_forced, compare
    from .mapper import native_prefill, transfer_prefill

    sm, _ = get(src)
    tm, tok = get(tgt)
    mapper = load_mapper(mapper_name, sm, tm)
    rows = []
    for L in lengths:
        seq = evals.load_heldout_text(tok, 1, L + 64, seed=5, split="eval", source="fineweb")[0]
        ids = torch.tensor([seq[:L]], device="cuda")
        cont = seq[L:L + 64]
        row = {"len": L}
        for gt in (False, True):
            f = lambda: transfer_prefill(sm, tm, mapper, ids, tail=tail, window=window, full_until=full_until,
                                         graph_tail=gt)
            f()
            row[f"ttft_graph{int(gt)}"] = timeit(f, reps)
            c, lg = f()
            row[f"logits_graph{int(gt)}"] = teacher_forced(tm, c, lg, cont)
            del c
        a, b = row.pop("logits_graph0"), row.pop("logits_graph1")
        row["graph_vs_eager"] = compare(a, b, cont)
        f = lambda: native_prefill(tm, ids)
        f()
        row["native"] = timeit(f, reps)
        row["speedup_graph"] = row["native"] / row["ttft_graph1"]
        print(json.dumps(row), flush=True)
        rows.append(row)
        torch.cuda.empty_cache()
    return {"gpu": torch.cuda.get_device_name(), "rows": rows}


def task_gen_profile(src="4b", tgt="27b", mapper_name="4b-27b-chat", B=8, L=5000, new=64, strategy="target",
                     vary=False, max_new_cap=None):
    """Where does one batched genrun call spend time? (prefill loop, decoder capture, decode)"""
    from .decode import Decoder
    from .experiments import get, load_mapper
    from .genrun import make_batch_gen
    from .mapper import native_prefill, transfer_prefill

    sm, _ = get(src)
    tm, tok = get(tgt)
    mapper = load_mapper(mapper_name, sm, tm)
    g = torch.Generator().manual_seed(0)
    prompts = [torch.randint(1000, 50000, (L + 37 * i,), generator=g).tolist() for i in range(B)]
    gen = make_batch_gen(strategy, sm, tm, mapper, tok, batch=B, full_until=32768)
    out = {}
    for rep in range(4):
        if vary:
            prompts = [torch.randint(1000, 50000, (L + 131 * rep + 37 * i,), generator=g).tolist() for i in range(B)]
        t = time.time()
        gen(prompts, new)
        torch.cuda.synchronize()
        out[f"gen_call_{rep}"] = time.time() - t
    t = time.time()
    for p in prompts:
        ids = torch.tensor([p], device="cuda")
        if strategy == "target":
            native_prefill(tm, ids)
        else:
            transfer_prefill(sm, tm, mapper, ids, tail=128, window=8192, full_until=32768)
    torch.cuda.synchronize()
    out["prefill_loop"] = time.time() - t
    t = time.time()
    dec = Decoder(tm, B, _bucket(L + 37 * B + new + 16))
    torch.cuda.synchronize()
    out["decoder_capture"] = time.time() - t
    print(json.dumps(out), flush=True)
    return out


# --------------------------------------------------------------------------------------
# serving under load
# --------------------------------------------------------------------------------------


def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def task_serve(src="4b", tgt="27b", mapper_name="4b-27b-chat", rates=(0.1, 0.2, 0.3, 0.45, 0.65, 0.9, 1.2),
               n_req=30, lengths=(8192, 16384, 32768), new=256, slots=8, arms=("native", "transfer"), seed=0,
               tail=128, window=8192, full_until=32768, slo_ttft=(2.0, 5.0), offline=True, chunk=8192):
    """Online serving on one GPU. Poisson arrivals at each rate; prompt length drawn from `lengths`; every request
    generates `new` tokens. Identical scheduler for both arms: FCFS, one prefill admitted per iteration into a free
    decode slot, then one batched decode step over all slots (CUDA graph, fixed `slots`). TTFT includes queueing.
    """
    import random

    from .decode import Decoder
    from .experiments import get, load_mapper
    from .mapper import native_prefill, transfer_prefill

    sm, _ = get(src)
    tm, _ = get(tgt)
    mapper = load_mapper(mapper_name, sm, tm)
    cap = _bucket(max(lengths) + new + 16)

    def prefill(arm, ids):
        if arm == "native":
            return native_prefill(tm, ids, chunk=chunk)
        return transfer_prefill(sm, tm, mapper, ids, tail=tail, window=window, full_until=full_until, chunk=chunk)

    def make_ids(L, g):
        return torch.randint(1000, 100000, (1, L), generator=g, device="cuda")

    # warm every path / length once (autotune, graph capture of the persistent tail)
    gw = torch.Generator(device="cuda").manual_seed(123)
    for L in lengths:
        for arm in arms:
            c, _ = prefill(arm, make_ids(L, gw))
            del c
    torch.cuda.empty_cache()

    def workload(rate, n, rng):
        t, reqs = 0.0, []
        for i in range(n):
            t += rng.expovariate(rate) if rate else 0.0
            reqs.append({"i": i, "arrival": t, "L": rng.choice(lengths)})
        return reqs

    res = {"gpu": torch.cuda.get_device_name(), "config": dict(n_req=n_req, lengths=list(lengths), new=new,
                                                               slots=slots, mapper=mapper_name, tail=tail),
           "runs": []}
    for arm in arms:
        dec = Decoder(tm, slots, cap)
        plan = [(r, n_req) for r in rates] + ([(0.0, n_req)] if offline else [])
        for rate, n in plan:
            rng = random.Random(seed + int(rate * 1000))
            g = torch.Generator(device="cuda").manual_seed(seed + int(rate * 1000))
            reqs = workload(rate, n, rng)
            queue, slot_req = list(reqs), [None] * slots
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            done = 0
            while done < n:
                now = time.perf_counter() - t0
                free = [b for b in range(slots) if slot_req[b] is None]
                if queue and queue[0]["arrival"] <= now and free:
                    r = queue.pop(0)
                    c, lg = prefill(arm, make_ids(r["L"], g))
                    r["first"] = time.perf_counter() - t0
                    dec.load(free[0], c, int(lg[0, -1].argmax()))
                    del c
                    r["count"] = 1
                    slot_req[free[0]] = r
                    continue_decode = any(s is not None for s in slot_req)
                else:
                    continue_decode = any(s is not None for s in slot_req)
                if continue_decode:
                    dec.g_plain().tolist()  # sync
                    tnow = time.perf_counter() - t0
                    for b, r in enumerate(slot_req):
                        if r is None:
                            continue
                        r["count"] += 1
                        if r["count"] >= new:
                            r["done"] = tnow
                            slot_req[b] = None
                            done += 1
                elif queue:
                    wait = queue[0]["arrival"] - (time.perf_counter() - t0)
                    if wait > 0:
                        time.sleep(wait)
            ttft = [r["first"] - r["arrival"] for r in reqs]
            e2e = [r["done"] - r["arrival"] for r in reqs]
            tpot = [(r["done"] - r["first"]) / (new - 1) for r in reqs]
            span = max(r["done"] for r in reqs) - min(r["arrival"] for r in reqs)
            run = {"arm": arm, "rate": rate or "offline", "n": n, "throughput_req_s": n / span,
                   "output_tok_s": n * new / span, "prompt_tok_s": sum(r["L"] for r in reqs) / span,
                   "ttft_p50": _pct(ttft, .5), "ttft_p90": _pct(ttft, .9), "ttft_p99": _pct(ttft, .99),
                   "tpot_p50_ms": 1000 * _pct(tpot, .5), "tpot_p90_ms": 1000 * _pct(tpot, .9),
                   "e2e_p50": _pct(e2e, .5), "e2e_p99": _pct(e2e, .99),
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
            for s in slo_ttft:
                run[f"goodput_req_s_ttft<={s}s"] = sum(t <= s for t in ttft) / span
            print(json.dumps(run), flush=True)
            res["runs"].append(run)
        del dec
        torch.cuda.empty_cache()
    return res
