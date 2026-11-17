"""Experiment tasks run inside Modal GPU containers (see modal_jobs/lab.py)."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import evals
from .mapper import Mapper, Stats, native_prefill, transfer_prefill
from .qwen35 import Hybrid, _flash, load

MODELS = {"0.8b": "Qwen/Qwen3.5-0.8B", "2b": "Qwen/Qwen3.5-2B", "4b": "Qwen/Qwen3.5-4B", "27b": "Qwen/Qwen3.8-27B"}
ART = Path("/artifacts")
_cache: dict = {}


def resolve(key):
    root = Path(os.environ.get("HF_HOME", "/weights/hf")) / "hub" / ("models--" + MODELS[key].replace("/", "--"))
    snaps = sorted((root / "snapshots").iterdir())
    return str(snaps[-1])


def get(key) -> tuple[Hybrid, object]:
    if key not in _cache:
        t = time.time()
        _cache[key] = load(resolve(key))
        print(f"[load] {key} {time.time() - t:.1f}s  mem={torch.cuda.memory_allocated() / 1e9:.1f}GB", flush=True)
    return _cache[key]


def T(ids, dev="cuda"):
    return torch.tensor([ids], device=dev)


@torch.no_grad()
def teacher_forced(model: Hybrid, cache, first_logits, cont: list[int]):
    """Logits predicting each token of `cont` given the cached prompt. Returns (n, V) float32."""
    out = [first_logits[0, -1:]]
    if len(cont) > 1:
        out.append(model.forward(T(cont[:-1]), cache, logits="all")[0])
    return torch.cat(out, dim=0)


def compare(ref, test, cont):
    y = torch.tensor(cont, device=ref.device)
    lr, lt = F.log_softmax(ref, -1), F.log_softmax(test, -1)
    return {
        "nll_ref": float(-lr.gather(1, y[:, None]).mean()),
        "nll": float(-lt.gather(1, y[:, None]).mean()),
        "top1_agree": float((ref.argmax(-1) == test.argmax(-1)).float().mean()),
        "kl": float((lr.exp() * (lr - lt)).sum(-1).mean()),
    }


def mean_dicts(ds):
    keys = ds[0].keys()
    return {k: sum(d[k] for d in ds) / len(ds) for k in keys}


# --------------------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------------------


def task_selftest(src="0.8b", tgt="2b", calib_seqs=8, calib_len=1024):
    res = {"flash_attn": _flash() is not None}
    try:
        import fla

        res["fla"] = fla.__version__
    except Exception as e:  # pragma: no cover
        res["fla"] = repr(e)
    sm, stok = get(src)
    tm, ttok = get(tgt)
    probe = "Hello world! The quick brown fox jumps over 13 lazy dogs. 你好，世界。 def f(x): return x**2"
    res["same_tokenizer"] = stok.encode(probe) == ttok.encode(probe) and len(stok) == len(ttok)
    seqs = evals.load_heldout_text(ttok, 2, 600, seed=1, split="eval", source="wikitext")
    ids = T(seqs[0][:512])
    with torch.no_grad():
        ref = tm.hf(input_ids=ids).logits.float()[0]
    c = tm.new_cache(1, 1024)
    ours = tm.prefill(ids, c, chunk=512, logits="all")[0]
    c2 = tm.new_cache(1, 1024)
    tm.prefill(ids[:, :511], c2, chunk=128, logits=None)
    step = tm.forward(ids[:, 511:], c2)[0, -1]
    res["parity_full_maxabs"] = float((ours - ref).abs().max())
    res["parity_full_top1"] = float((ours.argmax(-1) == ref.argmax(-1)).float().mean())
    res["parity_chunked_decode_maxabs"] = float((step - ref[-1]).abs().max())
    res["parity_chunked_decode_top1"] = bool(step.argmax() == ref[-1].argmax())
    # chunk boundary inside the conv history / scan state
    c3 = tm.new_cache(1, 1024)
    tm.prefill(ids[:, :509], c3, chunk=100, logits=None)
    step3 = tm.forward(ids[:, 509:], c3, logits="all")[0, -1]
    res["parity_odd_chunks_maxabs"] = float((step3 - ref[-1]).abs().max())
    print(json.dumps(res, indent=1), flush=True)

    # mini calibration + transfer smoke test
    cal = evals.load_heldout_text(ttok, calib_seqs, calib_len, seed=0, split="calib", source="wikitext")
    st = Stats(sm, tm)
    for s in cal:
        st.add(sm, tm, T(s))
    mapper, rep = st.solve()
    mapper.to(tm.device)
    res["r2"] = rep
    out = []
    for s in seqs:
        p, cont = s[:448], s[448:600]
        nc, nl = native_prefill(tm, T(p))
        ref = teacher_forced(tm, nc, nl, cont)
        row = {}
        for tail in (1, 16):
            tc, tl = transfer_prefill(sm, tm, mapper, T(p), tail=tail)
            row[f"tail{tail}"] = compare(ref, teacher_forced(tm, tc, tl, cont), cont)
        sc, sl = native_prefill(sm, T(p))
        row["source_native"] = compare(ref, teacher_forced(sm, sc, sl, cont), cont)
        out.append(row)
    res["smoke"] = {k: mean_dicts([r[k] for r in out]) for k in out[0]}
    return res


def task_calibrate(src="4b", tgt="27b", n_seqs=256, seq_len=4096, radius=2, lam=1e-3, window=None,
                   source="fineweb", name=None):
    sm, stok = get(src)
    tm, ttok = get(tgt)
    seqs = evals.load_heldout_text(ttok, n_seqs, seq_len, seed=0, split="calib", source=source)
    st = Stats(sm, tm, radius=radius)
    t = time.time()
    for k, s in enumerate(seqs):
        st.add(sm, tm, T(s), window=window)
        if k % 16 == 0:
            print(f"[calib] {k}/{len(seqs)} {time.time() - t:.0f}s mem={torch.cuda.max_memory_allocated() / 1e9:.1f}GB",
                  flush=True)
    mapper, rep = st.solve(lam=lam)
    name = name or f"{src}-{tgt}-r{radius}-n{n_seqs}x{seq_len}" + (f"-w{window}" if window else "")
    (ART / "maps").mkdir(parents=True, exist_ok=True)
    mapper.save(ART / "maps" / f"{name}.pt")
    # also keep raw stats small enough? (they are large; store only the report)
    return {"name": name, "tokens": st.n, "params": mapper.n_params(), "report": rep,
            "seconds": time.time() - t}


def load_mapper(name, sm, tm):
    return Mapper.load(ART / "maps" / f"{name}.pt", sm.spec, tm.spec, tm.device)


def task_quality(src="4b", tgt="27b", mapper_name=None, n_seqs=16, prefix=2048, cont=256, tails=(1, 8, 32, 64),
                 window=None, source="fineweb", mc_limit=0, mc_tails=(1, 32)):
    sm, stok = get(src)
    tm, ttok = get(tgt)
    mapper = load_mapper(mapper_name, sm, tm)
    seqs = evals.load_heldout_text(ttok, n_seqs, prefix + cont, seed=7, split="eval", source=source)
    rows = []
    for s in seqs:
        p, c = s[:prefix], s[prefix:prefix + cont]
        nc, nl = native_prefill(tm, T(p))
        ref = teacher_forced(tm, nc, nl, c)
        del nc
        row = {}
        for tail in tails:
            tc, tl = transfer_prefill(sm, tm, mapper, T(p), tail=tail, window=window)
            row[f"tail{tail}"] = compare(ref, teacher_forced(tm, tc, tl, c), c)
            del tc
        sc, sl = native_prefill(sm, T(p))
        row["source_native"] = compare(ref, teacher_forced(sm, sc, sl, c), c)
        rows.append(row)
    res = {"lm": {k: mean_dicts([r[k] for r in rows]) for k in rows[0]}, "n_seqs": len(rows),
           "prefix": prefix, "cont": cont}
    print(json.dumps(res["lm"], indent=1), flush=True)
    if mc_limit:
        res["mc"] = run_mc(sm, tm, mapper, ttok, mc_limit, mc_tails, window)
    return res


def make_scorer(strategy, sm, tm, mapper, tail=1, window=None):
    """Scorer over (ctx, cont) pairs with one prefill per distinct context (evals.Scorer)."""

    def prefill(ctx):
        if strategy == "target":
            return native_prefill(tm, T(ctx))
        if strategy == "source":
            return native_prefill(sm, T(ctx))
        return transfer_prefill(sm, tm, mapper, T(ctx), tail=min(tail, len(ctx)), window=window)

    model = sm if strategy == "source" else tm

    def scorer(pairs):
        out = [None] * len(pairs)
        for ctx, idxs in evals.group_by_context(pairs).items():
            cache, first = prefill(list(ctx))
            for k in idxs:
                cont = pairs[k][1]
                c = cache.clone()
                lg = teacher_forced(model, c, first, cont)
                lp = F.log_softmax(lg, -1)
                y = torch.tensor(cont, device=lp.device)
                out[k] = (float(lp.gather(1, y[:, None]).sum()), len(cont), bool((lg.argmax(-1) == y).all()))
        return out

    return scorer


def run_mc(sm, tm, mapper, tok, limit, tails, window):
    res = {}
    tasks = {"hellaswag": evals.run_hellaswag, "piqa": evals.run_piqa, "arc_challenge": evals.run_arc_challenge,
             "mmlu": evals.run_mmlu, "winogrande": evals.run_winogrande}
    strategies = [("target", None), ("source", None)] + [("transfer", t) for t in tails]
    for tname, fn in tasks.items():
        res[tname] = {}
        for strat, tail in strategies:
            key = strat if tail is None else f"transfer_tail{tail}"
            try:
                r = fn(tok, make_scorer(strat, sm, tm, mapper, tail or 1, window), limit=limit, seed=0)
            except Exception as e:  # dataset availability etc.
                r = {"error": repr(e)[:300]}
            res[tname][key] = r
            print(f"[mc] {tname} {key} {json.dumps({k: v for k, v in r.items() if k != 'outputs'})}", flush=True)
    return res


def task_bench(src="4b", tgt="27b", mapper_name=None, lengths=(8192, 16384, 32768, 65536), reps=3, tail=32,
               window=None, chunk=8192, native=True, warmup=True):
    sm, stok = get(src)
    tm, ttok = get(tgt)
    mapper = load_mapper(mapper_name, sm, tm) if mapper_name else None
    g = torch.Generator().manual_seed(0)
    res = {"gpu": torch.cuda.get_device_name(), "rows": []}
    if warmup:
        ids = torch.randint(1000, 100000, (1, 4096), generator=g).cuda()
        native_prefill(tm, ids)
        if mapper:
            transfer_prefill(sm, tm, mapper, ids, tail=tail, window=window)
    for L in lengths:
        ids = torch.randint(1000, 100000, (1, L), generator=g).cuda()
        row = {"len": L}
        if native:
            ts = []
            for _ in range(reps):
                tm_ = {}
                c, _ = native_prefill(tm, ids, chunk=chunk, timings=tm_)
                del c
                torch.cuda.empty_cache()
                ts.append(tm_["total"])
            row["native_s"] = sorted(ts)[len(ts) // 2]
        if mapper:
            ts, parts = [], []
            for _ in range(reps):
                tm_ = {}
                c, _ = transfer_prefill(sm, tm, mapper, ids, tail=tail, window=window, chunk=chunk, timings=tm_)
                del c
                torch.cuda.empty_cache()
                ts.append(tm_["total"])
                parts.append(tm_)
            row["transfer_s"] = sorted(ts)[len(ts) // 2]
            row["transfer_parts"] = parts[len(parts) // 2]
            if native:
                row["speedup"] = row["native_s"] / row["transfer_s"]
        row["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        print(json.dumps(row), flush=True)
        res["rows"].append(row)
    return res


def make_generator(strategy, sm, tm, mapper, tok, tail=32, window=None):
    stop = {i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id) if isinstance(i, int)}

    def gen(ctx, max_new):
        ids = T(ctx)
        if strategy == "target":
            c, lg = native_prefill(tm, ids)
            return tm.greedy(c, lg, max_new, stop)
        if strategy == "source":
            c, lg = native_prefill(sm, ids)
            return sm.greedy(c, lg, max_new, stop)
        c, lg = transfer_prefill(sm, tm, mapper, ids, tail=tail, window=window)
        return tm.greedy(c, lg, max_new, stop)

    return gen


def task_longctx(src="4b", tgt="27b", mapper_name=None, prefix=32768, cont=256, n_seqs=6, windows=(None, 8192, 2048),
                 tail=32, niah_len=32768, niah_n=20, num_keys=(1, 4), source="fineweb", lm=True,
                 niah_tails=None, baselines=True):
    sm, stok = get(src)
    tm, ttok = get(tgt)
    mapper = load_mapper(mapper_name, sm, tm)
    res = {"prefix": prefix, "niah_len": niah_len}
    if lm:
        seqs = evals.load_heldout_text(ttok, n_seqs, prefix + cont, seed=5, split="eval", source=source)
        rows = []
        for s in seqs:
            p, c = s[:prefix], s[prefix:prefix + cont]
            nc, nl = native_prefill(tm, T(p))
            ref = teacher_forced(tm, nc, nl, c)
            del nc
            row = {}
            for w in windows:
                tc, tl = transfer_prefill(sm, tm, mapper, T(p), tail=tail, window=w)
                row[f"w{w or 'full'}"] = compare(ref, teacher_forced(tm, tc, tl, c), c)
                del tc
            sc, sl = native_prefill(sm, T(p))
            row["source_native"] = compare(ref, teacher_forced(sm, sc, sl, c), c)
            del sc
            rows.append(row)
            torch.cuda.empty_cache()
        res["lm"] = {k: mean_dicts([r[k] for r in rows]) for k in rows[0]}
        print(json.dumps(res["lm"], indent=1), flush=True)
    res["niah"] = {}
    for nk in num_keys:
        samples = evals.make_niah(ttok, niah_len, niah_n, seed=3, num_keys=nk)
        strategies = ([("target", None, tail), ("source", None, tail)] if baselines else []) +             [("transfer", w, t) for w in windows for t in (niah_tails or [tail])]
        for strat, w, t in strategies:
            key = f"k{nk}-" + (strat if strat != "transfer" else f"transfer_w{w or 'full'}_t{t}")
            r = evals.run_niah(ttok, make_generator(strat, sm, tm, mapper, ttok, t, w), samples=samples,
                               max_new_tokens=24)
            res["niah"][key] = {k: v for k, v in r.items() if k != "outputs"}
            res["niah"][key]["sample_outputs"] = r.get("outputs", [])[:3]
            print(f"[niah] {key} acc={r['acc']:.3f} {r.get('acc_by_depth')}", flush=True)
            torch.cuda.empty_cache()
    return res


def task_profile(src="4b", tgt="27b", mapper_name=None, lengths=(8192, 32768), tail=32, reps=3, chunk=8192,
                 window=None):
    from . import mapper as M
    from .qwen35 import delta_scan, delta_state

    sm, _ = get(src)
    tm, _ = get(tgt)
    mapper = load_mapper(mapper_name, sm, tm)
    res = {"gpu": torch.cuda.get_device_name()}
    # state-only scan matches the full kernel
    gen = torch.Generator(device="cuda").manual_seed(0)
    B, L, H, K, V = 1, 1000, 8, 128, 128
    k = torch.randn(B, L, H, K, device="cuda", generator=gen).bfloat16()
    v = torch.randn(B, L, H, V, device="cuda", generator=gen).bfloat16()
    g = -torch.rand(B, L, H, device="cuda", generator=gen) * 0.1
    beta = torch.rand(B, L, H, device="cuda", generator=gen).bfloat16()
    s0 = torch.randn(B, H, K, V, device="cuda", generator=gen) * 0.1
    with torch.no_grad():
        ref = delta_scan(k, k, v, g, beta, s0.clone())[1]
        got = delta_state(k, v, g, beta, s0.clone())
    res["state_only_maxrel"] = float((ref - got).abs().max() / ref.abs().max())
    from .qwen35 import STATS
    res["kernel_stats"] = dict(STATS)

    def timeit(fn):
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
            torch.cuda.empty_cache()
        return sorted(ts)[len(ts) // 2]

    rows = []
    for L in lengths:
        ids = torch.randint(1000, 100000, (1, L), device="cuda")
        row = {"len": L}
        row["native_tgt"] = timeit(lambda: native_prefill(tm, ids, chunk=chunk))
        row["native_src"] = timeit(lambda: native_prefill(sm, ids, chunk=chunk))
        for mode in ("src", "gemm_only", "gemm", None):
            M.PROFILE["mode"] = mode
            row[f"transfer_{mode or 'full'}"] = timeit(
                lambda: transfer_prefill(sm, tm, mapper, ids, tail=tail, window=window, chunk=chunk))
        M.PROFILE.clear()
        print(json.dumps(row), flush=True)
        rows.append(row)
    res["rows"] = rows
    return res


class RetrievalSynth:
    """Synthetic lookup tasks for distillation: 'entry ABC-123: the access code is 482913.' records planted in
    real text, then a chat-formatted question. Deliberately different from the eval NIAH (template, code length,
    filler) so the needle eval stays out of distribution."""

    def __init__(self, tok, seed=0):
        import random

        self.tok, self.rng = tok, random.Random(seed)
        ph = "XXPLACEHOLDERXX"
        try:
            t = tok.apply_chat_template([{"role": "user", "content": ph}], add_generation_prompt=True,
                                        enable_thinking=False, tokenize=False)
        except TypeError:
            t = tok.apply_chat_template([{"role": "user", "content": ph}], add_generation_prompt=True, tokenize=False)
        self.pre, self.post = t.split(ph)
        self.end = tok.convert_tokens_to_ids("<|im_end|>")

    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def make(self, body: list[int], n_records: int):
        r = self.rng
        letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        recs = []
        for _ in range(n_records):
            name = "".join(r.choice(letters) for _ in range(3)) + f"-{r.randint(100, 999)}"
            code = str(r.randint(100000, 999999))
            recs.append((name, code))
        q_name, q_code = r.choice(recs)
        body = list(body)
        for name, code in recs:
            pos = r.randint(0, len(body))
            body[pos:pos] = self.enc(f" Note: entry {name}: the access code is {code}. ")
        q = self.enc(f"\n\nAccording to the text above, what is the access code for entry {q_name}? "
                     f"Reply with the code only.") + self.enc(self.post)
        prompt = self.enc(self.pre) + body + q
        return prompt, self.enc(q_code) + [self.end]


def task_distill(src="4b", tgt="27b", mapper_name=None, out_name=None, steps=1500, prefix=1024, cont=128,
                 tail=1, lr=2e-5, warmup=50, clip=1.0, eval_every=250, n_eval=8, source="fineweb", window=None,
                 prefix_choices=None, retrieval_frac=0.0, niah_len=0, niah_n=10, eval_tail=None):
    """Fine-tune the ridge maps end to end: minimize KL(native target || transferred target) on continuations.

    Only the maps are trained; both models stay frozen. Gradients flow from the target's continuation logits
    through its attention over the synthesized KV and through the DeltaNet scans into the maps.
    """
    import random

    sm, stok = get(src)
    tm, ttok = get(tgt)
    mapper = load_mapper(mapper_name, sm, tm)
    params = mapper.trainable()
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.98), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * max(0.1, 0.5 * (1 + math.cos(math.pi * s / steps))))
    lens = prefix_choices or [prefix]
    train = evals.load_heldout_text(ttok, steps, max(lens) + cont, seed=11, split="calib", source=source)
    held = evals.load_heldout_text(ttok, n_eval, prefix + cont, seed=7, split="eval", source=source)
    rng = random.Random(0)

    def evaluate():
        rows = []
        for s in held:
            p, c = s[:prefix], s[prefix:]
            nc, nl = native_prefill(tm, T(p))
            ref = teacher_forced(tm, nc, nl, c)
            del nc
            tc, tl = transfer_prefill(sm, tm, mapper, T(p), tail=eval_tail, window=window)
            rows.append(compare(ref, teacher_forced(tm, tc, tl, c), c))
        return mean_dicts(rows)

    synth = RetrievalSynth(ttok, seed=1) if retrieval_frac else None
    eval_tail = eval_tail or tail

    def niah_eval():
        if not niah_len:
            return {}
        samples = evals.make_niah(ttok, niah_len, niah_n, seed=3, num_keys=1)
        r = evals.run_niah(ttok, make_generator("transfer", sm, tm, mapper, ttok, max(eval_tail, 128), window),
                           samples=samples, max_new_tokens=24)
        return {"niah_acc": r["acc"]}

    hist = [{"step": 0, **evaluate(), **niah_eval()}]
    print(json.dumps(hist[-1]), flush=True)
    t0 = time.time()
    run = 0.0
    for step, s in enumerate(train):
        P = rng.choice(lens)
        if synth is not None and rng.random() < retrieval_frac:
            p, c = synth.make(s[:P - 80], rng.randint(1, 4))
        else:
            p, c = s[:P], s[P:P + cont]
        with torch.no_grad():
            nc, nl = native_prefill(tm, T(p))
            ref = teacher_forced(tm, nc, nl, c)
            del nc
            lr_ = F.log_softmax(ref, -1)
        tc, tl = transfer_prefill(sm, tm, mapper, T(p), tail=tail, window=window, grad=True)
        with torch.enable_grad():
            lg = torch.cat((tl[0, -1:], tm.forward(T(c[:-1]), tc, logits="all")[0]), dim=0)
            lt = F.log_softmax(lg, -1)
            loss = (lr_.exp() * (lr_ - lt)).sum(-1).mean()
            loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        run = 0.98 * run + 0.02 * float(loss) if step else float(loss)
        if step % 25 == 0:
            print(f"[distill] step {step} kl {float(loss):.4f} ema {run:.4f} gn {float(gn):.3f} "
                  f"{time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
        if (step + 1) % eval_every == 0 or step + 1 == len(train):
            last = step + 1 == len(train)
            hist.append({"step": step + 1, **evaluate(), **(niah_eval() if last else {})})
            print(json.dumps(hist[-1]), flush=True)
    out_name = out_name or f"{mapper_name}-distill{steps}"
    mapper.save(ART / "maps" / f"{out_name}.pt")
    return {"name": out_name, "history": hist, "seconds": time.time() - t0}


TASKS = {k[5:]: v for k, v in globals().items() if k.startswith("task_")}


def dispatch(task: str, args: dict):
    t = time.time()
    out = TASKS[task](**args)
    out = {"task": task, "args": args, "wall_s": time.time() - t, "result": out}
    (ART / "results").mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (ART / "results" / f"{task}-{stamp}.json").write_text(json.dumps(out, indent=1, default=str))
    return out
