"""Run the generation benchmarks (genevals) for native target, source alone, and transfer."""
from __future__ import annotations

import json
import os
import time

import torch

from .perf import _bucket


def stop_ids(tok):
    ids = set()
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            ids.add(i)
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return ids


def make_batch_gen(strategy, sm, tm, mapper, tok, tail=128, window=8192, batch=8, chunk=8192, log=None,
                   full_until=None):
    """genevals GenFn: batched greedy decoding after a per-prompt prefill with the given strategy."""
    from .decode import Decoder
    from .mapper import native_prefill, transfer_prefill

    stops = stop_ids(tok)
    if strategy.startswith("transfer@"):  # e.g. transfer@16: only the last 16 prompt tokens run natively
        strategy, tail = "transfer", int(strategy.split("@")[1])
    model = sm if strategy == "source" else tm

    def gen(prompts, max_new):
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        out = [None] * len(prompts)
        for g0 in range(0, len(order), batch):
            idx = order[g0:g0 + batch]
            cap = _bucket(max(len(prompts[i]) for i in idx) + max_new + 16)
            dec = Decoder(model, len(idx), cap)
            for b, i in enumerate(idx):
                ids = torch.tensor([prompts[i]], device="cuda")
                if strategy == "transfer":
                    c, lg = transfer_prefill(sm, tm, mapper, ids, tail=min(tail, ids.shape[1]), window=window,
                                             chunk=chunk, full_until=full_until)
                else:
                    c, lg = native_prefill(model, ids, chunk=chunk)
                dec.load(b, c, int(lg[0, -1].argmax()))
                del c
            outs, _ = dec.generate(max_new, stops)
            for b, i in enumerate(idx):
                out[i] = outs[b]
            del dec
            torch.cuda.empty_cache()
            if log:
                log(f"{strategy} {min(g0 + batch, len(order))}/{len(order)}")
        return out

    return gen


def task_gen_evals(src="4b", tgt="27b", mapper_name="4b-27b-w8192-distill-ret", benches=("arc_chat",),
                   strategies=("target", "transfer", "source"), limit=100, seed=0, tail=128, window=8192,
                   batch=8, max_ctx=None, bench_kw=None, full_until=None):
    from . import genevals as G
    from .experiments import ART, get, load_mapper

    os.environ.setdefault("XKV_GENEVALS_CACHE", str(ART / "genevals_cache"))
    sm, _ = get(src)
    tm, tok = get(tgt)
    mapper = load_mapper(mapper_name, sm, tm) if any(s.startswith("transfer") for s in strategies) else None
    bench_kw = bench_kw or {}
    res = {"config": {"mapper": mapper_name, "tail": tail, "window": window, "full_until": full_until,
                      "limit": limit, "seed": seed}}
    for bname in benches:
        res[bname] = {}
        for strat in strategies:
            t0 = time.time()
            gen = make_batch_gen(strat, sm, tm, mapper, tok, tail=tail, window=window,
                                 batch=bench_kw.get(bname, {}).get("batch", batch), full_until=full_until)
            kw = {k: v for k, v in bench_kw.get(bname, {}).items() if k != "batch"}
            try:
                r = getattr(G, f"run_{bname}")(gen, tok, limit=limit, seed=seed, max_ctx=max_ctx, **kw)
            except Exception as e:  # keep the other benchmarks going
                import traceback

                r = {"error": repr(e)[:500], "tb": traceback.format_exc()[-3000:]}
            r["seconds"] = time.time() - t0
            res[bname][strat] = r
            short = {k: v for k, v in r.items() if k in ("score", "n", "n_scored", "n_skipped", "error", "seconds")}
            print(f"[gen] {bname} {strat} {json.dumps(short)}", flush=True)
            (ART / "results").mkdir(parents=True, exist_ok=True)
            (ART / "results" / f"gen_partial_{bname}.json").write_text(json.dumps(res[bname], default=str))
    return res
