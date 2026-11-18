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
                   full_until=None, turn_tail_max=None):
    """genevals GenFn: batched greedy decoding after a per-prompt prefill with the given strategy."""
    from .decode import Decoder
    from .mapper import adaptive_tail, native_prefill, transfer_prefill

    stops = stop_ids(tok)
    sink, oracle, repair, rstate = 0, None, 0, "final"
    if "+rep" in strategy:  # e.g. transfer@auto+rep1024 (z suffix: re-encode from a zero DeltaNet state)
        strategy, r = strategy.split("+rep")
        rstate = "zero" if r.endswith("z") else "final"
        repair = int(r.rstrip("z"))
    if strategy.startswith("transfer") and "+oracle_" in strategy:  # diagnostic: transfer+oracle_attn|lin|both
        strategy, oracle = strategy.split("+oracle_")[0], strategy.split("+oracle_")[1]
    if strategy.startswith("transfer") and ":s" in strategy:  # e.g. transfer:s1 -> first token computed natively
        strategy, sink = strategy.split(":s")[0], int(strategy.split(":s")[1])
    if strategy.startswith("transfer@"):  # e.g. transfer@16: only the last 16 prompt tokens run natively
        strategy, tail = "transfer", strategy.split("@")[1]  # transfer@auto: mapper.adaptive_tail
        tail = tail if tail == "auto" else int(tail)
    model = sm if strategy == "source" else tm
    im_start = tok.convert_tokens_to_ids("<|im_start|>")

    def tail_for(p):
        """turn_tail_max: the target natively reads the whole newest turn (from the last non-assistant message
        start), up to turn_tail_max tokens; the mapped part is the system prompt and older history."""
        base = adaptive_tail(len(p)) if tail == "auto" else tail
        if not turn_tail_max:
            return base
        starts = [i for i, t in enumerate(p) if t == im_start]
        if len(starts) < 2:
            return base
        return max(base, min(turn_tail_max, len(p) - starts[-2]))

    def gen(prompts, max_new, spans=None):
        """spans: optional per-prompt list of (a, b) token ranges the target computes natively (transfer only)."""
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        out = [None] * len(prompts)
        for g0 in range(0, len(order), batch):
            idx = order[g0:g0 + batch]
            cap = _bucket(max(len(prompts[i]) for i in idx) + max_new + 16)
            dec = Decoder(model, len(idx), cap)
            for b, i in enumerate(idx):
                ids = torch.tensor([prompts[i]], device="cuda")
                if strategy == "transfer":
                    c, lg = transfer_prefill(sm, tm, mapper, ids, tail=min(tail_for(prompts[i]), ids.shape[1]), window=window,
                                             chunk=chunk, full_until=full_until, sink=sink,
                                             native_spans=spans[i] if spans else None, oracle=oracle,
                                             repair=repair, repair_state=rstate,
                                             graph_tail=False if oracle else None)
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
                   batch=8, max_ctx=None, bench_kw=None, full_until=None, exec_modal=False, limits=None,
                   env=None):
    from . import genevals as G
    from .experiments import ART, get, load_mapper

    os.environ.setdefault("XKV_GENEVALS_CACHE", str(ART / "genevals_cache"))
    if exec_modal:  # BigCodeBench tests run in the official evaluation image (modal_jobs/exec_app.py)
        os.environ["XKV_EXEC_MODAL"] = "1"
    limits = limits or {}
    for k, v in (env or {}).items():
        os.environ[k] = str(v)
    sm, _ = get(src)
    tm, tok = get(tgt)
    mapper = load_mapper(mapper_name, sm, tm) if any(s.startswith("transfer") for s in strategies) else None
    res_env = dict(env or {})
    bench_kw = bench_kw or {}
    res = {"config": {"mapper": mapper_name, "tail": tail, "window": window, "full_until": full_until,
                      "limit": limit, "seed": seed, "env": res_env}}
    for bname in benches:
        res[bname] = {}
        for strat in strategies:
            t0 = time.time()
            gen = make_batch_gen(strat, sm, tm, mapper, tok, tail=tail, window=window,
                                 batch=bench_kw.get(bname, {}).get("batch", batch), full_until=full_until)
            kw = {k: v for k, v in bench_kw.get(bname, {}).items() if k != "batch"}
            try:
                r = getattr(G, f"run_{bname}")(gen, tok, limit=limits.get(bname, limit), seed=seed, max_ctx=max_ctx,
                                               **kw)
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
