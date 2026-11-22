"""Extra src/xkv tasks and benches registered by modal_xkv.py (no edits to src/xkv).

Benches:
  mbpp_dev: MBPP "full" VALIDATION split (90 items, a held-out dev set: never in any test row), rendered and scored
            exactly like genevals' mbpp (EvalPlus instruction, chat template, assistant prefix, exec of test_list).
Tasks (name without the task_ prefix, as dispatch expects):
  agent_kl: KL(native || transfer) teacher-forced on agent-log answers (xkv agent_logs JSONL), per mapper, with the
            bench's transfer@cap tail (min(256, 25% of the prompt)); the held-out check of an on-policy distillation.
"""


def register():
    from xkv import experiments as E
    from xkv import genevals as G
    if "mbpp_dev" not in G._BENCH:
        def _build_mbpp_dev(tok, limit, seed, max_ctx, max_new=768):
            docs = G._load_hf("google-research-datasets/mbpp", "full", "validation")

            def build(i):
                d = dict(docs[i])
                d.setdefault("test_imports", [d["test_setup_code"]] if d.get("test_setup_code") else [])
                problem = f"{d['text']}\nYour code should pass these tests:\n\n" + "\n".join(d["test_list"])
                prompt = G.chat_prompt(tok, G._EVALPLUS_INSTR.format(problem=problem), assistant_prefix=G._code_prefix())
                return G._mk(tok, prompt, d["task_id"], max_new, oracle=d["code"] + "\n```", meta={"doc": d})
            return G._select(len(docs), build, limit, seed, max_ctx)
        G._BENCH["mbpp_dev"] = G.Bench("mbpp_dev", "pass@1", _build_mbpp_dev, G._score_mbpp)
        G.run_mbpp_dev = lambda gen, tok, limit=100, seed=0, max_ctx=None, batch=8, max_new=None: G._run(
            "mbpp_dev", gen, tok, limit, seed, max_ctx, batch, {"max_new": max_new})
    orig = E._extra_tasks
    if not getattr(orig, "_kvx", False):
        def extra():
            return {**orig(), "agent_kl": task_agent_kl}
        extra._kvx = True
        E._extra_tasks = extra


def task_agent_kl(src="4b", tgt="27b", mappers=(), log="", cont=256, limit=0):
    """Per mapper: mean KL / top-1 agreement / NLL of transfer@cap vs native on the answers of an agent log."""
    import json

    import torch

    from xkv.agentbench import render_prompt, to_template_messages
    from xkv.experiments import ART, T, compare, get, load_mapper, mean_dicts, teacher_forced
    from xkv.mapper import native_prefill, transfer_prefill
    sm, _ = get(src)
    tm, ttok = get(tgt)
    samples = []
    for line in (ART / log).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pre = render_prompt(ttok, r["messages"], r["tools"], False)
        full = ttok.apply_chat_template(to_template_messages(r["messages"] + [dict(r["out"], role="assistant")]),
                                        tools=r["tools"] or None, tokenize=False, add_generation_prompt=False,
                                        enable_thinking=False)
        assert full.startswith(pre)
        samples.append((ttok.encode(pre, add_special_tokens=False),
                        ttok.encode(full[len(pre):], add_special_tokens=False)[:cont]))
    samples = samples[:limit] if limit else samples
    out = {}
    with torch.no_grad():
        refs = []
        for p, c in samples:
            nc, nl = native_prefill(tm, T(p))
            refs.append(teacher_forced(tm, nc, nl, c).cpu())
            del nc
        for name in mappers:
            mp = load_mapper(name, sm, tm)
            rows = []
            for (p, c), ref in zip(samples, refs):
                tail = max(1, min(256, len(p) // 4))
                tc, tl = transfer_prefill(sm, tm, mp, T(p), tail=tail)
                rows.append(compare(ref.to(tl.device), teacher_forced(tm, tc, tl, c), c))
                del tc
            out[name] = dict(n=len(rows), **mean_dicts(rows))
            del mp
            torch.cuda.empty_cache()
    return out
