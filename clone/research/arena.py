"""Note 020 replication: 100 Chatbot Arena prompts read by Qwen3-1.7B, answered by Qwen3-8B through a mapped cache.

Their protocol (onetriangle.ai/blog/arena-prompts-through-a-translated-cache, data chat_arena_1.7B-to-8B.csv): one
sampled, seeded generation per arm per prompt, thinking disabled, 2,048-token budget, loop = generation stopped for a
repeated tail block (research/loops.py reproduces their flags on 499/500 of their published replies). Ours: same 100
prompts, Qwen3 non-thinking sampling defaults (T 0.7, top-p 0.8, top-k 20), arms: mapped (strict: target runs only
the last prompt token), recipe sAtN (native first A + last N prompt tokens), native 8B. Pairwise judging vs their
arms is done offline (research/arena_judge.py).
"""
import csv, io, json, os, time, urllib.request

import torch

from research import mc128, genmap
from research.loops import tail_loop

CSV_URL = "https://onetriangle.ai/research/october-2026/chat_arena_1.7B-to-8B.csv"


def arena_prompts():
    csv.field_size_limit(10 ** 9)
    req = urllib.request.Request(CSV_URL, headers={"User-Agent": "Mozilla/5.0"})
    rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(req, timeout=60).read().decode())))
    return [(r["id"], r["prompt"]) for r in rows]


def run_pair(src="Qwen/Qwen3-1.7B", tgt="Qwen/Qwen3-8B", n_calib=1000, L=512, ks="16,24", calib_src="mix",
             recipes="s4t64", out="", models=None, one_pass=0, bs=16, max_new=2048, seed=0, restore="rms", **_):
    from transformers import AutoTokenizer
    from kvx.models import model_path
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_path(tgt))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    S, T = models if models else (mc128.load(src, dev, torch.bfloat16), mc128.load(tgt, dev, torch.bfloat16))
    P = mc128.Pair(S, T, tok, dev, Lmax=8192)
    klist = [min(int(k), P.Ls) for k in str(ks).split(",")]
    res = dict(src=src, tgt=tgt, n_calib=n_calib, L=L, ks=klist, calib_src=calib_src, recipes=recipes, seed=seed,
               sampling=dict(temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=max_new))
    res["r2"] = P.fit(mc128.calibration(tok, n_calib, L, calib_src), klist, 1e-3, 4, one_pass=bool(one_pass))
    evals = mc128.fineweb(tok, 16, L, skip=mc128.EVAL_SKIP)
    res["heldout"] = mc128.heldout(P, evals, L - 64, klist, restore, direct=False)
    k = min(klist, key=lambda k_: res["heldout"][f"map_k{k_}"]["kl"])
    res["k_selected"] = k
    print(f"[{time.time()-t0:.0f}s] fit done k={k} heldout {json.dumps(res['heldout'])}", flush=True)
    items = arena_prompts()
    prompts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True, add_generation_prompt=True,
                                       enable_thinking=False) for _, p in items]
    res["prompt_tokens"] = dict(mean=sum(map(len, prompts)) / len(prompts), max=max(map(len, prompts)))

    def mapped_fn(ids):
        Ks, Vs = P.src_kv(ids)
        return P.mapped(Ks, Vs, k, restore)

    def recipe_fn(A):
        def f(ids):
            Km, Vm = mapped_fn(ids)
            if A:
                Ka, Va = P.native(ids[:, :A])
                Km[:, :, :, :A], Vm[:, :, :, :A] = Ka, Va
            return Km, Vm
        return f

    arms = [("native8b", None, 1), ("mapped", mapped_fn, 1)]
    for r in [x for x in recipes.split(",") if x]:
        A_, N_ = (int(x) for x in r[1:].split("t"))
        arms.append((f"recipe_{r}", recipe_fn(A_), N_))
    gen = {}
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    for name, fn, tail in arms:
        t1 = time.time()
        torch.manual_seed(seed)
        # prompts shorter than the tail: let the target read them natively (tail = len - 1)
        outs = []
        for b0 in range(0, len(prompts), bs):
            chunk = prompts[b0:b0 + bs]
            tl = min(tail, min(map(len, chunk)) - 1)
            outs += genmap.generate(P, chunk, fn, tail=tl, bs=bs, max_new_tokens=max_new, do_sample=True,
                                    temperature=0.7, top_p=0.8, top_k=20, eos_token_id=eos, pad_token_id=tok.pad_token_id)
        texts, loops, lens = [], [], []
        for o in outs:
            o = [t for t in o if t != tok.pad_token_id] if tok.pad_token_id not in eos else o
            cut = len(o)
            for j, t in enumerate(o):
                if t in eos:
                    cut = j
                    break
            o = o[:cut]
            lens.append(len(o)); loops.append(int(tail_loop(o, min_block=8, reps=3)))
            texts.append(tok.decode(o, skip_special_tokens=True))
        gen[name] = dict(texts=texts, loops=loops, lens=lens)
        res[name] = dict(loop_rate=sum(loops) / len(loops), mean_len=sum(lens) / len(lens), t_gen=time.time() - t1)
        print(f"[{time.time()-t0:.0f}s] {name}: loop {res[name]['loop_rate']:.2f} mean_len {res[name]['mean_len']:.0f}", flush=True)
    res["t_total"] = time.time() - t0
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        json.dump(res, open(out, "w"), indent=1)
        json.dump(dict(ids=[i for i, _ in items], **gen), open(out.replace(".json", "_items.json"), "w"))
    return res
