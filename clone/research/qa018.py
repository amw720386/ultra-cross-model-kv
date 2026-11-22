"""Note 018 replication: Qwen3-4B -> Ministral-3-14B cross-family transfer, answer F1 on 128 held-out questions
(64 HotpotQA + 64 Qasper; theirs: original transfer 53.59, 256-token replay 58.63, native receiver 63.16, SDPA).

Ours (labelled): LongBench v1 hotpotqa / qasper test items (first 64 each whose prompt fits 16K target tokens), the
official LongBench prompt templates and qa_f1 scorer, greedy decoding (64 new tokens), chat template of the target.
The context segment is read by the source and mapped (research/xalign contract); the template prefix and the question
suffix run natively on the target. Arms: native, mapped (strict), replayN (last N context tokens recomputed natively,
their 256-token reread), sink4+replay256. Maps: research/xfit ridge on FineWeb + PG19 text (plus optional mcmix-style
task text is NOT used here: generic only, as theirs).
"""
import collections, io, json, os, re, string, time, urllib.request, zipfile

import torch

from kvx import core
from research import mc128, xfit

LB_URL = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
PROMPTS = {
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
                "The following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me "
                "the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single "
              "phrase or sentence if possible. If the question cannot be answered based on the information in the article, "
              "write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not "
              "provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely "
              "as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the "
              "information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", "
              "\"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
}


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    pr, rc = ns / len(p), ns / len(g)
    return 2 * pr * rc / (pr + rc)


def longbench(task, root="/vol/data"):
    path = os.path.join(root, "longbench_data.zip")
    if not os.path.exists(path):
        os.makedirs(root, exist_ok=True)
        req = urllib.request.Request(LB_URL, headers={"User-Agent": "Mozilla/5.0"})
        open(path + ".part", "wb").write(urllib.request.urlopen(req, timeout=600).read())
        os.replace(path + ".part", path)
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.endswith(f"/{task}.jsonl") or n == f"{task}.jsonl"][0]
        return [json.loads(l) for l in io.TextIOWrapper(z.open(name), encoding="utf-8")]


def split_prompt(task, ex):
    """(head_text, context_text, tail_text): head and tail native, context mapped."""
    t = PROMPTS[task]
    i = t.index("{context}")
    return t[:i], ex["context"], t[i + len("{context}"):].replace("{input}", ex["input"])


@torch.no_grad()
def greedy(P, cache, ids, max_new=64, eos=()):
    """Continue from `cache` (covering a prefix) with `ids` (the rest of the prompt), greedy max_new tokens."""
    out = []
    x = torch.tensor([ids], device=P.dev)
    for _ in range(max_new):
        lg = P.T(input_ids=x, past_key_values=cache, use_cache=True).logits[:, -1]
        t = int(lg.argmax(-1))
        if t in eos:
            break
        out.append(t)
        x = torch.tensor([[t]], device=P.dev)
    return out


@torch.no_grad()
def answer(P, head_ids, ctx_text, tail_ids, arm, replay=0, sink=0, max_new=64, eos=()):
    from transformers import DynamicCache
    if arm == "native":
        s_ids, c_ids, idx = P.enc(ctx_text, 10 ** 7)
        cache = DynamicCache()
        ids = head_ids + c_ids + tail_ids
        return greedy(P, cache, ids, max_new, eos)
    s_ids, c_ids, idx = P.enc(ctx_text, 10 ** 7)
    n_map = len(c_ids) - replay
    Kh, Vh = P.native(torch.tensor([head_ids], device=P.dev)) if head_ids else (None, None)
    Ks, Vs, _ = core.capture_kv(P.S, torch.tensor([s_ids], device=P.dev), P.cs, P.ss)
    ii = idx[:n_map].to(P.dev)
    Km, Vm = P.mapped(Ks[:, :, :, ii], Vs[:, :, :, ii], P.k_use, "none" if P.kw is None else "rms")
    if sink:
        Kn, Vn = P.native(torch.tensor([head_ids + c_ids[:sink]], device=P.dev))
        Km[:, :, :, :sink], Vm[:, :, :, :sink] = Kn[:, :, :, len(head_ids):], Vn[:, :, :, len(head_ids):]
    K = torch.cat([Kh, Km], 3) if Kh is not None else Km
    V = torch.cat([Vh, Vm], 3) if Vh is not None else Vm
    cache = core.build_cache(K, V, P.ct, P.st)
    return greedy(P, cache, c_ids[n_map:] + tail_ids, max_new, eos)


def run_pair(src="Qwen/Qwen3-4B", tgt="mistralai/Ministral-3-14B-Instruct-2512-BF16", n_calib=1000, L=1024, ks="8,16,24",
             out="", models=None, n_per=64, max_ctx=16000, recipes="r256,s4r256", **_):
    from transformers import AutoTokenizer
    from kvx.models import model_path
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    tok_s, tok_t = AutoTokenizer.from_pretrained(model_path(src)), AutoTokenizer.from_pretrained(model_path(tgt))
    S, T = models if models else (mc128.load(src, dev, torch.bfloat16), mc128.load(tgt, dev, torch.bfloat16))
    bos_t = tok_t.bos_token_id
    P = xfit.XPair(S, T, tok_s, tok_t, dev, bos_s=None, bos_t=bos_t, Lmax=max_ctx + 2048)
    P.src, P.tgt = src, tgt
    klist = [min(int(k), P.Ls) for k in str(ks).split(",")]
    texts = xfit.texts_fineweb(n_calib // 2, stop=mc128.EVAL_SKIP) + xfit.texts_pg19(n_calib - n_calib // 2)
    res = dict(src=src, tgt=tgt, n_calib=len(texts), L=L, ks=klist, n_per=n_per, max_ctx=max_ctx,
               format="LongBench v1 hotpotqa/qasper prompts + qa_f1, chat template, greedy 64")
    res["r2"] = P.fit_texts(texts, klist, L)
    ev = xfit.texts_fineweb(16, skip=mc128.EVAL_SKIP)
    res["heldout"] = P.heldout_x(ev, L - 64, 64, klist, "none" if P.kw is None else "rms")
    P.k_use = min(klist, key=lambda k: res["heldout"][f"map_k{k}"]["kl"])
    res["k_selected"] = P.k_use
    print(f"[{time.time()-t0:.0f}s] fit k={P.k_use} heldout {json.dumps(res['heldout'])}", flush=True)
    msg = tok_t.apply_chat_template([{"role": "user", "content": "XXXX"}], tokenize=False, add_generation_prompt=True)
    pre_txt, post_txt = msg.split("XXXX")
    eos = {tok_t.eos_token_id} | ({tok_t.convert_tokens_to_ids("</s>")} - {None})
    arms = [("native", 0, 0), ("mapped", 0, 0)] + [(r, int(r.split("r")[-1]), int(r[1:r.index("r")]) if r.startswith("s") else 0)
                                                  for r in recipes.split(",") if r]
    scores = {a: {} for a, _, _ in arms}
    gens = {a: [] for a, _, _ in arms}
    for task in ("hotpotqa", "qasper"):
        data = longbench(task)
        used = 0
        for ex in data:
            head, ctx, tail = split_prompt(task, ex)
            head_ids = tok_t(pre_txt + head, add_special_tokens=False)["input_ids"]
            tail_ids = tok_t(tail + post_txt, add_special_tokens=False)["input_ids"]
            n_ctx = len(tok_t(ctx, add_special_tokens=False)["input_ids"])
            if n_ctx + len(head_ids) + len(tail_ids) > max_ctx:
                continue
            for a, rp, sk in arms:
                g = answer(P, head_ids, ctx, tail_ids, "native" if a == "native" else "mapped", rp, sk, 64, eos)
                text = tok_t.decode(g, skip_special_tokens=True).strip()
                f1 = max(qa_f1(text, ans) for ans in ex["answers"])
                scores[a].setdefault(task, []).append(f1)
                gens[a].append(dict(task=task, id=ex.get("_id"), pred=text, f1=f1))
            used += 1
            if used >= n_per:
                break
        print(f"[{time.time()-t0:.0f}s] {task}: " + json.dumps({a: round(100 * sum(v[task]) / len(v[task]), 2) for a, v in scores.items()}), flush=True)
    for a in scores:
        per = {t: 100 * sum(v) / len(v) for t, v in scores[a].items()}
        res[a] = dict(macro_f1=sum(per.values()) / len(per), by_task=per, n=sum(len(v) for v in scores[a].values()))
    res["t_total"] = time.time() - t0
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        json.dump(res, open(out, "w"), indent=1)
        json.dump(gens, open(out.replace(".json", "_items.json"), "w"))
    return res
