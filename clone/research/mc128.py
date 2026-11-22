"""Replicate OneTriangle research notes 001 / 016 / 019 with the kvx ridge mapper.

001 (onetriangle.ai/research/kv-cache-transfer-v0): Qwen3-0.6B -> 1.7B, held-out next-token top-1 agreement
    (theirs 61% learned k=1, 2% direct injection), mapped/oracle perplexity ratio (1.1402), HellaSwag / PIQA
    (54.6 / 70.2 vs oracle 59.4 / 74.0), chance-normalized retention (85.1%).
016 / 019 (two-way-qwen-memory.json, kv-memory-ablation.json): ten Qwen3 directions, 128 questions
    (32 each ARC-Challenge, MMLU-Redux, MedQA, OpenBookQA; inputs > 512 tokens excluded), teacher-forced summed
    log-prob of the answer-label continuation. Arms: native, mapped (both translated), native K + mapped V,
    mapped K + native V, wrong cache. Theirs: means 72.11 / 43.67 / 49.14 / 56.02; 4B->8B 71.88, 8B->4B 68.75.

Protocol (ours): the prompt minus its last token comes from the source through the maps; the target runs the
last prompt token and the label natively (the minimum any prefix handoff needs to score a continuation).
Maps: per-target-layer top-k ridge (kvx.core), FineWeb-Edu calibration only (never the test sets), k chosen on
held-out FineWeb KL, rms key restore (Qwen3 k_norm).
"""
import argparse, json, math, os, random, time

import torch
import torch.nn.functional as F

from kvx import core

EVAL_SKIP = 20000  # FineWeb doc index where held-out docs start (calibration stops before it)
LETTERS = "ABCDEFGH"


# ------------------------------------------------------------------ data

def fineweb(tok, n, L, skip=0, stop=None):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    out = []
    for i, ex in enumerate(ds):
        if i < skip:
            continue
        if stop is not None and i >= stop:
            raise RuntimeError(f"calibration reached doc {stop} (eval range) with {len(out)}/{n} seqs")
        ids = tok(ex["text"], add_special_tokens=False)["input_ids"]
        if len(ids) >= L:
            out.append(ids[:L])
            if len(out) >= n:
                break
    return torch.tensor(out)


def pg19_train(tok, n, L, per_book=32):
    """Calibration chunks from PG19 TRAIN books (never a test set)."""
    from datasets import load_dataset
    out = []
    for ex in load_dataset("emozilla/pg19", split="train", streaming=True):
        ids = tok(ex["text"][: L * per_book * 6], add_special_tokens=False)["input_ids"]
        for c in range(min(per_book, len(ids) // L)):
            out.append(ids[c * L:(c + 1) * L])
            if len(out) >= n:
                return torch.tensor(out)
    return torch.tensor(out)


def mc_train_text(seed=0):
    """MC-formatted TRAIN-split questions with their answers (ARC-C/ARC-E/OBQA/MedQA train, MMLU auxiliary_train).
    Disjoint from every test item scored here; the prompt format equals the eval format."""
    from datasets import load_dataset
    out = []
    for cfg in ("ARC-Challenge", "ARC-Easy"):
        for ex in load_dataset("allenai/ai2_arc", cfg, split="train"):
            lab = ex["choices"]["label"]
            if ex["answerKey"] in lab:
                out.append((ex["question"], ex["choices"]["text"], lab.index(ex["answerKey"])))
    for ex in load_dataset("allenai/openbookqa", "main", split="train"):
        out.append((ex["question_stem"], ex["choices"]["text"], ex["choices"]["label"].index(ex["answerKey"])))
    for ex in load_dataset("GBaker/MedQA-USMLE-4-options", split="train"):
        keys = sorted(ex["options"])
        out.append((ex["question"], [ex["options"][k] for k in keys], keys.index(ex["answer_idx"])))
    aux = load_dataset("cais/mmlu", "auxiliary_train", split="train").shuffle(seed=seed).select(range(20000))
    for ex in aux:
        ex = ex.get("train", ex)
        out.append((ex["question"], ex["choices"], int(ex["answer"])))
    random.Random(seed).shuffle(out)
    return [_mc_prompt(q, ch) + " " + LETTERS[g] + "\n\n" for q, ch, g in out]


def mc_train_seqs(tok, n, L, seed=0):
    ids, out = [], []
    for t in mc_train_text(seed):
        ids += tok(t, add_special_tokens=False)["input_ids"]
        while len(ids) >= L:
            out.append(ids[:L]); ids = ids[L:]
            if len(out) >= n:
                return torch.tensor(out)
    return torch.tensor(out)


def calibration(tok, n, L, src="fineweb"):
    if src == "mcmix":  # one third each: FineWeb-Edu, PG19-train, MC-format train questions (task-domain calibration)
        a = n // 3
        return torch.cat([fineweb(tok, a, L, stop=EVAL_SKIP), pg19_train(tok, a, L), mc_train_seqs(tok, n - 2 * a, L)])
    if src == "mix":  # kvx default recipe: half FineWeb-Edu, half PG19-train books
        return torch.cat([fineweb(tok, n // 2, L, stop=EVAL_SKIP), pg19_train(tok, n - n // 2, L)])
    return fineweb(tok, n, L, stop=EVAL_SKIP)


def _mc_prompt(q, choices):
    return "Question: " + q.strip() + "\n" + "".join(f"{LETTERS[i]}. {c.strip()}\n" for i, c in enumerate(choices)) + "Answer:"


def _arc():
    from datasets import load_dataset
    for ex in load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"):
        labels = ex["choices"]["label"]
        if ex["answerKey"] in labels:
            yield ex["id"], ex["question"], ex["choices"]["text"], labels.index(ex["answerKey"])


def _obqa():
    from datasets import load_dataset
    for ex in load_dataset("allenai/openbookqa", "main", split="test"):
        labels = ex["choices"]["label"]
        yield ex["id"], ex["question_stem"], ex["choices"]["text"], labels.index(ex["answerKey"])


def _medqa():
    from datasets import load_dataset
    for i, ex in enumerate(load_dataset("GBaker/MedQA-USMLE-4-options", split="test")):
        keys = sorted(ex["options"])
        yield f"medqa-{i}", ex["question"], [ex["options"][k] for k in keys], keys.index(ex["answer_idx"])


MMLU_REDUX_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics", "college_medicine", "college_physics",
    "computer_security", "conceptual_physics", "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics", "high_school_physics",
    "high_school_psychology", "high_school_statistics", "high_school_us_history", "high_school_world_history",
    "human_aging", "human_sexuality", "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
    "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions"]


def _mmlu_redux(seed):
    """One clean ('ok') item per subject in a seeded subject order, round-robin."""
    from datasets import load_dataset
    subs = list(MMLU_REDUX_SUBJECTS)
    random.Random(seed).shuffle(subs)
    pools = {}
    r = 0
    while True:
        for s in subs:
            if s not in pools:
                ds = load_dataset("edinburgh-dawg/mmlu-redux-2.0", s, split="test")
                pools[s] = [ex for ex in ds if ex.get("error_type", "ok") == "ok"]
                random.Random(seed).shuffle(pools[s])
            if r < len(pools[s]):
                ex = pools[s][r]
                yield f"mmlu-{s}-{r}", ex["question"], ex["choices"], int(ex["answer"])
        r += 1


def mc_items(tok, per_task=32, max_tokens=512, seed=0):
    """128 questions = 32 per task; seeded shuffle for ARC/OBQA/MedQA; prompts > max_tokens excluded."""
    out = []
    for task, gen in (("arc_challenge", _arc()), ("mmlu_redux", _mmlu_redux(seed)), ("medqa", _medqa()),
                      ("openbookqa", _obqa())):
        items = list(gen) if task != "mmlu_redux" else None
        if items is not None:
            random.Random(seed).shuffle(items)
            gen = iter(items)
        got = 0
        for qid, q, ch, gold in gen:
            p = _mc_prompt(q, ch)
            if len(tok(p, add_special_tokens=False)["input_ids"]) > max_tokens:
                continue
            out.append(dict(task=task, id=qid, ctx=p, conts=[" " + LETTERS[i] for i in range(len(ch))], gold=gold))
            got += 1
            if got >= per_task:
                break
    return out


def hellaswag_items(n):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    out = []
    for ex in ds.select(range(n)):
        ctx = ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
        out.append(dict(task="hellaswag", ctx=ctx, conts=[" " + e for e in ex["endings"]], gold=int(ex["label"])))
    return out


def piqa_items(n):
    from datasets import load_dataset
    ds = load_dataset("ybisk/piqa", split="validation", revision="refs/convert/parquet")
    out = []
    for ex in ds.select(range(n)):
        out.append(dict(task="piqa", ctx="Question: " + ex["goal"] + "\nAnswer:",
                        conts=[" " + ex["sol1"], " " + ex["sol2"]], gold=int(ex["label"])))
    return out


# ------------------------------------------------------------------ mapping

class Pair:
    def __init__(self, S, T, tok, dev, Lmax=2048):
        self.S, self.T, self.tok, self.dev = S, T, tok, dev
        self.cs, self.ss = core.rope_cos_sin(S, Lmax, dev)
        self.ct, self.st = core.rope_cos_sin(T, Lmax, dev)
        c = T.config
        self.Ht, self.Dt = c.num_key_value_heads, getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        self.Ls, self.Lt = S.config.num_hidden_layers, c.num_hidden_layers
        cs_ = S.config
        self.Hs, self.Ds = cs_.num_key_value_heads, getattr(cs_, "head_dim", None) or cs_.hidden_size // cs_.num_attention_heads
        try:
            self.kw = [l.self_attn.k_norm.weight.detach().float() for l in T.model.layers]
        except AttributeError:
            self.kw = None
        self.maps = {}

    def fit(self, calib, ks, lam_rel=1e-3, stride=4, bs=4, one_pass=False):
        """one_pass: K and V stats from the same forwards (2 Grams in memory); else K then V (one Gram at a time)."""
        r2 = {}
        pending = {}
        for kind in ("k", "v"):
          if kind in pending:
            st = pending.pop(kind)
          else:
            kinds = ("k", "v") if one_pass else (kind,)
            sts = {kk: core.GramStats(self.Ls, self.Hs * self.Ds, self.Lt, self.Ht * self.Dt, self.dev) for kk in kinds}
            for i in range(0, len(calib), bs):
                ids = calib[i:i + bs].to(self.dev)
                with torch.no_grad():
                    Ks, Vs, _ = core.capture_kv(self.S, ids, self.cs, self.ss)
                    Kt, Vt, _ = core.capture_kv(self.T, ids, self.ct, self.st)
                for kk in kinds:
                    X = (Ks if kk == "k" else Vs)[:, :, :, ::stride]
                    Y = (Kt if kk == "k" else Vt)[:, :, :, ::stride]
                    sts[kk].add(core.flat_tokens(X).reshape(-1, self.Ls * self.Hs * self.Ds).float(),
                                core.flat_tokens(Y).reshape(-1, self.Lt * self.Ht * self.Dt).float())
                    del X, Y
                del Ks, Vs, Kt, Vt
            st = sts.pop(kind)
            pending.update(sts)
          if True:
            st.finalize()
            R2 = core.layer_r2(st, 0.0, self.Ht, lam_rel)
            r2[kind] = {"best_single": R2.max(-1).values.mean().item()}
            for k in ks:
                ms = core.fit_maps(st, R2, k, 0.0, self.Ht, lam_rel)
                for m in ms:  # bf16 weights: maps for several k (Lt x k*ds x dt each) otherwise dominate GPU memory
                    m["W"] = m["W"].to(torch.bfloat16)
                self.maps[(kind, k)] = ms
                r2[kind][f"k{k}"] = torch.stack([m["r2"] for m in ms]).mean().item()
                if self.dev != "cpu":
                    torch.cuda.empty_cache()
            del st, R2
        return r2

    def src_kv(self, ids):
        Ks, Vs, _ = core.capture_kv(self.S, ids, self.cs, self.ss)
        return Ks, Vs

    @staticmethod
    def apply(maps, Xs):
        """core.apply_maps with bf16-stored W (fp32 accumulate via float upcast per layer)."""
        Ls, B, H, T, D = Xs.shape
        X = core.flat_tokens(Xs).float()
        outs = []
        for m in maps:
            Y = X[:, m["sel"]].reshape(B * T, -1) @ m["W"].float() + m["b"]
            outs.append(Y.view(B, T, m["H"], m["D"]).permute(0, 2, 1, 3))
        return torch.stack(outs)

    def mapped(self, Ks, Vs, k, restore="rms"):
        Km = self.apply(self.maps[("k", k)], Ks)
        if k in getattr(self, "norestore", ()):  # maps trained without the key restore (e.g. distilled from direct)
            restore = "none"
        if restore != "none" and not (restore == "rms" and self.kw is None):
            Km = core.restore_scale(Km, self.maps[("k", k)], restore, self.Ht, self.kw)
        return Km, self.apply(self.maps[("v", k)], Vs).float()

    def direct(self, Ks, Vs):
        """Direct injection: source layer at the nearest normalized depth, no learned map (needs equal H, D)."""
        assert (self.Hs, self.Ds) == (self.Ht, self.Dt), "direct injection needs equal KV head shapes"
        idx = [min(self.Ls - 1, round(t * (self.Ls - 1) / max(1, self.Lt - 1))) for t in range(self.Lt)]
        return Ks[idx].float(), Vs[idx].float()

    def native(self, ids):
        Kt, Vt, _ = core.capture_kv(self.T, ids, self.ct, self.st)
        return Kt.float(), Vt.float()


# ------------------------------------------------------------------ scoring

@torch.no_grad()
def score_choices(P, ctx_ids, conts_ids, kv, nt=1):
    """kv: None (native full prompt) or (K, V) pre-RoPE for ctx_ids[:-nt]; the target runs ctx_ids[-nt:] + the
    continuation natively. Returns summed log-prob per continuation."""
    n = len(conts_ids)
    pad = P.tok.pad_token_id or 0
    if kv is None:
        seqs = [ctx_ids + e for e in conts_ids]
        M = max(map(len, seqs))
        x = torch.tensor([s + [pad] * (M - len(s)) for s in seqs], device=P.dev)
        lg = P.T(input_ids=x).logits[:, len(ctx_ids) - 1:].float()
    else:
        K, V = kv
        cache = core.build_cache(K.expand(-1, n, -1, -1, -1).contiguous(), V.expand(-1, n, -1, -1, -1).contiguous(), P.ct, P.st)
        tails = [ctx_ids[-nt:] + e for e in conts_ids]
        M = max(map(len, tails))
        x = torch.tensor([t + [pad] * (M - len(t)) for t in tails], device=P.dev)
        lg = P.T(input_ids=x, past_key_values=cache).logits[:, nt - 1:].float()
    lp = F.log_softmax(lg, -1)
    return [lp[i, torch.arange(len(e), device=P.dev), torch.tensor(e, device=P.dev)].sum().item()
            for i, e in enumerate(conts_ids)]


def run_mc(P, items, k, arms, norm=False, restore="rms"):
    """Accuracy per arm. arms subset of native, mapped, nk_mv, mk_nv, wrong, direct, source."""
    tok = P.tok
    enc = []
    for it in items:
        c = tok(it["ctx"], add_special_tokens=False)["input_ids"]
        enc.append((c, [tok(e, add_special_tokens=False)["input_ids"] for e in it["conts"]]))
    res = {a: [] for a in arms}
    choice_kl = {a: [] for a in arms}
    for j, (it, (c, es)) in enumerate(zip(items, enc)):
        pre = torch.tensor([c[:-1]], device=P.dev)
        scores = {}
        Ks, Vs = P.src_kv(pre)
        Km, Vm = P.mapped(Ks, Vs, k, restore)
        need_nat = any(a in arms for a in ("nk_mv", "mk_nv", "oracle"))
        Kn, Vn = P.native(pre) if need_nat else (None, None)
        for a in arms:
            if a == "native":
                scores[a] = score_choices(P, c, es, None)
            elif a == "mapped":
                scores[a] = score_choices(P, c, es, (Km, Vm))
            elif a == "nk_mv":
                scores[a] = score_choices(P, c, es, (Kn, Vm))
            elif a == "mk_nv":
                scores[a] = score_choices(P, c, es, (Km, Vn))
            elif a == "oracle":  # target's own captured cache (roundtrip check of the native-capture path)
                scores[a] = score_choices(P, c, es, (Kn, Vn))
            elif a == "direct":
                scores[a] = score_choices(P, c, es, P.direct(Ks, Vs))
            elif a == "wrong":  # mapped cache of a different question (shifted by half the set)
                c2 = enc[(j + len(enc) // 2) % len(enc)][0]
                K2, V2 = P.src_kv(torch.tensor([c2[:-1]], device=P.dev))
                scores[a] = score_choices(P, c2[:-1] + c[-1:], es, P.mapped(K2, V2, k, restore))
            elif a.startswith("recipe"):  # kvx best recipe: native sink A + native tail N, mapped middle
                A_, N_ = (int(x) for x in a[len("recipe_s"):].split("t"))
                N_ = max(1, min(N_, len(c) - 1 - A_))
                pre2 = torch.tensor([c[:-N_]], device=P.dev)
                Ks2, Vs2 = P.src_kv(pre2)
                Km2, Vm2 = P.mapped(Ks2, Vs2, k, restore)
                if A_ > 0:
                    Ka, Va = P.native(pre2[:, :A_])
                    Km2[:, :, :, :A_], Vm2[:, :, :, :A_] = Ka, Va
                scores[a] = score_choices(P, c, es, (Km2, Vm2), nt=N_)
            elif a == "source":
                Tm = P.T
                P.T = P.S
                scores[a] = score_choices(P, c, es, None)
                P.T = Tm
        for a in arms:
            s = scores[a]
            if norm:
                s = [v / max(1, len(e)) for v, e in zip(s, it["conts"])]
            pred = max(range(len(s)), key=lambda i: s[i])
            res[a].append(dict(task=it["task"], id=it.get("id", j), correct=int(pred == it["gold"]), pred=pred,
                               gold_nll=-scores[a][it["gold"]], scores=[round(v, 4) for v in scores[a]]))
            if "native" in scores:
                pn = torch.log_softmax(torch.tensor(scores["native"]), 0)
                pa = torch.log_softmax(torch.tensor(scores[a]), 0)
                choice_kl[a].append(float((pn.exp() * (pn - pa)).sum()))
    out = {}
    for a in arms:
        acc = sum(r["correct"] for r in res[a]) / len(res[a])
        by = {}
        for r in res[a]:
            by.setdefault(r["task"], []).append(r["correct"])
        out[a] = dict(acc=acc, n=len(res[a]), by_task={t: sum(v) / len(v) for t, v in by.items()},
                      gold_nll=sum(r["gold_nll"] for r in res[a]) / len(res[a]),
                      choice_kl=(sum(choice_kl[a]) / len(choice_kl[a])) if choice_kl[a] else None,
                      ci95=1.96 * math.sqrt(max(acc * (1 - acc), 1e-9) / len(res[a])))
    return out, res


@torch.no_grad()
def heldout(P, evals, Pfx, ks, restore="rms", direct=True):
    """Prefix Pfx tokens from the source (mapped / direct), continuation natively by the target.
    Agreement = target argmax equals the full-prefill target argmax at each continuation position."""
    acc = {}

    def add(name, lg, lp_ref, top_ref, cont):
        lp = torch.log_softmax(lg.float(), -1)
        kl = (lp_ref.exp() * (lp_ref - lp)).sum(-1)
        nll = -lp[:, :-1].gather(-1, cont[:, 1:, None]).squeeze(-1)
        a = acc.setdefault(name, dict(kl=0.0, agree=0.0, nll=0.0, n=0, ntok=0))
        a["kl"] += kl.mean(1).sum().item()
        a["agree"] += (lg.argmax(-1) == top_ref).float().mean(1).sum().item()
        a["nll"] += nll.sum().item(); a["ntok"] += nll.numel()
        a["n"] += lg.shape[0]

    for i in range(len(evals)):
        ids = evals[i:i + 1].to(P.dev)
        pre, cont = ids[:, :Pfx], ids[:, Pfx:]
        lg_ref = P.T(input_ids=ids).logits[:, Pfx:].float()
        lp_ref = torch.log_softmax(lg_ref, -1)
        top = lg_ref.argmax(-1)
        add("native", lg_ref, lp_ref, top, cont)
        Ks, Vs = P.src_kv(pre)
        for k in ks:
            add(f"map_k{k}", P.T(input_ids=cont, past_key_values=core.build_cache(*P.mapped(Ks, Vs, k, restore), P.ct, P.st)).logits, lp_ref, top, cont)
        if direct and (P.Hs, P.Ds) == (P.Ht, P.Dt):
            add("direct", P.T(input_ids=cont, past_key_values=core.build_cache(*P.direct(Ks, Vs), P.ct, P.st)).logits, lp_ref, top, cont)
        add("nocontext", P.T(input_ids=cont).logits, lp_ref, top, cont)
    out = {}
    for n_, a in acc.items():
        out[n_] = dict(kl=a["kl"] / a["n"], agree=a["agree"] / a["n"], ppl=math.exp(a["nll"] / a["ntok"]), n=a["n"])
    for n_ in out:
        out[n_]["ppl_ratio"] = out[n_]["ppl"] / out["native"]["ppl"]
    return out


# ------------------------------------------------------------------ main

def load(name, dev, dtype):
    from transformers import AutoConfig, AutoModelForCausalLM
    from kvx.models import model_path, load_lm
    cfg = AutoConfig.from_pretrained(model_path(name))
    if getattr(cfg, "architectures", None) and "ConditionalGeneration" in cfg.architectures[0]:  # e.g. Mistral3 (TextLM)
        m = load_lm(name, dtype=dtype).to(dev).eval()
        m.requires_grad_(False)
        return m
    m = AutoModelForCausalLM.from_pretrained(model_path(name), dtype=dtype, attn_implementation="sdpa").to(dev).eval()
    m.requires_grad_(False)
    return m


def run_pair(src, tgt, n_calib=500, L=512, ks="1,4,8,16", n_heldout=32, per_task=32, hs_n=0, piqa_n=0,
             out="", dev=None, dtype="bfloat16", restore="rms", lam_rel=1e-3, stride=4, mc=1, models=None, one_pass=0,
             calib_src="fineweb", recipes="", distill_steps=0, distill_lrs="1e-5,3e-5,1e-4", distill_n=512, maps_from="",
             sdp="default"):
    from transformers import AutoTokenizer
    from kvx.models import model_path
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    dt = getattr(torch, dtype)
    if sdp != "default" and torch.cuda.is_available():  # debug of run f: which SDPA backend breaks mixed caches on B200
        torch.backends.cuda.enable_cudnn_sdp(sdp not in ("nocudnn", "math"))
        if sdp == "math":
            torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_path(tgt))
    S, T = models if models else (load(src, dev, dt), load(tgt, dev, dt))
    P = Pair(S, T, tok, dev, Lmax=max(L, 2048))
    klist = [int(x) for x in str(ks).split(",")]
    klist = [min(k, P.Ls) for k in klist]
    calib = calibration(tok, n_calib, L, calib_src) if not maps_from else None
    evals = fineweb(tok, n_heldout, L, skip=EVAL_SKIP)
    res = dict(sdp=sdp, calib_src=calib_src, recipes=recipes, src=src, tgt=tgt, n_calib=n_calib, L=L, ks=klist, lam_rel=lam_rel, stride=stride, restore=restore,
               dev=str(dev), dtype=dtype, t_load=time.time() - t0)
    t1 = time.time()
    if maps_from:  # reuse another seat's kvx-format maps (e.g. /kvx/runs/pairs-q32-q8/maps_k24.pt), no fit
        d = torch.load(maps_from, map_location=dev)
        assert d["src"] == src and d["tgt"] == tgt and d.get("feat", "kv") == "kv", (d["src"], d["tgt"], d.get("feat"))
        kk = int(d["k_layers"])
        for kind in ("k", "v"):
            P.maps[(kind, kk)] = [{a: (b.to(dev) if torch.is_tensor(b) else b) for a, b in m.items()} for m in d[kind]]
        klist = [kk]
        res["maps_from"] = maps_from
        res["r2"] = None
    else:
        res["r2"] = P.fit(calib, klist, lam_rel, stride, one_pass=bool(one_pass))
    res["t_fit"] = time.time() - t1
    print(f"[{time.time()-t0:.0f}s] fit {res['r2']}", flush=True)
    res["heldout"] = heldout(P, evals, L - 64, klist, restore)
    kbest = min(klist, key=lambda k: res["heldout"][f"map_k{k}"]["kl"])
    res["k_selected"] = kbest
    if distill_steps:  # KV-Lingo-style KL distillation of the selected map on calibration seqs (train text only)
        from research import kvdistill
        va = fineweb(tok, 16, L, skip=EVAL_SKIP + 5000)
        mk, mv, rep = kvdistill.distill(P, kbest, calib[torch.randperm(len(calib), generator=torch.Generator().manual_seed(0))[:distill_n]],
                                         va, L - 64, [float(x) for x in distill_lrs.split(",")], distill_steps, 4, restore,
                                         log=lambda m: print(m, flush=True))
        res["distill"] = rep
        if rep["chosen"] is not None:
            P.maps[("k", "d")], P.maps[("v", "d")] = mk, mv
            res["heldout"]["map_kd"] = heldout(P, evals, L - 64, ["d"], restore, direct=False)["map_kd"]
            kbest = "d"
            res["k_mc"] = "d"
    print(f"[{time.time()-t0:.0f}s] heldout {json.dumps(res['heldout'])}", flush=True)
    detail = {}
    if mc:
        items = mc_items(tok, per_task)
        arms = ["native", "mapped", "nk_mv", "mk_nv", "oracle", "wrong", "source"] + (["direct"] if (P.Hs, P.Ds) == (P.Ht, P.Dt) else [])
        arms += [f"recipe_{r}" for r in recipes.split(",") if r]
        res["mc128"], detail["mc128"] = run_mc(P, items, kbest, arms, restore=restore)
        print(f"[{time.time()-t0:.0f}s] mc128 " + json.dumps({a: round(v['acc'], 4) for a, v in res['mc128'].items()}), flush=True)
    for name, fn, n in (("hellaswag", hellaswag_items, hs_n), ("piqa", piqa_items, piqa_n)):
        if n:
            res[name], detail[name] = run_mc(P, fn(n), kbest, ["native", "mapped", "direct", "source"], norm=True, restore=restore)
            print(f"[{time.time()-t0:.0f}s] {name} " + json.dumps({a: round(v['acc'], 4) for a, v in res[name].items()}), flush=True)
    res["t_total"] = time.time() - t0
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        json.dump(res, open(out, "w"), indent=1)
        json.dump(detail, open(out.replace(".json", "_items.json"), "w"))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--tgt", required=True)
    ap.add_argument("--n-calib", type=int, default=500); ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--ks", default="1,4,8,16"); ap.add_argument("--n-heldout", type=int, default=32)
    ap.add_argument("--per-task", type=int, default=32); ap.add_argument("--hs-n", type=int, default=0)
    ap.add_argument("--piqa-n", type=int, default=0); ap.add_argument("--out", default="")
    ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--mc", type=int, default=1)
    a = ap.parse_args()
    run_pair(a.src, a.tgt, a.n_calib, a.L, a.ks, a.n_heldout, a.per_task, a.hs_n, a.piqa_n, a.out, dtype=a.dtype, mc=a.mc)
