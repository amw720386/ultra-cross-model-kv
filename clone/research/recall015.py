"""Note 015 replication: TinyLlama-1.1B reads a record, a cross-family map carries its KV cache into frozen
Qwen3-4B-Instruct, which answers questions with 0 target prefill of the record (their 187/192 held-out recall,
191/192 counterfactual facts followed).

Their record/question format is unpublished; ours (labelled): 8 held-out synthetic person records x 8 fields x
3 phrasings = 192 questions. Chat template prefix ("<|im_start|>user\\n") and the question are target-native; the
record tokens are mapped (none prefilled by the target). Counterfactual: the record shown to the source has a
different value for the asked field; "followed" = the answer contains the swapped value.
Maps: research/xfit ridge (generic FineWeb+PG19 text) and a task-calibrated variant (+ synthetic TRAIN records with
disjoint names/values -- same idea as their supervision on native target prefill).
"""
import json, os, random, time

import torch

from kvx import core
from research import mc128, xfit

FIRST = ["Alice", "Bruno", "Chiara", "Dmitri", "Elena", "Farid", "Greta", "Hiro", "Ines", "Jonas", "Kalani", "Lucia",
         "Mateo", "Nadia", "Oskar", "Priya", "Quentin", "Rosa", "Soren", "Tamsin", "Umar", "Vera", "Wen", "Ximena",
         "Yusuf", "Zora", "Anders", "Bianca", "Cyrus", "Delphine", "Emeka", "Freya", "Gideon", "Hana", "Ivo", "Jun"]
LAST = ["Moreno", "Albrecht", "Okafor", "Lindqvist", "Tanaka", "Haddad", "Novak", "Castellanos", "Mbeki", "Fontaine",
        "Kowalski", "Ferreira", "Nakamura", "Rossi", "Iversen", "Dubois", "Sato", "Varga", "Quispe", "Brennan"]
FIELDS = {
    "city": ["Lisbon", "Tromso", "Valparaiso", "Osaka", "Marrakesh", "Tallinn", "Cusco", "Hobart", "Ljubljana", "Accra",
             "Bergen", "Cebu", "Gdansk", "Leon", "Perth", "Quito", "Sapporo", "Turku", "Utrecht", "Windhoek"],
    "job": ["glassblower", "cartographer", "beekeeper", "luthier", "actuary", "falconer", "sommelier", "welder",
            "archivist", "glaciologist", "puppeteer", "locksmith", "perfumer", "ferry pilot", "botanist", "jeweler"],
    "pet": ["a parrot named Ziggy", "a tortoise named Basil", "a ferret named Pip", "a husky named Tundra",
            "a cat named Saffron", "a goat named Waffles", "an iguana named Rex", "a rabbit named Clover",
            "a pony named Juniper", "a gecko named Noodle"],
    "color": ["teal", "maroon", "ochre", "lavender", "chartreuse", "indigo", "coral", "olive", "magenta", "amber"],
    "number": [str(n) for n in (7, 13, 21, 34, 47, 58, 62, 79, 86, 93, 101, 117, 128, 144)],
    "car": ["a blue Volvo", "a red Fiat", "a green Subaru", "a white Skoda", "a black Saab", "a yellow Mini",
            "a silver Lada", "an orange Citroen"],
    "month": ["January", "March", "April", "June", "August", "September", "October", "December"],
    "food": ["paella", "ramen", "pierogi", "injera", "ceviche", "goulash", "laksa", "falafel", "risotto", "bibimbap"],
}
ANSWER = {"pet": lambda v: v.split(" named ")[1]}  # graded on the distinctive string
QUESTIONS = {
    "city": ["Which city does {n} live in?", "Where is {n}'s home city?", "In what city is {n} based?"],
    "job": ["What is {n}'s occupation?", "What does {n} do for a living?", "What is {n}'s job?"],
    "pet": ["What is the name of {n}'s pet?", "What is {n}'s pet called?", "What name does {n}'s pet have?"],
    "color": ["What is {n}'s favorite color?", "Which color does {n} like best?", "What color is {n}'s favorite?"],
    "number": ["What is {n}'s lucky number?", "Which number does {n} consider lucky?", "What lucky number does {n} have?"],
    "car": ["What car does {n} drive?", "Which car does {n} own?", "What kind of car does {n} have?"],
    "month": ["In which month was {n} born?", "What is {n}'s birth month?", "Which month is {n}'s birthday in?"],
    "food": ["What is {n}'s favorite food?", "Which dish does {n} love most?", "What food does {n} like best?"],
}


def make_record(rng, first_pool, vals=None):
    name = f"{rng.choice(first_pool)} {rng.choice(LAST)}"
    v = vals or {f: rng.choice(opts) for f, opts in FIELDS.items()}
    text = (f"Record: {name} lives in {v['city']} and works as a {v['job']}. {name.split()[0]} has {v['pet']}. "
            f"Favorite color: {v['color']}. Lucky number: {v['number']}. {name.split()[0]} drives {v['car']}, "
            f"was born in {v['month']}, and loves {v['food']}.")
    return name, v, text


def heldout_set(seed=1):
    """8 test records (first names from the second half of the pool, disjoint from TRAIN records) x 24 questions."""
    rng = random.Random(seed)
    recs, qs = [], []
    for _ in range(8):
        name, v, text = make_record(rng, FIRST[18:])
        recs.append((name, v, text))
        for f in FIELDS:
            for q in QUESTIONS[f]:
                qs.append(dict(rec=len(recs) - 1, field=f, q=q.format(n=name.split()[0]),
                               answer=ANSWER.get(f, lambda x: x)(v[f])))
    return recs, qs


def train_texts(n, seed=7):
    """Synthetic TRAIN records (first names from the first half of the pool) with a question/answer appended."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        name, v, text = make_record(rng, FIRST[:18])
        f = rng.choice(list(FIELDS))
        out.append(text + "\n\n" + rng.choice(QUESTIONS[f]).format(n=name.split()[0]) + " " + v[f] + ".")
    return out


@torch.no_grad()
def answer(P, prefix_ids, rec_text, q_ids, mapped_k=None, max_new=16):
    """prefix_ids: target-native template tokens; rec_text: record (mapped, or native when mapped_k is None);
    q_ids: target-native question + assistant prefix."""
    s_ids, r_ids, idx = P.enc(rec_text, 10 ** 6)
    ids = prefix_ids + r_ids + q_ids
    x = torch.tensor([ids], device=P.dev)
    if mapped_k is None:
        g = P.T.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=max_new, do_sample=False,
                         pad_token_id=P.tok.pad_token_id or 0)
    else:
        Kn, Vn = P.native(torch.tensor([prefix_ids], device=P.dev))
        Ks, Vs, _ = core.capture_kv(P.S, torch.tensor([s_ids], device=P.dev), P.cs, P.ss)
        Ka, Va = Ks[:, :, :, idx.to(P.dev)], Vs[:, :, :, idx.to(P.dev)]
        Km, Vm = P.mapped(Ka, Va, mapped_k, "rms" if P.kw is not None else "none")
        K = torch.cat([Kn, Km], 3); V = torch.cat([Vn, Vm], 3)
        cache = core.build_cache(K, V, P.ct, P.st)
        g = P.T.generate(x, attention_mask=torch.ones_like(x), past_key_values=cache, max_new_tokens=max_new,
                         do_sample=False, pad_token_id=P.tok.pad_token_id or 0)
    return P.tok.decode(g[0, len(ids):], skip_special_tokens=True)


def run_pair(src="TinyLlama/TinyLlama-1.1B-Chat-v1.0", tgt="Qwen/Qwen3-4B-Instruct-2507", n_calib=1000, L=512,
             ks="4,8,16", out="", models=None, n_task=2000, **_):
    from transformers import AutoTokenizer
    from kvx.models import model_path
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    tok_s, tok_t = AutoTokenizer.from_pretrained(model_path(src)), AutoTokenizer.from_pretrained(model_path(tgt))
    S, T = models if models else (mc128.load(src, dev, torch.bfloat16), mc128.load(tgt, dev, torch.bfloat16))
    P = xfit.XPair(S, T, tok_s, tok_t, dev, bos_s=tok_s.bos_token_id, bos_t=None)
    klist = [min(int(k), P.Ls) for k in str(ks).split(",")]
    generic = xfit.texts_fineweb(n_calib // 2, stop=mc128.EVAL_SKIP) + xfit.texts_pg19(n_calib - n_calib // 2)
    recs, qs = heldout_set()
    # target-native chat framing around the mapped record
    msg = tok_t.apply_chat_template([{"role": "user", "content": "XXXX"}], tokenize=False, add_generation_prompt=True)
    pre_txt, post_txt = msg.split("XXXX")
    prefix_ids = tok_t(pre_txt, add_special_tokens=False)["input_ids"]
    res = dict(src=src, tgt=tgt, ks=klist, n_calib=len(generic), n_task=n_task, n_q=len(qs), prefix_tokens=len(prefix_ids),
               format="ours (theirs unpublished): 8 records x 8 fields x 3 phrasings")

    def grade(arm_fn, swap=False):
        ok = []
        rng = random.Random(3)
        for q in qs:
            name, v, text = recs[q["rec"]]
            want = q["answer"]
            if swap:  # counterfactual: the source reads a record with a different value for this field
                alt = [o for o in FIELDS[q["field"]] if o != v[q["field"]]]
                v2 = dict(v); v2[q["field"]] = rng.choice(alt)
                text = make_record(random.Random(0), [name.split()[0]], v2)[2].replace(
                    make_record(random.Random(0), [name.split()[0]], v2)[0], name)
                want = ANSWER.get(q["field"], lambda x: x)(v2[q["field"]])
            q_ids = tok_t("\n\n" + q["q"] + " Answer with just the value." + post_txt, add_special_tokens=False)["input_ids"]
            a = arm_fn(text, q_ids)
            ok.append(int(want.lower() in a.lower()))
        return sum(ok), ok

    arms = {"native": lambda text, q_ids: answer(P, prefix_ids, text, q_ids, None)}
    for tag, texts in (("generic", generic), ("task", generic[: n_calib // 2] + train_texts(n_task))):
        res[f"r2_{tag}"] = P.fit_texts(texts, klist, L)
        for k in klist:
            P.maps[("k", f"{tag}{k}")], P.maps[("v", f"{tag}{k}")] = P.maps[("k", k)], P.maps[("v", k)]
            arms[f"mapped_{tag}_k{k}"] = (lambda kk: (lambda text, q_ids: answer(P, prefix_ids, text, q_ids, kk)))(f"{tag}{k}")
        print(f"[{time.time()-t0:.0f}s] fit {tag} {res[f'r2_{tag}']}", flush=True)
    detail = {}
    for name, fn in arms.items():
        c, ok = grade(fn)
        cs, oks = grade(fn, swap=True)
        res[name] = dict(correct=c, n=len(qs), counterfactual_followed=cs)
        detail[name] = dict(ok=ok, ok_swap=oks)
        print(f"[{time.time()-t0:.0f}s] {name}: {c}/{len(qs)} correct, counterfactual {cs}/{len(qs)}", flush=True)
    res["t_total"] = time.time() - t0
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        json.dump(res, open(out, "w"), indent=1)
        json.dump(detail, open(out.replace(".json", "_items.json"), "w"))
    return res
