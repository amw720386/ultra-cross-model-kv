"""Benchmark loaders for OneTriangle's bundle rows (Qwen3.5-4B -> Qwen3.8-27B) and the benchmark page.

Each item: dict(id, bench, context, question, kind, answer, meta). `context` is the long part the source model
reads (mapped); `question` is the instruction that follows it (native tail in the mapped arm).
kind: "letter" (answer = letter), "label" (answer = one of meta["labels"]), "code" (meta: tests), "math", "summary".
Raw files are fetched once into `root` (public URLs only) and parsed as data, never executed.
"""
import ast
import io
import json
import os
import urllib.request
import zipfile

URLS = {
    "arc_c_test.parquet": "https://huggingface.co/api/datasets/allenai/ai2_arc/parquet/ARC-Challenge/test/0.parquet",
    "humaneval.parquet": "https://huggingface.co/api/datasets/openai/openai_humaneval/parquet/openai_humaneval/test/0.parquet",
    "mbpp_test.parquet": "https://huggingface.co/api/datasets/google-research-datasets/mbpp/parquet/full/test/0.parquet",
    "math500.parquet": "https://huggingface.co/api/datasets/HuggingFaceH4/MATH-500/parquet/default/test/0.parquet",
    "paperqa_16k.jsonl": "https://raw.githubusercontent.com/RUCAIBox/BAMBOO/main/datasets/paperqa_16k.jsonl",
    "longhealth.json": "https://raw.githubusercontent.com/kbressem/LongHealth/main/data/benchmark_v5.json",
    "contract_nli.zip": "https://huggingface.co/datasets/tau/scrolls/resolve/main/contract_nli.zip",
    "qmsum.zip": "https://huggingface.co/datasets/tau/scrolls/resolve/main/qmsum.zip",
    "hellaswag_val.parquet": "https://huggingface.co/api/datasets/Rowan/hellaswag/parquet/default/validation/0.parquet",
    "piqa_val.parquet": "https://huggingface.co/api/datasets/baber/piqa/parquet/default/validation/0.parquet",
    "quality_val.parquet": "https://huggingface.co/api/datasets/emozilla/quality/parquet/default/validation/0.parquet",
    # allenai/ZebraLogicBench (public) has its grid solutions BLANKED ("___"); WildEval/ZebraLogic carries the same
    # 1000 puzzles (same ids) with real solutions. The allenai -private copy is gated.
    "zebra_grid_wildeval.parquet": "https://huggingface.co/api/datasets/WildEval/ZebraLogic/parquet/grid_mode/test/0.parquet",
    "medqa_test.parquet": "https://huggingface.co/api/datasets/GBaker/MedQA-USMLE-4-options/parquet/default/test/0.parquet",
    "obqa_test.parquet": "https://huggingface.co/api/datasets/allenai/openbookqa/parquet/main/test/0.parquet",
    "longbench_v2.json": "https://huggingface.co/datasets/THUDM/LongBench-v2/resolve/main/data.json",
    "bigcodebench.parquet": "https://huggingface.co/api/datasets/bigcode/bigcodebench/parquet/default/v0.1.4/0.parquet",
}
LETTERS = "ABCDEFGHIJ"


def fetch(root, name):
    path = os.path.join(root, name)
    if not os.path.exists(path):
        os.makedirs(root, exist_ok=True)
        tmp = path + ".part"
        with urllib.request.urlopen(URLS[name], timeout=300) as r, open(tmp, "wb") as f:
            f.write(r.read())
        os.replace(tmp, path)
    return path


def _parquet(root, name):
    import pyarrow.parquet as pq
    return pq.read_table(fetch(root, name)).to_pylist()


def _zip_jsonl(root, name, member):
    with zipfile.ZipFile(fetch(root, name)) as z:
        return [json.loads(l) for l in io.TextIOWrapper(z.open(member), encoding="utf-8")]


def _opts(options):
    return "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))


LETTER_ASK = "Answer with the letter of the correct option only."


def arc_chat(root, n):
    out = []
    for r in _parquet(root, "arc_c_test.parquet")[:n]:
        labels = list(r["choices"]["label"])
        texts = list(r["choices"]["text"])
        ans = LETTERS[labels.index(r["answerKey"])]
        q = f"{r['question']}\n\n{_opts(texts)}\n\n{LETTER_ASK}"
        out.append(dict(id=r["id"], bench="arc_chat", context="", question=q, kind="letter", answer=ans,
                        meta=dict(n_opts=len(texts))))
    return out


NLI_LABELS = ["Entailment", "Contradiction", "Not mentioned"]


def contract_nli(root, n):
    rows = _zip_jsonl(root, "contract_nli.zip", "contract_nli/validation.jsonl")[:n]
    out = []
    for r in rows:
        hyp, _, doc = r["input"].partition("\n\n")
        q = (f"Hypothesis: {hyp.strip()}\n\nBased only on the contract above, is the hypothesis entailed by the contract, "
             "contradicted by it, or not mentioned in it? Answer with exactly one of: Entailment, Contradiction, Not mentioned.")
        out.append(dict(id=r["id"], bench="contract_nli", context=f"Contract:\n{doc.strip()}\n\n", question=q, kind="label",
                        answer=r["output"].strip(), meta=dict(labels=NLI_LABELS)))
    return out


def bamboo_paperqa(root, n):
    out = []
    rows = []
    with open(fetch(root, "paperqa_16k.jsonl")) as f:
        for l in f:  # upstream line 25 has unescaped quotes in "evidence": skipped (99 of 100 parse)
            try:
                rows.append(json.loads(l))
            except json.JSONDecodeError:
                continue
    rows = rows[:n]
    for i, r in enumerate(rows):
        opts = "\n".join(o.strip() for o in r["options"])  # options already carry "A. " prefixes
        q = f"Question: {r['question'].strip()}\n\n{opts}\n\n{LETTER_ASK}"
        out.append(dict(id=f"paperqa16k-{i}", bench="bamboo_paperqa", context=f"Paper:\n{r['content'].strip()}\n\n",
                        question=q, kind="letter", answer=r["answer"].strip()[:1], meta=dict(n_opts=len(r["options"]))))
    return out


def longhealth(root, n):
    with open(fetch(root, "longhealth.json")) as f:
        d = json.load(f)
    out = []
    for pid in sorted(d):
        p = d[pid]
        texts = p["texts"] if isinstance(p["texts"], dict) else ast.literal_eval(p["texts"])
        qs = p["questions"] if isinstance(p["questions"], list) else ast.literal_eval(p["questions"])
        ctx = "Patient records:\n\n" + "\n\n".join(f"<{k}>\n{v}\n</{k}>" for k, v in texts.items()) + "\n\n"
        for qd in qs:
            opts = [qd[f"answer_{c}"] for c in "abcde"]
            ans = LETTERS[opts.index(qd["correct"])]
            q = f"Question: {qd['question']}\n\n{_opts(opts)}\n\n{LETTER_ASK}"
            out.append(dict(id=f"{pid}-{qd['No']}", bench="longhealth", context=ctx, question=q, kind="letter",
                            answer=ans, meta=dict(n_opts=5)))
            if len(out) >= n:
                return out
    return out


def qmsum(root, n):
    rows = _zip_jsonl(root, "qmsum.zip", "qmsum/validation.jsonl")[:n]
    out = []
    for r in rows:
        query, _, tr = r["input"].partition("\n\n")
        q = f"Query: {query.strip()}\n\nAnswer the query based on the meeting transcript above, in a short paragraph."
        out.append(dict(id=r["id"], bench="qmsum", context=f"Meeting transcript:\n{tr.strip()}\n\n", question=q,
                        kind="summary", answer=r["output"].strip(), meta={}))
    return out


CODE_ASK = "Write the complete Python solution in a single ```python code block."


def humaneval(root, n):
    out = []
    for r in _parquet(root, "humaneval.parquet")[:n]:
        q = f"Complete the following Python function.\n\n```python\n{r['prompt']}```\n\n{CODE_ASK}"
        out.append(dict(id=r["task_id"], bench="humaneval", context="", question=q, kind="code", answer=None,
                        meta=dict(prompt=r["prompt"], test=r["test"], entry_point=r["entry_point"])))
    return out


def mbpp(root, n):
    out = []
    for r in _parquet(root, "mbpp_test.parquet")[:n]:
        tests = list(r["test_list"])
        q = f"{r['text']}\nYour code should pass these tests:\n" + "\n".join(tests) + f"\n\n{CODE_ASK}"
        out.append(dict(id=f"mbpp-{r['task_id']}", bench="mbpp", context="", question=q, kind="code", answer=None,
                        meta=dict(test="\n".join([r.get("test_setup_code") or ""] + tests), entry_point=None)))
    return out


def math500(root, n):
    out = []
    for r in _parquet(root, "math500.parquet")[:n]:
        q = f"{r['problem']}\n\nSolve the problem and put the final answer in \\boxed{{}}."
        out.append(dict(id=r["unique_id"], bench="math500", context="", question=q, kind="math", answer=r["answer"],
                        meta=dict(level=r.get("level"), subject=r.get("subject"))))
    return out


def _lst(x):
    return list(x) if not isinstance(x, str) else ast.literal_eval(x)


def hellaswag(root, n):
    out = []
    for r in _parquet(root, "hellaswag_val.parquet")[:n]:
        ends = _lst(r["endings"])
        q = (f"Choose the most plausible continuation.\n\nContext: {r['activity_label']}: {r['ctx']}\n\n{_opts(ends)}\n\n"
             f"{LETTER_ASK}")
        out.append(dict(id=f"hs-{r['ind']}", bench="hellaswag", context="", question=q, kind="letter",
                        answer=LETTERS[int(r["label"])], meta=dict(n_opts=len(ends))))
    return out


def piqa(root, n):
    out = []
    for i, r in enumerate(_parquet(root, "piqa_val.parquet")[:n]):
        q = f"Goal: {r['goal']}\n\nWhich solution is better?\n\n{_opts([r['sol1'], r['sol2']])}\n\n{LETTER_ASK}"
        out.append(dict(id=f"piqa-{i}", bench="piqa", context="", question=q, kind="letter",
                        answer=LETTERS[int(r["label"])], meta=dict(n_opts=2)))
    return out


def arc_c(root, n):
    return [dict(it, bench="arc_c") for it in arc_chat(root, n)]


def quality(root, n):
    out = []
    for i, r in enumerate(_parquet(root, "quality_val.parquet")[:n]):
        opts = _lst(r["options"])
        q = f"Question: {r['question']}\n\n{_opts(opts)}\n\n{LETTER_ASK}"
        out.append(dict(id=f"quality-{i}", bench="quality", context=f"Story:\n{r['article'].strip()}\n\n", question=q,
                        kind="letter", answer=LETTERS[int(r["answer"])], meta=dict(n_opts=len(opts), hard=r.get("hard"))))
    return out


def longbench_v2(root, n, max_chars=None):
    """THUDM LongBench-v2 (503). OneTriangle's n=97 is unstated; our reading: the n shortest items (the 97 shortest are
    <= ~105K chars, i.e. what fits a 32K-token window). Order = by context length."""
    with open(fetch(root, "longbench_v2.json")) as f:
        d = json.load(f)
    d = sorted(d, key=lambda x: len(x["context"]))[:n]
    out = []
    for r in d:
        opts = [r[f"choice_{c}"] for c in "ABCD"]
        q = f"Question: {r['question']}\n\n{_opts(opts)}\n\n{LETTER_ASK}"
        out.append(dict(id=r["_id"], bench="longbench_v2", context=f"Document:\n{r['context'].strip()}\n\n", question=q,
                        kind="letter", answer=r["answer"], meta=dict(n_opts=4, length=r["length"], difficulty=r["difficulty"])))
    return out


ZEBRA_ASK = ("Solve the puzzle. Output your final answer as JSON at the end, in this exact form:\n"
             '{"solution": {"House 1": {"<attribute>": "<value>", ...}, "House 2": {...}, ...}}')


def zebralogic(root, n):
    rows = _parquet(root, "zebra_grid_wildeval.parquet")
    out = []
    for r in rows[:n]:
        sol = r["solution"] if isinstance(r["solution"], dict) else ast.literal_eval(r["solution"])
        header = list(sol["header"])
        grid = {f"House {row[0]}": {h: v for h, v in zip(header[1:], row[1:])} for row in [list(x) for x in sol["rows"]]}
        if any(v == "___" for d in grid.values() for v in d.values()):
            raise ValueError(f"ZebraLogic {r['id']}: blanked solution (wrong dataset copy)")
        q = f"{r['puzzle']}\n\nAttributes: {', '.join(header[1:])}.\n\n{ZEBRA_ASK}"
        out.append(dict(id=r["id"], bench="zebralogic", context="", question=q, kind="zebra", answer=grid,
                        meta=dict(size=r["size"])))
    return out


def medqa(root, n):
    out = []
    for i, r in enumerate(_parquet(root, "medqa_test.parquet")[:n]):
        o = r["options"] if isinstance(r["options"], dict) else ast.literal_eval(r["options"])
        keys = sorted(o)
        q = f"{r['question']}\n\n" + "\n".join(f"{k}. {o[k]}" for k in keys) + f"\n\n{LETTER_ASK}"
        out.append(dict(id=f"medqa-{i}", bench="medqa", context="", question=q, kind="letter", answer=r["answer_idx"],
                        meta=dict(n_opts=len(keys))))
    return out


def obqa(root, n):
    out = []
    for r in _parquet(root, "obqa_test.parquet")[:n]:
        ch = r["choices"] if isinstance(r["choices"], dict) else ast.literal_eval(r["choices"])
        texts, labels = list(ch["text"]), list(ch["label"])
        q = f"{r['question_stem']}\n\n{_opts(texts)}\n\n{LETTER_ASK}"
        out.append(dict(id=r["id"], bench="obqa", context="", question=q, kind="letter",
                        answer=LETTERS[labels.index(r["answerKey"])], meta=dict(n_opts=len(texts))))
    return out


def bigcodebench(root, n):
    """BigCodeBench v0.1.4 (1140 tasks), instruct split prompts (chat models), first n. Scored by running the task's
    unittest TestCases against the solution (exec_code.py) in an image with BigCodeBench's eval requirements."""
    out = []
    for r in _parquet(root, "bigcodebench.parquet")[:n]:
        q = f"{r['instruct_prompt'].strip()}\n\n{CODE_ASK}"
        out.append(dict(id=r["task_id"], bench="bigcodebench", context="", question=q, kind="code", answer=None,
                        meta=dict(prompt=r["code_prompt"], test=r["test"], entry_point=r["entry_point"])))
    return out


def deveval(root, n):
    """DevEval (arXiv 2405.19856) local-file completion prompts (official gpt-4-1106 prompt file): the file context above
    the function is the mapped part, "The code to be completed is: ..." the native tail. Items: the first n namespaces
    (data.jsonl order) listed in deveval_eligible.txt (reference solution passes in our sandbox; OneTriangle's n=500
    subset is unstated). Scored outside (tools/deveval_score.py in a python:3.9 sandbox)."""
    elig = [l.strip() for l in open(os.path.join(root, "deveval_eligible.txt")) if l.strip()]
    keep = set(elig[:n])
    sep = "\n\nThe code to be completed is:"
    out = []
    with open(os.path.join(root, "deveval_local_completion_prompt.jsonl")) as f:
        for l in f:
            r = json.loads(l)
            if r["namespace"] not in keep:
                continue
            ctx, _, q = r["prompt"].partition(sep)
            out.append(dict(id=r["namespace"], bench="deveval", context=ctx, question=sep.lstrip() + q, kind="code",
                            answer=None, meta=dict()))
    order = {ns: i for i, ns in enumerate(elig)}
    return sorted(out, key=lambda x: order[x["id"]])


BENCHES = dict(arc_chat=arc_chat, contract_nli=contract_nli, bamboo_paperqa=bamboo_paperqa, longhealth=longhealth,
               qmsum=qmsum, humaneval=humaneval, mbpp=mbpp, math500=math500,
               hellaswag=hellaswag, piqa=piqa, arc_c=arc_c, quality=quality, longbench_v2=longbench_v2,
               zebralogic=zebralogic, medqa=medqa, obqa=obqa, bigcodebench=bigcodebench, deveval=deveval)
# generation budget per bench (non-thinking)
MAX_TOKENS = dict(arc_chat=16, contract_nli=16, bamboo_paperqa=16, longhealth=16, qmsum=512, humaneval=1024,
                  mbpp=1024, math500=4096, hellaswag=16, piqa=16, arc_c=16, quality=16, longbench_v2=16,
                  zebralogic=4096, medqa=16, obqa=16, bigcodebench=1536, deveval=1024)
THINK_TOKENS = 16384  # extra generation budget per item when --thinking


def load(bench, root, n):
    return BENCHES[bench](root, n)
