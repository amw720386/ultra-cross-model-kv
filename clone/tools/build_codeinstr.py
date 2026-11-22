"""CPU Modal job: eval-format code instruction prompts for an on-policy code distillation (code map v2).

Each item is one user turn in exactly the bench's format (src/xkv genevals: EvalPlus instruction, chat template,
thinking off); the answer is filled in later by the native target (tools/gen_native_answers.py):
  mbpp:  MBPP train / validation / prompt splits (text + its tests), as _build_mbpp renders the test split;
  hefmt: Python functions with docstrings from codeparrot-clean-valid, problem = signature + docstring (HumanEval form);
  instr: (off by default) Magicoder-OSS-Instruct-75K problems; dropped: gpt-3.5-generated, licence for this use unclear.
Decontamination: a prompt sharing a 13-gram (whitespace words) with any MBPP / HumanEval / BigCodeBench test item is
dropped (tools/build_codemix.test_grams); the build asserts 0 overlaps after filtering.
Held-out dev (never trained on, asserted): MBPP validation (90) and the first --n-dev hefmt items.
Output on KVX_VOLUME: xkv/data/<name>.jsonl (one item per line: id, kind, split, user, prompt_text) + <name>.stats.json.

    cd clone && modal run tools/build_codeinstr.py::main --name codeinstr-v1
"""
import ast
import json
import os
import random

import modal

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("transformers==5.24.0", "tokenizers", "datasets>=4.0", "huggingface_hub", "torch==2.8.0", "jinja2")
         .env({"HF_HOME": "/tmp/hf", **{k: os.environ[k] for k in ("KVX_VOLUME", "KVX_WEIGHTS_VOLUME") if k in os.environ}})
         .add_local_python_source("build_codemix"))
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"), create_if_missing=True)
weights = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
app = modal.App(os.environ.get("KVX_APP", "kvx-codeinstr"), image=image)

# the bench's instruction (src/xkv/genevals.py _EVALPLUS_INSTR), copied so this job needs no GPU-side package
EVALPLUS_INSTR = ("Please provide a self-contained Python script that solves the following problem "
                  "in a markdown code block:\n```\n{problem}\n```\n")


def snapshot(name):
    rd = f"/hf/hub/models--{name.replace('/', '--')}"
    return f"{rd}/snapshots/{open(f'{rd}/refs/main').read().strip()}"


def doc_functions(src, max_lines=40):
    """Top-level functions with a docstring: (signature + docstring text, body line count)."""
    try:
        tree = ast.parse(src)
    except Exception:
        return []
    lines = src.splitlines()
    out = []
    for n in tree.body:
        if not isinstance(n, ast.FunctionDef) or not ast.get_docstring(n) or n.name.startswith("_"):
            continue
        ds = n.body[0]
        if not isinstance(ds, ast.Expr) or len(n.body) < 2:
            continue
        nbody = n.end_lineno - ds.end_lineno
        if not 3 <= nbody <= max_lines or len(ast.get_docstring(n)) < 40:
            continue
        head = "\n".join(lines[n.lineno - 1 - len(n.decorator_list):ds.end_lineno])
        out.append(head + "\n")
    return out


@app.function(cpu=8.0, memory=32768, timeout=3000, volumes={"/vol": vol, "/hf": weights.read_only()})
def build(name: str, tok_name: str, n_hefmt: int, n_instr: int, n_dev: int, max_files: int):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from build_codemix import grams, test_grams
    tok = AutoTokenizer.from_pretrained(snapshot(tok_name))
    tg, n_test = test_grams()
    rng = random.Random(0)
    items, dropped = [], {}

    def add(kind, split, iid, problem):
        user = EVALPLUS_INSTR.format(problem=problem.strip())
        if grams(user) & tg:
            dropped[kind] = dropped.get(kind, 0) + 1
            return
        text = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False,
                                       enable_thinking=False)
        items.append(dict(id=iid, kind=kind, split=split, user=user, problem=problem.strip(), prompt_tokens=len(tok(text, add_special_tokens=False)["input_ids"])))

    m = load_dataset("google-research-datasets/mbpp", "full")
    for sp in ("train", "prompt", "validation"):
        for ex in m[sp]:
            problem = f"{ex['text']}\nYour code should pass these tests:\n\n" + "\n".join(ex["test_list"])
            add("mbpp", "dev" if sp == "validation" else "train", f"mbpp/{sp}/{ex['task_id']}", problem)
    fns = []
    for i, ex in enumerate(load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)):
        if i >= max_files or len(fns) >= 3 * n_hefmt:
            break
        fns += [(f"cp/{i}/{k}", f) for k, f in enumerate(doc_functions(ex["content"]))]
    rng.shuffle(fns)
    seen = set()
    for k, (iid, f) in enumerate(fns):
        if sum(1 for x in items if x["kind"] == "hefmt") >= n_hefmt + n_dev:
            break
        if f in seen:
            continue
        seen.add(f)
        add("hefmt", "dev" if sum(1 for x in items if x["kind"] == "hefmt") < n_dev else "train", iid, f)
    mg = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train") if n_instr else []
    idx = [i for i, ex in enumerate(mg) if (ex.get("lang") or "").lower() == "python"]
    rng.shuffle(idx)
    for i in idx[:int(n_instr * 1.2)]:
        if sum(1 for x in items if x["kind"] == "instr") >= n_instr:
            break
        add("instr", "train", f"magicoder/{i}", mg[i]["problem"])
    leak = sum(1 for x in items if grams(x["user"]) & tg)
    assert leak == 0, leak
    # held-out dev must not be in training: MBPP validation ids never in the train split, and no train prompt shares
    # a 13-gram with a dev prompt (lead rule, code-map-v2)
    val_ids = {f"mbpp/validation/{ex['task_id']}" for ex in m["validation"]}
    assert not any(x["id"] in val_ids for x in items if x["split"] == "train")
    # (13-grams of the PROBLEM text: the shared EvalPlus instruction is itself a 16-word common string)
    dev_g = set().union(*(grams(x["problem"]) for x in items if x["split"] == "dev"))
    n0 = len(items)
    items = [x for x in items if x["split"] != "train" or not (grams(x["problem"]) & dev_g)]
    dropped["train_overlapping_dev"] = n0 - len(items)
    dev_leak = sum(1 for x in items if x["split"] == "train" and grams(x["problem"]) & dev_g)
    assert dev_leak == 0, f"{dev_leak} train prompts share a 13-gram with the dev set"
    counts = {}
    for x in items:
        counts[f"{x['kind']}/{x['split']}"] = counts.get(f"{x['kind']}/{x['split']}", 0) + 1
    pt = sorted(x["prompt_tokens"] for x in items)
    stats = dict(test_items=n_test, test_13grams=len(tg), dropped=dropped, counts=counts, overlap_after=leak,
                 dev_overlap_after=dev_leak, dev="mbpp validation (90) + first n_dev hefmt items",
                 prompt_tokens=dict(min=pt[0], median=pt[len(pt) // 2], max=pt[-1]), tokenizer=tok_name,
                 instr_template="src/xkv genevals _EVALPLUS_INSTR, chat template, enable_thinking=False")
    os.makedirs("/vol/xkv/data", exist_ok=True)
    with open(f"/vol/xkv/data/{name}.jsonl", "w") as f:
        for x in items:
            f.write(json.dumps(x) + "\n")
    json.dump(stats, open(f"/vol/xkv/data/{name}.stats.json", "w"), indent=1)
    vol.commit()
    return stats


@app.local_entrypoint()
def main(name: str = "codeinstr-v1", tok_name: str = "Qwen/Qwen3.8-27B", n_hefmt: int = 1000, n_instr: int = 0,
         n_dev: int = 100, max_files: int = 20000):
    # n_instr default 0: Magicoder's problems are gpt-3.5 outputs under OpenAI's usage policy (dataset card), so its
    # licence for this use is unclear -> dropped (lead rule: unclear -> drop, no substitute)
    print(json.dumps(build.remote(name, tok_name, n_hefmt, n_instr, n_dev, max_files), indent=1))
