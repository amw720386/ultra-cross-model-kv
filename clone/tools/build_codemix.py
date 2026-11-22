"""CPU Modal job: build a code-heavy distillation set for src/xkv task_distill ("file:" source in modal_xkv.py).

Mix: FineWeb-Edu (xkv's calib split) + public code with identifiers and tests:
  MBPP full train / prompt / validation splits (text + code + asserts), the-stack-smol Python files.
Decontamination: any training document sharing a 13-gram (whitespace words) with an MBPP / HumanEval / BigCodeBench
TEST item is dropped; the build asserts 0 remaining overlaps and records the counts.
Output (on KVX_VOLUME): xkv/data/<name>.pt = {"calib": [[ids]], "eval": [[ids]], "stats": {...}}; documents are
EOS-packed into seq_len-token sequences (like xkv's FineWeb loader).

    cd clone && modal run tools/build_codemix.py::main --name codemix-v1 --n-calib 900 --seq-len 2176
"""
import os
import random
import re

import modal

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("transformers==5.20.2", "tokenizers", "datasets==3.6.0", "huggingface_hub", "torch==2.8.0")
         .env({"HF_HOME": "/vol/hf", **{k: os.environ[k] for k in ("KVX_VOLUME", "KVX_WEIGHTS_VOLUME") if k in os.environ}}))
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"), create_if_missing=True)
weights = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
app = modal.App(os.environ.get("KVX_APP", "kvx-codemix"), image=image)

WORD = re.compile(r"\S+")


def grams(text, n=13):
    w = WORD.findall(text)
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def test_grams():
    from datasets import load_dataset
    g, n_items = set(), 0
    for name, cfg, split, fields in (
            ("google-research-datasets/mbpp", "full", "test", ("text", "code", "test_list")),
            ("google-research-datasets/mbpp", "sanitized", "test", ("prompt", "code", "test_list")),
            ("openai/openai_humaneval", None, "test", ("prompt", "canonical_solution", "test")),
            ("bigcode/bigcodebench", None, None, ("complete_prompt", "canonical_solution", "test"))):
        ds = load_dataset(name, cfg) if cfg else load_dataset(name)
        for sp in ([split] if split else list(ds.keys())):
            for ex in ds[sp]:
                t = "\n".join(str(ex[f]) if not isinstance(ex[f], list) else "\n".join(ex[f]) for f in fields if f in ex)
                g |= grams(t)
                n_items += 1
    return g, n_items


def code_docs(max_stack):
    from datasets import load_dataset
    docs = []
    m = load_dataset("google-research-datasets/mbpp", "full")
    for sp in ("train", "prompt", "validation"):
        for ex in m[sp]:
            docs.append(f"# {ex['text']}\n{ex['code']}\n" + "\n".join(ex["test_list"]))
    st = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train")
    for i, ex in enumerate(st):
        if i >= max_stack:
            break
        docs.append(ex["content"])
    return docs


def snapshot(name):
    rd = f"/hf/hub/models--{name.replace('/', '--')}"
    ref = open(f"{rd}/refs/main").read().strip()
    return f"{rd}/snapshots/{ref}"


@app.function(cpu=8.0, memory=32768, timeout=3300, volumes={"/vol": vol, "/hf": weights.read_only()})
def build(name: str, tok_name: str, n_calib: int, n_eval: int, seq_len: int, code_frac: float, max_stack: int):
    import torch
    from transformers import AutoTokenizer
    from datasets import load_dataset
    weights.reload()
    tok = AutoTokenizer.from_pretrained(snapshot(tok_name))
    eos = tok.eos_token_id
    tg, n_test = test_grams()
    code = code_docs(max_stack)
    kept, dropped = [], 0
    for d in code:
        if grams(d) & tg:
            dropped += 1
        else:
            kept.append(d)
    rng = random.Random(0)
    rng.shuffle(kept)
    # FineWeb-Edu documents (stream; every 7th doc goes to eval, the rest to calib, so the splits never share docs)
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True).shuffle(seed=11, buffer_size=10000)

    def pack(doc_iter, n_seqs):
        out, buf = [], []
        for d in doc_iter:
            buf += tok(d, add_special_tokens=False)["input_ids"] + [eos]
            while len(buf) >= seq_len:
                out.append(buf[:seq_len]); buf = buf[seq_len:]
                if len(out) >= n_seqs:
                    return out
        return out
    n_code_c = int(n_calib * code_frac)
    n_code_e = int(n_eval * code_frac)
    k_eval = max(1, len(kept) // 10)
    code_eval_docs, code_calib_docs = kept[:k_eval], kept[k_eval:]
    fw_it = iter(fw)
    fw_calib = (ex["text"] for i, ex in enumerate(fw_it) if i % 7)
    calib = pack(iter(code_calib_docs), n_code_c) + pack(fw_calib, n_calib - n_code_c)
    fw_eval = (ex["text"] for i, ex in enumerate(iter(fw)) if i % 7 == 0)
    evals = pack(iter(code_eval_docs), n_code_e) + pack(fw_eval, n_eval - n_code_e)
    rng.shuffle(calib)
    # recheck: decoded training sequences vs test grams (must be 0)
    leak = sum(1 for s in calib if grams(tok.decode(s)) & tg)
    assert leak == 0, f"{leak} training sequences overlap test items after filtering"
    stats = dict(test_items=n_test, test_13grams=len(tg), code_docs=len(code), code_dropped=dropped, code_kept=len(kept),
                 calib=len(calib), calib_code=n_code_c, eval=len(evals), eval_code=n_code_e, seq_len=seq_len,
                 overlap_after=leak, sources="mbpp full train/prompt/validation + the-stack-smol python + fineweb-edu")
    os.makedirs("/vol/xkv/data", exist_ok=True)
    torch.save({"calib": calib, "eval": evals, "stats": stats}, f"/vol/xkv/data/{name}.pt")
    vol.commit()
    return stats


@app.local_entrypoint()
def main(name: str = "codemix-v1", tok_name: str = "Qwen/Qwen3.8-27B", n_calib: int = 900, n_eval: int = 16,
         seq_len: int = 2176, code_frac: float = 0.6, max_stack: int = 6000):
    print(build.remote(name, tok_name, n_calib, n_eval, seq_len, code_frac, max_stack))
