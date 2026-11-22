"""Decontamination check of the distilled 4B -> 27B xkv maps, CPU on Modal (xkv's FineWeb streaming loader needs > 10 GB).
The local entrypoint hashes every bundle item's 13-word grams (prompt + reference, as kvx.evals.data builds them); the
remote CPU job rebuilds the exact distillation sequences (xkv.evals.load_heldout_text: FineWeb-Edu sample-10BT, split
calib, seed 11, as task_distill calls it) and returns the shared hashes. Output: items per benchmark sharing >= 1 13-gram.

cd clone && PYTHONPATH=. python tools/modal_decontam.py <bench files> bench_grams.json   (any python with pyarrow)
PYTHONPATH=.:../src modal run tools/modal_decontam.py --bench-file bench_grams.json --out decontam.json
"""
import hashlib
import json
import os
import re

import modal

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
         .pip_install("transformers", "datasets", "huggingface_hub", "numpy")
         .env({"HF_HUB_CACHE": "/tmp/hub", "HF_DATASETS_CACHE": "/tmp/ds"})
         .add_local_python_source("xkv"))
hf_shared = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
app = modal.App(os.environ.get("KVX_APP", "kvx-qcpu-decontam"), image=image)
BENCHES = ["arc_chat", "contract_nli", "bamboo_paperqa", "longhealth", "qmsum", "humaneval", "mbpp", "math500",
           "bigcodebench"]


def gram_hashes(text, n=13):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {int.from_bytes(hashlib.blake2b(" ".join(w[i:i + n]).encode(), digest_size=8).digest(), "little")
            for i in range(len(w) - n + 1)}


@app.function(cpu=4, memory=32768, timeout=3000, volumes={"/hf": hf_shared.read_only()})
def train_hits(bench_hashes: list, n_seqs: int, seq_len: int, tok_name: str):
    from transformers import AutoTokenizer
    from xkv import evals
    allg = set(bench_hashes)
    tok = AutoTokenizer.from_pretrained(tok_name, cache_dir="/hf/hub")  # read-only weights volume
    seqs = evals.load_heldout_text(tok, n_seqs, seq_len, seed=11, split="calib", source="fineweb")
    hit, ntrain = set(), 0
    for s in seqs:
        g = gram_hashes(tok.decode(s.tolist() if hasattr(s, "tolist") else list(s)))
        ntrain += len(g)
        hit |= g & allg
    return dict(train_seqs=len(seqs), train_13grams=ntrain, hits=sorted(hit))


def bench_hashes(data_root, n):
    """{bench: [(item id, [hashes])]}; also runnable without modal: python tools/modal_decontam.py <root> <out.json>"""
    from kvx.evals import data
    return {b: [(it["id"], sorted(gram_hashes(it["context"] + " " + it["question"] + " " + str(it.get("answer") or ""))))
                for it in data.load(b, data_root, n)] for b in BENCHES}


@app.local_entrypoint()
def main(out: str, data_root: str = "", bench_file: str = "", n_seqs: int = 1500, seq_len: int = 4224, tok: str = "Qwen/Qwen3.8-27B", n: int = 100):
    bench = json.load(open(bench_file)) if bench_file else bench_hashes(data_root, n)
    bench = {b: [(i, set(g)) for i, g in v] for b, v in bench.items()}
    allg = sorted(set().union(*(g for v in bench.values() for _, g in v)))
    r = train_hits.remote(allg, n_seqs, seq_len, tok)
    hit = set(r["hits"])
    res = {b: dict(items=len(v), items_with_13gram_overlap=sum(1 for _, g in v if g & hit),
                   ids=[i for i, g in v if g & hit][:10]) for b, v in bench.items()}
    o = dict(train_seqs=r["train_seqs"], train_13grams=r["train_13grams"], bench_13grams=len(allg), benches=res)
    json.dump(o, open(out, "w"), indent=1)
    print(json.dumps(o, indent=1))


if __name__ == "__main__":
    import sys
    json.dump(bench_hashes(sys.argv[1], 100), open(sys.argv[2], "w"))
