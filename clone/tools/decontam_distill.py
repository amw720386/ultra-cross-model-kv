"""Decontamination check for the distilled 4B -> 27B xkv maps: rebuild the exact distillation training sequences
(xkv.evals.load_heldout_text, as task_distill calls it: FineWeb-Edu sample-10BT, split calib, seed 11) and count
13-word n-gram overlaps with the bundle benchmarks' items (prompts + references, as the clone's loaders build them).

cd clone && PYTHONPATH=.:../src python tools/decontam_distill.py --data-root <bench files> --n-seqs 1500 --len 4224
(runs on CPU; streams FineWeb-Edu). Output: JSON with per-benchmark item counts that share >= 1 13-gram.
"""
import argparse
import json
import re


def grams(text, n=13):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--n-seqs", type=int, default=1500)
    ap.add_argument("--len", type=int, default=4224)
    ap.add_argument("--tok", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from xkv import evals
    from kvx.evals import data
    tok = AutoTokenizer.from_pretrained(a.tok)
    seqs = evals.load_heldout_text(tok, a.n_seqs, a.len, seed=11, split="calib", source="fineweb")
    train = set()
    for s in seqs:
        ids = s.tolist() if hasattr(s, "tolist") else list(s)
        train |= grams(tok.decode(ids))
    res = {}
    for b in ["arc_chat", "contract_nli", "bamboo_paperqa", "longhealth", "qmsum", "humaneval", "mbpp", "math500",
              "bigcodebench"]:
        items = data.load(b, a.data_root, a.n)
        hits = []
        for it in items:
            g = grams(it["context"] + " " + it["question"] + " " + str(it.get("answer") or ""))
            if g & train:
                hits.append(it["id"])
        res[b] = dict(items=len(items), items_with_13gram_overlap=len(hits), ids=hits[:10])
    print(json.dumps(dict(train_seqs=len(seqs), train_13grams=len(train), benches=res), indent=1))


if __name__ == "__main__":
    main()
