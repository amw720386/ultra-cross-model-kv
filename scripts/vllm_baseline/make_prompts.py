"""Real ~32K-token prompts for the TTFT comparison: LongHealth questions over one patient's letters plus other
patients' letters as distractors (LongHealth's task-2 design), in seeded order. Writes their token ids as JSON, so
the engine and vLLM time exactly the same tokens.

    PYTHONPATH=src python scripts/vllm_baseline/make_prompts.py --out prompts.json [--n 6] [--tokens 32000]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

SEED = "xkv-demo-2026-11-20"


def build_prompts(tok, n: int, target_tokens: int) -> list[dict]:
    from xkv import genevals as G

    bench = json.loads(G._fetch("longhealth").read_text(encoding="utf-8"))
    pids = sorted(bench)
    flat = [(pid, qi) for pid in pids for qi in range(len(bench[pid]["questions"]))]
    random.Random(SEED).shuffle(flat)
    enc = lambda s: G._encode(tok, s)  # noqa: E731
    sep_n = len(enc(G._LH_SEP))

    def build(pid: str, qi: int) -> dict | None:
        p, q = bench[pid], bench[pid]["questions"][qi]
        opts = [q[f"answer_{letter}"] for letter in "abcde"]
        if q["correct"] not in opts:
            return None
        options = "\n".join(f"{L}: {o}" for L, o in zip("ABCDE", opts))
        empty = G._LH_TEMPLATE.format(documents="", question_text=q["question"], options=options)
        budget = target_tokens - len(enc(G.chat_prompt(tok, empty, system=G._LH_SYSTEM)))
        own = [(f"{pid}:{k}", t) for k, t in p["texts"].items()]
        used = sum(len(enc(t)) + sep_n for _, t in own)
        if used > budget:
            return None
        others = [(f"{op}:{k}", t) for op in pids if op != pid for k, t in bench[op]["texts"].items()]
        random.Random(f"xkv-demo-distract-{pid}-{qi}").shuffle(others)
        docs = list(own)
        for k, t in others:
            need = len(enc(t)) + sep_n
            if used + need <= budget:
                docs.append((k, t))
                used += need
        random.Random(f"xkv-demo-order-{pid}-{qi}").shuffle(docs)
        user = G._LH_TEMPLATE.format(documents=G._LH_SEP.join(t for _, t in docs), question_text=q["question"],
                                     options=options)
        ids = enc(G.chat_prompt(tok, user, system=G._LH_SYSTEM))
        return {"pid": pid, "qi": qi, "gold": "ABCDE"[opts.index(q["correct"])], "n_docs": len(docs),
                "n_tokens": len(ids), "ids": ids}

    out = []
    for pid, qi in flat:
        if (c := build(pid, qi)) is not None:
            c["index"] = len(out)
            out.append(c)
        if len(out) >= n:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=32000)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    from xkv.experiments import resolve

    tok = AutoTokenizer.from_pretrained(resolve("27b"))
    prompts = build_prompts(tok, a.n, a.tokens)
    for c in prompts:
        print(f"prompt {c['index']}: {c['pid']}/q{c['qi']} {c['n_tokens']} tokens, {c['n_docs']} letters", flush=True)
    Path(a.out).write_text(json.dumps(prompts))
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
