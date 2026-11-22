"""Frozen QMSum judge (kvx.evals.judge, rubric v2frac, gpt-5.5 reasoning low) on xkv gen_evals outputs.

python tools/xkv_qmsum_judge.py <gen_partial_qmsum.json> <strategy> out.jsonl [--workers 8]
xkv saves per item prompt_tail (ends "Query: ...<|im_end|>..."), output, gold; items are in seeded order, so arms of the
same run line up by index (checked through the gold). Prints mean, 95% CI half-width, judge cost.
Needs OPENAI_API_KEY.
"""
import argparse
import importlib.util
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kvx.evals import judge  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("partial")
    ap.add_argument("strategy")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="gpt-5.5")
    a = ap.parse_args()
    path = os.path.join(os.path.dirname(__file__), "..", "research", "qmsum_calib.py")
    spec = importlib.util.spec_from_file_location("qmsum_calib", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    judge.PROMPT = mod.VARIANTS["v2frac"]
    outs = json.load(open(a.partial))[a.strategy]["outputs"]
    rows, queries = [], {}
    for i, o in enumerate(outs):
        m = re.search(r"Query: (.*?)<\|im_end\|>", o["prompt_tail"], flags=re.S)
        assert m, f"no query in item {i}"
        queries[i] = m.group(1).strip()
        rows.append(dict(id=i, arm=a.strategy, text=o["output"], answer=o["gold"]))
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda r: judge.judge_one(r, queries[r["id"]], a.model), rows))
    with open(a.out, "w") as f:
        for r, row in zip(res, rows):
            f.write(json.dumps(dict(r, gold=row["answer"][:200])) + "\n")
    ok = [r["judge"] for r in res if r["judge"] is not None]
    m = sum(ok) / len(ok)
    sd = math.sqrt(sum((x - m) ** 2 for x in ok) / max(1, len(ok) - 1))
    print(json.dumps(dict(strategy=a.strategy, n=len(res), scored=len(ok), mean=round(m, 4),
                          ci_half=round(1.96 * sd / math.sqrt(len(ok)), 4),
                          cost_usd=round(sum(r["cost"] for r in res), 3), judge=a.model, rubric="v2frac")))


if __name__ == "__main__":
    main()
