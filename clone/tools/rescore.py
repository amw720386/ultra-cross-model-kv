"""Re-score a finished arm from its saved generations (no GPU) and rewrite gens.jsonl + summary.json on the volume.
python tools/rescore.py RUN_ID ARM [--data-root DIR] [--note "why"]
Keeps the old summary as summary.orig.json. Used after the ZebraLogic gold fix (b4dbf73)."""
import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvx.evals import data, score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("arm")
    ap.add_argument("--data-root", default=os.path.expanduser("~/kvx-data/dl"))
    ap.add_argument("--note", default="rescored after the ZebraLogic gold fix (b4dbf73)")
    a = ap.parse_args()
    d = tempfile.mkdtemp()
    base = f"runs/{a.run_id}/{a.arm}"
    for f in ("gens.jsonl", "summary.json"):
        subprocess.run(["modal", "volume", "get", os.environ.get("KVX_VOLUME", "kvx"), f"{base}/{f}", d], check=True, capture_output=True)
    rows = [json.loads(l) for l in open(f"{d}/gens.jsonl")]
    summ = json.load(open(f"{d}/summary.json"))
    items = {}
    for b in {r["bench"] for r in rows}:
        n = summ["benches"][b]["n"]
        items.update({(b, it["id"]): it for it in data.load(b, a.data_root, n)})
    by = {}
    for r in rows:
        it = items[(r["bench"], r["id"])]
        if it["kind"] == "code":
            by.setdefault(r["bench"], []).append(r)
            continue
        r.update(score.score_item(it, r["text"]))
        r["answer"] = it["answer"]
        by.setdefault(r["bench"], []).append(r)
    for b, rs in by.items():
        sc = [x["score"] for x in rs if x.get("score") is not None]
        ent = summ["benches"][b]
        if sc and rs[0]["kind"] in ("letter", "label", "math", "zebra"):
            k = int(sum(sc))
            p, lo, hi = score.wilson(k, len(sc))
            ent.update(score=p, k=k, ci_lo=lo, ci_hi=hi)
        ent["rescored"] = a.note
    os.replace(f"{d}/summary.json", f"{d}/summary.orig.json")
    json.dump(summ, open(f"{d}/summary.json", "w"), indent=1)
    with open(f"{d}/gens.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for f in ("gens.jsonl", "summary.json", "summary.orig.json"):
        subprocess.run(["modal", "volume", "put", "-f", os.environ.get("KVX_VOLUME", "kvx"), f"{d}/{f}", f"{base}/{f}"], check=True,
                       capture_output=True)
    print(json.dumps({b: {k: v for k, v in e.items() if k in ("score", "n", "ci_lo", "ci_hi")} for b, e in summ["benches"].items()}))


if __name__ == "__main__":
    main()
