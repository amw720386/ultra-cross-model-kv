"""Merge kvx.evals results into the lead's seats/kvx-quality/ours.json (scoreboard ids -> our value, n, CI, retention).

python tools/quality_ours.py specs/q17-q32.json
spec: {"run_id": ..., "rig": "...", "cost_usd": x, "rows": {"<scoreboard id>": {"arm": "<mapped arm>", "bench": "...",
       "native": "<native arm>", "note": "..."}}}
Summaries are fetched from the Modal volume kvx (/vol/runs/<run_id>/<arm>/summary.json).
"""
import json
import os
import subprocess
import sys
import tempfile

OURS = os.path.expanduser("~/kvx-data/seats/kvx-quality/ours.json")


def summary(run_id, arm, cache):
    key = (run_id, arm)
    if key not in cache:
        d = tempfile.mkdtemp()
        subprocess.run(["modal", "volume", "get", os.environ.get("KVX_VOLUME", "kvx"), f"runs/{run_id}/{arm}/summary.json", d],
                       capture_output=True, check=True)
        cache[key] = json.load(open(os.path.join(d, "summary.json")))
    return cache[key]


def main(spec_path):
    spec = json.load(open(spec_path))
    ours = json.load(open(OURS)) if os.path.exists(OURS) else {}
    cache = {}
    for rid, r in spec["rows"].items():
        run_id = r.get("run_id", spec["run_id"])
        m = summary(run_id, r["arm"], cache)["benches"][r["bench"]]
        ent = dict(value=round(m["score"], 4), n=m["n"], rig=spec["rig"], run_id=f"{run_id}/{r['arm']}",
                   cost_usd=spec.get("cost_usd"))
        if "ci_lo" in m:
            ent["ci"] = round((m["ci_hi"] - m["ci_lo"]) / 2, 4)
        elif "ci_half" in m:
            ent["ci"] = round(m["ci_half"], 4)
        note = [r.get("note", "")]
        if r.get("native"):
            nat = summary(r.get("native_run", run_id), r["native"], cache)["benches"][r["bench"]]
            ent["native_value"] = round(nat["score"], 4)
            ent["retention"] = round(m["score"] / nat["score"], 4) if nat["score"] else None
            note.append(f"native {nat['score']:.4f} (n={nat['n']}) on the same rig/protocol, retention {ent['retention']}")
        if "mapped_frac" in m:
            ent["mapped_token_frac"] = round(m["mapped_frac"], 4)
            note.append(f"{100 * m['mapped_frac']:.1f}% of prompt tokens mapped from the source")
        ent["note"] = "; ".join(x for x in note if x)
        ours[rid] = ent
    os.makedirs(os.path.dirname(OURS), exist_ok=True)
    json.dump(ours, open(OURS, "w"), indent=1, sort_keys=True)
    print(json.dumps({k: (v["value"], v.get("native_value"), v["n"]) for k, v in ours.items()}, indent=0))


if __name__ == "__main__":
    main(sys.argv[1])
