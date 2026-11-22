"""Write the pooled Terminal-Bench 2.0 mapped row from tools/agentic_rows.py's tb2 output (refuses unless quality_ok).

python tools/agentic_rows.py tb2 <jobs root> <logs dir> > tb2.json
python tools/tb2_row.py tb2.json --cost 14.5 [--dry]
"""
import argparse
import json
import os

OURS = os.path.expanduser("~/clone-push/data/clone-push-kvx/seats/kvx-quality/ours.json")
THEIRS = 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agg")
    ap.add_argument("--cost", type=float, required=True)
    ap.add_argument("--native", default="9/18 on the first 18 tasks (same harness)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    d = json.load(open(a.agg))["tb2"]
    if d["quality_ok"] is not True:
        raise SystemExit(f"not writing: quality_ok={d['quality_ok']} ({d.get('invalid') or d.get('unverified')})")
    verdict = "BEATEN" if d["lo"] > THEIRS else ("behind" if d["hi"] < THEIRS else "TIE (theirs inside our CI)")
    note = ("routed by task category (declared 13:55Z): code rows use the code-distilled map "
            "4b-27b-r2-distill-code+attn (code-mix: 60% code + 40% FineWeb; decontam per kvx-pairs: 0 overlaps vs "
            "MBPP/HumanEval/BigCodeBench test items); "
            f"Terminal-Bench 2.0 (Harbor 0.27, terminus-2 agent, Modal sandboxes), all 89 tasks, 1 trial each: "
            f"{d['passed']}/{d['n']} scored = {d['value']:.4f}, Wilson 95% [{d['lo']:.3f}, {d['hi']:.3f}] -> {verdict} "
            f"vs theirs {THEIRS} (1/10 scored); native 27B {a.native}; map_once (multi-turn, lead rule 12:4xZ): each "
            f"conversation's first request maps all but its last min(256, 25%) tokens from Qwen3.5-4B, later turns are "
            f"appended natively; mapped_frac first requests {d['mapped_frac_first']:.3f} (rule: >= .75), all request "
            f"tokens {d['mapped_frac_all']:.3f}; guard: {d['later_requests']} of ~{d['later_expected']} later requests "
            f"reused the stored prefix; solved: {', '.join(d['solved'])}; ablation (notes only): the earlier re-map "
            f"protocol on the first 18 with the plain distilled map gave 3/18")
    row = dict(value=d["value"], n=d["n"], ci=d["ci"], mapped_token_frac=d["mapped_frac_first"],
               mapped_token_frac_all=d["mapped_frac_all"], quality_ok=True,
               rig="Harbor 0.27 terminal-bench@2.0, terminus-2, Modal CPU sandboxes; agent Qwen3.8-27B on vLLM 0.34, "
                   "1x H200 per batch (Modal) + kvx HybridCrossModelConnector (verified xkv adapter, map_once, FLA fast "
                   "path), batches of 18 tasks",
               run_id="tb2-a/xkvcode27mo2-b1..b6", cost_usd=a.cost, note=note)
    print(json.dumps(row, indent=1)[:2500])
    if not a.dry:
        o = json.load(open(OURS))
        o["bundle-terminal-bench-2-0-mapped"] = row
        json.dump(o, open(OURS, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
