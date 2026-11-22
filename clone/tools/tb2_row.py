"""Write the pooled Terminal-Bench 2.0 mapped row from tools/agentic_rows.py's tb2 output (refuses unless quality_ok).

python tools/agentic_rows.py tb2 <jobs root> <logs dir> plans/tb2_fit1800.json > tb2.json
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
    ap.add_argument("--native", default="9/18")
    ap.add_argument("--fit", default=os.path.join(os.path.dirname(__file__), "..", "plans", "tb2_fit1800.json"))
    ap.add_argument("--gpu", required=True, help="gpu_measured for the row (decision rig-h200-hold), e.g. 'NVIDIA H200'")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    d = json.load(open(a.agg))["tb2"]
    if d["quality_ok"] is not True:
        raise SystemExit(f"not writing: quality_ok={d['quality_ok']} ({d.get('invalid') or d.get('unverified')})")
    verdict = "BEATEN" if d["lo"] > THEIRS else ("behind" if d["hi"] < THEIRS else "TIE (theirs inside our CI)")
    wc = d["worst_case"]
    fitf = json.load(open(a.fit))
    b1 = d["per_batch"].get("b1", [0, 0])
    nat_k, nat_n = (int(x) for x in a.native.split("/"))
    note = ("decision tb2-long-excluded: the row is the 73 TB2 2.0 tasks with an agent timeout <= 1800 s (task.toml); "
            f"{len(fitf['excluded_long'])} longer tasks are excluded by our serve limit (55-min boot, Modal 1-h app "
            f"limit), NOT by result: {', '.join(sorted(fitf['excluded_long']))}; "
            "routed by task category (declared 13:55Z): code rows use the code-distilled map "
            "4b-27b-r2-distill-code+attn (code-mix: 60% code + 40% FineWeb; decontam per kvx-pairs: 0 overlaps vs "
            "MBPP/HumanEval/BigCodeBench test items); "
            f"Terminal-Bench 2.0 (Harbor 0.27, terminus-2 agent, Modal sandboxes), 1 trial each: {d['passed']}/{d['n']} "
            f"scored ({d['unscored_fit']} of the 73 not scored: cut by the boot) = {d['value']:.4f}, Wilson 95% "
            f"[{d['lo']:.3f}, {d['hi']:.3f}] -> {verdict} vs theirs {THEIRS} (1/10 scored); WORST CASE over all "
            f"{wc['n']} tasks (long tasks as scored inside the boot: solved "
            f"{', '.join(wc['long_solved']) or 'none'}; every other long task and every unscored trial counted as a "
            f"failure): {wc['passed']}/{wc['n']} = "
            f"{wc['value']:.4f}, Wilson [{wc['lo']:.3f}, {wc['hi']:.3f}] "
            f"({'still above' if wc['lo'] > THEIRS else 'NOT above'} theirs); "
            f"retention vs native on the subset with a same-harness native arm (batch 1, 18 tasks): mapped "
            f"{b1[0]}/{b1[1]} vs native {a.native} -> {b1[0] / nat_k:.3f}; "
            f"map_once (multi-turn, lead rule 12:4xZ): each conversation's first request maps all but its last "
            f"min(256, 25%) tokens from Qwen3.5-4B, later turns are appended natively; mapped_frac first requests "
            f"{d['mapped_frac_first']:.3f} (rule: >= .75), all request tokens {d['mapped_frac_all']:.3f}; guard: "
            f"{d['later_requests']} of ~{d['later_expected']} later requests reused the stored prefix; solved: "
            f"{', '.join(d['solved'])}; ablation (notes only): the re-map protocol on batch 1 with the plain distilled "
            f"map gave 3/18")
    row = dict(gpu_measured=a.gpu, value=d["value"], n=d["n"], ci=d["ci"], native_value=round(nat_k / nat_n, 4), retention=round(b1[0] / nat_k, 4),
               worst_case=wc, mapped_token_frac=d["mapped_frac_first"],
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
