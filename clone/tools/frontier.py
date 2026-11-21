"""Source-size frontier -> Qwen3-32B: quality (KL) next to the 32K compute ceiling. Prints a markdown table."""
import json, os
OUT = os.path.join(os.path.dirname(__file__), "..", "out")
CEIL = {"0.6B": 8.0, "1.7B": 6.8, "4B": 3.7, "8B": 2.8, "14B": 1.9}
RUNS = {"0.6B": "q0.6B-q32-c2k", "1.7B": "q17-q32-cal4000", "4B": "q4B-q32-c2k", "8B": "q8B-q32-c2k", "14B": "q14-q32-cal2000"}
print("| Source | 32K TTFT ceiling | small alone | mapped (k16/24 + rms) | + tail 64 | + 4 sink + tail 256 (or 64) |")
print("|---|---|---|---|---|---|")
for s, run in RUNS.items():
    p = os.path.join(OUT, run, "result.json")
    if not os.path.exists(p):
        continue
    e = json.load(open(p))["eval"]
    g = lambda pre: min((v["kl"] for k, v in e.items() if k.startswith("map") and k.endswith(pre)), default=None)
    mapped = min(v["kl"] for k, v in e.items() if k.startswith("map_k") and k.endswith("_rms") and "sink" not in k)
    t64 = g("_rms_tail64")
    sk = g("_rms_sink4_tail256") or g("_rms_sink4_tail64")
    f = lambda x: f"{x:.3f}" if x is not None else "-"
    print(f"| {s} | {CEIL[s]}x | {e['ref_source_model']['kl']:.3f} | {mapped:.3f} | {f(t64)} | {f(sk)} |")
