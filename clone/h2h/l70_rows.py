"""Rows for kvx-cell-llama70 from one container's run dir (local copy), written through magni.feed.validate.append
(mode "raise": a serving row that is not recipe-ready is refused, never written).

  MAGNI_TOOLS=<magni checkout> python3 h2h/l70_rows.py RUN_DIR {32k|128k} LEDGER.jsonl [--usd-per-h 7.90] [--dry]

Per arm (stock, ours): ttft_p50_s, ttft_p99_s (nearest-rank over every timed request of the 3 reps; n stated),
ttft_cost_usd_per_req (TTFT p50 x box $/h), and quality rows: gate needles k/n (+ Wilson), HellaSwag-100, first-token
KL (ours vs stock, top-20 support) + top-1 agreement. recipe = the cell id on every row.
"""
import argparse, json, math, os, statistics, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("length", choices=["32k", "128k"]); ap.add_argument("ledger")
ap.add_argument("--usd-per-h", type=float, default=7.90)
ap.add_argument("--usd-source", default="Modal list H100 x2 (magni session 504642675a billed 2x H100 at $7.90/h)")
ap.add_argument("--dry", action="store_true")
a = ap.parse_args()
sys.path.insert(0, os.environ["MAGNI_TOOLS"])
from magni.feed.validate import append, check  # noqa: E402

RD, L = a.run_dir, a.length
CELL = f"meta-llama/Llama-3.3-70B-Instruct/h100/c1/{L}"
log = json.load(open(f"{RD}/run_log.json"))
here = os.path.dirname(os.path.abspath(__file__))


def j(p):
    return json.load(open(p)) if os.path.exists(p) else None


def ttfts(arm):
    v = []
    for rep in range(3):
        b = j(f"{RD}/{arm}/ttft_rep{rep}.json")
        if b:
            v += [r["ttft_s"] for r in b["records"] if r.get("ttft_s") is not None]
    return v


def p_nearest(v, q):
    s = sorted(v)
    return s[max(0, math.ceil(q * len(s)) - 1)]


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]


fz = j(f"{RD}/frozen_config.json") or j(f"{RD}/frozen_config_used.json")
isl = {"32k": 32632, "128k": 130936}[L]
base = {"recipe": CELL, "model": "meta-llama/Llama-3.3-70B-Instruct",
        "weights_repo": "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic@f50dbad2c84590ca17dc51e207c34321b65ff14b",
        "weights_note": "magni-hq/llama70b-h100-recipe pins this FP8-dynamic checkpoint as the H100 stock (BF16 cannot serve on 2x H100)",
        "chip": "h100", "gpus": 2, "gpu_names": log.get("gpu_names"), "box": "Modal H100!:2", "engine": "vllm", "engine_version": "0.34.0",
        "image": log.get("image"), "concurrency": 1, "isl": isl, "osl": 32, "context": {"32k": 32768, "128k": 131072}[L],
        "workload": {"kind": "needle retrieval, chat template, c=1", "n_prompts": 24, "osl_note": "max_tokens 32; TTFT metric"},
        "run_id": log["run_id"], "run_dir": f"kvx-h2h-runs:{log['run_id']}", "ts": log.get("ended") or log.get("started"),
        "precision": ("fp8 weights (FP8-dynamic), " + ("fp8 KV cache (pre-registered deviation forced by our arm's memory at 128K; "
                      "magni fp8kv arm, session 504642675a), both arms" if L == "128k" else "bf16 KV cache")),
        "deviation": "--max-model-len set to the cell (magni's argv has 16384)" + ("; --kv-cache-dtype fp8 on both arms" if L == "128k" else "")}
rows = []
for arm in ("stock", "ours"):
    argv = j(f"{RD}/argv_{arm}.json") or j(f"{RD}/argv_{'ours_dev0' if arm == 'ours' else arm}.json")
    v = ttfts(arm)
    if not v:
        continue
    a_ = dict(base, arm=arm, launch=argv,
              levers=(None if arm == "stock" else {"frozen_config": fz.get("pick") if fz else None,
                                                    "frozen_sha256": fz.get("pick_sha256") if fz else None}),
              lever_id=("stock" if arm == "stock" else "kvx-map-8b-to-70b"))
    p50, p99 = statistics.median(v), p_nearest(v, 0.99)
    rows += [dict(a_, metric="ttft_p50_s", value=round(p50, 4), n=len(v)),
             dict(a_, metric="ttft_p99_s", value=round(p99, 4), n=len(v), p99_note=f"nearest-rank over n={len(v)} requests (3 reps)"),
             dict(a_, metric="ttft_cost_usd_per_req", value=round(p50 * a.usd_per_h / 3600, 6), n=len(v),
                  usd_per_h=a.usd_per_h, usd_source=a.usd_source, cost_basis="TTFT p50 x box $/h (both GPUs held by the request)")]
    q = j(f"{RD}/{arm}/gate_needles/quality.json")
    if q:
        rows.append(dict(a_, metric=f"needle_acc_{L}", value=round(q["correct"] / q["n"], 4), correct=q["correct"], n=q["n"],
                         wilson95=wilson(q["correct"], q["n"]), prompts_sha256=q.get("prompts_sha256")))
    h = j(f"{RD}/{arm}/hellaswag/quality.json")
    if h:
        rows.append(dict(a_, metric="hellaswag100_acc", value=round(h["correct"] / h["n"], 4), correct=h["correct"], n=h["n"],
                         note=("ours serves it through the map (min_len 1, labelled)" if arm == "ours" else None)))
kl = None
if os.path.exists(f"{RD}/stock/first_token_logprobs.jsonl") and os.path.exists(f"{RD}/ours/first_token_logprobs.jsonl"):
    kl = json.loads(subprocess.run([sys.executable, f"{here}/kl_first_token.py", f"{RD}/stock/first_token_logprobs.jsonl",
                                    f"{RD}/ours/first_token_logprobs.jsonl"], capture_output=True, text=True).stdout)
    rows.append(dict(base, arm="ours", lever_id="kvx-map-8b-to-70b", metric="first_token_kl_vs_stock", value=kl["kl_mean"],
                     n=kl["n"], kl=kl, launch=j(f"{RD}/argv_ours.json") or j(f"{RD}/argv_ours_dev0.json")))
for r in rows:
    miss = check(r)
    print(r["metric"], r.get("arm"), r["value"], "n", r.get("n"), "READY" if not miss else f"MISSING {miss}")
    if not a.dry:
        append(a.ledger, r, mode="raise")
print(json.dumps({"rows": len(rows), "ledger": a.ledger, "dry": a.dry}))
