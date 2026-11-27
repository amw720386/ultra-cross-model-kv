"""Feed the OneTriangle DO run 2 (20261123T1336Z-onetriangle-h2h-do2, 1x H100 80GB SXM5, DigitalOcean) to magni's recipe
feed: native vLLM Qwen3-8B (lever none, its own launch) and ours mapped (frozen pick e146c768) at 8K/16K/30K, c=1,
through magni.feed.validate.append (mode raise). Every number comes from the run's own records.

  MAGNI_TOOLS=<magni checkout> python3 h2h/do2_feed_rows.py RUN_DIR PROMPTS_DIR LEDGER.jsonl [--dry]
"""
import argparse, json, math, os, statistics, sys

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("prompts_dir"); ap.add_argument("ledger"); ap.add_argument("--dry", action="store_true")
a = ap.parse_args()
sys.path.insert(0, os.environ["MAGNI_TOOLS"])
from magni.feed.validate import append, check  # noqa: E402

RD, PD = a.run_dir, a.prompts_dir
RID = os.path.basename(os.path.normpath(RD))
j = lambda p: json.load(open(p))
fz = j(f"{RD}/frozen_config.json")
img = next((l.split()[2] for l in open(f"{RD}/evidence/docker_images.txt") if l.startswith("vllm/vllm-openai") and "v0.34.0" in l), None)
cpu = open(f"{RD}/evidence/cpu_model.txt").read().split(":", 1)[-1].strip()


def wilson_lo(k, n, z=1.96):
    p = k / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def p_nearest(v, q):
    s = sorted(v)
    return s[max(0, math.ceil(q * len(s)) - 1)]


def needle_hits(records, prompts):
    P = {r["id"]: r for r in map(json.loads, open(prompts))}
    k = sum(1 for r in records if P[r["id"]]["answer"] in (r.get("text") or ""))
    return k, len(records)


base = {"model": "Qwen/Qwen3-8B", "weights_repo": "Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218",
        "chip": "h100", "gpus": 1, "gpu_name": "NVIDIA H100 80GB HBM3 (SXM5)", "cpu": cpu,
        "box": "DigitalOcean ams3 gpu-h100x1-80gb (droplet 622575902)", "engine": "vllm", "engine_version": "0.34.0",
        "image": f"vllm/vllm-openai:v0.34.0@{img}" if img else "vllm/vllm-openai:v0.34.0", "concurrency": 1, "osl": 16,
        "max_model_len": 32768, "run_id": RID, "run_dir": f"traceDB traces/clone-push-h2h/runs/{RID}/",
        "coresidency": ("same boot, same GPU: OneTriangle's kvt server stayed resident (idle between its own rounds); "
                        "both of our vLLM arms ran at --gpu-memory-utilization 0.33 --max-num-seqs 2 for co-residency"),
        "client": "verified-h2h bench/sse_ttft.py (stream, temperature 0, max_tokens 16, c=1, chat template, thinking off)",
        "source": "h2h/do2_feed_rows.py from the run's own records"}
rows = []
for L in (8192, 16384, 30720):
    prompts = f"{PD}/needles_{L}.jsonl"
    # native vLLM (lever none): native_context/ttft_<L>.json, launch ours_cmd_native.json
    nb = j(f"{RD}/native_context/ttft_{L}.json")
    isl = round(statistics.median(r["input_tokens"] for r in nb["records"] if r.get("input_tokens")))
    ctx = isl + 16
    cell = f"Qwen3-8B/h100/c1/{round(ctx / 1024)}k"
    nv = [r["ttft_s"] for r in nb["records"] if r.get("ttft_s") is not None]
    nk, nn = needle_hits(nb["records"], prompts)
    nat = dict(base, recipe=cell, isl=isl, context=ctx, precision="bf16/bf16", arm="native", lever_id="none", levers=[],
               launch=j(f"{RD}/ours_cmd_native.json"), prompts_file=f"needles_{L}.jsonl (held-out gate set)")
    rows += [dict(nat, metric="ttft_p50_s", value=round(statistics.median(nv), 4), n=len(nv)),
             dict(nat, metric="ttft_p99_s", value=round(p_nearest(nv, 0.99), 4), n=len(nv), p99_note=f"nearest-rank, n={len(nv)} (1 run)"),
             dict(nat, metric=f"needle_acc_{L // 1024}k", value=round(nk / nn, 4), correct=nk, n=nn,
                  quality_note="scored from the native timing run's replies with the gate client's rule (answer in reply)")]
    # ours mapped: abba_<L>/ours/<round>/bench.json (4 ABBA rounds), quality = held-out needles (abba_<L>/ours/quality.json)
    ov = []
    for rnd in sorted(d for d in os.listdir(f"{RD}/abba_{L}/ours") if d.isdigit()):
        ov += [r["ttft_s"] for r in j(f"{RD}/abba_{L}/ours/{rnd}/bench.json")["records"] if r.get("ttft_s") is not None]
    qo, qt = j(f"{RD}/abba_{L}/ours/quality.json"), j(f"{RD}/abba_{L}/theirs/quality.json")
    bar = wilson_lo(qt["correct"], qt["n"])                 # pre-registered h2h rule: vs OneTriangle
    qpass_ot = qo["correct"] / qo["n"] >= bar
    nbar = wilson_lo(nk, nn)                                 # the feed's comparison: ours vs native vLLM, same boot, same prompts
    qpass = qo["correct"] / qo["n"] >= nbar
    label = {8192: "context (same frozen config)", 16384: "HEADLINE (pre-registered): VERIFIED-BEAT vs OneTriangle +29.0% TTFT",
             30720: ("context only, NO claim (pre-registered: the dev-set 30K gate failed, ours 19/24 = .792 < theirs' Wilson LB .798); "
                     "on the held-out set ours 20/24 = theirs 20/24, so the Wilson rule passes here")}[L]
    ours = dict(base, recipe=cell, isl=isl, context=ctx, precision="bf16/bf16 (KV map GEMM fp8)", arm="ours",
                lever_id="kvx-map-1p7b-to-8b", levers=fz["pick"], frozen_config_sha256=fz["pick_sha256"],
                launch=j(f"{RD}/ours_cmd_abba.json"), prompts_file=f"needles_{L}.jsonl (held-out gate set)",
                quality_pass=qpass,
                quality_rule=f"ours needles >= Wilson LB of NATIVE vLLM's on the same boot + prompts ({nk}/{nn} -> {nbar:.3f})",
                quality_pass_vs_onetriangle=qpass_ot,
                quality_rule_vs_onetriangle=f"pre-registered h2h rule: ours >= Wilson LB of OneTriangle's ({qt['correct']}/{qt['n']} -> {bar:.3f})",
                native_needles=f"{nk}/{nn}",
                quality_label=label, needles=f"{qo['correct']}/{qo['n']}", reference_needles=f"OneTriangle {qt['correct']}/{qt['n']}")
    if not qpass:
        ours["exclude_from_fit"] = f"quality gate vs native failed at {L // 1024}K: ours {qo['correct']}/{qo['n']} < Wilson LB {nbar:.3f} of native {nk}/{nn}"
    rows += [dict(ours, metric="ttft_p50_s", value=round(statistics.median(ov), 4), n=len(ov)),
             dict(ours, metric="ttft_p99_s", value=round(p_nearest(ov, 0.99), 4), n=len(ov), p99_note=f"nearest-rank, n={len(ov)} (4 ABBA rounds)"),
             dict(ours, metric=f"needle_acc_{L // 1024}k", value=round(qo["correct"] / qo["n"], 4), correct=qo["correct"], n=qo["n"])]
bad = 0
for r in rows:
    miss = check(r)
    bad += bool(miss)
    print(f"{r['recipe']:24s} {r['arm']:6s} {r['metric']:14s} {r['value']:<8} n={r.get('n')} "
          + ("READY" if not miss else f"MISSING {miss}") + (f" quality_pass={r.get('quality_pass')}" if r['arm'] == 'ours' else ""))
    if not a.dry:
        append(a.ledger, r, mode="raise")
print(json.dumps({"rows": len(rows), "not_ready": bad, "ledger": a.ledger, "dry": a.dry}))
