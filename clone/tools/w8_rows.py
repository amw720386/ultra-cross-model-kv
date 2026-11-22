"""ours.json rows from the 8xH100 1M window (decision w8-1m-window): per mapped config, TTFT at the 256K point and the 1M
peak (speedup vs the SAME-BOOT native vLLM at the same length), with quality vs that native on the same prompt files
(refused otherwise) and the spec labels. python tools/w8_rows.py out/speed/w8b [--write] [--cost-usd X]
Reads <dir>/<i>.json (modal_speed per-config entries: {cfg, result} or {cfg, error, partial_rows})."""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from speed_quality import cmp  # noqa: E402

OURS = os.path.expanduser("~/clone-push/data/clone-push-kvx/seats/kvx-speed/ours.json")
L256, L1M = 262144, 1048320
# card maxima (kvx-research context_limits.json): (source, target)
CARDS = {("Qwen/Qwen3.5-4B", "Qwen/Qwen3.8-27B"): (1010000, 1000000),
         ("Qwen/Qwen3.5-2B", "Qwen/Qwen3.8-27B"): (262144, 1000000),
         ("google/gemma-4-E4B", "google/gemma-4-31B"): (131072, 262144),
         ("google/gemma-4-E2B", "google/gemma-4-31B"): (131072, 262144)}
PAIR_ID = {("Qwen/Qwen3.5-4B", "Qwen/Qwen3.8-27B"): "home-qwen3-5-4b-qwen3-8-27b",
           ("Qwen/Qwen3.5-2B", "Qwen/Qwen3.8-27B"): "home-qwen3-5-2b-qwen3-8-27b",
           ("google/gemma-4-E4B", "google/gemma-4-31B"): "home-gemma-4-e4b-gemma-4-31b",
           ("google/gemma-4-E2B", "google/gemma-4-31B"): "home-gemma-4-e2b-gemma-4-31b"}


def spec_label(src, tgt, L):
    s, t = CARDS[(src, tgt)]
    over = [f"source card max {s:,}" for _ in [0] if L > s] + [f"target card max {t:,}" for _ in [0] if L > t]
    return f"beyond model spec ({'; '.join(over)})" if over else "within model spec"


def rows_of(e):
    r = e.get("result") or {}
    return {x["len"]: x["ttft_s"] for x in (r.get("rows") or e.get("partial_rows") or [])}


def quality(nat, m):
    """vs same-boot native on the same prompt files; None if refused (files differ / missing)."""
    if not nat or not m:
        return None, "no quality (config failed before its quality phase)"
    qa, qb = nat.get("quality_files"), m.get("quality_files")
    keys = ("fw", "long", "long_fw")
    if not qa or not qb or {k: qa.get(k) for k in keys} != {k: qb.get(k) for k in keys}:
        return None, "quality REFUSED: prompt files differ or unrecorded"
    q = {}
    for k in ("first_token_top20", "long_first_token_top20"):
        if nat.get(k) and m.get(k):
            q[k] = cmp(nat[k], m[k])
    if nat.get("needle") and m.get("needle"):
        q["needles"] = (sum(i["correct"] for i in m["needle"]), sum(i["correct"] for i in nat["needle"]), len(m["needle"]))
    parts = []
    if "first_token_top20" in q:
        a, n, kl = q["first_token_top20"]; parts.append(f"FineWeb first token top1 {a}/{n} KL~{kl:.3f}")
    if "long_first_token_top20" in q:
        a, n, kl = q["long_first_token_top20"]
        parts.append(f"{n} long prompts @{m.get('long_len')} (PG19 {m.get('long_n_pg')} + FineWeb {m.get('long_n_fw')}) "
                     f"top1 {a}/{n} KL~{kl:.3f}")
    if "needles" in q:
        a, b, n = q["needles"]; parts.append(f"needles @{m.get('long_len')} mapped {a}/{n} vs native {b}/{n}")
    files = f"prompt files fw {qb['fw'][1]}, long {qb.get('long', ['', ''])[1]}, long_fw {qb.get('long_fw', ['', ''])[1]} (sha256[:16], both arms)"
    return q, "; ".join(parts) + "; " + files


def main(d, write=False, cost=None):
    ents = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "*.json")), key=lambda f: int(os.path.basename(f).split(".")[0]))]
    natives = {e["cfg"]["tgt"]: e for e in ents if e["cfg"]["mode"] == "native"}
    out = {}
    for e in ents:
        c = e["cfg"]
        if c["mode"] != "mapped":
            print("native", c["tgt"], rows_of(e), "rc-ok" if e.get("result") else "FAILED/partial")
            continue
        pid = PAIR_ID.get((c["src"], c["tgt"]))
        ne = natives.get(c["tgt"])
        if pid is None or ne is None:
            print("skip", c["tag"], "no pair id or no same-boot native"); continue
        rm, rn = rows_of(e), rows_of(ne)
        q, qnote = quality(ne.get("result"), e.get("result"))
        rig = (f"8xH100 SXM (Modal), vLLM 0.34 TP=8, {c['extra'].get('connector')} connector, source TP-split"
               + (", YaRN x4 from 262144 on the target (both arms)" if c.get("yarn") else ""))
        share = round(cost / len(ents), 2) if cost else None
        base = dict(n=1, ci=None, rig=rig, run_id="w8-peaks-1m-r2", value_hf_baseline=None, cost_usd=share, blocked=None)
        if L256 in rm and L256 in rn:
            out[f"{pid}-256K-ttft"] = dict(base, value=round(rm[L256], 4), baseline_vllm=round(rn[L256], 4), context=L256,
                                           quality_ok=q is not None,
                                           note=f"{spec_label(c['src'], c['tgt'], L256)}. Quality vs same-boot native: {qnote}. "
                                                f"Speedup vs our native {rn[L256] / rm[L256]:.2f}x.")
        if L1M in rm and L1M in rn:
            sp = rn[L1M] / rm[L1M]
            out[f"{pid}-peak"] = dict(base, value=round(sp, 3), baseline_vllm=round(rn[L1M], 4), mapped_ttft_s=round(rm[L1M], 4),
                                      context=L1M, quality_ok=q is not None,
                                      note=(f"1M = 2^20 - 256 (context {L1M}, native arm the same length); "
                                            f"{spec_label(c['src'], c['tgt'], L1M)}. TTFT native {rn[L1M]:.2f} s / mapped "
                                            f"{rm[L1M]:.2f} s, same boot. Supported-max point {L256} quality (gate): {qnote}"))
    for k, v in out.items():
        print(k, v["value"], "base", v["baseline_vllm"], "q_ok", v["quality_ok"], "|", v["note"][:260])
    if write and out:
        o = json.load(open(OURS))
        o.update(out)
        json.dump(o, open(OURS, "w"), indent=1, sort_keys=True)
        print("wrote", len(out), "rows to", OURS)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    cost = float(a[a.index("--cost-usd") + 1]) if "--cost-usd" in a else None
    main(a[0], write="--write" in a, cost=cost)
