"""Bundle rows from xkv gen_evals result files (the xkv engine, clone/modal_xkv.py) into seats/kvx-quality/ours.json.

python tools/xkv_ours.py --mapped gen_evals-<distill>.json --target gen_evals-<r2 a>.json gen_evals-<r2 b>.json \
    --r2 gen_evals-<r2 a>.json gen_evals-<r2 b>.json --cost 3.1 [--decontam decontam.json] [--dry]
Row = the mapped file's transfer@cap arm; native_value = the target arm on the same seeded items; the r2 arm goes in
the note. Wilson 95% CI half-width; mapped_token_frac = the per-prompt floor under the cap (tail = min(256, n // 4)).
QMSum is skipped here (its row is the frozen v2frac judge mean, written by the judge step).
"""
import argparse
import json
import math
import os

OURS = os.path.expanduser("~/clone-push/data/clone-push-kvx/seats/kvx-quality/ours.json")
ROWS = {"arc_chat": "bundle-arc-chat-mapped", "contract_nli": "bundle-contractnli-mapped",
        "bamboo_paperqa": "bundle-bamboo-paperqa-mapped", "longhealth": "bundle-longhealth-mapped",
        "humaneval": "bundle-humaneval-mapped", "mbpp": "bundle-mbpp-mapped", "math500": "bundle-math-500-mapped"}
THEIRS = {"arc_chat": 0.97, "contract_nli": 0.61, "bamboo_paperqa": 0.893, "longhealth": 0.76, "humaneval": 0.91,
          "mbpp": 0.88, "math500": 0.59}


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def frac(n):
    return 1 - min(256, n // 4) / n


def arms(paths):
    out = {}
    for p in paths:
        for b, r in json.load(open(p))["result"].items():
            for s, m in r.items():
                out[(b, s)] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapped", required=True)
    ap.add_argument("--target", nargs="+", required=True)
    ap.add_argument("--r2", nargs="+", default=[])
    ap.add_argument("--cost", type=float, required=True)
    ap.add_argument("--decontam", default="")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    mp = json.load(open(a.mapped))
    mapper = mp["args"]["mapper_name"]
    m, t, r2 = arms([a.mapped]), arms(a.target), arms(a.r2)
    dc = json.load(open(a.decontam))["benches"] if a.decontam else {}
    ours = json.load(open(OURS))
    out = {}
    for b, rid in ROWS.items():
        x = m.get((b, "transfer@cap"))
        if not x:
            continue
        n, p = x["n"], x["score"]
        lo, hi = wilson(p, n)
        nat = t.get((b, "target"))
        pt = x["prompt_tokens"]
        th = THEIRS[b]
        verdict = "BEATEN" if lo > th else ("behind" if hi < th else "tie (theirs inside our CI)")
        note = [f"xkv engine, mapper {mapper} (FineWeb-distilled), strategy transfer@cap: the last min(256, n//4) "
                f"prompt tokens native, the rest mapped from Qwen3.5-4B; {verdict} vs theirs {th}"]
        if nat:
            note.append(f"native 27B (target arm, same seeded items) {nat['score']:.4f}, retention "
                        f"{p / nat['score']:.3f}")
        for (bb, s), y in r2.items():
            if bb == b and s != "target":
                note.append(f"r2 (non-distilled) map {s}: {y['score']:.4f}")
        note.append(f"prompt tokens min/median/max {pt['min']}/{pt['median']}/{pt['max']}; mapped_token_frac >= "
                    f"{frac(pt['min']):.3f} per prompt (median-length prompt {frac(pt['median']):.3f})")
        if b in dc:
            note.append(f"decontam: {dc[b]['items_with_13gram_overlap']}/{dc[b]['items']} items share a 13-gram with "
                        f"the distillation data (FineWeb-Edu, {1500} x 4224 tokens)")
        out[rid] = dict(value=round(p, 4), n=n, ci=round((hi - lo) / 2, 4), native_value=round(nat["score"], 4)
                        if nat else None, retention=round(p / nat["score"], 4) if nat else None,
                        mapped_token_frac=round(frac(pt["min"]), 4),
                        rig="1x H200 (Modal), xkv engine (HF transformers, bf16), greedy, xkv bundle loaders",
                        run_id=f"xkv/{os.path.basename(a.mapped)}", cost_usd=a.cost, note="; ".join(note))
    print(json.dumps({k: (v["value"], v["ci"], v["native_value"], v["note"].split("; ")[0][-40:]) for k, v in
                      out.items()}, indent=0))
    if not a.dry:
        ours.update(out)
        json.dump(ours, open(OURS, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
