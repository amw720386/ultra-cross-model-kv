"""First-token agreement of mapped vs native vLLM in modal_speed results: top-1 agreement and approx KL(native||mapped)
over the top-20 logprobs, on the 16 FineWeb prefixes and the 4 long PG19 prompts. python tools/speed_quality.py out/speed/X.json"""
import json
import math
import sys


def cmp(A, B):
    agree, kls = 0, []
    for a, b in zip(A, B):
        a = {int(k): v for k, v in a.items()}
        b = {int(k): v for k, v in b.items()}
        agree += max(a, key=a.get) == max(b, key=b.get)
        fl = min(b.values())
        kls.append(sum(math.exp(lp) * (lp - b.get(t, fl)) for t, lp in a.items()))
    return agree, len(A), sum(kls) / max(1, len(kls))


def main(paths):
    res = sum((json.load(open(p)) for p in paths), [])
    ok = [dict(x["result"], tag=x["result"].get("tag") or x["cfg"].get("tag")) for x in res if "result" in x]
    out = {}
    for tgt in sorted({r["tgt"] for r in ok}):
        nat = [r for r in ok if r["tgt"] == tgt and r["mode"] == "native"]
        if not nat:
            continue
        nat = nat[0]
        for r in [r for r in ok if r["tgt"] == tgt and r["mode"] == "mapped"]:
            row = {}
            for key in ("first_token_top20", "long_first_token_top20"):
                if nat.get(key) and r.get(key):
                    row[key] = cmp(nat[key], r[key])
            if nat.get("needle") and r.get("needle"):
                row["needle_acc"] = (round(r["needle_acc"], 3), round(nat["needle_acc"], 3), len(r["needle"]))
            out[r["tag"]] = row
            print(tgt, r["tag"], {k: (f"top1 {v[0]}/{v[1]} KL~{v[2]:.3f}" if k != "needle_acc" else f"mapped {v[0]} native {v[1]} n={v[2]}")
                                  for k, v in row.items()})
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
