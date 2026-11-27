"""Offline first-token KL(stock || ours) over the top-k support, per prompt (stdlib).
Both distributions are renormalised over the UNION of their top-k tokens; a token missing from one side gets that
side's k-th logprob minus log(2) (an upper bound on its mass), so missing tokens raise KL, never hide it.
Also reports top-1 agreement. usage: kl_first_token.py STOCK.jsonl OURS.jsonl"""
import json, math, statistics, sys

def load(p):
    return {r["id"]: dict((t, lp) for t, lp in r["top"]) for r in map(json.loads, open(p))}

S, O = load(sys.argv[1]), load(sys.argv[2])
kls, agree = [], 0
for i in S:
    if i not in O or not S[i] or not O[i]:
        continue
    s, o = S[i], O[i]
    fs, fo = min(s.values()) - math.log(2), min(o.values()) - math.log(2)
    sup = set(s) | set(o)
    ps = {t: math.exp(s.get(t, fs)) for t in sup}
    po = {t: math.exp(o.get(t, fo)) for t in sup}
    zs, zo = sum(ps.values()), sum(po.values())
    kls.append(sum((ps[t] / zs) * math.log((ps[t] / zs) / (po[t] / zo)) for t in sup))
    agree += max(s, key=s.get) == max(o, key=o.get)
print(json.dumps({"n": len(kls), "kl_mean": statistics.mean(kls) if kls else None,
                  "kl_median": statistics.median(kls) if kls else None, "kl_max": max(kls) if kls else None,
                  "top1_agree": agree, "top1_agree_frac": agree / len(kls) if kls else None}))
