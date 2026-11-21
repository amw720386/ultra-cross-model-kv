"""Compare native vs mapped vLLM runs: approx KL on top-20 first-token logprobs, top-1 agreement, greedy-32 prefix match.
python tools/vllm_quality.py out/vllm/native_Qwen3-32B.json out/vllm/mapped_Qwen3-32B.json"""
import json, math, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
kls, agree = [], 0
for p, q in zip(a["first_token_top20"], b["first_token_top20"]):
    p = {int(k): v for k, v in p.items()}; q = {int(k): v for k, v in q.items()}
    floor = min(q.values()) - 1.0  # tokens outside q's top-20 get a pessimistic floor
    kls.append(sum(math.exp(lp) * (lp - q.get(t, floor)) for t, lp in p.items()))
    agree += max(p, key=p.get) == max(q, key=q.get)
n = len(kls)
m = [sum(1 for _ in __import__("itertools").takewhile(lambda xy: xy[0] == xy[1], zip(x, y))) for x, y in zip(a["greedy32_ids"], b["greedy32_ids"])]
res = dict(n=n, approx_kl_top20=sum(kls) / n, top1_agree=agree / n, greedy32_common_prefix_mean=sum(m) / len(m))
if a.get("long_first_token_top20") and b.get("long_first_token_top20"):
    kl2, ag2 = [], 0
    for p, q in zip(a["long_first_token_top20"], b["long_first_token_top20"]):
        p = {int(k): v for k, v in p.items()}; q = {int(k): v for k, v in q.items()}
        floor = min(q.values()) - 1.0
        kl2.append(sum(math.exp(lp) * (lp - q.get(t, floor)) for t, lp in p.items()))
        ag2 += max(p, key=p.get) == max(q, key=q.get)
    res.update(long_len=a.get("long_len"), long_n=len(kl2), long_approx_kl_top20=sum(kl2) / len(kl2), long_top1_agree=ag2 / len(kl2))
print(json.dumps(res))
