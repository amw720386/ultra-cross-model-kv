"""First-token logprobs per prompt (stdlib): POST /v1/chat/completions, temperature 0, max_tokens 1, top_logprobs 20.
Writes JSONL: {id, top: [[token, logprob], ...], text}. Both arms use this same client on the same prompts; the
first-token KL is computed offline over the shared top-20 support (kl_first_token.py).
  first_token_logprobs.py --base-url URL --model M --prompts FILE.jsonl --out OUT.jsonl"""
import argparse, json, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--top", type=int, default=20)
ap.add_argument("--extra-json", default="")
a = ap.parse_args()
extra = json.loads(a.extra_json) if a.extra_json else {}
n = 0
with open(a.out, "w") as fo:
    for line in open(a.prompts):
        r = json.loads(line)
        msgs = r.get("messages") or [{"role": "user", "content": r["prompt"]}]
        body = dict({"model": a.model, "messages": msgs, "temperature": 0, "max_tokens": 1, "logprobs": True,
                     "top_logprobs": a.top}, **extra)
        req = urllib.request.Request(a.base_url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            d = json.load(resp)
        c = d["choices"][0]
        lp = (c.get("logprobs") or {}).get("content") or []
        top = [[t["token"], t["logprob"]] for t in (lp[0]["top_logprobs"] if lp else [])]
        fo.write(json.dumps({"id": r.get("id", n), "top": top, "text": c["message"]["content"]}) + "\n")
        n += 1
print(json.dumps({"n": n, "out": a.out}))
