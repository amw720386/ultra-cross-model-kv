"""Needle gate from a TIMED run's replies (no second prefill pass at 128K): score sse_ttft.py's bench.json records with
the gate client's own scorer (bench/quality_chat_client.py score()), write ARM_DIR/quality.json + quality_raw.jsonl in
the gate client's format so bench/audit_quality.py re-scores it.
  needles_from_sse.py BENCH.json PROMPTS.jsonl ARM_DIR [--h2h /opt/h2h]"""
import argparse, hashlib, importlib.util, json, os

ap = argparse.ArgumentParser()
ap.add_argument("bench"); ap.add_argument("prompts"); ap.add_argument("arm_dir"); ap.add_argument("--h2h", default="/opt/h2h")
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("qcc", os.path.join(a.h2h, "bench", "quality_chat_client.py"))
qcc = importlib.util.module_from_spec(spec); spec.loader.exec_module(qcc)
P = {r["id"]: r for r in map(json.loads, open(a.prompts))}
recs = json.load(open(a.bench))["records"]
os.makedirs(a.arm_dir, exist_ok=True)
k = n = 0
with open(os.path.join(a.arm_dir, "quality_raw.jsonl"), "w") as raw:
    for r in recs:
        p = P[r["id"]]
        ok = bool(qcc.score(p.get("kind", "needle"), p["answer"], r.get("text") or ""))
        k += ok; n += 1
        raw.write(json.dumps(dict(id=r["id"], file=os.path.basename(a.prompts), body_sha256=None, status=r.get("status"),
                                  reply=r.get("text"), finish=None, latency=r.get("e2e_s"), correct=ok,
                                  source="timed run (sse_ttft.py bench.json), scored by quality_chat_client.score")) + "\n")
fsha = hashlib.sha256(open(a.prompts, "rb").read()).hexdigest()
q = {"metric": os.path.basename(a.prompts).removesuffix(".jsonl"), "correct": k, "n": n, "failed_requests": sum(1 for r in recs if r.get("ttft_s") is None),
     "client": "bench/sse_ttft.py (timed rep 0) + quality_chat_client.score", "prompts_sha256": fsha, "from": a.bench}
json.dump(q, open(os.path.join(a.arm_dir, "quality.json"), "w"), indent=1)
print(json.dumps(q))
