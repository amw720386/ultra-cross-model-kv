"""Steps for kvx-cell-llama70 (meta-llama/Llama-3.3-70B-Instruct/h100/c1/{32k,128k}), one container per length.

32K container: stock up -> stock DEV needles (the devpick bar) -> stock GATE: 3 reps x 24 needles timed (rep 1 also the
needle gate via the quality client), first-token logprobs, HellaSwag-100 -> stock down -> ours DEVPICK (candidates
cheapest first; first passing = frozen, sha written BEFORE any gate timing) -> ours GATE (same steps) -> down.
128K container: the same with --kv-cache-dtype fp8 on both arms and ours = the 32K frozen config (up_frozen); timing
reps 2-3 use 12 prompts each (p99 over 48 samples).

  python h2h/make_steps_llama70.py 32k  --prompts DIR --maps-k16 P --maps-k24 P > steps_32k.json
  python h2h/make_steps_llama70.py 128k --prompts DIR --frozen /runs/<32k run>/frozen_config.json > steps_128k.json
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("length", choices=["32k", "128k"])
ap.add_argument("--prompts", required=True, help="dir with gate/ and dev/ needles + hellaswag100.jsonl (kvx-pairs)")
ap.add_argument("--maps-k16", default="")
ap.add_argument("--maps-k24", default="")
ap.add_argument("--frozen", default="")
a = ap.parse_args()

L = {"32k": 32768, "128k": 131072}[a.length]
FP8KV = a.length == "128k"
P = a.prompts
NT = {"32k": "needles_32640", "128k": "needles_130944"}[a.length]  # kvx-pairs' files (<= mml - 32 tokens)
GATE = f"{P}/prompts-{a.length}/{NT}.jsonl"
DEV = f"{P}/prompts-{a.length}-dev/{NT}.jsonl"
HS = f"{P}/prompts-{a.length}/hellaswag100.jsonl"
LEVERS = {"src_lean": 2, "async_load": 1, "mapper": "group", "group": 4}
VARGS = ["--async-scheduling", "--max-num-batched-tokens", "32768"]


def cand(maps):
    # min_len 1 (lead): the gate's short task (HellaSwag) goes through the map too; needles are >= 32K tokens either way
    return {"maps": maps, "tail": 1, "util": 0.66, "restore": "none", "min_len": 1, "extra": dict(LEVERS), "vllm_args": VARGS}


def gate(arm):
    s = []
    # rep 0 = all 24 gate needles, timed AND scored (needles_from_sse: same scorer as the gate client, no second prefill)
    for rep in range(3):
        s.append({"do": "client", "prompts": GATE, "max_tokens": 32, "warmup": 1 if rep == 0 else 0,
                  "limit": (12 if (FP8KV and rep > 0) else None), "out": f"{arm}/ttft_rep{rep}.json", "continue_on_error": True})
        s.append({"do": "metrics", "out": f"{arm}/metrics_rep{rep}.txt", "continue_on_error": True})
    s += [{"do": "needles_from_sse", "bench": f"{arm}/ttft_rep0.json", "prompts": GATE, "out_dir": f"{arm}/gate_needles",
           "continue_on_error": True},
          {"do": "logprobs", "prompts": GATE, "limit": (8 if FP8KV else None), "out": f"{arm}/first_token_logprobs.jsonl",
           "continue_on_error": True},
          {"do": "quality", "files": [HS], "out_dir": f"{arm}/hellaswag", "continue_on_error": True},
          {"do": "nvsmi"}]
    return s


steps = [{"do": "up", "arm": "stock", "mml": L, "kv_fp8": FP8KV, "tag": "stock"}, {"do": "nvsmi"}]
if a.length == "32k":
    steps.append({"do": "quality", "files": [DEV], "out_dir": "stock/dev_needles"})  # the devpick bar (same box)
steps += gate("stock") + [{"do": "down"}]
if a.length == "32k":
    cands = [cand(m) for m in (a.maps_k16, a.maps_k24) if m]
    assert cands, "pass --maps-k16 / --maps-k24"
    steps.append({"do": "devpick", "mml": L, "kv_fp8": FP8KV, "candidates": cands, "stock_dev": "stock/dev_needles",
                  "dev_files": DEV})
else:
    assert a.frozen, "128k uses the 32K frozen config"
    steps.append({"do": "up_frozen", "mml": L, "kv_fp8": FP8KV, "frozen": a.frozen})
steps += [{"do": "nvsmi"}] + gate("ours") + [{"do": "down"}]
print(json.dumps(steps, indent=1))
