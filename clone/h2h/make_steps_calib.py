"""Steps for the calibration boot (1x H100!): their package numbers on our replica, the dev tail pick for ours
(h2h-tail-frozen: dev files only), and a TTFT screen per length (not the verified ABBA run).

  python h2h/make_steps_calib.py > /tmp/steps_calib.json
"""
import json

P, D = "/vol/h2h/prompts", "/vol/h2h/prompts-dev"
DEV = f"{D}/needles_*.jsonl {D}/hellaswag*.jsonl"
TAILS = [1, 64, 256]
MAPS = "/vol/runs/pairs-q17-q8/maps_k24.pt"

steps = [{"do": "theirs_up"}, {"do": "nvsmi"},
         # lead: probe that their server accepts the identical body (chat_template_kwargs) before anything else
         {"do": "client", "arm": "theirs", "prompts": f"{P}/short_france.jsonl", "max_tokens": 32, "stream": False,
          "warmup": 0, "out": "calib/theirs/probe_body.json", "continue_on_error": True}]
# calibration vs their package: long retrieval e2e (non-streaming, as their acceptance test), short completion e2e,
# and the same long prompt streamed (their TTFT). 5 repeats each; the first call of each kind warms up.
for k in range(0):
    steps.append({"do": "client", "arm": "theirs", "prompts": f"{P}/long_retrieval_13837.jsonl", "max_tokens": 32,
                  "stream": False, "warmup": 1 if k == 0 else 0, "out": f"calib/theirs/long_e2e_{k}.json"})
for k in range(0):
    steps.append({"do": "client", "arm": "theirs", "prompts": f"{P}/short_france.jsonl", "max_tokens": 32,
                  "stream": False, "warmup": 1 if k == 0 else 0, "out": f"calib/theirs/short_e2e_{k}.json"})
for k in range(0):
    steps.append({"do": "client", "arm": "theirs", "prompts": f"{P}/long_retrieval_13837.jsonl", "max_tokens": 32,
                  "stream": True, "warmup": 1 if k == 0 else 0, "out": f"calib/theirs/long_sse_{k}.json"})
steps.append({"do": "quality", "arm": "theirs", "files": [DEV], "out_dir": "dev/theirs", "continue_on_error": True})
# ours: one server per tail candidate, dev quality + a TTFT screen at each length
for t in TAILS:
    cfg = {"maps": MAPS, "tail": t, "util": 0.45, "fp8": 1, "src_sdpa": "vllm_fa"}
    steps += [{"do": "ours_up", "cfg": cfg, "tag": f"tail{t}"}, {"do": "nvsmi"},
              {"do": "quality", "arm": "ours", "files": [DEV], "out_dir": f"dev/ours_tail{t}", "continue_on_error": True}]
    for L in (8192, 16384, 30720):  # lead: TTFT screens used for picking run on the DEV prompt files only
        steps.append({"do": "client", "arm": "ours", "prompts": f"{D}/needles_{L}.jsonl", "max_tokens": 16,
                      "stream": True, "out": f"devscreen/ours_tail{t}/ttft_{L}.json", "continue_on_error": True})
    steps.append({"do": "ours_down"})
for L in (8192, 16384, 30720):
    steps.append({"do": "client", "arm": "theirs", "prompts": f"{D}/needles_{L}.jsonl", "max_tokens": 16,
                  "stream": True, "out": f"devscreen/theirs/ttft_{L}.json", "continue_on_error": True})
steps.append({"do": "theirs_down"})
print(json.dumps(steps, indent=1))
