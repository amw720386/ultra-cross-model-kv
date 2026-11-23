"""Dev pick 3 (ot-k8-window), 1x H100!, DEV prompts only. Theirs resident and re-timed.
kvx-pairs' window-selection k8 maps (their mapper's depth-nearest windows), two lambdas from one fit, on lean2 +
group mapper + async_load + async-scheduling/mnbt (the dev-pick-2 candidate c's settings). Same per-length rule.

  python h2h/make_steps_devpick3.py TAG=/vol/path/maps.pt [TAG=...] > steps_devpick3.json
"""
import json
import sys

D = "/vol/h2h/prompts-dev"
FAST = ["--async-scheduling", "--max-num-batched-tokens", "32768"]
CANDS = [a.split("=", 1) for a in sys.argv[1:]]
assert CANDS, "pass TAG=maps_path"


def cfg(maps):
    return {"maps": maps, "tail": 1, "util": 0.33, "fp8": 1, "src_sdpa": "vllm_fa", "vllm_args": FAST,
            "extra": {"src_lean": 2, "async_load": 1, "mapper": "group", "group": 4}}


def screen(arm, tag, lengths=(8192, 16384, 30720)):
    return [{"do": "client", "arm": arm, "prompts": f"{D}/needles_{L}.jsonl", "max_tokens": 16, "stream": True,
             "out": f"devpick3/{tag}/ttft_{L}.json", "continue_on_error": True} for L in lengths]


steps = [{"do": "theirs_up"}, {"do": "nvsmi"}] + screen("theirs", "theirs_head", (8192, 16384))
for tag, maps in CANDS:
    steps += [{"do": "ours_up", "cfg": cfg(maps), "tag": tag, "continue_on_error": True}] + screen("ours", tag)
    steps.append({"do": "quality", "arm": "ours", "files": [f"{D}/needles_*.jsonl"], "out_dir": f"devpick3/{tag}/dev_needles",
                  "continue_on_error": True})
    steps.append({"do": "ours_down"})
steps += screen("theirs", "theirs_tail", (8192, 16384)) + [{"do": "theirs_down"}]
print(json.dumps(steps, indent=1))
