"""Dev pick 2 (ot-margin-first + ot-longcal), 1x H100!, DEV prompts only. Theirs resident and re-timed.

Every candidate: lean2 (lean source + torch.compile) on the frozen tail 1 / fp8 map / vLLM FA source / util .33.
Maps: the frozen 1K-calibration k24 and kvx-pairs' long-calibration (8K) k24 / k16 / k8 (pairs-q17-q8-cal8k).
k16/k8 only with mapper=group (the dense mapper costs the same at every k). Each candidate gets the dev needles at
8K/16K/30K (the per-length map gate) and a TTFT screen. Rule (h2h-tail-frozen, per length): ours >= theirs - CI.

  python h2h/make_steps_devpick2.py > steps_devpick2.json
"""
import json

D = "/vol/h2h/prompts-dev"
K1, K8 = "/vol/runs/pairs-q17-q8", "/vol/runs/pairs-q17-q8-cal8k"
FAST = ["--async-scheduling", "--max-num-batched-tokens", "32768"]


def cfg(maps, fast=True, **extra):
    c = {"maps": maps, "tail": 1, "util": 0.33, "fp8": 1, "src_sdpa": "vllm_fa", "extra": dict({"src_lean": 2}, **extra)}
    if fast:
        c["extra"]["async_load"] = 1
        c["vllm_args"] = FAST
    return c


CANDS = [
    ("a_k24_1k", cfg(f"{K1}/maps_k24.pt")),
    ("b_k24_8k", cfg(f"{K8}/maps_k24.pt")),
    ("c_k16_8k_group", cfg(f"{K8}/maps_all/maps_k16.pt", mapper="group", group=4)),
    ("d_k8_8k_group", cfg(f"{K8}/maps_all/maps_k8.pt", mapper="group", group=4)),
    ("c2_k16_8k_group_plain", cfg(f"{K8}/maps_all/maps_k16.pt", fast=False, mapper="group", group=4)),
    ("d2_k8_8k_group_plain", cfg(f"{K8}/maps_all/maps_k8.pt", fast=False, mapper="group", group=4)),
]


def screen(arm, tag, lengths=(8192, 16384, 30720)):
    return [{"do": "client", "arm": arm, "prompts": f"{D}/needles_{L}.jsonl", "max_tokens": 16, "stream": True,
             "out": f"devpick2/{tag}/ttft_{L}.json", "continue_on_error": True} for L in lengths]


steps = [{"do": "theirs_up"}, {"do": "nvsmi"}] + screen("theirs", "theirs_head", (8192, 16384))
for tag, c in CANDS:
    steps += [{"do": "ours_up", "cfg": c, "tag": tag, "continue_on_error": True}]
    steps += screen("ours", tag)  # the first call warms torch.compile; each client run also has 1 warmup request
    steps.append({"do": "quality", "arm": "ours", "files": [f"{D}/needles_*.jsonl"], "out_dir": f"devpick2/{tag}/dev_needles",
                  "continue_on_error": True})
    steps.append({"do": "ours_down"})
steps += screen("theirs", "theirs_tail", (8192, 16384)) + [{"do": "theirs_down"}]
print(json.dumps(steps, indent=1))
