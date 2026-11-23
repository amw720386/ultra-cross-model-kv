"""Second ours-side lever screen (ot-margin-first), 1x H100!, DEV prompts only. Theirs resident and re-timed.

Screen 1 (20261123T1035Z-onetriangle-levers) found one lever: src_lean=2 (lean source + torch.compile), source
prefill 97 -> 60 ms at 8K. It also showed the dense mapper costs the same at k8 and k24 (unselected source layers
are zero blocks in one GEMM), so k only saves time with mapper=group. Here, on top of lean2:
  - lean2 itself with the dev needles gate (compile may move numerics);
  - mapper=group G=4 at k8/k16: TTFT only (1K-calibration k8/k16 fail dev needles; this sizes the long-cal maps);
  - async_load, --async-scheduling + --max-num-batched-tokens 32768;
  - vLLM /metrics after every length (engine-side TTFT/prefill/queue vs client TTFT) and a short native prompt
    (fixed HTTP + scheduler + one step), to split our "rest" (~69 ms at 8K).

  python h2h/make_steps_levers2.py > steps_levers2.json
"""
import json

D, P = "/vol/h2h/prompts-dev", "/vol/h2h/prompts"
K = "/vol/runs/pairs-q17-q8"
BASE = {"maps": f"{K}/maps_k24.pt", "tail": 1, "util": 0.33, "fp8": 1, "src_sdpa": "vllm_fa", "extra": {"src_lean": 2}}


def cfg(maps=None, vllm_args=None, **extra):
    c = json.loads(json.dumps(BASE))
    if maps:
        c["maps"] = maps
    if vllm_args:
        c["vllm_args"] = vllm_args
    c["extra"].update(extra)
    return c


LEVERS = [  # (tag, config, dev needles gate?)
    ("lean2", cfg(), True),
    ("lean2_group_k8", cfg(maps=f"{K}/maps_all/maps_k8.pt", mapper="group", group=4), False),
    ("lean2_group_k16", cfg(maps=f"{K}/maps_all/maps_k16.pt", mapper="group", group=4), False),
    ("lean2_async", cfg(async_load=1), False),
    ("lean2_sched", cfg(vllm_args=["--async-scheduling", "--max-num-batched-tokens", "32768"]), False),
]


def screen(arm, tag, lengths=(8192, 16384, 30720), metrics=False):
    out = []
    for L in lengths:
        out.append({"do": "client", "arm": arm, "prompts": f"{D}/needles_{L}.jsonl", "max_tokens": 16, "stream": True,
                    "out": f"levers/{tag}/ttft_{L}.json", "continue_on_error": True})
        if metrics:
            out.append({"do": "metrics", "out": f"levers/{tag}/metrics_after_{L}.txt", "continue_on_error": True})
    return out


steps = [{"do": "theirs_up"}, {"do": "nvsmi"}] + screen("theirs", "theirs_head", (8192, 16384))
for tag, c, gate in LEVERS:
    steps += [{"do": "ours_up", "cfg": c, "tag": tag, "continue_on_error": True},
              {"do": "metrics", "out": f"levers/{tag}/metrics_start.txt", "continue_on_error": True}]
    steps += screen("ours", tag, metrics=True)
    if tag == "lean2":  # fixed overhead: a 19-token prompt (< min_len, served natively)
        steps += [{"do": "client", "arm": "ours", "prompts": f"{P}/short_france.jsonl", "max_tokens": 16, "stream": True,
                   "out": f"levers/{tag}/ttft_short.json", "continue_on_error": True},
                  {"do": "metrics", "out": f"levers/{tag}/metrics_after_short.txt", "continue_on_error": True}]
    if gate:
        steps.append({"do": "quality", "arm": "ours", "files": [f"{D}/needles_*.jsonl"], "out_dir": f"levers/{tag}/dev_needles",
                      "continue_on_error": True})
    steps.append({"do": "ours_down"})
steps += screen("theirs", "theirs_tail", (8192, 16384)) + [{"do": "theirs_down"}]
print(json.dumps(steps, indent=1))
