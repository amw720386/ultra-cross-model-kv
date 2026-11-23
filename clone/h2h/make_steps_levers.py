"""Steps for the ours-side lever screen (decision ot-margin-first), 1x H100!, DEV prompts only.

Their server stays resident (as in the resident ABBA) and is re-timed in the same boot at 8K/16K (head and tail).
One ours server per lever on top of the frozen config (tail 1, k24 dense, fp8 map, vLLM FA source); each gets a
TTFT screen at 8K/16K/30K, and the configs that change numerics get the dev needles (the per-length map gate).
A labelled diagnostic runs dev HellaSwag with min_len 1 (mapped HellaSwag); it never changes the frozen config.

  python h2h/make_steps_levers.py > steps_levers.json
"""
import json

D = "/vol/h2h/prompts-dev"
NEEDLES = f"{D}/needles_*.jsonl"
K = "/vol/runs/pairs-q17-q8"
BASE = {"maps": f"{K}/maps_k24.pt", "tail": 1, "util": 0.33, "fp8": 1, "src_sdpa": "vllm_fa"}


def cfg(**kw):
    c = dict(BASE)
    extra = kw.pop("extra", {})
    c.update(kw)
    if extra:
        c["extra"] = extra
    return c


# (tag, config, run the dev needles gate?)
LEVERS = [
    ("base", cfg(), False),
    ("lean1", cfg(extra={"src_lean": 1}), False),
    ("lean2", cfg(extra={"src_lean": 2}), False),          # lean source forward + torch.compile
    ("fp8src", cfg(extra={"fp8_source": True}), True),     # e4m3 linears in the source prefill
    ("fa3fp8", cfg(extra={"src_fp8_attn": 1}), True),      # FA3 e4m3 attention in the source prefill
    ("async", cfg(extra={"async_load": 1}), False),        # map/write on a side stream, per-layer events
    ("bf16map", cfg(fp8=0), True),
    ("k16", cfg(maps=f"{K}/maps_all/maps_k16.pt"), True),
    ("k12", cfg(maps=f"{K}/maps_all/maps_k12.pt"), True),
    ("k8", cfg(maps=f"{K}/maps_all/maps_k8.pt"), True),
    ("asched", cfg(vllm_args=["--async-scheduling"]), False),
    ("mnbt", cfg(vllm_args=["--max-num-batched-tokens", "32768"]), False),
    ("stack16", cfg(maps=f"{K}/maps_all/maps_k16.pt", vllm_args=["--async-scheduling"],
                    extra={"src_lean": 2, "fp8_source": True, "async_load": 1}), True),
    ("stack8", cfg(maps=f"{K}/maps_all/maps_k8.pt", vllm_args=["--async-scheduling"],
                   extra={"src_lean": 2, "fp8_source": True, "async_load": 1}), True),
]


def screen(arm, tag, lengths=(8192, 16384, 30720)):
    return [{"do": "client", "arm": arm, "prompts": f"{D}/needles_{L}.jsonl", "max_tokens": 16, "stream": True,
             "out": f"levers/{tag}/ttft_{L}.json", "continue_on_error": True} for L in lengths]


steps = [{"do": "theirs_up"}, {"do": "nvsmi"}] + screen("theirs", "theirs_head", (8192, 16384))
for tag, c, gate in LEVERS:
    steps += [{"do": "ours_up", "cfg": c, "tag": tag, "continue_on_error": True}] + screen("ours", tag)
    if gate:
        steps.append({"do": "quality", "arm": "ours", "files": [NEEDLES], "out_dir": f"levers/{tag}/dev_needles",
                      "continue_on_error": True})
    steps.append({"do": "ours_down"})
# diagnostic (kvx-pairs): mapped HellaSwag on our arm, labelled; not a config candidate
steps += [{"do": "ours_up", "cfg": cfg(extra={"min_len": 1}), "tag": "diag_minlen1", "continue_on_error": True},
          {"do": "quality", "arm": "ours", "files": [f"{D}/hellaswag*.jsonl"], "out_dir": "levers/diag_minlen1/dev_hellaswag",
           "continue_on_error": True},
          {"do": "ours_down"}]
steps += screen("theirs", "theirs_tail", (8192, 16384)) + [{"do": "theirs_down"}]
print(json.dumps(steps, indent=1))
