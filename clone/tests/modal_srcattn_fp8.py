"""Source prefill attention: bf16 FA3 vs fp8 FA3 (kvx/fa.py) on one GPU, with the top CUDA kernels, to see where fp8 loses
(quantization passes vs the attention kernel itself). KVX_GPU=H100 modal run tests/modal_srcattn_fp8.py --lens 131072,262144"""
import json
import os

import modal

GPU = os.environ.get("KVX_GPU", "H100")
from modal_vllm import image, vol, hf_shared

app = modal.App(os.environ.get("KVX_APP", "kvx-speed-srcattn-fp8"), image=image)


@app.function(gpu=GPU, timeout=1800, volumes={"/vol": vol, "/hf": hf_shared.read_only()})
def bench(src: str, lens: str, yarn: float):
    try:
        return _bench(src, lens, yarn)
    except BaseException:
        import traceback
        return {"error": traceback.format_exc()[-4000:]}


def _bench(src, lens, yarn):
    import time
    import torch
    from torch.profiler import ProfilerActivity, profile
    from transformers import AutoConfig
    from kvx import fa
    from kvx.lean_source import LeanSource, fa_attn
    from kvx.models import apply_yarn
    from transformers import AutoModelForCausalLM
    torch.set_grad_enabled(False)
    cfg = AutoConfig.from_pretrained(src)
    if yarn:
        apply_yarn(cfg, yarn, 32768)
    m = AutoModelForCausalLM.from_pretrained(src, config=cfg, dtype=torch.bfloat16).cuda().eval()
    from kvx import core
    out = {"gpu": torch.cuda.get_device_name(0), "rows": []}
    lean = LeanSource(m)
    for n in [int(x) for x in lens.split(",")]:
        ids = torch.randint(1000, 30000, (n,), device="cuda")
        cos, sin = core.rope_cos_sin(m, n, "cuda")
        row = {"len": n}
        for name, fp8 in (("bf16", False), ("fp8", True)):
            attn = fa_attn(fp8=fp8)
            ts = []
            for rep in range(3):
                torch.cuda.synchronize(); t = time.perf_counter()
                lean.forward(ids, cos, sin, attn)
                torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
            row[name + "_s"] = round(min(ts[1:]), 4)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                lean.forward(ids, cos, sin, attn)
                torch.cuda.synchronize()
            top = sorted(prof.key_averages(), key=lambda e: -e.device_time_total)[:8]
            row[name + "_kernels"] = [(e.key[:70], round(e.device_time_total / 1e3, 1), e.count) for e in top]
        out["rows"].append(row)
        print(json.dumps(row)[:2000], flush=True)
    return out


@app.local_entrypoint()
def main(src: str = "Qwen/Qwen3-1.7B", lens: str = "131072,262144", yarn: float = 8.0):
    r = bench.remote(src, lens, yarn)
    os.makedirs("out/speed", exist_ok=True)
    json.dump(r, open("out/speed/srcattn_fp8.json", "w"), indent=1)
    print(json.dumps(r, indent=1)[:8000])
