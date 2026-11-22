"""GLM-4.7-Flash source prefill cost on one GPU (HF, bf16, vLLM's FA varlen kernel via kvx/fa.py): full causal vs an
8K sliding window, at the given lengths. Decides whether a GLM-4.7-Flash -> GLM-5.3 transfer can beat GLM-5.3's DSA
native prefill at 1M. No connector, no target.
  KVX_GPU=H200 modal run tests/modal_glm_src_probe.py --lens 32768,65536,131072
  modal run tests/modal_glm_src_probe.py --dry 1     (CPU: image + a tiny Glm4MoeLite forward, sdpa)"""
import json
import os

import modal

GPU = os.environ.get("KVX_GPU", "H200")
from modal_vllm import image, vol, hf_shared

app = modal.App(os.environ.get("KVX_APP", "kvx-speed-glm-src-probe"), image=image)
SRC = "zai-org/GLM-4.7-Flash"


def _probe(src, lens, window, dry):
    import time
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    torch.set_grad_enabled(False)
    cfg = AutoConfig.from_pretrained(src)
    out = {"rows": [], "cfg": {k: getattr(cfg, k, None) for k in ("num_hidden_layers", "num_attention_heads",
                                                                    "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
                                                                    "v_head_dim", "n_routed_experts")}}
    if dry:  # tiny random model with the same architecture
        for k, v in dict(num_hidden_layers=2, hidden_size=64, num_attention_heads=4, n_routed_experts=4,
                         num_experts_per_tok=2, moe_intermediate_size=32, intermediate_size=64, q_lora_rank=32,
                         kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        m = AutoModelForCausalLM.from_config(cfg).eval()
        dev = "cpu"
    else:
        m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).cuda().eval()
        dev = "cuda"
        out["gpu"] = torch.cuda.get_device_name(0)
    attn_kind = "sdpa"
    if not dry:
        from kvx.fa import use_vllm_fa
        use_vllm_fa(m)
        attn_kind = "vllm_fa"
    out["attn"] = attn_kind
    from kvx import fa
    for n in [int(x) for x in lens.split(",")]:
        ids = torch.randint(1000, 30000, (1, n), device=dev)
        row = {"len": n}
        for name, w in (("full", 0), (f"win{window}", window)):
            if dry and w:
                continue
            fa.WINDOW = w
            ts = []
            for rep in range(2 if not dry else 1):
                if dev == "cuda":
                    torch.cuda.synchronize()
                t = time.perf_counter()
                m.model(input_ids=ids, use_cache=False)
                if dev == "cuda":
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - t)
            row[name + "_s"] = round(min(ts), 3)
            row[name + "_all"] = [round(x, 3) for x in ts]
            if dev == "cuda":
                row[name + "_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
                torch.cuda.reset_peak_memory_stats()
        fa.WINDOW = 0
        out["rows"].append(row)
        print(json.dumps(row), flush=True)
    return out


@app.function(gpu=GPU, timeout=2400, volumes={"/vol": vol, "/hf": hf_shared.read_only()})
def probe(src: str, lens: str, window: int):
    try:
        return _probe(src, lens, window, False)
    except BaseException:
        import traceback
        return {"error": traceback.format_exc()[-4000:]}


@app.function(cpu=4.0, memory=16384, timeout=900, volumes={"/hf": hf_shared.read_only()})
def probe_cpu(src: str):
    try:
        return _probe(src, "64", 0, True)
    except BaseException:
        import traceback
        return {"error": traceback.format_exc()[-4000:]}


@app.local_entrypoint()
def main(lens: str = "32768,65536,131072", window: int = 8192, dry: int = 0):
    r = probe_cpu.remote(SRC) if dry else probe.remote(SRC, lens, window)
    os.makedirs("out/speed", exist_ok=True)
    if not dry:
        json.dump(r, open("out/speed/glm_src_probe.json", "w"), indent=1)
    print(json.dumps(r, indent=1)[:6000])
