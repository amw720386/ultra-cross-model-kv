"""Source prefill at long context under each torch SDPA backend, and the connector's KV write (per-token indexing vs
block index_copy_) at 128K x 64 layers. No target model. KVX_GPU=B200 modal run tests/modal_src_attn_bench.py"""
import os
import modal

GPU = os.environ.get("KVX_GPU", "B200")
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
         .pip_install("vllm==0.34.0", "hf_transfer")
         .env({"HF_HUB_CACHE": "/hf/hub", "HF_HUB_OFFLINE": "1"})
         .add_local_python_source("kvx"))
hf_shared = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
app = modal.App(os.environ.get("KVX_APP", "kvx-srcattn"), image=image)


@app.function(gpu=GPU, timeout=1500, volumes={"/hf": hf_shared.read_only()})
def bench(src: str, lens: str, yarn: float):
    try:
        return _bench(src, lens, yarn)
    except BaseException:  # torch exception types cannot be unpickled by the torch-less local client
        import traceback
        return {"error": traceback.format_exc()[-3000:]}


def _bench(src, lens, yarn):
    import time, torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoConfig, AutoModelForCausalLM
    from kvx.models import apply_yarn
    from kvx.vllm_connector import write_kv_fa
    torch.set_grad_enabled(False)
    dev = "cuda:0"
    cfg = AutoConfig.from_pretrained(src)
    if yarn > 0:
        apply_yarn(cfg, yarn, 32768)
    S = AutoModelForCausalLM.from_pretrained(src, config=cfg, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()

    def sync():
        torch.cuda.synchronize()
        return time.perf_counter()

    out = dict(gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__), prefill={}, write={})
    backends = {"cudnn": SDPBackend.CUDNN_ATTENTION} if os.environ.get("KVX_SKIP_SDPA") != "1" else {}
    for n in [int(x) for x in lens.split(",")]:
        x = torch.randint(1000, 30000, (1, n), device=dev)
        for name, be in backends.items():
            try:
                ts = []
                for rep in range(3):
                    t0 = sync()
                    if be is None:
                        o = S(input_ids=x, use_cache=True, logits_to_keep=1)
                    else:
                        with sdpa_kernel([be]):
                            o = S(input_ids=x, use_cache=True, logits_to_keep=1)
                    ts.append(sync() - t0)
                    del o
                out["prefill"][f"{name}_{n}"] = round(min(ts[1:]), 4)
            except Exception as e:
                out["prefill"][f"{name}_{n}"] = f"ERR {type(e).__name__}: {str(e)[:120]}"
            torch.cuda.empty_cache()
        print(out["prefill"], flush=True)
    # vLLM's flash-attn varlen kernel through a transformers attention function; accuracy vs SDPA at the shortest length
    from kvx.fa import use_vllm_fa
    from kvx import core
    n0 = min(int(x) for x in lens.split(","))
    x0 = torch.randint(1000, 30000, (1, n0), device=dev)
    cs, ss = core.rope_cos_sin(S, n0, dev)
    K0, V0, _ = core.capture_kv(S, x0, cs, ss)
    try:
        use_vllm_fa(S)
        K1, V1, _ = core.capture_kv(S, x0, cs, ss)
        rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
        out["vllm_fa_vs_sdpa"] = dict(k_rel=round(rel(K1, K0), 5), v_rel=round(rel(V1, V0), 5))
        del K1, V1
        for n in [int(x) for x in lens.split(",")]:
            x = torch.randint(1000, 30000, (1, n), device=dev)
            ts = []
            for rep in range(3):
                t0 = sync(); o = S(input_ids=x, use_cache=True, logits_to_keep=1); ts.append(sync() - t0); del o
            out["prefill"][f"vllm_fa_{n}"] = round(min(ts[1:]), 4)
    except Exception as e:
        import traceback
        out["vllm_fa_error"] = traceback.format_exc()[-1500:]
    return out


@app.local_entrypoint()
def main(src: str = "Qwen/Qwen3-1.7B", lens: str = "32768,131072", yarn: float = 4.0):
    import json
    print(json.dumps(bench.remote(src, lens, yarn), indent=1))
