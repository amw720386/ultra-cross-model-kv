"""Smoke-test modal_fit_eval.fit_eval_impl on a CPU container with tiny models (no GPU, outside the GPU cap).
modal run tests/modal_fit_smoke_cpu.py --args '{"src": "Qwen/Qwen3-0.6B", ...}'  (kwargs of fit_eval_impl as JSON)"""
import os
import json
import modal

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "transformers==4.56.2", "accelerate", "datasets==3.6.0", "numpy")
         .add_local_python_source("kvx", "modal_fit_eval"))
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"))
hf_shared = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
app = modal.App(os.environ.get("KVX_APP", "kvx-fitsmoke-cpu"), image=image)


@app.function(cpu=8.0, memory=32768, timeout=3000, volumes={"/vol": vol, "/hf": hf_shared.read_only()})
def smoke(kw: dict):
    from modal_fit_eval import fit_eval_safe
    return fit_eval_safe(**kw)


@app.local_entrypoint()
def main(args: str):
    r = smoke.remote(json.loads(args))
    print(json.dumps({k: v for k, v in r.items() if k in ("eval", "maps_from", "gpu", "t_total")}, indent=1))
