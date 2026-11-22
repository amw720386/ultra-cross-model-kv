"""Run src/xkv experiment tasks (amw's Qwen3.5 -> Qwen3.8 engine) against an existing read-only weights volume.

modal_jobs/lab.py downloads weights into its own volume; this wrapper instead reads them from a shared HF-cache
volume (KVX_WEIGHTS_VOLUME, mounted at /hf) and keeps maps/results on KVX_VOLUME under /vol/xkv (xkv's /artifacts).
Several tasks run in one boot so the 27B loads once.

    cd clone && PYTHONPATH=../src KVX_GPU=H200 modal run modal_xkv.py::main --run-id xkv-4b-27b-ridge \
        --tasks '[["calibrate", {"src": "4b", "tgt": "27b"}], ["quality", {"mapper_name": "4b-27b-r2-n256x4096"}]]'
"""
import json
import os
import time

import modal

GPU = os.environ.get("KVX_GPU", "H200")
GPU_SPEC = GPU.split(",") if "," in GPU else GPU
APP = os.environ.get("KVX_APP", "kvx-xkv")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install("torch==2.8.0", "transformers==5.20.2", "accelerate", "safetensors", "huggingface_hub[hf_transfer]",
                 "datasets", "numpy", "flash-linear-attention==0.4.1", "fla-core==0.4.1", "kernels",
                 "math-verify[antlr4_13_2]>=0.7", "rouge-score>=0.1.2", "nltk>=3.8")
    .pip_install("https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
                 "flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl")
    # xkv resolves weights under $HF_HOME/hub (read-only mount); datasets and hub downloads go to the writable volume
    .env({"HF_HOME": "/hf", "HF_HUB_CACHE": "/vol/hf/hub", "HF_DATASETS_CACHE": "/vol/hf/datasets",
          "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          **{k: os.environ[k] for k in ("KVX_VOLUME", "KVX_WEIGHTS_VOLUME") if k in os.environ}})
    .add_local_python_source("xkv")
)
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"), create_if_missing=True)
weights = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
app = modal.App(APP, image=image)


def run_tasks(run_id: str, tasks: list):
    import torch
    weights.reload()
    os.makedirs("/vol/xkv", exist_ok=True)
    if not os.path.exists("/artifacts"):
        os.symlink("/vol/xkv", "/artifacts")
    from xkv.experiments import dispatch
    from xkv import evals as _xev
    _orig = _xev.load_heldout_text

    def _load(tokenizer, n_seqs, seq_len, seed=0, split="eval", source="wikitext", **kw):
        # source "file:<path under /vol>": prebuilt token sequences {"calib": [...], "eval": [...]} (tools/build_codemix.py)
        if isinstance(source, str) and source.startswith("file:"):
            d = torch.load(f"/vol/{source[5:]}")
            seqs = [s[:seq_len] for s in d[split] if len(s) >= seq_len]
            assert len(seqs) >= min(n_seqs, 1), (split, len(seqs), seq_len)
            return [seqs[i % len(seqs)] for i in range(n_seqs)]
        return _orig(tokenizer, n_seqs, seq_len, seed=seed, split=split, source=source, **kw)
    _xev.load_heldout_text = _load
    torch.backends.cuda.matmul.allow_tf32 = True  # as modal_jobs/lab.py
    out = {"run_id": run_id, "gpu": torch.cuda.get_device_name(), "tasks": []}
    t0 = time.time()
    try:
        for name, args in tasks:
            t = time.time()
            r = dispatch(name, args)
            out["tasks"].append({"task": name, "args": args, "seconds": time.time() - t, "result": r.get("result")})
            print(f"[{time.time() - t0:.0f}s] {name} done", flush=True)
            os.makedirs(f"/vol/runs/{run_id}", exist_ok=True)
            json.dump(out, open(f"/vol/runs/{run_id}/result.json", "w"), indent=1, default=str)
            vol.commit()
    finally:
        out["seconds"] = time.time() - t0
        os.makedirs(f"/vol/runs/{run_id}", exist_ok=True)
        json.dump(out, open(f"/vol/runs/{run_id}/result.json", "w"), indent=1, default=str)
        vol.commit()
    return json.loads(json.dumps(out, default=str))


def run_safe(run_id, tasks):
    try:
        return run_tasks(run_id, tasks)
    except Exception:
        import traceback
        raise RuntimeError(traceback.format_exc()[-3000:]) from None


run = app.function(gpu=GPU_SPEC, volumes={"/vol": vol, "/hf": weights.read_only()}, timeout=3300, cpu=8.0,
                   memory=65536)(run_safe)


@app.local_entrypoint()
def main(run_id: str, tasks: str):
    r = run.remote(run_id, json.loads(tasks))
    os.makedirs(f"out/{run_id}", exist_ok=True)
    json.dump(r, open(f"out/{run_id}/result.json", "w"), indent=1)
    print(json.dumps([{"task": t["task"], "seconds": t["seconds"]} for t in r["tasks"]], indent=1))
