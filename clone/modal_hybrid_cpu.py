"""CPU-only jobs for modal_fit_hybrid.py (no GPU argument here: vast-guard counts GPU args in a launched file).
modal run modal_hybrid_cpu.py::smoke --src Qwen/Qwen3.5-0.8B --tgt Qwen/Qwen3.5-2B
modal run modal_hybrid_cpu.py::prime_main --toks Qwen/Qwen3.8-27B"""
import json, os, time

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.0", "transformers==5.24.0", "accelerate", "datasets==3.6.0", "numpy", "hf_transfer",
                 "einops")  # no flash-linear-attention: its Triton kernels need a GPU (HF uses the torch reference)
    .env({"HF_HOME": "/vol/hf", **{k: os.environ[k] for k in ("KVX_VOLUME", "KVX_WEIGHTS_VOLUME") if k in os.environ}})  # datasets cache writable on the seat volume; weights via snapshot(cache_dir=/hf/hub)
    .add_local_python_source("kvx", "modal_fit_hybrid")
)
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"), create_if_missing=True)
hf_shared = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
app = modal.App(os.environ.get("KVX_APP", "kvx-pairs-cpu"), image=image)
mounts = {"/vol": vol, "/hf": hf_shared.read_only()}


@app.function(cpu=8.0, memory=65536, timeout=3300, volumes=mounts)
def fit_cpu(kw):
    import modal_fit_hybrid as M
    return M.fit_safe(**kw)


@app.function(cpu=4.0, memory=16384, timeout=3300, volumes=mounts)
def prime(tok_name, n_calib=4000, n_eval=64, L=1024, eval_l=0):
    import modal_fit_hybrid as M
    from transformers import AutoTokenizer
    hf_shared.reload()
    tok = AutoTokenizer.from_pretrained(M.snapshot(tok_name))
    tok.name_or_path = tok_name
    bos = int(M._bos(tok))
    half = n_calib // 2
    a = M.cached(tok, f"calib_{half}x{L}_bos{bos}", lambda: M.load_texts(tok, half, L, exclude=M.EVAL_DOCS))
    b = M.cached(tok, f"pg19train_{half}x{L}_bos{bos}", lambda: M.load_pg19_train(tok, half, L))
    EL = eval_l or L
    e = M.cached(tok, f"eval_{n_eval}x{EL}_bos{bos}", lambda: M.load_texts(tok, n_eval, EL, skip_docs=20000))
    return f"{tok_name}: calib {len(a)}+{len(b)} eval {len(e)} bos {bos}"


@app.local_entrypoint()
def smoke(src: str = "Qwen/Qwen3.5-0.8B", tgt: str = "Qwen/Qwen3.5-2B", run_id: str = "pairs-smoke",
          n_calib: int = 8, n_eval: int = 4, sinks: str = "2", tails: str = "16", split_kv: int = 0, w: int = 3):
    r = fit_cpu.remote(dict(src=src, tgt=tgt, run_id=run_id, n_calib=n_calib, n_eval=n_eval, bs=2, sinks=sinks, tails=tails,
                            split_kv=split_kv, w=w))
    os.makedirs(f"out/{run_id}", exist_ok=True)
    json.dump(r, open(f"out/{run_id}/result.json", "w"), indent=1)
    print(json.dumps(r.get("eval", r), indent=1))


@app.local_entrypoint()
def prime_main(toks: str = "Qwen/Qwen3.8-27B", n_calib: int = 4000, n_eval: int = 64, eval_l: int = 0):
    for r in prime.starmap([(t, n_calib, n_eval, 1024, eval_l) for t in toks.split(",")]):
        print(r)


@app.function(cpu=4.0, memory=98304, timeout=1800, volumes=mounts)
def extract_template(run_id: str):
    """Write runs/<run_id>/sink_template.pt from its hybrid_units.pt / hybrid_maps.pt (loading those needs RAM)."""
    import torch
    vol.reload()
    d = f"/vol/runs/{run_id}"
    src = f"{d}/hybrid_units.pt" if os.path.exists(f"{d}/hybrid_units.pt") else f"{d}/hybrid_maps.pt"
    m = torch.load(src, map_location="cpu", mmap=True)
    t = {int(l): {k: v.float() for k, v in dd.items()} for l, dd in m["sink_template"].items()}
    torch.save(t, f"{d}/sink_template.pt")
    vol.commit()
    return f"{run_id}: {len(t)} layers, A={next(iter(next(iter(t.values())).values())).shape[0]}"


@app.local_entrypoint()
def template_main(run_ids: str):
    for r in extract_template.map(run_ids.split(",")):
        print(r)
