"""CPU-only Modal runner (no GPU function in this file, so the team GPU guard does not count it) for research/mc128.py (OneTriangle notes 001 / 016 / 019 replication).

CPU (outside the team GPU cap):  modal run research/modal_mc128.py --pairs Qwen/Qwen3-0.6B:Qwen/Qwen3-1.7B --cpu 1
GPU:  KVX_GPU=H200 modal run research/modal_mc128.py --pairs a:b,c:d --cpu 0
Results: volume kvx-research /runs/<run_id>/<src>__<tgt>.json, also written locally to ./out/<run_id>/.
"""
import json, os

import modal

CPUS = float(os.environ.get("KVX_CPUS", "32"))
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.0", "transformers==4.57.1", "accelerate", "datasets==5.0.1", "numpy", "hf_transfer",
                 extra_index_url="https://download.pytorch.org/whl/cpu" if os.environ.get("KVX_CPU_IMAGE") else None)
    .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("kvx", "research")
)
vol = modal.Volume.from_name("kvx-research", create_if_missing=True)
hf_shared = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
app = modal.App(os.environ.get("KVX_APP", "kvx-research-mc128cpu"), image=image)
VOLS = {"/vol": vol, "/hf": hf_shared.read_only()}


def _impl(src, tgt, run_id, kw):
    hf_shared.reload()
    from research import mc128
    out = f"/vol/runs/{run_id}/{src.split('/')[-1]}__{tgt.split('/')[-1]}.json"
    try:
        r = mc128.run_pair(src, tgt, out=out, **kw)
    finally:
        vol.commit()
    items = json.load(open(out.replace(".json", "_items.json")))
    return r, items


@app.function(cpu=CPUS, memory=65536, timeout=3300, volumes=VOLS)
def mc_cpu(src, tgt, run_id, kw):
    import torch
    torch.set_num_threads(int(CPUS))
    return _impl(src, tgt, run_id, kw)


@app.local_entrypoint()
def main(pairs: str, run_id: str = "mc128", cpu: int = 0, n_calib: int = 500, length: int = 512, ks: str = "1,4,8,16",
         n_heldout: int = 32, per_task: int = 32, hs_n: int = 0, piqa_n: int = 0, one_pass: int = 1, dtype: str = "bfloat16"):
    kw = dict(n_calib=n_calib, L=length, ks=ks, n_heldout=n_heldout, per_task=per_task, hs_n=hs_n, piqa_n=piqa_n,
              one_pass=one_pass, dtype=dtype)
    assert cpu, "GPU runs: research/modal_mc128.py"
    fn = mc_cpu
    plist = [p.split(":") for p in pairs.split(",")]
    os.makedirs(f"out/{run_id}", exist_ok=True)
    for (s, t), res in zip(plist, fn.starmap([(s, t, run_id, kw) for s, t in plist], return_exceptions=True)):
        name = f"{s.split('/')[-1]}__{t.split('/')[-1]}"
        if isinstance(res, Exception):
            print(f"FAILED {name}: {res!r}")
            continue
        r, items = res
        json.dump(r, open(f"out/{run_id}/{name}.json", "w"), indent=1)
        json.dump(items, open(f"out/{run_id}/{name}_items.json", "w"))
        mc = r.get("mc128", {})
        print(name, "k", r.get("k_selected"), {a: round(v["acc"], 4) for a, v in mc.items()},
              {k: (round(v["agree"], 3), round(v["ppl_ratio"], 4)) for k, v in r["heldout"].items()})
