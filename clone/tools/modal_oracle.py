"""Hybrid connector identity oracle on Modal, TP1 vs TP2 and connector A/B switches, in ONE boot (2 GPUs).
Identity kvx-hybrid-v1 maps (source = target = Qwen3.5-4B) must reproduce the native engine at every TP size: any
mismatch is a connector bug (sharding, streaming, fast GDN), not map quality.

modal run tools/modal_oracle.py [--gpu L40S] [--variants '<json list of [name, tp, extra]>']
Default variants: tp1 | tp2 | tp2 stream_feats=0 | tp2 fast_gdn=0. Output: per variant approx KL (top-20), top1
agreement and greedy-32 common prefix vs the native engine at the same TP.
"""
import json
import os
import subprocess

import modal

GPU = os.environ.get("KVXQ_ORACLE_GPU", "L40S")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.34.0", "pyarrow")
    .env({"HF_HUB_CACHE": "/hf/hub", "HF_HUB_OFFLINE": "1"})
    .add_local_file(os.path.expanduser("~/clone-push/kvx-quality/dl/quality_val.parquet"), "/data/quality_val.parquet")
    .add_local_dir(os.path.join(os.path.dirname(__file__)), "/root/tools")
    .add_local_python_source("kvx")
)
hf_shared = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
app = modal.App(os.environ.get("KVX_APP", "kvx-quality-oracle"), image=image)
MODEL = "Qwen/Qwen3.5-4B"
DEFAULT = [["tp1", 1, {}], ["tp2", 2, {}], ["tp2-nostream", 2, {"stream_feats": 0}], ["tp2-nofastgdn", 2, {"fast_gdn": 0}]]


@app.function(gpu=f"{GPU}:2", volumes={"/hf": hf_shared.read_only()}, timeout=3000)
def oracle(variants: list):
    env = dict(os.environ, PYTHONPATH="/root")
    sh = lambda *a: subprocess.run(["python", "/root/tools/hybrid_oracle_local.py", *a], env=env, capture_output=True,
                                   text=True, timeout=1500)
    out, logs = {}, {}
    r = sh("maps", MODEL, "/tmp/maps.pt")
    if r.returncode:
        return dict(error="maps", log=r.stderr[-3000:])
    for tp in sorted({v[1] for v in variants}):
        r = sh("run", "native", MODEL, "/tmp/maps.pt", "/data/quality_val.parquet", f"/tmp/native{tp}.json", str(tp))
        if r.returncode:
            return dict(error=f"native tp{tp}", log=r.stderr[-3000:])
    for name, tp, extra in variants:
        r = sh("run", "mapped", MODEL, "/tmp/maps.pt", "/data/quality_val.parquet", f"/tmp/{name}.json", str(tp),
               json.dumps(extra))
        if r.returncode:
            out[name] = dict(error="mapped run failed")
            logs[name] = r.stderr[-2500:]
            continue
        c = sh("compare", f"/tmp/native{tp}.json", f"/tmp/{name}.json")
        out[name] = json.loads(c.stdout.strip().splitlines()[-1]) if c.returncode == 0 else dict(error=c.stderr[-500:])
        logs[name] = "\n".join(l for l in r.stdout.splitlines() + r.stderr.splitlines() if "[kvx]" in l)[-1500:]
    return dict(results=out, kvx_lines=logs)


@app.local_entrypoint()
def main(variants: str = ""):
    v = json.loads(variants) if variants else DEFAULT
    print(json.dumps(oracle.remote(v), indent=1))
