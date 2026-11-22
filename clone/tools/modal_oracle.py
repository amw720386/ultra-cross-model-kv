"""Hybrid connector identity oracle on Modal, TP1 vs TP2 and connector A/B switches, in ONE boot (2 GPUs).
Identity kvx-hybrid-v1 maps (source = target = Qwen3.5-4B) must reproduce the native engine at every TP size: any
mismatch is a connector bug (sharding, streaming, fast GDN), not map quality.

KVXQ_ORACLE_GPU=H200 modal run tools/modal_oracle.py [--variants '<json list of [name, tp, extra, model]>']
Variants "<x>@tp1" and "<x>@tp2" are also compared with each other (same maps, TP1 vs TP2).
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
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"))  # xkv maps at /vol/xkv/maps
app = modal.App(os.environ.get("KVX_APP", "kvx-quality-oracle"), image=image)
MODEL = "Qwen/Qwen3.5-4B"
DEFAULT = [["tp1", 1, {}], ["tp2", 2, {}], ["tp2-nostream", 2, {"stream_feats": 0}], ["tp2-nofastgdn", 2, {"fast_gdn": 0}]]


@app.function(gpu=f"{GPU}:2", volumes={"/hf": hf_shared.read_only(), "/vol": vol}, timeout=3000)
def oracle(variants: list):
    env = dict(os.environ, PYTHONPATH="/root")
    sh = lambda *a: subprocess.run(["python", "/root/tools/hybrid_oracle_local.py", *a], env=env, capture_output=True,
                                   text=True, timeout=1500)
    out, logs = {}, {}
    r = sh("maps", MODEL, "/tmp/maps.pt")
    if r.returncode:
        return dict(error="maps", log=r.stderr[-3000:])
    variants = [list(v) + [MODEL] * (4 - len(v)) for v in variants]  # [name, tp, extra, model]
    tag = lambda m, tp: f"{m.split('/')[-1]}-tp{tp}"
    for m, tp in sorted({(v[3], v[1]) for v in variants}):
        r = sh("run", "native", m, "/tmp/maps.pt", "/data/quality_val.parquet", f"/tmp/native-{tag(m, tp)}.json", str(tp))
        if r.returncode:
            return dict(error=f"native {tag(m, tp)}", log=r.stderr[-3000:])
    for name, tp, extra, m in variants:
        r = sh("run", "mapped", m, "/tmp/maps.pt", "/data/quality_val.parquet", f"/tmp/{name}.json", str(tp),
               json.dumps(extra))
        if r.returncode:
            out[name] = dict(error="mapped run failed")
            logs[name] = r.stderr[-2500:]
            continue
        c = sh("compare", f"/tmp/native-{tag(m, tp)}.json", f"/tmp/{name}.json")
        out[name] = json.loads(c.stdout.strip().splitlines()[-1]) if c.returncode == 0 else dict(error=c.stderr[-500:])
        logs[name] = "\n".join(l for l in r.stdout.splitlines() + r.stderr.splitlines() if "[kvx]" in l)[-1500:]
    # mapped vs mapped (e.g. TP1 vs TP2 of the same maps): pairs named "<a>~<b>" in the result
    names = [v[0] for v in variants]
    for a in names:
        for b in names:
            if a < b and a.split("@")[0] == b.split("@")[0] and os.path.exists(f"/tmp/{a}.json") and os.path.exists(f"/tmp/{b}.json"):
                c = sh("compare", f"/tmp/{a}.json", f"/tmp/{b}.json")
                if c.returncode == 0:
                    out[f"{a}~{b}"] = json.loads(c.stdout.strip().splitlines()[-1])
    return dict(results=out, kvx_lines=logs)


@app.local_entrypoint()
def main(variants: str = ""):
    v = json.loads(variants) if variants else DEFAULT
    print(json.dumps(oracle.remote(v), indent=1))
