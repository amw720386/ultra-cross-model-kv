"""Terminal-Bench 2.0 via the official Harbor harness: the agent model is served by vLLM 0.34 in this container
(native, or mapped through kvx.vllm_hybrid_connector), Harbor's terminus-2 agent drives each task, and every task
environment is a Modal sandbox (`harbor run -e modal`), so no task images touch local disk.

KVXQ_GPU=H200 modal run modal_tb2.py --arm native27b --run-id tb2-a --n-tasks 12
Mapped arm: --connector '{"_class": "HybridCrossModelConnector", "src": ..., "maps": ..., "tail": 256, ...}'.
Results: /vol/runs/<run_id>/<arm>/ (harbor job dir + summary.json). OneTriangle's protocol: >= 10 scored trials,
frozen early stop below 20%; their mapped arm 1/10 (3 infra errors), native-small 1/11.
"""
import glob
import json
import os
import subprocess
import time

import modal

GPU = os.environ.get("KVXQ_GPU", "H200").split(",")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "tmux")
    .pip_install("vllm==0.34.0", "hf_transfer", "harbor[modal]==0.27.0")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_CACHE": "/hf/hub", "HF_HUB_OFFLINE": "1"})
    .env({k: v for k, v in os.environ.items() if k.startswith("KVX_") and k.endswith(("_VOLUME", "_SECRET"))})
    .add_local_python_source("kvx")
)
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"))
hf_shared = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
app = modal.App(os.environ.get("KVX_APP", "kvx-quality-tb2"), image=image)


def _serve(model, connector, max_len, log):
    c = ["vllm", "serve", model, "--port", "8000", "--max-model-len", str(max_len), "--gpu-memory-utilization", "0.72",
         "--max-num-seqs", "32", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser",
         "qwen3", "--served-model-name", "agent", "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0})]
    if connector:
        cc = dict(connector)
        c += ["--no-enable-prefix-caching", "--kv-transfer-config", json.dumps(dict(
            kv_connector=cc.pop("_class", "HybridCrossModelConnector"),
            kv_connector_module_path=cc.pop("_module", "kvx.vllm_hybrid_connector"), kv_role="kv_both",
            kv_connector_extra_config=cc))]
    return subprocess.Popen(c, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPATH="/root"))


@app.function(gpu=GPU, volumes={"/vol": vol, "/hf": hf_shared.read_only()}, timeout=3300)
def run(arm: str, run_id: str, model: str, connector: str, n_tasks: int, concurrency: int, max_len: int,
        budget_s: int, tasks: str):
    import urllib.request
    t0 = time.time()
    hf_shared.reload()
    vol.reload()
    out = f"/vol/runs/{run_id}/{arm}"
    os.makedirs(out, exist_ok=True)
    srv = _serve(model, json.loads(connector) if connector else None, max_len, f"{out}/vllm.log")
    for _ in range(180):
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5)
            break
        except Exception:
            if srv.poll() is not None:
                return dict(error="vllm exited", log=open(f"{out}/vllm.log").read()[-4000:])
            time.sleep(5)
    cmd = ["harbor", "run", "-d", "terminal-bench@2.0", "-a", "terminus-2", "-m", "hosted_vllm/agent",
           "--ak", "api_base=http://127.0.0.1:8000/v1", "-e", "modal", "-n", str(concurrency), "-o", f"{out}/jobs"]
    for t in [x for x in tasks.split(",") if x][:n_tasks] if tasks else []:
        cmd += ["-i", t]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=max(60, budget_s - (time.time() - t0)),
                       env=dict(os.environ, HOSTED_VLLM_API_BASE="http://127.0.0.1:8000/v1", HOSTED_VLLM_API_KEY="none"))
    open(f"{out}/harbor.log", "w").write(p.stdout[-40000:] + "\n---\n" + p.stderr[-40000:])
    srv.terminate()
    res = dict(arm=arm, rc=p.returncode, wall_s=round(time.time() - t0), results_files=glob.glob(f"{out}/jobs/**/result.json",
                                                                                                    recursive=True)[:5])
    json.dump(res, open(f"{out}/summary.json", "w"), indent=1)
    vol.commit()
    return res


@app.local_entrypoint()
def main(arm: str, run_id: str, model: str = "Qwen/Qwen3.8-27B", connector: str = "", n_tasks: int = 12,
         concurrency: int = 6, max_len: int = 65536, budget_s: int = 3100, tasks: str = ""):
    print(json.dumps(run.remote(arm, run_id, model, connector, n_tasks, concurrency, max_len, budget_s, tasks),
                     indent=1)[:5000])
