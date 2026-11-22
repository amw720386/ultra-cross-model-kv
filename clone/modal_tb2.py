"""Terminal-Bench 2.0 via the official Harbor harness. The GPU function only SERVES the agent model (vLLM 0.34; native,
or mapped through kvx.vllm_hybrid_connector) behind a Modal tunnel; Harbor runs on the launching machine (terminus-2
agent loop there, every task environment a Modal CPU sandbox via `harbor run -e modal`), so no Modal credentials go
into any container and no task image touches local disk.

KVXQ_GPU=H200 modal run modal_tb2.py --arm native27b --run-id tb2-a --tasks-file first_batch.json
Mapped arm: --connector '{"_class": "HybridCrossModelConnector", "src": ..., "maps": ..., "tail": 256, ...}'.
Needs harbor[modal]==0.27.0 on the launching machine (--harbor PATH to its executable).
OneTriangle's protocol: >= 10 scored trials, frozen early stop below 20%; their mapped arm 1/10 (3 infra errors).
"""
import json
import os
import subprocess
import time

import modal

GPU = os.environ.get("KVXQ_GPU", "H200").split(",")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.34.0", "hf_transfer")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_CACHE": "/hf/hub", "HF_HUB_OFFLINE": "1"})
    .env({k: v for k, v in os.environ.items() if k.startswith("KVX_") and k.endswith(("_VOLUME", "_SECRET"))})
    .add_local_python_source("kvx")
)
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"))
hf_shared = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
app = modal.App(os.environ.get("KVX_APP", "kvx-quality-tb2"), image=image)
CTL = os.environ.get("KVX_TB2_CTL", "kvx-quality-tb2-ctl")


def _serve_cmd(model, connector, max_len):
    c = ["vllm", "serve", model, "--port", "8000", "--max-model-len", str(max_len), "--gpu-memory-utilization", "0.72",
         "--max-num-seqs", "32", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser",
         "qwen3", "--served-model-name", "agent", "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0})]
    if connector:
        cc = dict(connector)
        c += ["--no-enable-prefix-caching", "--kv-transfer-config", json.dumps(dict(
            kv_connector=cc.pop("_class", "HybridCrossModelConnector"),
            kv_connector_module_path=cc.pop("_module", "kvx.vllm_hybrid_connector"), kv_role="kv_both",
            kv_connector_extra_config=cc))]
    return c


@app.function(gpu=GPU, volumes={"/vol": vol, "/hf": hf_shared.read_only()}, timeout=3300)
def serve(key: str, model: str, connector: str, max_len: int, run_dir: str):
    """Serve until ctl[key + ':stop'] is set or the timeout; ctl[key] = tunnel URL once healthy."""
    import urllib.request
    hf_shared.reload()
    os.makedirs(run_dir, exist_ok=True)
    ctl = modal.Dict.from_name(CTL, create_if_missing=True)
    log = open(f"{run_dir}/vllm.log", "w")
    srv = subprocess.Popen(_serve_cmd(model, json.loads(connector) if connector else None, max_len), stdout=log,
                           stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPATH="/root"))
    with modal.forward(8000) as tunnel:
        for _ in range(240):
            try:
                urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5)
                ctl[key] = tunnel.url
                break
            except Exception:
                if srv.poll() is not None:
                    ctl[key] = "ERROR: vllm exited"
                    vol.commit()
                    return dict(error="vllm exited")
                time.sleep(5)
        while not ctl.get(key + ":stop", False):
            if srv.poll() is not None:
                ctl[key] = "ERROR: vllm exited while serving"
                break
            time.sleep(10)
    srv.terminate()
    vol.commit()
    return dict(ok=True)


@app.local_entrypoint()
def main(arm: str, run_id: str, model: str = "Qwen/Qwen3.8-27B", connector: str = "", tasks_file: str = "",
         concurrency: int = 6, max_len: int = 65536, harbor: str = "harbor", out_dir: str = "out/tb2"):
    key = f"{run_id}/{arm}/{int(time.time())}"
    ctl = modal.Dict.from_name(CTL, create_if_missing=True)
    call = serve.spawn(key, model, connector, max_len, f"/vol/runs/{run_id}/{arm}")
    url = None
    for _ in range(360):  # boot + weights + engine (up to 30 min of queue/boot)
        v = ctl.get(key)
        if v:
            url = v
            break
        time.sleep(5)
    if not url or url.startswith("ERROR"):
        ctl[key + ":stop"] = True
        raise SystemExit(f"model server did not come up: {url}")
    tasks = json.load(open(tasks_file))["tasks"] if tasks_file else []
    jobs = os.path.join(out_dir, run_id, arm)
    os.makedirs(jobs, exist_ok=True)
    cmd = [harbor, "run", "-d", "terminal-bench@2.0", "-a", "terminus-2", "-m", "hosted_vllm/agent",
           "--ak", f"api_base={url}/v1",
           # Harbor's hosted_vllm validation needs token limits + costs (our own server: cost 0)
           "--ak", "model_info=" + json.dumps(dict(max_input_tokens=max_len, max_output_tokens=8192,
                                                   input_cost_per_token=0.0, output_cost_per_token=0.0)),
           "-e", "modal", "-n", str(concurrency), "-o", jobs]
    for t in tasks:
        cmd += ["-i", t]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, env=dict(os.environ, HOSTED_VLLM_API_BASE=f"{url}/v1", HOSTED_VLLM_API_KEY="none"))
        rc = p.returncode
    finally:
        ctl[key + ":stop"] = True
    call.get(timeout=600)
    print(json.dumps(dict(arm=arm, rc=rc, wall_s=round(time.time() - t0), jobs=jobs, tasks=len(tasks))))
