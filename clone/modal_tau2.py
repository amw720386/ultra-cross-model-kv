"""tau2-bench (official, rev 989d3285) with the agent served by vLLM 0.34 in the same container, user simulator
GPT-5.5 (as OneTriangle's bundle: text-v1, English, seed 300, 1 trial/task). One domain per boot (< 55 min).

KVXQ_GPU=H100 modal run modal_tau2.py --domain airline --arm native27b --run-id tau2-a
Mapped arm: --connector '{"src": "Qwen/Qwen3.5-4B", "maps": "/vol/runs/<id>/hybrid_maps.pt", "tail": 256, ...}'
(kvx.vllm_hybrid_connector; prefix caching off). Results: /vol/runs/<run_id>/<arm>/<domain>/ (tau2 json + summary).
"""
import json
import os
import subprocess
import sys
import time

import modal

GPU = os.environ.get("KVXQ_GPU", "H100").split(",")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .pip_install("vllm==0.34.0", "hf_transfer", "pyarrow")
    .run_commands("git clone https://github.com/sierra-research/tau2-bench /opt/tau2 && cd /opt/tau2 && "
                  "git checkout 989d3285 && pip install -e .")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_CACHE": "/hf/hub", "HF_HUB_OFFLINE": "1", "TAU2_DATA_DIR": "/opt/tau2/data"})
    # resource names from the launching shell (kvx-env), so the container's module import resolves the same objects
    .env({k: v for k, v in os.environ.items() if k.startswith("KVX_") and k.endswith(("_VOLUME", "_SECRET"))})
    .add_local_python_source("kvx")
)
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"))
hf_shared = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
app = modal.App(os.environ.get("KVX_APP", "kvx-quality-tau2"), image=image)


def _serve_cmd(model, connector, max_len, gpu_util):
    c = ["vllm", "serve", model, "--port", "8000", "--max-model-len", str(max_len), "--gpu-memory-utilization",
         str(gpu_util), "--max-num-seqs", "64", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3",
         "--served-model-name", "agent", "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0})]
    if connector:
        c += ["--no-enable-prefix-caching", "--kv-transfer-config", json.dumps(dict(
            kv_connector=connector.pop("_class", "HybridCrossModelConnector"),
            kv_connector_module_path=connector.pop("_module", "kvx.vllm_hybrid_connector"), kv_role="kv_both",
            kv_connector_extra_config=connector))]
    return c


@app.function(gpu=GPU, volumes={"/vol": vol, "/hf": hf_shared.read_only()}, timeout=3300,
              secrets=[modal.Secret.from_name(os.environ.get("KVX_OPENAI_SECRET", "kvx-openai"))])
def run(domain: str, arm: str, run_id: str, model: str, connector: str, num_tasks: int, concurrency: int,
        agent_args: str, max_len: int, budget_s: int, task_ids: str = ""):
    import urllib.request
    t0 = time.time()
    hf_shared.reload()
    vol.reload()
    out = f"/vol/runs/{run_id}/{arm}/{domain}"
    os.makedirs(out, exist_ok=True)
    conn = json.loads(connector) if connector else None
    srv = subprocess.Popen(_serve_cmd(model, conn, max_len, 0.90), stdout=open(f"{out}/vllm.log", "w"),
                           stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPATH="/root"))
    for _ in range(180):
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5)
            break
        except Exception:
            if srv.poll() is not None:
                return dict(error="vllm exited", log=open(f"{out}/vllm.log").read()[-4000:])
            time.sleep(5)
    else:
        srv.kill()
        return dict(error="vllm not healthy in 15 min", log=open(f"{out}/vllm.log").read()[-4000:])
    t_ready = time.time() - t0
    aa = json.loads(agent_args)
    aa.update(api_base="http://127.0.0.1:8000/v1", api_key="none")
    cmd = ["tau2", "run", "--domain", domain, "--agent-llm", "hosted_vllm/agent", "--agent-llm-args", json.dumps(aa),
           "--user-llm", "gpt-5.5", "--user-llm-args", "{}",  # gpt-5.5 rejects tau2's default temperature=0
           "--num-trials", "1", "--seed", "300",
           "--max-concurrency", str(concurrency), "--save-to", f"{out}/tau2.json"]
    # explicit ids (e.g. OneTriangle's 51-task mapped slice) or the first num_tasks of the base split
    cmd += (["--task-ids"] + task_ids.split(",")) if task_ids else ["--num-tasks", str(num_tasks)]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd="/opt/tau2",
                       timeout=max(60, budget_s - (time.time() - t0)))
    open(f"{out}/tau2.log", "w").write(p.stdout[-30000:] + "\n---\n" + p.stderr[-30000:])
    srv.terminate()
    res = dict(domain=domain, arm=arm, rc=p.returncode, t_ready_s=round(t_ready), wall_s=round(time.time() - t0))
    try:
        rp = f"{out}/tau2.json"
        d = json.load(open(os.path.join(rp, "results.json") if os.path.isdir(rp) else rp))  # --save-to makes a dir
        sims = d.get("simulations", [])
        rw = [s.get("reward_info", {}).get("reward") for s in sims]
        ok = [r for r in rw if r is not None]
        res.update(n=len(sims), scored=len(ok), passed=sum(1 for r in ok if r >= 1.0),
                   pass_rate=(sum(1 for r in ok if r >= 1.0) / len(ok)) if ok else None,
                   terminations={k: sum(1 for s in sims if s.get("termination_reason") == k)
                                 for k in set(s.get("termination_reason") for s in sims)},
                   user_cost_usd=sum((s.get("user_cost") or 0) for s in sims),
                   agent_cost_reported=sum((s.get("agent_cost") or 0) for s in sims))
    except Exception as e:
        res["parse_error"] = repr(e)
        res["tail"] = (p.stdout + p.stderr)[-3000:]
    json.dump(res, open(f"{out}/summary.json", "w"), indent=1)
    vol.commit()
    return res


@app.local_entrypoint()
def main(domain: str, arm: str, run_id: str, model: str = "Qwen/Qwen3.8-27B", connector: str = "",
         num_tasks: int = 50, concurrency: int = 25, agent_args: str = '{"temperature": 1.0, "top_p": 0.95}',
         max_len: int = 65536, budget_s: int = 3100, task_ids: str = "", slice_file: str = ""):
    if slice_file and not task_ids:  # kvx-research's tau2_slice.json: OneTriangle's mapped 51-task slice
        task_ids = ",".join(str(x) for x in json.load(open(slice_file))["mapped_slice_task_ids"][domain])
    print(json.dumps(run.remote(domain, arm, run_id, model, connector, num_tasks, concurrency, agent_args, max_len,
                                budget_s, task_ids), indent=1)[:5000])
