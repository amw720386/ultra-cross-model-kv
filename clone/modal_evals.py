"""Quality evals on Modal: one GPU boot runs several vLLM arms (native / mapped through kvx.vllm_connector) of kvx.evals.

KVXQ_GPU=H200,B200 modal run modal_evals.py --plan plans/evals-q17-q32.json --run-id evals-q17-q32-a
Each arm is a subprocess (fresh vLLM engine, GPU memory freed between arms). Outputs: /vol/runs/<run_id>/<arm>/
(gens.jsonl, summary.json, code_scored.jsonl). Benchmark files: /vol/data/evals (upload once with modal volume put).
Arms past `deadline_s` are skipped so the app stays under the reaper's 1 h (queue time counts).
"""
import json
import os
import subprocess
import sys
import time

import modal

GPU = os.environ.get("KVXQ_GPU", "H100").split(",")  # Modal takes a fallback list, not a comma string
from modal_vllm import vol, hf_shared

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.34.0", "hf_transfer", "pyarrow", "math-verify")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_CACHE": "/hf/hub", "HF_HUB_OFFLINE": "1"})
    .add_local_python_source("kvx", "modal_vllm")
)
app = modal.App(os.environ.get("KVX_APP", "kvx-quality-evals"), image=image)


def _arm_cmd(arm, out_dir):
    c = [sys.executable, "-m", "kvx.evals.run", "--model", arm["model"], "--arm", arm["arm"], "--benches", arm["benches"],
         "--n", str(arm.get("n", 100)), "--out", out_dir, "--data-root", "/vol/data/evals",
         "--max-model-len", str(arm.get("max_model_len", 32768)), "--gpu-util", str(arm.get("gpu_util", 0.88))]
    if arm.get("thinking"):
        c.append("--thinking")
    if arm.get("connector"):
        c += ["--connector", json.dumps(arm["connector"]), "--attn-backend", arm.get("backend", "FLASH_ATTN")]
    if arm.get("question_tail"):
        c.append("--question-tail")
    if arm.get("ns"):
        c += ["--ns", json.dumps(arm["ns"])]
    return c


@app.function(cpu=8, memory=16384, timeout=1800)
def score_code(gens: str):
    import tempfile
    from kvx.evals import exec_code
    d = tempfile.mkdtemp()
    src, dst = f"{d}/g.jsonl", f"{d}/s.jsonl"
    open(src, "w").write(gens)
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec_code.main(src, dst)
    return dict(rows=open(dst).read(), summary=buf.getvalue().strip())


@app.function(gpu=GPU, volumes={"/vol": vol, "/hf": hf_shared.read_only()}, timeout=3300)
def run_arms(arms: list, run_id: str, deadline_s: int = 3000):
    hf_shared.reload()
    vol.reload()
    t0 = time.time()
    res = []
    for arm in arms:
        out_dir = f"/vol/runs/{run_id}/{arm['arm']}"
        os.makedirs(out_dir, exist_ok=True)
        if time.time() - t0 > deadline_s:
            res.append(dict(arm=arm["arm"], error="skipped: deadline"))
            continue
        ts = time.time()
        if arm.get("script"):  # a pre-step (e.g. kvx.evals.chat_sink) instead of an eval arm
            p = subprocess.run([sys.executable, "-m"] + arm["script"], capture_output=True, text=True, timeout=1500)
            open(f"{out_dir}/log.txt", "w").write(p.stdout[-20000:] + "\n---stderr---\n" + p.stderr[-20000:])
            r = dict(arm=arm["arm"], rc=p.returncode, wall_s=round(time.time() - ts), out=p.stdout[-1500:],
                     err=p.stderr[-1500:] if p.returncode else "")
            vol.commit()
            print(json.dumps(r), flush=True)
            res.append(r)
            continue
        p = subprocess.run(_arm_cmd(arm, out_dir), capture_output=True, text=True,
                           timeout=max(60, deadline_s - (time.time() - t0)))
        open(f"{out_dir}/log.txt", "w").write(p.stdout[-20000:] + "\n---stderr---\n" + p.stderr[-40000:])
        r = dict(arm=arm["arm"], rc=p.returncode, wall_s=round(time.time() - ts))
        if os.path.exists(f"{out_dir}/summary.json"):
            r["summary"] = json.load(open(f"{out_dir}/summary.json"))
            if any(b in arm["benches"] for b in ("humaneval", "mbpp")):
                # generated code runs in a separate CPU container with no volumes and no secrets
                code_rows = [l for l in open(f"{out_dir}/gens.jsonl") if '"kind": "code"' in l]
                scored = score_code.remote("".join(code_rows))
                open(f"{out_dir}/code_scored.jsonl", "w").write(scored["rows"])
                r["code"] = scored["summary"]
        else:
            txt = p.stdout + p.stderr
            key = [l for l in txt.splitlines() if ("Error" in l or "error:" in l) and "Processed prompts" not in l]
            r["error"] = ("\n".join(key[-6:]) or txt[-1500:])[-2500:]
        vol.commit()
        print(json.dumps(r), flush=True)
        res.append(r)
    return res


@app.local_entrypoint()
def main(plan: str, run_id: str):
    arms = json.load(open(plan))
    r = run_arms.remote(arms, run_id)
    os.makedirs("out/evals", exist_ok=True)
    json.dump(r, open(f"out/evals/{run_id}.json", "w"), indent=1)
    for x in r:
        s = x.get("summary", {}).get("benches", {})
        print(x["arm"], x.get("rc"), {b: (round(v.get("score", -1), 4), v["n"]) for b, v in s.items()}, x.get("code", ""),
              x.get("error", "")[-800:])
