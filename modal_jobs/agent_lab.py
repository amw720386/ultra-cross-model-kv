"""Run an agentic benchmark task (xkv.agentbench) on a Modal GPU.

    cd modal_jobs && PYTHONPATH=../src modal run agent_lab.py --task tau2 --gpu H100 --minutes 600 \
        --args '{"strategies": ["target", "transfer"], "n_per_domain": 50}'

Terminal-Bench 2.0 creates its task sandboxes on Modal from inside the container, so it needs a Modal token as a
secret (create once: `modal secret create xkv-harbor-modal MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...`) and the
launch must set XKV_SECRETS=xkv-harbor-modal so the secret is attached:

    cd modal_jobs && XKV_SECRETS=xkv-harbor-modal PYTHONPATH=../src modal run agent_lab.py --task terminal_bench \
        --gpu H100 --minutes 600 --args '{"strategies": ["transfer"], "n_tasks": 10}'
"""
import json
import os
import time
from pathlib import Path

import modal

from agent_common import VOLUMES, agent_image, artifacts, charge, weights

app = modal.App("xkv-agent-lab")
# Optional Modal secrets to attach (comma-separated names), e.g. the Harbor Modal token or an OpenAI key.
_SECRETS = [x for x in os.environ.get("XKV_SECRETS", "").split(",") if x]
KW = dict(image=agent_image, volumes=VOLUMES, retries=0, max_containers=1, timeout=12 * 3600,
          secrets=[modal.Secret.from_name(x) for x in _SECRETS])


def _run(task, args):
    import torch

    from xkv import agentbench, experiments

    torch.backends.cuda.matmul.allow_tf32 = True
    experiments.TASKS.update(agentbench.TASKS)
    try:
        return experiments.dispatch(task, args)
    finally:
        artifacts.commit()
        weights.commit()


@app.function(gpu="H100", **KW)
def run_h100(task, args):
    return _run(task, args)


@app.function(gpu="H200", **KW)
def run_h200(task, args):
    return _run(task, args)


FN = {"H100": run_h100, "H200": run_h200}


@app.function(image=agent_image, cpu=2, memory=4096, timeout=900, retries=0)
def probe():
    """CPU check of the image: harness CLIs resolve the official frames and agentbench imports.

        cd modal_jobs && PYTHONPATH=../src modal run agent_lab.py::probe
    """
    import subprocess

    from xkv import agentbench as A

    env = A.tau2_env("http://127.0.0.1:1/v1")
    tau2 = subprocess.run(A.tau2_command("retail", "http://127.0.0.1:1/v1", "http://127.0.0.1:2/v1",
                                         "http://127.0.0.1:2/v1", "/tmp/probe_tau2", num_tasks=2,
                                         extra=["--dry-run"]),
                          env=env, capture_output=True, text=True, cwd=str(A.TAU2_HOME))
    hb = subprocess.run(A.harbor_command("http://127.0.0.1:1/v1", "/tmp/probe_tb", "probe", n_tasks=2,
                                         extra=["--dry-run"]),
                        env=dict(os.environ, OPENAI_API_KEY="xkv"), capture_output=True, text=True)
    out = {"tau2_rc": tau2.returncode, "tau2": (tau2.stdout + tau2.stderr)[-1500:],
           "harbor_rc": hb.returncode, "harbor": (hb.stdout + hb.stderr)[-1500:]}
    print(json.dumps(out, indent=1))
    return out


@app.local_entrypoint()
def main(task: str, gpu: str = "H100", minutes: int = 600, args: str = "{}"):
    charge(f"xkv-agent-lab:{task}", gpu, minutes, args[:200])
    fn = FN[gpu].with_options(timeout=minutes * 60)
    out = fn.remote(task, json.loads(args))
    d = Path(__file__).resolve().parents[1] / "results" / "raw"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{task}-{gpu}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str)[:20000])
    print("saved", p)
