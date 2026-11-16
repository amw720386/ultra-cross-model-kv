"""Run one experiment task on a Modal GPU.

    cd modal_jobs && PYTHONPATH=../src modal run lab.py --task selftest --gpu L40S --minutes 20 --args '{}'
"""
import json
import time
from pathlib import Path

import modal

from common import VOLUMES, artifacts, charge, image, weights

app = modal.App("xkv-lab")
KW = dict(image=image, volumes=VOLUMES, retries=0, max_containers=1, timeout=3 * 3600)


def _run(task, args):
    import torch

    from xkv.experiments import dispatch

    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        return dispatch(task, args)
    finally:
        artifacts.commit()
        weights.commit()


@app.function(gpu="L40S", **KW)
def run_l40s(task, args):
    return _run(task, args)


@app.function(gpu="A100-80GB", **KW)
def run_a100(task, args):
    return _run(task, args)


@app.function(gpu="H100", **KW)
def run_h100(task, args):
    return _run(task, args)


@app.function(gpu="H200", **KW)
def run_h200(task, args):
    return _run(task, args)


@app.function(gpu="B200", **KW)
def run_b200(task, args):
    return _run(task, args)


FN = {"L40S": run_l40s, "A100-80GB": run_a100, "H100": run_h100, "H200": run_h200, "B200": run_b200}


@app.local_entrypoint()
def main(task: str, gpu: str = "L40S", minutes: int = 20, args: str = "{}"):
    charge(f"xkv-lab:{task}", gpu, minutes, args[:200])
    fn = FN[gpu].with_options(timeout=minutes * 60)
    out = fn.remote(task, json.loads(args))
    d = Path(__file__).resolve().parents[1] / "results" / "raw"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{task}-{gpu}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str)[:20000])
    print("saved", p)
