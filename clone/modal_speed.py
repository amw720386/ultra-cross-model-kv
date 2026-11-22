"""One GPU boot, several vLLM engines: native + mapped variants (leave-one-out ablations) back to back.

Each config runs modal_vllm.serve_impl in its own subprocess (a fresh vLLM engine, GPU memory released between
configs), so the boot and the weight page cache are paid once.

KVX_GPU=H100:2 modal run modal_speed.py --plan plans/q17-q32-h100x2.json --tag h100x2
"""
import json
import os
import subprocess
import sys
import time

import modal

GPU = os.environ.get("KVX_GPU", "H100:2")
from modal_vllm import image, vol, hf_shared  # same image/volumes as modal_vllm

app = modal.App(os.environ.get("KVX_APP", "kvx-speed"), image=image)

RUNNER = ("import json, sys; from modal_vllm import serve_impl; "
          "r = serve_impl(**json.loads(sys.argv[1])); json.dump(r, open(sys.argv[2], 'w'))")


TIMEOUT = int(os.environ.get("KVX_PLAN_TIMEOUT", "3300"))  # long windows (8xH100 1M: decision w8-1m-window) raise it


@app.function(gpu=GPU, volumes={"/vol": vol, "/hf": hf_shared.read_only()}, timeout=TIMEOUT)
def run_plan(configs: list, deadline_s: int = 3000, hard_s: int = 0, res_dir: str = ""):
    """deadline_s: start no new config after this; hard_s > 0: kill the config in flight at this wall time (cost cap),
    keeping what finished. res_dir: each config's entry is written to /vol/<res_dir>/<i>.json and committed as soon as
    it ends, so results are pulled per config, not only at the end."""
    hf_shared.reload()
    t0 = time.time()
    out = []
    for i, cfg in enumerate(configs):
        if time.time() - t0 > deadline_s:
            out.append(dict(cfg=cfg, error="skipped: deadline"))
            continue
        path = f"/tmp/res{i}.json"
        ts = time.time()
        left = (hard_s - (time.time() - t0)) if hard_s else None
        try:
            p = subprocess.run([sys.executable, "-c", RUNNER, json.dumps(cfg), path], capture_output=True, text=True,
                               timeout=max(30, left) if left is not None else None)
            full = p.stdout + p.stderr
        except subprocess.TimeoutExpired as e:
            dec = lambda b: b.decode(errors="replace") if isinstance(b, bytes) else (b or "")
            full = dec(e.stdout) + dec(e.stderr) + f"\n[run_plan] KILLED at the hard cap ({hard_s} s)"
            p = type("P", (), {"returncode": 124, "stdout": full, "stderr": ""})()
        os.makedirs("/vol/runs/speed_logs", exist_ok=True)
        with open(f"/vol/runs/speed_logs/{cfg.get('tag', i)}.log", "w") as fh:
            fh.write(full)
        # the engine's root cause is "above" the EngineDeadError: keep every traceback/error line plus the tail
        import re as _re
        errs = [l for l in full.splitlines() if _re.search(r"Error|Traceback|File \"|raise |assert", l)]
        log = "\n".join(errs[-150:]) + "\n---- tail ----\n" + full[-4000:]
        kvx_lines = [l for l in (p.stdout + p.stderr).splitlines() if "kvx components" in l or "kvx connector: n=" in l]
        if p.returncode == 0 and os.path.exists(path):
            r = json.load(open(path))
            r["wall_s"] = time.time() - ts
            r["connector_log"] = kvx_lines
            out.append(dict(cfg=cfg, result=r))
        else:
            import re as _re2
            m = _re2.findall(r"PARTIAL_ROWS=(\[.*\])", full)
            part = json.loads(m[-1]) if m else []
            out.append(dict(cfg=cfg, error=log, partial_rows=part, connector_log=kvx_lines))
        print(json.dumps(dict(i=i, tag=cfg.get("tag"), rc=p.returncode, wall_s=round(time.time() - ts),
                              rows=out[-1].get("result", {}).get("rows") or out[-1].get("partial_rows"))), flush=True)
        if res_dir:
            os.makedirs(f"/vol/{res_dir}", exist_ok=True)
            json.dump(out[-1], open(f"/vol/{res_dir}/{i}.json", "w"))
            try:
                vol.commit()
            except Exception:
                pass
    try:
        vol.commit()
    except Exception:
        pass
    return out


@app.local_entrypoint()
def main(plan: str, tag: str = ""):
    configs = json.load(open(plan))
    r = run_plan.remote(configs, deadline_s=int(os.environ.get("KVX_PLAN_DEADLINE", "3000")),
                        hard_s=int(os.environ.get("KVX_PLAN_HARD", "0")), res_dir=os.environ.get("KVX_PLAN_RESDIR", ""))
    os.makedirs("out/speed", exist_ok=True)
    name = f"out/speed/{os.path.basename(plan).rsplit('.', 1)[0]}{tag}.json"
    json.dump(r, open(name, "w"), indent=1)
    for x in r:
        print(x["cfg"].get("tag"), x.get("result", {}).get("rows") or x.get("error", "")[-1500:])
    print("saved", name)
