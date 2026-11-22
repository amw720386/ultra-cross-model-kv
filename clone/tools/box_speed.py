"""modal_speed's plan loop on a rented box (Vast): python tools/box_speed.py plans/X.json out/speed/X.json
Expects /vol (runs/<maps>, data/<eval caches>) and HF_HUB_CACHE with the models; one subprocess (fresh vLLM engine) per config."""
import json
import os
import re
import subprocess
import sys
import time

RUNNER = ("import json, sys; from modal_vllm import serve_impl; "
          "r = serve_impl(**json.loads(sys.argv[1])); json.dump(r, open(sys.argv[2], 'w'))")


def main(plan, out_path):
    configs = json.load(open(plan))
    out = []
    os.makedirs("/vol/runs/speed_logs", exist_ok=True)
    for i, cfg in enumerate(configs):
        path = f"/tmp/res{i}.json"
        ts = time.time()
        p = subprocess.run([sys.executable, "-c", RUNNER, json.dumps(cfg), path], capture_output=True, text=True)
        full = p.stdout + p.stderr
        open(f"/vol/runs/speed_logs/{cfg.get('tag', i)}.log", "w").write(full)
        kvx_lines = [l for l in full.splitlines() if "kvx components" in l or "kvx connector: n=" in l]
        if p.returncode == 0 and os.path.exists(path):
            r = json.load(open(path))
            r["wall_s"], r["connector_log"] = time.time() - ts, kvx_lines
            out.append(dict(cfg=cfg, result=r))
        else:
            errs = [l for l in full.splitlines() if re.search(r"Error|Traceback|File \"|raise |assert", l)]
            m = re.findall(r"PARTIAL_ROWS=(\[.*\])", full)
            out.append(dict(cfg=cfg, error="\n".join(errs[-150:]) + "\n---- tail ----\n" + full[-4000:],
                            partial_rows=json.loads(m[-1]) if m else [], connector_log=kvx_lines))
        print(json.dumps(dict(i=i, tag=cfg.get("tag"), rc=p.returncode, wall_s=round(time.time() - ts),
                              rows=out[-1].get("result", {}).get("rows") or out[-1].get("partial_rows"))), flush=True)
        json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
