"""GLM-5.3 native TTFT probe on 8xH200 (vLLM 0.34, DSA + MLA, default KV dtype), decision glm-probe-sameboot:
  1. native GLM-5.3 at 8K (warm-up), 128K, 512K, 1M (card max 1,048,576), one vLLM engine in a subprocess;
  2. if native(1M) >= GATE_S (35 s) and a gate command is given, the SAME container then runs kvx-pairs' windowed-GLM
     quality gate (its own subprocess, after vLLM has released the GPUs), so the 755 GB checkpoint is read from the
     volume once (page cache: the container asks for enough RAM to keep it);
  3. otherwise the container exits (Modal tears it down) and the gate is skipped.
The gate command is a JSON list (argv) that runs in this image with /vol and /hf mounted, e.g.
  ["python", "-m", "kvx.glm_gate", "--maps-from", "pairs-glm47f-glm53", ...]; it must write its own results under /vol.
  KVX_GPU=H200:8 modal run tests/modal_glm53_native_probe.py [--gate-cmd '<json argv>']
  CPU check first: modal run tests/modal_glm53_native_check.py [--gate-cmd ...]"""
import json
import os
import subprocess
import sys
import time

import modal

GPU = os.environ.get("KVX_GPU", "H200:8")
from modal_vllm import image, vol, hf_shared

app = modal.App(os.environ.get("KVX_APP", "kvx-speed-glm53-native"), image=image)
TGT = "zai-org/GLM-5.3"
GATE_S = 35.0
RUNNER = ("import json, sys; from modal_vllm import serve_impl; "
          "r = serve_impl(**json.loads(sys.argv[1])); json.dump(r, open(sys.argv[2], 'w'))")
CFG = dict(mode="native", src=TGT, tgt=TGT, maps="", lens="8192,131072,524288,1048320", tail=0, reps=1,
           backend="", tp=8, util=0.9, maxlen=1048576, lens_quality=0, quality_long=0, long_n=0, needle_n=0,
           max_num_seqs=4, tag="-h200x8-tp8-glm53-native-1m")


def _sh(argv, log):
    t = time.time()
    p = subprocess.run(argv, capture_output=True, text=True)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    open(log, "w").write(p.stdout + p.stderr)
    return p.returncode, round(time.time() - t, 1), (p.stdout + p.stderr)[-3000:]


@app.function(gpu=GPU, timeout=7200, memory=900_000, volumes={"/vol": vol, "/hf": hf_shared.read_only()})
def probe(gate_cmd: str = ""):
    hf_shared.reload()
    out = {"cfg": CFG}
    rc, wall, tail = _sh([sys.executable, "-c", RUNNER, json.dumps(CFG), "/tmp/native.json"],
                         f"/vol/runs/speed_logs/{CFG['tag']}.log")
    out.update(native_rc=rc, native_wall_s=wall)
    if rc != 0 or not os.path.exists("/tmp/native.json"):
        out["native_error"] = tail
        return out
    res = json.load(open("/tmp/native.json"))
    out["native"] = res
    rows = {r["len"]: r["ttft_s"] for r in res.get("rows", [])}
    t1m = rows.get(1048320)
    out["native_1m_s"] = t1m
    out["gate_threshold_s"] = GATE_S
    if t1m is None or t1m < GATE_S or not gate_cmd:
        out["gate"] = "skipped" + ("" if gate_cmd else " (no command)") + (f": native 1M {t1m} s < {GATE_S}" if t1m and t1m < GATE_S else "")
        return out
    grc, gwall, gtail = _sh(json.loads(gate_cmd), "/vol/runs/speed_logs/glm53-window-gate.log")
    out.update(gate_rc=grc, gate_wall_s=gwall, gate_tail=gtail)
    return out


@app.local_entrypoint()
def main(gate_cmd: str = ""):
    r = probe.remote(gate_cmd)
    os.makedirs("out/speed", exist_ok=True)
    json.dump(r, open("out/speed/glm53_native_probe.json", "w"), indent=1)
    print(json.dumps(r, indent=1)[:6000])
