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
from modal_vllm import glm_image as image, vol, hf_shared

app = modal.App(os.environ.get("KVX_APP", "kvx-speed-glm53-native"), image=image)
TGT = "zai-org/GLM-5.3"
GATE_S = 35.0
PLAIN_LENS, PLAIN_MAXLEN = "8192,131072,360448", 362000  # vLLM's plain-TP max at 1M settings: 370,688
PLAIN_BUDGET_S = 2100  # glm-dcp-once: native part <= $25 at ~$36/h (~41 min): no plain point after 35 min
RUNNER = ("import json, sys; from modal_vllm import serve_impl; "
          "r = serve_impl(**json.loads(sys.argv[1])); json.dump(r, open(sys.argv[2], 'w'))")
CFG = dict(mode="native", src=TGT, tgt=TGT, maps="", lens="8192,131072,524288,1048320", tail=0, reps=1,
           backend="", tp=8, util=0.9, maxlen=1048576, lens_quality=0, quality_long=0, long_n=0, needle_n=0,
           max_num_seqs=4, tag="-h200x8-tp8-glm53-native-1m")


OUT_DIR = "/vol/runs/glm53-native-probe"
BACKEND_PAT = ("backend", "Using", "indexer", "index_topk", "sparse", "MLA", "kv_cache_dtype", "KV cache", "max_model_len")


def _sh(argv, log, timeout=None):
    t = time.time()
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        rc, txt = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:  # lead rule: the gate must fit the 2 h budget
        rc = 124
        txt = ((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")) + \
              ((e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")) + \
              f"\n[probe] TIMEOUT after {timeout} s"
    os.makedirs(os.path.dirname(log), exist_ok=True)
    open(log, "w").write(txt)
    vol.commit()
    return rc, round(time.time() - t, 1), txt


def _save(out):
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(OUT_DIR, "result.json"), "w"), indent=1)
    vol.commit()  # visible to the laptop poller before the container ends (lead rule 4)


@app.function(gpu=GPU, timeout=7200, memory=900_000, ephemeral_disk=524288, volumes={"/vol": vol, "/hf": hf_shared.read_only()})
def probe(gate_cmd: str = "", dcp: int = 0):
    t0 = time.time()
    cfg = dict(CFG, engine_kw={"decode_context_parallel_size": dcp}, tag=CFG["tag"] + f"-dcp{dcp}") if dcp else CFG
    hf_shared.reload()
    names = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                           text=True).stdout.strip().splitlines()
    out = {"cfg": cfg, "stage": "native", "gpu_names": names}
    if len(names) != 8 or not all("H200" in n for n in names):  # glm-rig-exempt covers 8xH200 only
        out.update(stage="done", native_error=f"ABORT: need 8x H200, container has {names}")
        _save(out)
        return out
    _save(out)
    rc, wall, txt = _sh([sys.executable, "-c", RUNNER, json.dumps(cfg), "/tmp/native.json"],
                        f"/vol/runs/speed_logs/{cfg['tag']}.log", timeout=5400)
    out.update(native_rc=rc, native_wall_s=wall,
               vllm_settings=sorted({l.strip()[-300:] for l in txt.splitlines() if any(k in l for k in BACKEND_PAT)})[:60])
    if rc != 0 or not os.path.exists("/tmp/native.json"):
        out.update(native_error=txt[-3000:], stage="done")
        _save(out)
        return out
    res = json.load(open("/tmp/native.json"))
    out["native"] = res
    rows = {r["len"]: r["ttft_s"] for r in res.get("rows", [])}
    t1m = rows.get(1048320)
    out.update(native_1m_s=t1m, native_ttft=rows, gate_threshold_s=GATE_S)
    if dcp and time.time() - t0 < PLAIN_BUDGET_S:  # glm-dcp-once: a plain-TP point at the largest length that fits
        pcfg = dict(CFG, lens=PLAIN_LENS, maxlen=PLAIN_MAXLEN, tag=CFG["tag"].replace("-1m", "-plain360k"))
        out["stage"] = "plain"
        _save(out)
        prc, pwall, ptxt = _sh([sys.executable, "-c", RUNNER, json.dumps(pcfg), "/tmp/plain.json"],
                               f"/vol/runs/speed_logs/{pcfg['tag']}.log", timeout=int(PLAIN_BUDGET_S + 600 - (time.time() - t0)))
        out.update(plain_cfg=pcfg, plain_rc=prc, plain_wall_s=pwall)
        if prc == 0 and os.path.exists("/tmp/plain.json"):
            pres = json.load(open("/tmp/plain.json"))
            out["plain_ttft"] = {r["len"]: r["ttft_s"] for r in pres.get("rows", [])}
        else:
            out["plain_error"] = ptxt[-2000:]
    elif dcp:
        out["plain"] = f"skipped: {time.time() - t0:.0f} s elapsed > {PLAIN_BUDGET_S} s (native $ cap)"
    if t1m is None or t1m < GATE_S or not gate_cmd:
        out["gate"] = "skipped" + ("" if gate_cmd else " (no command)") + (f": native 1M {t1m} s < {GATE_S}" if t1m and t1m < GATE_S else "")
        out["stage"] = "done"
        _save(out)
        return out
    out["stage"] = "gate"
    _save(out)  # native numbers are on the volume now, before the gate runs
    left = int(7200 - (time.time() - t0) - 180)  # keep 3 min to save and return
    grc, gwall, gtxt = _sh(json.loads(gate_cmd), "/vol/runs/speed_logs/glm53-window-gate.log", timeout=max(60, left))
    out.update(gate_rc=grc, gate_wall_s=gwall, gate_tail=gtxt[-3000:], gate_timeout_s=left, stage="done")
    _save(out)  # nonzero gate: return at once, no retry (lead rule 3)
    return out


@app.local_entrypoint()
def main(gate_cmd: str = "", dcp: int = 0):
    r = probe.remote(gate_cmd, dcp)
    os.makedirs("out/speed", exist_ok=True)
    json.dump(r, open("out/speed/glm53_native_probe.json", "w"), indent=1)
    print(json.dumps(r, indent=1)[:6000])
