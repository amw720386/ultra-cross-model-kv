"""kvx-cell-llama70: meta-llama/Llama-3.3-70B-Instruct/h100/c1/{32k,128k}, stock vs our cross-model KV map, ONE Modal
container per length with both arms (same launch/box), 2x H100 asserted.

Both arms: magni's pinned target RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic @f50dbad2 in magni's pinned image
vllm/vllm-openai:v0.34.0@sha256:20fdb089..., magni's stock argv verbatim except --max-model-len = the cell (labelled);
at 128K both arms add --kv-cache-dtype fp8 (pre-registered deviation forced by our arm's memory; magni's fp8kv arm in
session 504642675a). Ours = the same argv + our connector + our levers (frozen config) + an explicit
--gpu-memory-utilization (vLLM 0.34 resolves the default to 1.0 with the extensible KV cache, leaving the connector no room).
Arms run one at a time on :8000 (two TP2 70B servers do not fit 2x H100 together).

  OURS_SRC=<git archive dir> H2H_SRC=<harness dir> modal run h2h/modal_cell_llama70.py::check
  ... modal run h2h/modal_cell_llama70.py::main --run-id R --steps-file S.json
"""
import json
import os
import subprocess
import time

import modal

OURS_SRC = os.environ.get("OURS_SRC", "")
H2H_SRC = os.environ.get("H2H_SRC", "")
IMAGE = "vllm/vllm-openai:v0.34.0@sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1"  # magni spec.image
MODELS = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "gap-push-models-xl-hf"))
VOL = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "clone-push-kvx"))
RUNS = modal.Volume.from_name(os.environ.get("H2H_RUNS_VOLUME", "kvx-h2h-runs"), create_if_missing=True)

image = modal.Image.from_registry(IMAGE).entrypoint([])
if OURS_SRC:
    image = image.add_local_dir(OURS_SRC, "/opt/ours-src", copy=True)
if H2H_SRC:
    image = image.add_local_dir(H2H_SRC, "/opt/h2h", copy=True)
app = modal.App(os.environ.get("KVX_APP", "clone-push-kvx-speed-llama70"), image=image)
MOUNTS = {"/hf": MODELS.read_only(), "/vol": VOL, "/runs": RUNS}
ENV = {"HF_HUB_CACHE": "/hf/hub", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

TARGET = "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"
TARGET_REV = "f50dbad2c84590ca17dc51e207c34321b65ff14b"
TARGET_DIR = f"/hf/hub/models--RedHatAI--Llama-3.3-70B-Instruct-FP8-dynamic/snapshots/{TARGET_REV}"
SOURCE_DIR = "/hf/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/4699cc75b550f9c6f3173fb80f4703b62d946aa5"
URL = "http://127.0.0.1:8000"


def stock_argv(mml, kv_fp8):
    """magni-hq/llama70b-h100-recipe spec.stock.argv verbatim, --max-model-len set to the cell, + --revision (magni's
    measured cells pin it), + --kv-cache-dtype fp8 at 128K only (pre-registered)."""
    a = ["vllm", "serve", TARGET, "--port", "8000", "--tensor-parallel-size", "2", "--max-model-len", str(mml),
         "--revision", TARGET_REV]
    return a + (["--kv-cache-dtype", "fp8"] if kv_fp8 else [])


def ours_argv(mml, kv_fp8, cfg):
    kv = {"kv_connector": "CrossModelConnector", "kv_connector_module_path": "kvx.vllm_connector", "kv_role": "kv_both",
          "kv_connector_extra_config": dict({"src": SOURCE_DIR, "maps": cfg["maps"], "tail": cfg["tail"], "min_len": cfg.get("min_len", 512),
                                             "restore": cfg.get("restore", "none"), "src_sdpa": "vllm_fa", "src_tp": 1, "fp8": True},
                                            **cfg.get("extra", {}))}
    return (stock_argv(mml, kv_fp8) + ["--gpu-memory-utilization", str(cfg["util"]), "--kv-transfer-config", json.dumps(kv)]
            + [str(x) for x in cfg.get("vllm_args", [])])


def _sh(cmd, timeout=900):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)[-4000:]


@app.function(cpu=4.0, memory=16384, timeout=1800, volumes=MOUNTS)
def check():
    """CPU check: image + vLLM version, both argv lists parse, connector imports, weights + maps + prompts exist."""
    out = {"image": IMAGE}
    out["vllm"] = _sh("python3 -c 'import vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__)'")
    out["connector"] = _sh("cd /opt/ours-src && PYTHONPATH=/opt/ours-src python3 -c 'import kvx.vllm_connector as c; print(c.CrossModelConnector)'")
    out["lean_group"] = _sh("cd /opt/ours-src && PYTHONPATH=/opt/ours-src python3 -c 'import kvx.lean_source, kvx.vllm_connector; print(\"ok\")'")
    out["weights"] = _sh(f"ls {TARGET_DIR} | head -20; ls {SOURCE_DIR} | head; python3 -c \"import json;c=json.load(open('{TARGET_DIR}/config.json'));"
                         f"print(c['num_hidden_layers'], c['num_key_value_heads'], c.get('head_dim'), c['max_position_embeddings'], c.get('quantization_config',{{}}).get('quant_method'))\"")
    out["source_cfg"] = _sh(f"python3 -c \"import json;c=json.load(open('{SOURCE_DIR}/config.json'));print(c['num_hidden_layers'], c['num_key_value_heads'], c['vocab_size'])\"")
    out["tokenizers_match"] = _sh(f"cmp {TARGET_DIR}/tokenizer.json {SOURCE_DIR}/tokenizer.json && echo identical || echo DIFFER")
    out["maps"] = _sh("ls -la /vol/runs/pairs-l31-8b-l33-70b-fp8*/ 2>&1 | head; ls -la /vol/runs/pairs-l31-8b-l33-70b-fp8*/maps_all 2>&1 | head")
    out["prompts"] = _sh("ls -la /vol/h2h-llama70/* 2>&1 | head -20")
    # every argv this app will launch, parsed by vLLM's own arg parser (no GPU)
    for name, argv in (("stock32k", stock_argv(32768, False)), ("stock128k", stock_argv(131072, True)),
                       ("ours128k", ours_argv(131072, True, {"maps": "/vol/x.pt", "tail": 1, "util": 0.66,
                                                             "extra": {"src_lean": 2, "async_load": 1, "mapper": "group", "group": 4},
                                                             "vllm_args": ["--async-scheduling", "--max-num-batched-tokens", "32768"]}))):
        code = ("import sys; from vllm.entrypoints.openai.cli_args import make_arg_parser; from vllm.utils.argparse_utils import "
                "FlexibleArgumentParser as P; p=make_arg_parser(P()); a=p.parse_args(sys.argv[1:]); "
                "print('ok', a.model, a.max_model_len, a.tensor_parallel_size, a.kv_cache_dtype, a.gpu_memory_utilization, a.revision)")
        out[f"argv_{name}"] = {"argv": argv, "parse": _sh("python3 -c " + json.dumps(code) + " " + " ".join(json.dumps(x) for x in argv[2:]))}
    out["fp8_cache_write"] = _sh("cd /opt/ours-src && PYTHONPATH=/opt/ours-src python3 h2h/fp8_cache_write_check.py")
    out["harness"] = _sh("ls /opt/h2h/bench; python3 /opt/h2h/bench/sse_ttft.py --help | head -3; python3 /opt/ours-src/h2h/first_token_logprobs.py --help | head -3")
    return out


def _wait(timeout, proc):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited rc={proc.returncode} before healthy")
        try:
            with urllib.request.urlopen(URL + "/health", timeout=5) as r:
                if r.status == 200:
                    return round(time.time() - t0, 1)
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"not healthy after {timeout}s")


def _stop(p):
    if p is not None and p.poll() is None:
        p.terminate()
        try:
            p.wait(90)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(30)
    _sh("for i in $(seq 1 30); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 2; done")


@app.function(gpu="H100!:2", cpu=16.0, memory=196608, timeout=3600, volumes=MOUNTS)
def run(run_id: str, steps: list):
    """Steps: {"do": "up", "arm": "stock"|"ours", "mml", "kv_fp8", "cfg"?} | {"do": "down"} | {"do": "client", "prompts",
    "out", "max_tokens", "warmup"} | {"do": "logprobs", "prompts", "out"} | {"do": "quality", "files", "out_dir"} |
    {"do": "metrics", "out"} | {"do": "nvsmi"}. Everything lands in /runs/<run_id>/ and is committed after each step."""
    rd = f"/runs/{run_id}"
    os.makedirs(rd, exist_ok=True)
    names = _sh("nvidia-smi --query-gpu=name --format=csv,noheader")[1].strip().splitlines()
    assert len(names) == 2 and all("H100" in n for n in names), f"device assert failed: {names}"
    log = {"run_id": run_id, "gpu_names": names, "image": IMAGE, "steps": [], "started": time.strftime("%FT%TZ", time.gmtime())}
    _sh(f"nvidia-smi -q > {rd}/nvidia-smi-q.txt; lscpu > {rd}/lscpu.txt; pip freeze > {rd}/pip_freeze.txt; "
        f"cp /opt/ours-src/SNAPSHOT_SHA {rd}/ours_sha.txt; cp /opt/h2h/SNAPSHOT_SHA {rd}/h2h_sha.txt")

    def save():
        json.dump(log, open(f"{rd}/run_log.json", "w"), indent=1)
        RUNS.commit()

    proc = None
    env = dict(os.environ, **ENV, PYTHONPATH="/opt/ours-src")
    try:
        for st in steps:
            t0 = time.time()
            ent = dict(st)
            try:
                if st["do"] == "up":
                    argv = (stock_argv(st["mml"], st["kv_fp8"]) if st["arm"] == "stock" else
                            ours_argv(st["mml"], st["kv_fp8"], st["cfg"]))
                    tag = st.get("tag", st["arm"])
                    open(f"{rd}/argv_{tag}.json", "w").write(json.dumps(argv, indent=1))
                    proc = subprocess.Popen(argv, env=env, stdout=open(f"{rd}/server_{tag}.log", "ab"), stderr=subprocess.STDOUT)
                    ent["health_s"] = _wait(1800, proc)
                elif st["do"] == "down":
                    _stop(proc)
                    proc = None
                elif st["do"] == "client":
                    out = f"{rd}/{st['out']}"
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    ent["result"] = dict(zip(("rc", "tail"), _sh(
                        f"python3 /opt/h2h/bench/sse_ttft.py --base-url {URL} --model {TARGET} --prompts {st['prompts']} "
                        f"--max-tokens {st.get('max_tokens', 32)} --out {out} --warmup {st.get('warmup', 1)}", timeout=3000)))
                elif st["do"] == "logprobs":
                    out = f"{rd}/{st['out']}"
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    ent["result"] = dict(zip(("rc", "tail"), _sh(
                        f"python3 /opt/ours-src/h2h/first_token_logprobs.py --base-url {URL} --model {TARGET} "
                        f"--prompts {st['prompts']} --out {out}", timeout=3000)))
                elif st["do"] == "quality":
                    od = f"{rd}/{st['out_dir']}"
                    os.makedirs(od, exist_ok=True)
                    ent["result"] = dict(zip(("rc", "tail"), _sh(
                        f"python3 /opt/h2h/bench/quality_chat_client.py --base-url {URL} --model {TARGET} --out-dir {od} "
                        + " ".join(st["files"]), timeout=3000)))
                elif st["do"] == "metrics":
                    out = f"{rd}/{st['out']}"
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    ent["result"] = _sh(f"curl -s {URL}/metrics > {out}; grep -c '^vllm:' {out}")
                elif st["do"] == "nvsmi":
                    ent["result"] = _sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv")
            except Exception as e:
                ent["error"] = repr(e)[:1500]
                if not st.get("continue_on_error"):
                    log["steps"].append(ent)
                    raise
            ent["wall_s"] = round(time.time() - t0, 1)
            log["steps"].append(ent)
            save()
    finally:
        _stop(proc)
        log["ended"] = time.strftime("%FT%TZ", time.gmtime())
        save()
    return log


@app.local_entrypoint()
def main(run_id: str = "", steps_file: str = "", what: str = "run"):
    if what == "check":
        print(json.dumps(check.remote(), indent=1))
        return
    steps = json.load(open(steps_file))
    out = run.remote(run_id, steps)
    print(json.dumps({k: out[k] for k in ("run_id", "gpu_names", "started", "ended")}, indent=1))
