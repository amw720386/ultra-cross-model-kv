"""Head-to-head vs OneTriangle's public Docker release (Qwen3-1.7B -> Qwen3-8B), one Modal container, both arms on GPU0.

Their side, unchanged: the image is THEIR Dockerfile built from their repo at a pinned revision (downloaded into its own
dir, SHA256SUMS checked); their process is THEIR runtime-entrypoint script, run as-is. Modal process wrapper only:
the image ENTRYPOINT is cleared so Modal can start its own process, and the script is started as a subprocess with
their documented env (LOCAL_FILES_ONLY=1, MAX_NEW_TOKENS default). The pinned Qwen snapshots are mounted read-only at
their HF cache path. Their nginx API-key proxy is a deployment wrapper and is not used (their runtime serves :8000).

Our side, a separate venv (/opt/ours): vLLM 0.34.0 + our clone at a pinned sha (PYTHONPATH), `vllm serve` of the same
pinned Qwen3-8B snapshot with our cross-model KV connector and our 1.7B->8B maps.

  OT_REPO=<their repo dir> OURS_SRC=<git archive dir> H2H_SRC=<harness dir> modal run h2h/modal_ot_h2h.py::check
  OT_REPO=... modal run h2h/modal_ot_h2h.py::run --plan-json '<run_ab plans>'     # 1x H100!, device asserted
"""
import json
import os
import subprocess
import sys
import time

import modal

OT_REPO = os.environ.get("OT_REPO", "")
OURS_SRC = os.environ.get("OURS_SRC", "")
H2H_SRC = os.environ.get("H2H_SRC", "")
WEIGHTS = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
VOL = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"))
RUNS = modal.Volume.from_name(os.environ.get("H2H_RUNS_VOLUME", "kvx-h2h-runs"), create_if_missing=True)

image = None
if OT_REPO:
    image = (
        modal.Image.from_dockerfile(os.path.join(OT_REPO, "Dockerfile"), context_dir=OT_REPO)
        .entrypoint([])  # Modal process wrapper only: their runtime-entrypoint is started as a subprocess below
        .run_commands(
            # Modal needs a `python` on PATH to detect the image's Python; their base has only python3. A symlink to the
            # same /usr/bin/python3 adds a name and changes no interpreter their entrypoint (python3 -m kvt) uses.
            "[ -e /usr/bin/python ] || ln -s /usr/bin/python3 /usr/bin/python",
            "python3 -m venv /opt/ours",
            "/opt/ours/bin/pip install --no-cache-dir vllm==0.34.0",
            # OUR side only: a C compiler for vLLM's torch.compile/Inductor (their runtime never invokes one)
            "apt-get update -qq && apt-get install -y -qq --no-install-recommends gcc g++ python3-dev && rm -rf /var/lib/apt/lists/*",
        )
    )
    if OURS_SRC:
        image = image.add_local_dir(OURS_SRC, "/opt/ours-src", copy=True)
    if H2H_SRC:
        image = image.add_local_dir(H2H_SRC, "/opt/h2h", copy=True)

app = modal.App(os.environ.get("KVX_APP", "kvx-speed-ot-h2h"), image=image)
MOUNTS = {"/cache/huggingface": WEIGHTS.read_only(), "/vol": VOL, "/runs": RUNS}
QWEN8 = "/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
QWEN17 = "/cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
THEIR_ENV = {"LOCAL_FILES_ONLY": "1", "CUDA_VISIBLE_DEVICES": "0"}  # their .env.example / compose values


def _sh(cmd, timeout=600):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)[-4000:]


@app.function(cpu=4.0, memory=16384, timeout=1800, volumes=MOUNTS)
def check():
    """$0-ish CPU check: both stacks import, their CLI parses, our connector imports, the paths they need exist."""
    out = {}
    out["python3"] = _sh("command -v python3; python3 --version; ls -la $(command -v python3)")
    out["their_python_is_system"] = _sh("python3 -c 'import sys; print(sys.executable, sys.prefix)'")
    out["their_pkgs"] = _sh("python3 -c 'import torch, transformers, accelerate; print(torch.__version__, "
                            "transformers.__version__, accelerate.__version__)'")
    out["their_kvt_help"] = _sh("cd /app && python3 -m kvt serve --help | head -40")
    out["their_entrypoint"] = _sh("ls -la /usr/local/bin/runtime-entrypoint /usr/bin/tini; id kvt; "
                                  "sha256sum /usr/local/bin/runtime-entrypoint")
    out["their_env"] = _sh("env | grep -E 'KVT_|HF_HOME|TRANSFORMERS_CACHE|PIP_' | sort")
    out["weights"] = _sh(f"ls {QWEN8} {QWEN17} | head -30")
    out["ours_pkgs"] = _sh("/opt/ours/bin/python -c 'import vllm, torch, transformers; print(vllm.__version__, "
                           "torch.__version__, transformers.__version__)'")
    out["ours_connector"] = _sh("cd /opt/ours-src && PYTHONPATH=/opt/ours-src /opt/ours/bin/python -c "
                                "'import kvx.vllm_connector as c; print(c.CrossModelConnector)'")
    out["ours_vllm_serve_help"] = _sh("/opt/ours/bin/vllm serve --help=kv-transfer-config 2>&1 | head -20; "
                                      "/opt/ours/bin/vllm serve --help 2>&1 | grep -i -E 'default-chat-template|served-model' | head")
    out["maps"] = _sh("ls -la /vol/runs/pairs-q17-q8/maps_k24.pt")
    out["vllm_C"] = _sh("ls -la /opt/ours/lib/python3.12/site-packages/vllm/_C* /opt/ours/lib/python3.12/site-packages/vllm/*.so 2>&1 | head; "
                        "/opt/ours/bin/python -c 'import importlib.util as u; print(u.find_spec(\"vllm._C\"))'; "
                        "/opt/ours/bin/pip show vllm | head -3; /opt/ours/bin/pip download vllm==0.34.0 --no-deps -d /tmp/w -q 2>&1 | tail -1; ls /tmp/w")
    out["vllm_serve_flags"] = _sh("/opt/ours/bin/vllm serve --help=all 2>/dev/null | grep -E -- '--(attention-backend|kv-transfer-config|"
                                  "default-chat-template-kwargs|served-model-name|no-enable-prefix-caching|max-num-seqs)' | head")
    out["harness"] = _sh("ls /opt/h2h/bench /opt/h2h/tools && python3 /opt/h2h/bench/sse_ttft.py --help | head -5")
    return out


THEIRS_URL, OURS_URL = "http://127.0.0.1:8000", "http://127.0.0.1:8001"
THEIRS_MODEL, OURS_MODEL = "kvt-qwen3-1.7b-to-8b", "qwen3-8b-kvx"
CLIENT = "python3 /opt/h2h/bench/sse_ttft.py"
EXTRA = json.dumps({"chat_template_kwargs": {"enable_thinking": False}})


def _wait(url, timeout, proc=None):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited rc={proc.returncode} before {url} was healthy")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return round(time.time() - t0, 1)
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"{url} not healthy after {timeout}s")


def _start_theirs(rd):
    env = dict(os.environ, **THEIR_ENV)
    log = open(os.path.join(rd, "theirs_server.log"), "ab")
    # their runtime-entrypoint, unchanged (Modal process wrapper only)
    p = subprocess.Popen(["/usr/bin/tini", "--", "/usr/local/bin/runtime-entrypoint"], env=env, stdout=log,
                         stderr=subprocess.STDOUT)
    return p, _wait(THEIRS_URL + "/health", 1800, p)


def _ours_cmd(cfg):
    kv = {"kv_connector": "CrossModelConnector", "kv_connector_module_path": "kvx.vllm_connector", "kv_role": "kv_both",
          "kv_connector_extra_config": dict({"src": QWEN17, "maps": cfg["maps"], "tail": cfg["tail"], "min_len": 512,
                                             "restore": "rms", "src_sdpa": cfg.get("src_sdpa", "vllm_fa"), "src_tp": 1},
                                            **({"fp8": True} if cfg.get("fp8", 1) else {}), **cfg.get("extra", {}))}
    args = ["/opt/ours/bin/vllm", "serve", QWEN8, "--served-model-name", OURS_MODEL, "--port", "8001",
            "--max-model-len", "32768", "--gpu-memory-utilization", str(cfg.get("util", 0.45)), "--dtype", "bfloat16",
            "--no-enable-prefix-caching", "--max-num-seqs", "4"]
    if cfg.get("mapped", True):
        args += ["--kv-transfer-config", json.dumps(kv)]
    return args


def _start_ours(rd, cfg, tag):
    env = dict(os.environ, PYTHONPATH="/opt/ours-src", HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="0")
    log = open(os.path.join(rd, f"ours_server_{tag}.log"), "ab")
    args = _ours_cmd(cfg)
    open(os.path.join(rd, f"ours_cmd_{tag}.json"), "w").write(json.dumps(args, indent=1))
    p = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT)
    return p, _wait(OURS_URL + "/health", 1200, p)


def _stop(p):
    if p is not None and p.poll() is None:
        p.terminate()
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            p.kill()


def _client(url, model, prompts, out, max_tokens=16, stream=True, warmup=1):
    cmd = (f"{CLIENT} --base-url {url} --model {model} --prompts {prompts} --max-tokens {max_tokens} --out {out} "
           f"--warmup {warmup} --extra-json '{EXTRA}'" + ("" if stream else " --no-stream"))
    rc, txt = _sh(cmd, timeout=3600)
    return {"cmd": cmd, "rc": rc, "tail": txt[-1500:]}


@app.function(gpu="H100!", cpu=8.0, memory=65536, timeout=7200, volumes=MOUNTS)
def run(run_id: str, steps: list):
    """Steps (dicts, in order): {"do": "theirs_up"} | {"do": "theirs_down"} | {"do": "ours_up", "cfg": {...}, "tag"} |
    {"do": "ours_down"} | {"do": "client", "arm": "theirs"|"ours", "prompts", "max_tokens", "stream", "out"} |
    {"do": "quality", "arm", "files": [...], "out_dir"}. Everything lands in /runs/<run_id>/ and is committed."""
    rd = f"/runs/{run_id}"
    os.makedirs(rd, exist_ok=True)
    names = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                           text=True).stdout.strip().splitlines()
    log = {"run_id": run_id, "gpu_names": names, "steps": [], "started": time.strftime("%FT%TZ", time.gmtime()),
           "notes": ["their image = their Dockerfile @ pinned sha, ENTRYPOINT cleared: Modal process wrapper only; "
                     "their runtime-entrypoint started unchanged as a subprocess",
                     "their weights: documented LOCAL_FILES_ONLY=1 path, pinned snapshots mounted read-only at "
                     "/cache/huggingface", "their nginx API-key proxy not used (deployment wrapper, not compute)",
                     "image adds /usr/bin/python -> /usr/bin/python3 (Modal Python detection); their python3 unchanged"]}

    def save():
        json.dump(log, open(os.path.join(rd, "run_log.json"), "w"), indent=1)
        RUNS.commit()

    if len(names) != 1 or "H100" not in names[0]:
        log["abort"] = f"need 1x H100, got {names}"
        save()
        return log
    _sh(f"bash /opt/h2h/bench/evidence.sh {rd}/evidence/start")
    _sh(f"python3 -m pip freeze > {rd}/evidence/pip_freeze_theirs.txt; /opt/ours/bin/pip freeze > {rd}/evidence/pip_freeze_ours.txt")
    _sh(f"cp /opt/ours-src/SNAPSHOT_SHA {rd}/evidence/ours_sha.txt; cp /opt/h2h/SNAPSHOT_SHA {rd}/evidence/h2h_sha.txt; "
        f"sha256sum /usr/local/bin/runtime-entrypoint > {rd}/evidence/their_entrypoint_sha256.txt")
    theirs = ours = None
    try:
        for st in steps:
            t0 = time.time()
            ent = dict(st)
            try:
                if st["do"] == "theirs_up":
                    theirs, ent["health_s"] = _start_theirs(rd)
                elif st["do"] == "theirs_down":
                    _stop(theirs); theirs = None
                elif st["do"] == "ours_up":
                    ours, ent["health_s"] = _start_ours(rd, st["cfg"], st.get("tag", "ours"))
                elif st["do"] == "ours_down":
                    _stop(ours); ours = None
                elif st["do"] == "client":
                    url, model = (THEIRS_URL, THEIRS_MODEL) if st["arm"] == "theirs" else (OURS_URL, OURS_MODEL)
                    out = os.path.join(rd, st["out"])
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    ent["result"] = _client(url, model, st["prompts"], out, st.get("max_tokens", 16), st.get("stream", True),
                                            st.get("warmup", 1))
                elif st["do"] == "quality":
                    url, model = (THEIRS_URL, THEIRS_MODEL) if st["arm"] == "theirs" else (OURS_URL, OURS_MODEL)
                    od = os.path.join(rd, st["out_dir"])
                    os.makedirs(od, exist_ok=True)
                    ent["result"] = dict(zip(("rc", "tail"), _sh(
                        f"python3 /opt/h2h/bench/quality_chat_client.py --base-url {url} --model {model} --out-dir {od} "
                        + " ".join(st["files"]), timeout=5400)))
                elif st["do"] == "nvsmi":
                    ent["result"] = _sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv")
            except Exception as e:  # keep going to the next step only if asked; always record
                ent["error"] = repr(e)[:1500]
                if not st.get("continue_on_error"):
                    log["steps"].append(ent)
                    raise
            ent["wall_s"] = round(time.time() - t0, 1)
            log["steps"].append(ent)
            save()
    finally:
        _stop(ours)
        _stop(theirs)
        _sh(f"bash /opt/h2h/bench/evidence.sh {rd}/evidence/end")
        log["ended"] = time.strftime("%FT%TZ", time.gmtime())
        save()
    return log


@app.local_entrypoint()
def main(what: str = "check", run_id: str = "", steps_file: str = ""):
    if what == "check":
        r = check.remote()
        for k, (rc, txt) in r.items():
            print(f"===== {k}: rc={rc}\n{txt}")
        return
    r = run.remote(run_id, json.load(open(steps_file)))
    os.makedirs("out/h2h", exist_ok=True)
    json.dump(r, open(f"out/h2h/{run_id}.json", "w"), indent=1)
    print(json.dumps(r, indent=1)[:6000])
