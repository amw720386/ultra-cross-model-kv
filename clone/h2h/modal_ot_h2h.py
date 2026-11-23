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
            "python3 -m venv /opt/ours",
            "/opt/ours/bin/pip install --no-cache-dir vllm==0.34.0",
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
    out["harness"] = _sh("ls /opt/h2h/bench /opt/h2h/tools && python3 /opt/h2h/bench/sse_ttft.py --help | head -5")
    return out


@app.local_entrypoint()
def main(what: str = "check"):
    r = check.remote()
    for k, (rc, txt) in r.items():
        print(f"===== {k}: rc={rc}\n{txt}")
