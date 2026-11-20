"""Modal image for agentic benchmarks: the xkv GPU stack plus tau2-bench and Harbor in their own Python 3.12 venvs.

The harnesses need Python >= 3.12 while the model stack runs on 3.11, so they live in separate uv-managed venvs and
run as subprocesses that talk to the in-process OpenAI-compatible server (xkv.agentbench).
"""
from __future__ import annotations

import modal

from common import MODELS, VOLUMES, artifacts, weights  # noqa: F401  (re-exported for agent_lab)

TAU2_REPO = "https://github.com/sierra-research/tau2-bench"
TAU2_COMMIT = "c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4"  # tau2 2.0.0 (text-v1 manifest, revision regional-release-data-v16)
HARBOR_VERSION = "0.26.0"

# Same base layers as common.image (kept in sync by hand; common.py is not edited), then the harness venvs.
# add_local_python_source(copy=False) must stay the last step.
agent_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.8.0",
        "transformers==5.20.2",
        "accelerate",
        "safetensors",
        "huggingface_hub[hf_transfer]",
        "datasets",
        "numpy",
        "flash-linear-attention==0.4.1", "fla-core==0.4.1",
        "kernels",
        "math-verify[antlr4_13_2]>=0.7", "rouge-score>=0.1.2", "nltk>=3.8",
    )
    .pip_install("https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl")
    .pip_install("uv")
    .run_commands(
        "uv python install 3.12",
        f"git clone {TAU2_REPO} /opt/tau2-bench && git -C /opt/tau2-bench checkout {TAU2_COMMIT}",
        "uv venv --python 3.12 /opt/tau2env && uv pip install --python /opt/tau2env/bin/python /opt/tau2-bench",
        "uv venv --python 3.12 /opt/harborenv && "
        f"uv pip install --python /opt/harborenv/bin/python 'harbor[modal]=={HARBOR_VERSION}'",
        # Fail the build early if the frozen text-v1 frame does not verify (digests/bindings of the checkout).
        "cd /opt/tau2-bench && TAU2_DATA_DIR=/opt/tau2-bench/data /opt/tau2env/bin/python -m tau2.cli run "
        "--benchmark text --language en --domain airline --num-tasks 1 --agent-llm openai/x --save-to /tmp/dry "
        "--dry-run > /dev/null",
        "/opt/harborenv/bin/harbor --help > /dev/null",
    )
    .env({"HF_HOME": "/weights/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "XKV_TAU2_HOME": "/opt/tau2-bench", "XKV_TAU2_PY": "/opt/tau2env/bin/python",
          "XKV_HARBOR_BIN": "/opt/harborenv/bin/harbor"})
    .add_local_python_source("xkv", "common", "agent_common", copy=False)
)
