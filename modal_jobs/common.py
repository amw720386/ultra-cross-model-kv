"""Shared Modal image, volumes and model registry (MODELS) for xkv jobs."""
from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]


weights = modal.Volume.from_name("xkv-weights", create_if_missing=True)
artifacts = modal.Volume.from_name("xkv-artifacts", create_if_missing=True)
VOLUMES = {"/weights": weights, "/artifacts": artifacts}

image = (
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
    .env({"HF_HOME": "/weights/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("xkv", "common", copy=False)
)

MODELS = {
    "0.8b": "Qwen/Qwen3.5-0.8B",
    "2b": "Qwen/Qwen3.5-2B",
    "4b": "Qwen/Qwen3.5-4B",
    "27b": "Qwen/Qwen3.8-27B",
}

