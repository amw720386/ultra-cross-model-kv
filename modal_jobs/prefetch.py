"""Download model weights into the xkv-weights Volume (CPU only)."""
import modal

from common import MODELS, VOLUMES, image, weights

app = modal.App("xkv-prefetch")


@app.function(image=image, volumes=VOLUMES, timeout=3600, cpu=8, memory=16384, retries=0, max_containers=4)
def fetch(repo: str):
    from huggingface_hub import snapshot_download

    p = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    weights.commit()
    return p


@app.local_entrypoint()
def main(which: str = "0.8b,2b,4b,27b"):
    for r in fetch.map([MODELS[k] for k in which.split(",")]):
        print(r)
