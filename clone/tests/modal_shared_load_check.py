"""CPU-only: load a model + tokenizer from the shared weights volume via kvx.models (no GPU).
modal run tests/modal_shared_load_check.py --name Qwen/Qwen3-0.6B"""
import modal

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "transformers==4.56.2", "accelerate", "numpy")
         .add_local_python_source("kvx"))
hf_shared = modal.Volume.from_name("gap-push-models-xl-hf")

app = modal.App("kvx-sharedload-cpu", image=image)


@app.function(cpu=4.0, memory=16384, timeout=900, volumes={"/hf": hf_shared.read_only()})
def check(name: str):
    import time, torch
    from transformers import AutoTokenizer
    from kvx.models import load_lm, model_path
    t = time.time()
    path = model_path(name)
    tok = AutoTokenizer.from_pretrained(path)
    m = load_lm(name, dtype=torch.float32)
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    out = m(input_ids=ids, logits_to_keep=1)
    return dict(path=path, load_s=round(time.time() - t, 1), next_token=tok.decode(out.logits[0, -1].argmax()))


@app.local_entrypoint()
def main(name: str = "Qwen/Qwen3-0.6B"):
    print(check.remote(name))
