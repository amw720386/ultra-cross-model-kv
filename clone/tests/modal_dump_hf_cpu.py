"""Dump the transformers modeling files of the vLLM image (Qwen3.5 / Gemma-4) to read them locally (CPU, no GPU).
modal run tests/modal_dump_hf_cpu.py"""
import modal
from modal_vllm import image

app = modal.App("kvx-speed-dump-hf", image=image)


@app.function(cpu=2, timeout=600)
def dump():
    import os
    import transformers
    d = os.path.dirname(transformers.__file__) + "/models"
    out = {"version": transformers.__version__}
    for m in ("qwen3_5", "qwen3_next", "gemma4"):
        p = os.path.join(d, m, f"modeling_{m}.py")
        if os.path.exists(p):
            out[m] = open(p).read()
    return out


@app.local_entrypoint()
def main():
    import os
    r = dump.remote()
    os.makedirs("out/hf_src", exist_ok=True)
    for k, v in r.items():
        if k != "version":
            open(f"out/hf_src/modeling_{k}.py", "w").write(v)
    print(r["version"], [k for k in r if k != "version"])
