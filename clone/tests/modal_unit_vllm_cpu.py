"""Run tests/test_core_cpu.py inside the vLLM serving image (same transformers/torch as modal_vllm.py) on a CPU container.
Catches library-version breakage in the connector before a GPU run. modal run tests/modal_unit_vllm_cpu.py"""
import modal

image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
         .pip_install("vllm==0.34.0", "hf_transfer")
         .pip_install("pytest")
         .env({"TRITON_INTERPRET": "1"})
         .add_local_python_source("kvx").add_local_dir("tests", "/root/tests"))
app = modal.App("kvx-unit-vllmimg-cpu", image=image)


@app.function(cpu=4.0, memory=8192, timeout=900)
def run():
    import subprocess
    v = subprocess.run(["python", "-c", "import transformers, torch, vllm; print(transformers.__version__, torch.__version__, vllm.__version__)"],
                       capture_output=True, text=True)
    p = subprocess.run(["python", "-m", "pytest", "-q", "-rs", "-x", "/root/tests/test_core_cpu.py"], capture_output=True,
                       text=True, cwd="/root")
    return p.returncode, v.stdout + v.stderr[-500:] + p.stdout[-6000:] + p.stderr[-2000:]


@app.local_entrypoint()
def main():
    rc, out = run.remote()
    print(out)
    print("rc", rc)
