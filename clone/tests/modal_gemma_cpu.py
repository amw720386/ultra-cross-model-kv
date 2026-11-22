import os, modal
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
         .pip_install("vllm==0.34.0", "pytest").env({"TRITON_INTERPRET": "1"})
         .add_local_python_source("kvx").add_local_dir("tests", "/root/tests"))
app = modal.App("kvx-speed-gemma-cpu", image=image)
@app.function(cpu=4.0, memory=16384, timeout=900)
def run():
    import subprocess
    p = subprocess.run(["python", "-m", "pytest", "-q", "-x", "-rs", "/root/tests/test_gemma_connector_cpu.py"], capture_output=True, text=True, cwd="/root")
    return p.stdout[-6000:] + p.stderr[-2000:]
@app.local_entrypoint()
def main():
    print(run.remote())
