"""Run tests/test_core_cpu.py on a Modal CPU container (no GPU). modal run tests/modal_unit_cpu.py"""
import os
import modal

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "transformers==4.56.2", "numpy", "pytest", "triton==3.4.0")
         .env({"TRITON_INTERPRET": "1"})
         .add_local_python_source("kvx").add_local_dir("tests", "/root/tests"))
app = modal.App(os.environ.get("KVX_APP", "kvx-unit-cpu"), image=image)


@app.function(cpu=2.0, memory=4096, timeout=600)
def run():
    import subprocess
    p = subprocess.run(["python", "-m", "pytest", "-q", "-rs", "/root/tests/test_core_cpu.py"], capture_output=True,
                       text=True, cwd="/root")
    return p.returncode, p.stdout[-6000:] + p.stderr[-2000:]


@app.local_entrypoint()
def main():
    rc, out = run.remote()
    print(out)
    print("rc", rc)
