"""CPU sandbox for BigCodeBench test execution inside the official BigCodeBench evaluation image.

    cd modal_jobs && PYTHONPATH=../src modal deploy exec_app.py
GPU jobs call it when XKV_EXEC_MODAL=1 is set (see genevals._run_many).
"""
import modal

# The evaluation image ships Python 3.10 with the pinned test dependencies; Modal's runtime needs its own
# Python 3.11, and test programs run under the image's original interpreter (XKV_EXEC_PYTHON).
image = (modal.Image.from_registry("bigcodebench/bigcodebench-evaluate:latest", add_python="3.11")
         .entrypoint([])
         .run_commands("ls -l /usr/bin/python3* /usr/local/bin/python3* 2>/dev/null || true")
         .add_local_python_source("xkv", copy=False))
app = modal.App("xkv-exec")


@app.function(image=image, cpu=8, memory=16384, timeout=1800, max_containers=4, retries=0)
def run_many(programs):
    import os
    import shutil

    for cand in ("/usr/local/bin/python3.10", "/usr/bin/python3.10", shutil.which("python3.10")):
        if cand and os.path.exists(cand):
            os.environ["XKV_EXEC_PYTHON"] = cand
            break
    from xkv.genevals import _run_many_local

    return _run_many_local(programs, workers=8)


@app.function(image=image, cpu=2, timeout=600)
def probe():
    import os, subprocess
    py = [p for p in ("/usr/local/bin/python3.10", "/usr/bin/python3.10") if os.path.exists(p)]
    out = subprocess.run([py[0], "-c", "import numpy, pandas, tensorflow, sklearn; print(numpy.__version__)"],
                         capture_output=True, text=True) if py else None
    return {"py": py, "stdout": out and out.stdout, "stderr": out and out.stderr[-500:]}
