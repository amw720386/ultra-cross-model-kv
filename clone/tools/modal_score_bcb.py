"""CPU scorer for BigCodeBench generations (no GPU in this file): runs kvx.evals.exec_code in BigCodeBench's official
evaluation image (all task libraries installed), no volumes, no secrets.
modal run tools/modal_score_bcb.py --gens out/gens.jsonl --out out/bcb_scored.jsonl
modal run tools/modal_score_bcb.py --canonical 100   # environment check: canonical solutions must pass
"""
import json
import os
import sys

import modal

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Modal's image builder rejects the official image's Python 3.10, so: Python 3.11 + BigCodeBench's eval requirements,
# pins relaxed only where no 3.11 wheel exists (tools/bcb_requirements_py311.txt). Validate with --canonical first.
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("libgl1", "libglib2.0-0", "tesseract-ocr", "libsndfile1", "gdal-bin", "libgdal-dev")
         .pip_install_from_requirements(os.path.join(HERE, "tools/bcb_requirements_py311.txt"))
         .add_local_file(os.path.join(HERE, "kvx/evals/exec_code.py"), "/opt/kvx_exec_code.py"))
app = modal.App(os.environ.get("KVX_APP", "kvx-quality-bcb"), image=image)


@app.function(cpu=8, memory=16384, timeout=3000)
def score(rows_jsonl: str):
    import contextlib
    import io
    import importlib.util
    import tempfile
    spec = importlib.util.spec_from_file_location("exec_code", "/opt/kvx_exec_code.py")
    ec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ec)
    d = tempfile.mkdtemp()
    src, dst = f"{d}/g.jsonl", f"{d}/s.jsonl"
    open(src, "w").write(rows_jsonl)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ec.main(src, dst, workers=8)
    return dict(rows=open(dst).read(), summary=buf.getvalue().strip())


@app.local_entrypoint()
def main(gens: str = "", out: str = "", canonical: int = 0):
    if canonical:
        sys.path.insert(0, HERE)
        from kvx.evals import data
        items = data.bigcodebench(os.path.expanduser("~/kvx-data/dl"), canonical)
        import pyarrow.parquet as pq
        sol = {r["task_id"]: r["canonical_solution"] for r in
               pq.read_table(os.path.expanduser("~/kvx-data/dl/bigcodebench.parquet")).to_pylist()}
        rows = "".join(json.dumps(dict(id=it["id"], bench=it["bench"], kind="code", arm="canonical", meta=it["meta"],
                                       text="```python\n" + it["meta"]["prompt"] + sol[it["id"]] + "\n```")) + "\n"
                       for it in items)
    else:
        rows = "".join(l for l in open(gens) if '"bigcodebench"' in l)
    r = score.remote(rows)
    if out:
        open(out, "w").write(r["rows"])
    print(r["summary"])
    fails = [json.loads(l) for l in r["rows"].splitlines() if json.loads(l)["score"] == 0]
    for f in fails[:5]:
        print("FAIL", f["id"], (f.get("err") or "")[-200:].replace("\n", " | "))
