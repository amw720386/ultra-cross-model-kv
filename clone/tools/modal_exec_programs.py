"""Execute canonical test programs (tools/canon_programs.py output, HumanEval/MBPP) in a Modal CPU container with no
volumes and no secrets; each program runs as its own `python -I` subprocess with a timeout in a fresh temp dir.

modal run tools/modal_exec_programs.py --programs out.jsonl --scored scored.jsonl
scored.jsonl: {id, bench, passed, status, old_score}; prints pass rate and how many items changed vs old_score.
(BigCodeBench rows go through tools/modal_score_bcb.py, which builds BCB's own test program.)
"""
import json
import os

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")  # HumanEval/MBPP use the stdlib (+ numpy guard)
app = modal.App(os.environ.get("KVX_APP", "kvx-qcpu-exec"), image=image)


@app.function(cpu=8, memory=8192, timeout=1800, block_network=True)
def run_all(rows: list, timeout_s: float = 10.0) -> list:
    import subprocess
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    def one(r):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "prog.py")
            open(p, "w").write(r["program"])
            try:
                q = subprocess.run(["python", "-I", p], cwd=d, capture_output=True, text=True, timeout=timeout_s)
                st = "pass" if q.returncode == 0 else "fail"
                err = q.stderr[-300:]
            except subprocess.TimeoutExpired:
                st, err = "timeout", ""
        return dict(id=r["id"], bench=r["bench"], passed=st == "pass", status=st, err=err, old_score=r.get("old_score"))

    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(one, rows))


@app.local_entrypoint()
def main(programs: str, scored: str):
    rows = [json.loads(l) for l in open(programs)]
    res = run_all.remote(rows)
    with open(scored, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    n = len(res)
    k = sum(r["passed"] for r in res)
    ch = [r["id"] for r in res if r["old_score"] is not None and bool(r["old_score"]) != r["passed"]]
    print(json.dumps(dict(n=n, passed=k, pass_rate=k / n if n else None, changed_vs_old=len(ch), changed=ch[:20])))
