"""DevEval pass@1 scorer (Python 3.9, standalone: no kvx imports). Mirrors the official pass_k.py: splice the
completion body at body_position (re-indented to `indent`), run each listed test with `python setup.py pytest
--addopts <test>` in the repo (60 s per sample, 5 GB RSS cap), restore the file. Samples of one repo run serially
(the official harness warns against concurrent runs in a repo); repos run in parallel.

Run ONLY in a throwaway sandbox (python:3.9 container, --network none):
  python deveval_score.py --data data.jsonl --src Source_Code --completions gens.jsonl --out scored.jsonl
  python deveval_score.py --data data.jsonl --src Source_Code --reference --out ref.jsonl   # eligibility check
gens.jsonl rows: {"id": namespace, "text": raw model output}.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import textwrap
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor


def _pos(x):
    return json.loads(x) if isinstance(x, str) else x


def body_from_generation(text, func_name):
    """Raw generation -> function body (no signature). Takes the longest fenced block (or the whole text), finds
    `def func_name(`, skips the signature up to the line that ends it, returns the indented body that follows."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blocks = re.findall(r"```(?:[Pp]ython|py)?\s*\n(.*?)```", text, flags=re.S)
    code = max(blocks, key=len) if blocks else text
    lines = code.splitlines()
    for i, l in enumerate(lines):
        if re.match(rf"\s*(async\s+)?def\s+{re.escape(func_name)}\s*\(", l):
            j = i
            while j < len(lines) and not re.search(r"\)\s*(->\s*[^:]+)?:\s*(#.*)?$", lines[j]):
                j += 1
            ind = len(l) - len(l.lstrip())
            body = []
            for k in range(j + 1, len(lines)):
                s = lines[k]
                if s.strip() and (len(s) - len(s.lstrip())) <= ind:
                    break
                body.append(s)
            return "\n".join(body).rstrip() + "\n"
    return code.rstrip() + "\n"  # already a body


def adjust_indent(code, n):
    return textwrap.indent(textwrap.dedent(code), " " * int(n))


def run_tests(proj, tests, timeout=60):
    t0 = time.time()
    for t in tests:
        left = timeout - (time.time() - t0)
        if left <= 0:
            return "TimeOut"
        try:
            p = subprocess.run(["python", "setup.py", "pytest", "--addopts", t], cwd=proj, capture_output=True,
                               timeout=left)
        except subprocess.TimeoutExpired:
            return "TimeOut"
        if p.returncode != 0:
            return "Error"
    return "Pass"


def score_sample(d, body, src):
    if body.strip() in ("", "pass"):
        return "Error"
    path = os.path.join(src, d["completion_path"])
    tmp = path + ".kvx_orig"
    shutil.copyfile(path, tmp)
    try:
        lines = open(path).readlines()
        sos, eos = _pos(d["body_position"])[0] - 1, _pos(d["body_position"])[1]
        lines = lines[:sos] + ["\n", adjust_indent(body, d["indent"]), "\n"] + lines[eos:]
        open(path, "w").write("".join(lines))
        return run_tests(os.path.join(src, d["project_path"]), _pos(d["tests"]))
    finally:
        shutil.move(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--completions")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--only", help="file with one namespace per line (restrict the sample set)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    data = {json.loads(l)["namespace"]: json.loads(l) for l in open(a.data)}
    if a.reference:
        jobs = {}
        for ns, d in data.items():
            lines = open(os.path.join(a.src, d["completion_path"])).readlines()
            s, e = _pos(d["body_position"])
            jobs[ns] = "".join(lines[s - 1:e])
    else:
        jobs = {}
        for l in open(a.completions):
            r = json.loads(l)
            ns = r["id"]
            jobs[ns] = body_from_generation(r["text"], ns.split(".")[-1])
    if a.only:
        keep = set(x.strip() for x in open(a.only) if x.strip())
        jobs = {k: v for k, v in jobs.items() if k in keep}
    by_repo = defaultdict(list)
    for ns in jobs:
        by_repo[data[ns]["project_path"]].append(ns)

    def run_repo(repo):
        out = []
        for ns in by_repo[repo]:
            t0 = time.time()
            res = score_sample(data[ns], jobs[ns], a.src)
            out.append(dict(id=ns, repo=repo, result=res, score=float(res == "Pass"), s=round(time.time() - t0, 1)))
        return out

    with ThreadPoolExecutor(a.workers) as ex:
        res = [r for rs in ex.map(run_repo, sorted(by_repo)) for r in rs]
    with open(a.out, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    n = len(res)
    print(json.dumps(dict(n=n, pass_rate=sum(r["score"] for r in res) / max(1, n),
                          results={k: sum(1 for r in res if r["result"] == k) for k in ("Pass", "Error", "TimeOut")})))


if __name__ == "__main__":
    main()
