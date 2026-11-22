"""Execute generated code against HumanEval / MBPP tests. Run ONLY inside a throwaway sandbox, e.g.
docker run --rm --network none --memory 8g --pids-limit 512 -v $PWD:/w -w /w python:3.12-slim \
    python exec_code.py gens.jsonl scored.jsonl
Reads the generations file written by run.py; writes one row per code item with pass/fail."""
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor


def extract_code(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if blocks:
        return max(blocks, key=len)
    return text


def program(row):
    m = row["meta"]
    code = extract_code(row["text"])
    if row["bench"] == "humaneval":
        ep = m["entry_point"]
        if not re.search(rf"def\s+{re.escape(ep)}\s*\(", code):
            code = m["prompt"] + code
        else:  # keep the prompt's imports
            imports = "\n".join(l for l in m["prompt"].splitlines() if l.startswith(("import ", "from ")))
            code = imports + "\n" + code
        return code + "\n\n" + m["test"] + f"\n\ncheck({ep})\n"
    if row["bench"] == "bigcodebench":
        ep = m["entry_point"]
        if not re.search(rf"def\s+{re.escape(ep)}\s*\(", code):
            code = m["prompt"] + "\n" + code
        return (code + "\n\n" + m["test"] + "\n\nif __name__ == '__main__':\n    import unittest\n"
                "    unittest.main(argv=['bcb'], exit=True)\n")
    return code + "\n\n" + m["test"] + "\n"


def run_one(row, timeout=15):
    if row["bench"] == "bigcodebench":
        timeout = 120  # BigCodeBench's own harness allows long-running tasks (plots, sklearn fits)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "prog.py")
        with open(p, "w") as f:
            f.write(program(row))
        try:
            r = subprocess.run([sys.executable, p], cwd=d, capture_output=True, timeout=timeout, text=True)
            ok, err = r.returncode == 0, r.stderr[-300:]
        except subprocess.TimeoutExpired:
            ok, err = False, "timeout"
    return dict(id=row["id"], bench=row["bench"], arm=row.get("arm"), score=float(ok), err=None if ok else err)


def main(src, dst, workers=8):
    rows = [json.loads(l) for l in open(src)]
    rows = [r for r in rows if r["kind"] == "code"]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(run_one, rows))
    with open(dst, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    by = {}
    for r in res:
        by.setdefault(r["bench"], []).append(r["score"])
    print(json.dumps({b: dict(n=len(v), pass_rate=sum(v) / len(v)) for b, v in by.items()}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
