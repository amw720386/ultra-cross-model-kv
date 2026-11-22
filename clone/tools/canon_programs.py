"""Canonical extraction (decision code-extract-canonical, 2026-11-22): build each code item's test program from the
FULL saved model output with the benchmark's own official sanitizer, pinned; everything else as the harness built it.

  HumanEval / MBPP: evalplus.sanitize (evalplus==0.3.1) on the assistant turn as generated (EvalPlus answer prefix +
                    output), entry point = the item's entry_point (MBPP: none); program assembly as xkv's scorer
                    (_humaneval_program / _mbpp_program), only the extraction differs.
  BigCodeBench:     bigcodebench.sanitize (bigcodebench==0.2.5), entry point task_func.

Runs on CPU with the official packages:
  uv run --no-project --with evalplus==0.3.1 --with bigcodebench==0.2.5 python -I tools/canon_programs.py \
      xkv <gen_partial_<bench>.json> <strategy> <bench> out.jsonl
  ... python -I tools/canon_programs.py kvx <gens.jsonl> - bigcodebench out.jsonl
out.jsonl: one row per item {id, bench, entry_point, raw, sanitized, program}; execute with tools/modal_exec_programs.py.
"""
import json
import re
import sys

EVALPLUS_PREFIX = ("Below is a Python script with a self-contained function that solves the problem and "
                   "passes corresponding tests:\n```python\n")  # == xkv genevals._EVALPLUS_PREFIX


def _import_header(src):
    return "\n".join(l for l in src.splitlines() if re.match(r"\s*(import|from)\s+\S+", l) and not l.startswith(" "))


def prompt_helpers(prompt, code, entry_point):
    """HumanEval prompts can define helper functions the ORIGINAL check() calls (HE/38 encode_cyclic, HE/50
    encode_shift). evalplus.sanitize keeps only the entry point and its dependencies, and EvalPlus's own harness
    tests the entry point alone; with the original HumanEval tests (ours) a helper the model did not redefine must
    come from the prompt, as in HumanEval's prompt + completion convention. Returns the missing helpers' source."""
    import ast
    try:
        pt = ast.parse(prompt)
    except SyntaxError:
        return ""
    have = set()
    try:
        have = {n.name for n in ast.parse(code).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    except SyntaxError:
        pass
    out = [ast.get_source_segment(prompt, n) for n in pt.body
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name != entry_point and n.name not in have]
    return "\n\n".join(x for x in out if x)


def he_program(d, code):  # == xkv genevals._humaneval_program
    ep = d["entry_point"]
    if not re.search(rf"^\s*def\s+{re.escape(ep)}\s*\(", code, re.M):
        code = d["prompt"] + code
    else:
        code = _import_header(d["prompt"]) + "\n" + code
    return code + "\n\n" + d["test"] + f"\n\ncheck({ep})\n"


def mbpp_program(d, code):  # == xkv genevals._mbpp_program
    return "\n".join(d["test_imports"]) + "\n" + code + "\n\n" + "\n".join(d["test_list"]) + "\n"


def main():
    src, path, strategy, bench, out = sys.argv[1:6]
    rows = []
    if src == "xkv":
        from evalplus.sanitize import sanitize
        for o in json.load(open(path))[strategy]["outputs"]:
            d = o["doc"]
            raw = EVALPLUS_PREFIX + o["output"]
            ep = d.get("entry_point") if bench == "humaneval" else None
            s = sanitize(raw, ep)
            helpers = prompt_helpers(d["prompt"], s, ep) if bench == "humaneval" else ""
            prog = he_program(d, (helpers + "\n\n" + s) if helpers else s) if bench == "humaneval" else mbpp_program(d, s)
            rows.append(dict(id=str(d["task_id"]), bench=bench, entry_point=ep, raw=raw, sanitized=s, program=prog,
                             prompt_helpers_added=bool(helpers), old_score=o.get("score")))
    else:  # kvx gens.jsonl (BigCodeBench rows); the BCB scorer assembles the test program from `sanitized`
        from bigcodebench.sanitize import sanitize
        for l in open(path):
            g = json.loads(l)
            if g.get("bench") != "bigcodebench":
                continue
            raw = re.sub(r"<think>.*?</think>", "", g["text"], flags=re.S)
            s = sanitize(raw, "task_func")
            # gens-compatible: tools/modal_score_bcb.py --gens out.jsonl extracts the fenced block == `sanitized`
            rows.append(dict(id=g["id"], bench=bench, kind="code", arm=g.get("arm"), entry_point="task_func", raw=raw,
                             sanitized=s, program=None, meta=g.get("meta"), text="```python\n" + s + "\n```",
                             old_score=g.get("score")))
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(dict(items=len(rows), empty_sanitized=sum(1 for r in rows if not r["sanitized"].strip()))))


if __name__ == "__main__":
    main()
