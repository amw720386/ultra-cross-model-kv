"""Generation benchmarks (chat prompts, greedy decoding, task-specific scoring).

Every benchmark here is written against one abstract batched generator, so the
same code measures the native target prefill, a mapped (cross-model) cache, or a
mapped cache with the last N prompt tokens recomputed by the target::

    gen(prompts: list[list[int]], max_new: int) -> list[list[int]]

``prompts`` are full prompt token ids (chat template already applied, thinking
disabled); the generator greedily decodes up to ``max_new`` new tokens per
prompt and stops at EOS / ``<|im_end|>`` (the harness additionally cuts each
output at the first stop token, so returning stop tokens is harmless).

Prompt construction
-------------------
``tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
enable_thinking=False)`` (falls back to a template without the kwarg, then to a
plain ``User:/Assistant:`` format), optionally followed by an assistant prefix
(e.g. EvalPlus/BigCodeBench "Below is a Python script ... ```python" or a RULER
answer prefix). For long-document tasks the document always comes first and the
question/instructions LAST in the user message, so a recompute window over the
final prompt tokens covers the whole question.

Benchmarks (``run_<name>(gen, tok, limit=100, seed=0, max_ctx=None, batch=8)``)
------------------------------------------------------------------------------
Short / code / math : arc_chat, humaneval, mbpp, bigcodebench, math500, zebralogic, deveval
Long documents      : contract_nli, bamboo_paperqa, longhealth, qmsum, quality,
                      longbench_v2, locomo, mrcr, ruler

See :data:`BENCHMARKS` for dataset ids, metrics and context policies. Each run
returns ``{"task", "metric", <metric>, "score", "n", "n_skipped", "by_<group>",
"prompt_tokens", "outputs"}``; ``outputs`` holds the first few (prompt tail,
output, gold, score) records.

Sampling and ``max_ctx``
------------------------
Items are drawn from a seeded permutation of the full split (``limit<=0`` = all).
With ``max_ctx`` set, an item is admissible only if ``prompt_tokens + max_new <=
max_ctx``. Tasks whose official harness truncates (LongHealth: distractor
documents are cut; LongBench v2: middle truncation; LoCoMo: oldest turns are
dropped) truncate to fit; every other task skips over-long items and draws the
next one from the permutation ("skip-and-refill"). ``n_skipped`` counts skips
and ``prompt_tokens`` reports min/median/mean/max of the evaluated prompts.

Code execution
--------------
HumanEval / MBPP / BigCodeBench programs run in a fresh ``python`` subprocess in a
temporary directory with a wall-clock timeout (process group killed on
timeout) and an address-space limit on POSIX. This is isolation against
accidents, not a security boundary: run untrusted generations inside a
container. BigCodeBench needs the packages in :data:`BIGCODEBENCH_PIP`.

``python -m xkv.genevals --selfcheck`` (with ``PYTHONPATH=src``) loads two items
per benchmark, builds prompts with the Qwen3.5 tokenizer (tokenizer files only)
and checks every scorer with an oracle generator (gold answers / canonical code
must score 1) and a dummy generator (must score below 1).
"""

from __future__ import annotations

import ast
import difflib
import json
import math
import os
import random
import re
import statistics
import string
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "GenFn",
    "BENCHMARKS",
    "GENEVALS_PIP",
    "BIGCODEBENCH_PIP",
    "chat_prompt",
    "run_program",
    "extract_code",
    "run_arc_chat",
    "run_contract_nli",
    "run_bamboo_paperqa",
    "run_longhealth",
    "run_qmsum",
    "run_humaneval",
    "run_mbpp",
    "run_bigcodebench",
    "run_math500",
    "run_zebralogic",
    "run_deveval",
    "run_quality",
    "run_longbench_v2",
    "run_ruler",
    "run_locomo",
    "run_mrcr",
]

#: Batched greedy generator: (prompt token ids, max_new_tokens) -> generated ids.
GenFn = Callable[[list[list[int]], int], list[list[int]]]

#: Packages used by the scorers (besides ``datasets`` / ``transformers``).
GENEVALS_PIP = ["datasets>=2.19", "math-verify[antlr4_13_2]>=0.7", "rouge-score>=0.1.2", "nltk>=3.8"]

#: BigCodeBench execution environment (bigcode-project/bigcodebench Requirements/requirements-eval.txt).
BIGCODEBENCH_PIP = [
    "beautifulsoup4==4.8.2", "blake3==0.4.1", "chardet==5.2.0", "cryptography==38.0.0",
    "datetime==5.5", "Django==4.2.7", "dnspython==2.6.1", "docxtpl==0.11.5", "Faker==20.1.0",
    "flask_login==0.6.3", "flask_restful==0.3.10", "flask_wtf==1.2.1", "Flask-Mail==0.9.1",
    "flask==3.0.3", "folium==0.16.0", "gensim==4.3.2", "geopandas==0.13.2", "geopy==2.4.1",
    "holidays==0.29", "keras==2.11.0", "Levenshtein==0.25.0", "librosa==0.10.1", "lxml==4.9.3",
    "matplotlib==3.7.0", "mechanize==0.4.9", "natsort==7.1.1", "networkx==2.6.3", "nltk==3.8",
    "numba==0.55.0", "numpy==1.21.2", "opencv-python-headless==4.9.0.80", "openpyxl==3.1.2",
    "pandas==2.0.3", "Pillow==10.3.0", "prettytable==3.10.0", "psutil==5.9.5",
    "pycryptodome==3.14.1", "pyfakefs==5.4.1", "pyquery==1.4.3", "pytesseract==0.3.10",
    "pytest==8.2.0", "python_http_client==3.3.7", "python-dateutil==2.9.0", "python-docx==1.1.0",
    "python-Levenshtein-wheels", "pytz==2023.3.post1", "PyYAML==6.0.1", "requests_mock==1.11.0",
    "requests==2.31.0", "rsa==4.9", "scikit-image==0.18.0", "scikit-learn==1.3.1", "scipy==1.7.2",
    "seaborn==0.13.2", "selenium==4.15", "sendgrid==6.11.0", "shapely==2.0.4", "soundfile==0.12.1",
    "statsmodels==0.14.0", "sympy==1.12", "tensorflow==2.11.0", "textblob==0.18.0",
    "texttable==1.7.0", "Werkzeug==3.0.1", "wikipedia==1.4.0", "wordcloud==1.9.3",
    "wordninja==2.0.0", "WTForms==3.1.2", "xlrd==2.0.1", "xlwt==1.3.0", "xmltodict==0.13.0",
]
# The upstream file pins Python-3.8-era wheels (numpy 1.21, tensorflow 2.11); build the
# BigCodeBench image on python3.10 or use the official bigcodebench evaluate docker image.

# Raw files not hosted on the HF hub (pinned commits of the official repositories).
_GH = "https://raw.githubusercontent.com"
URLS = {
    "bamboo_paperqa": f"{_GH}/RUCAIBox/BAMBOO/f230f206148396c12774ef3a632df291d96dc6ab/datasets/paperqa_16k.jsonl",
    "longhealth": f"{_GH}/kbressem/LongHealth/479922279ce8fa528fe2cd87d854c6a2f8365d21/data/benchmark_v5.json",
    "locomo": f"{_GH}/snap-research/locomo/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json",
    "deveval_data": f"{_GH}/seketeam/DevEval/02e768c172171fc17c96e5366381232cab77d6c4/data.tar.gz",
    "deveval_local_completion": f"{_GH}/seketeam/DevEval/02e768c172171fc17c96e5366381232cab77d6c4/"
                                "Experiments/prompt/local_completion/gpt-4-1106_prompt.jsonl",
    "deveval_without_context": f"{_GH}/seketeam/DevEval/02e768c172171fc17c96e5366381232cab77d6c4/"
                               "Experiments/prompt/without_context/gpt-4-1106_prompt.jsonl",
}

CACHE_DIR = Path(os.environ.get("XKV_GENEVALS_CACHE", Path.home() / ".cache" / "xkv_genevals"))

#: name -> (source, split, metric, prompt format / context policy)
BENCHMARKS: dict[str, dict[str, str]] = {
    "arc_chat": dict(source="allenai/ai2_arc:ARC-Challenge", split="test (1172)", metric="acc (letter)",
                     prompt="question + lettered options, 'respond with only the letter'"),
    "contract_nli": dict(source="tasksource/contract-nli:contractnli_b", split="test (2091, full NDAs)",
                         metric="acc (3-way)", prompt="contract, then hypothesis + 3 labels; skip-and-refill"),
    "bamboo_paperqa": dict(source=URLS["bamboo_paperqa"], split="paperqa_16k: 84 parseable 4-option items",
                           metric="acc (BAMBOO qa_accuracy)", prompt="official BAMBOO prompt; skip-and-refill"),
    "longhealth": dict(source=URLS["longhealth"], split="task 1: 20 patients x 20 = 400",
                       metric="acc (5-way)", prompt="official LongHealth system+template; distractor docs truncated"),
    "qmsum": dict(source="Ahren09/QMSum", split="test (35 meetings, 281 queries)",
                  metric="ROUGE-L F1 (rouge_score, stemmed); also R1/R2/geo-mean",
                  prompt="transcript, then query; skip-and-refill"),
    "humaneval": dict(source="openai/openai_humaneval", split="test (164)", metric="pass@1 (exec)",
                      prompt="EvalPlus chat prompt + '```python' assistant prefix"),
    "mbpp": dict(source="google-research-datasets/mbpp:sanitized", split="test (257)", metric="pass@1 (exec)",
                 prompt="EvalPlus-style chat prompt with the 3 asserts shown"),
    "bigcodebench": dict(source="bigcode/bigcodebench (or bigcode/bigcodebench-hard)", split="v0.1.4 (1140 / 148)",
                         metric="pass@1 (official unittest TestCases)", prompt="official instruct chat prompt"),
    "math500": dict(source="HuggingFaceH4/MATH-500", split="test (500)", metric="acc (math_verify)",
                    prompt="problem + 'reason step by step ... \\boxed{}'"),
    "zebralogic": dict(source="WildEval/ZebraLogic:grid_mode", split="test (1000)",
                       metric="puzzle acc (all cells); also cell acc", prompt="official ZeroEval ZEBRA_GRID template"),
    "deveval": dict(source=URLS["deveval_local_completion"], split="1825 (data.jsonl)",
                    metric="pass@1 (repo tests; needs DevEval Source_Code + env)",
                    prompt="official local_completion instruct prompt (GPT-4 variant)"),
    "quality": dict(source="emozilla/quality", split="validation (2086)", metric="acc (letter)",
                    prompt="article, then question + options; skip-and-refill"),
    "longbench_v2": dict(source="zai-org/LongBench-v2", split="train (503)", metric="acc (official extractor)",
                         prompt="official 0-shot template; middle truncation to max_ctx"),
    "ruler": dict(source="synthetic (RULER recipe) + rajpurkar/squad + hotpotqa/hotpot_qa + Salesforce/wikitext",
                  split="generated per length", metric="string_match_all / string_match_part",
                  prompt="official RULER templates + answer prefixes"),
    "locomo": dict(source=URLS["locomo"], split="10 conversations, 1986 QA", metric="token F1 (official)",
                   prompt="official LoCoMo conversation format + QA prompt; oldest turns dropped"),
    "mrcr": dict(source="openai/mrcr", split="2needle (400) / 4needle / 8needle", metric="SequenceMatcher ratio (official)",
                 prompt="the dataset's multi-turn messages; skip-and-refill"),
}


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def _encode(tok: Any, text: str) -> list[int]:
    try:
        return list(tok.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tok.encode(text))


def _decode(tok: Any, ids: Sequence[int]) -> str:
    try:
        return tok.decode(list(ids), skip_special_tokens=True)
    except TypeError:
        return tok.decode(list(ids))


def _stop_ids(tok: Any) -> set[int]:
    out: set[int] = set()
    eos = getattr(tok, "eos_token_id", None)
    if isinstance(eos, int):
        out.add(eos)
    elif isinstance(eos, (list, tuple)):
        out.update(int(e) for e in eos)
    for t in ("<|im_end|>", "<|endoftext|>"):
        try:
            i = tok.convert_tokens_to_ids(t)
        except Exception:  # noqa: BLE001
            continue
        unk = getattr(tok, "unk_token_id", None)
        if isinstance(i, int) and i >= 0 and i != unk:
            out.add(i)
    return out


def chat_prompt(tok: Any, user: str | None = None, system: str | None = None,
                assistant_prefix: str = "", messages: list[dict] | None = None) -> str:
    """Render messages with the chat template (thinking disabled when supported)."""
    if messages is None:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user or ""}]
    try:
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:  # noqa: BLE001 - tokenizer without a chat template
        text = "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in messages) + "Assistant:"
    return text + assistant_prefix


def _fetch(key: str) -> Path:
    """Download a pinned raw file into :data:`CACHE_DIR` (once)."""
    url = URLS[key]
    dest = CACHE_DIR / (key + "_" + url.rsplit("/", 1)[-1])
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.replace(dest)
    return dest


def _load_hf(path: str, name: str | None, split: str, **kw: Any) -> Any:
    import datasets  # lazy

    return datasets.load_dataset(path, name, split=split, **kw)


def _order(n: int, seed: int) -> list[int]:
    return random.Random(seed).sample(range(n), n)


def _len_stats(lengths: Sequence[int]) -> dict:
    if not lengths:
        return {}
    return {"min": min(lengths), "median": int(statistics.median(lengths)),
            "mean": round(statistics.fmean(lengths), 1), "max": max(lengths)}


def _letters(n: int) -> str:
    return string.ascii_uppercase[:n]


# --------------------------------------------------------------------------- #
# Items, selection, generation and aggregation
# --------------------------------------------------------------------------- #
@dataclass
class Item:
    prompt: str                      # rendered prompt text (chat template + assistant prefix)
    ids: list[int]                   # prompt token ids
    gold: Any                        # whatever the scorer needs
    max_new: int
    groups: dict[str, str] = field(default_factory=dict)   # e.g. {"category": "temporal"}
    oracle: str | None = None        # a perfect model response (for --selfcheck)
    meta: dict = field(default_factory=dict)


def _mk(tok: Any, prompt: str, gold: Any, max_new: int, **kw: Any) -> Item:
    return Item(prompt=prompt, ids=_encode(tok, prompt), gold=gold, max_new=max_new, **kw)


def _fits(item: Item, max_ctx: int | None) -> bool:
    return max_ctx is None or len(item.ids) + item.max_new <= max_ctx


def _select(n_total: int, build: Callable[[int], Item | None], limit: int, seed: int,
            max_ctx: int | None) -> tuple[list[Item], int]:
    """Skip-and-refill selection over a seeded permutation of ``range(n_total)``."""
    items: list[tuple[int, Item]] = []
    skipped = 0
    for i in _order(n_total, seed):
        if 0 < limit <= len(items):
            break
        it = build(i)
        if it is None:
            continue
        if not _fits(it, max_ctx):
            skipped += 1
            continue
        items.append((i, it))
    items.sort(key=lambda x: x[0])
    return [it for _, it in items], skipped


def _generate(gen: GenFn, tok: Any, items: Sequence[Item], batch: int) -> list[str]:
    """Run ``gen`` over items (grouped by max_new, longest prompts first) and decode."""
    stops = _stop_ids(tok)
    outs: list[str | None] = [None] * len(items)
    by_new: dict[int, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_new[it.max_new].append(i)
    for max_new, idx in by_new.items():
        idx.sort(key=lambda i: -len(items[i].ids))
        for s in range(0, len(idx), max(1, batch)):
            chunk = idx[s:s + max(1, batch)]
            res = gen([items[i].ids for i in chunk], max_new)
            if len(res) != len(chunk):
                raise ValueError(f"generator returned {len(res)} outputs for {len(chunk)} prompts")
            for i, ids in zip(chunk, res):
                ids = list(ids)
                for j, t in enumerate(ids):
                    if t in stops:
                        ids = ids[:j]
                        break
                outs[i] = _decode(tok, ids)
    return [o or "" for o in outs]


def _aggregate(task: str, metric: str, items: Sequence[Item], outputs: Sequence[str],
               scores: Sequence[dict], n_skipped: int, n_show: int = 3, **extra: Any) -> dict:
    res: dict[str, Any] = {"task": task, "metric": metric, "n": len(items), "n_skipped": n_skipped}
    keys = sorted({k for s in scores for k, v in s.items() if v is not None})
    valid = [s for s in scores if s.get("score") is not None]
    for k in keys:
        vals = [s[k] for s in scores if s.get(k) is not None]
        res[k if k != "score" else "score"] = (sum(vals) / len(vals)) if vals else None
    res[metric] = res.get("score")
    res["n_scored"] = len(valid)
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for it, s in zip(items, scores):
        if s.get("score") is None:
            continue
        for g, v in it.groups.items():
            groups[g][str(v)].append(float(s["score"]))
    for g, d in groups.items():
        res[f"by_{g}"] = {v: {"n": len(xs), "score": sum(xs) / len(xs)} for v, xs in sorted(d.items())}
    res["prompt_tokens"] = _len_stats([len(it.ids) for it in items])
    res["outputs"] = [
        {"prompt_tail": it.prompt[-400:], "output": out[:2000], "gold": str(it.gold)[:500],
         "score": s.get("score")}
        for it, out, s in list(zip(items, outputs, scores))[:n_show]
    ]
    res.update(extra)
    return res


@dataclass
class Bench:
    name: str
    metric: str
    build: Callable[..., tuple[list[Item], int]]
    score: Callable[[list[Item], list[str]], list[dict]]


def _evaluate(bench: Bench, gen: GenFn, tok: Any, items: list[Item], skipped: int, batch: int,
              **extra: Any) -> dict:
    outputs = _generate(gen, tok, items, batch)
    scores = bench.score(items, outputs)
    return _aggregate(bench.name, bench.metric, items, outputs, scores, skipped, **extra)


# --------------------------------------------------------------------------- #
# Answer parsing helpers
# --------------------------------------------------------------------------- #
def _parse_letter(text: str, letters: str) -> str | None:
    """Extract a multiple-choice letter from a free-form chat answer."""
    t = text.strip()
    L = re.escape(letters)
    pats = [
        rf"(?i:answer|option|choice)\s*(?i:is)?\s*[:\-]?\s*\(?\**\s*([{L}])\b",
        rf"^\s*[\(\[\*]*\s*([{L}])\s*[\)\]\.:\*,]",
        rf"^\s*([{L}])\s*$",
        rf"^\s*([{L}])\b",
        rf"\(([{L}])\)",
        rf"\b([{L}])\b",
    ]
    for p in pats:
        m = re.search(p, t)
        if m:
            return m.group(1)
    return None


def _mc_block(options: Sequence[str], letters: str | None = None) -> str:
    letters = letters or _letters(len(options))
    return "\n".join(f"{l}. {o}" for l, o in zip(letters, options))


def _mc_score(items: list[Item], outputs: list[str]) -> list[dict]:
    out = []
    for it, o in zip(items, outputs):
        pred = _parse_letter(o, it.meta["letters"])
        out.append({"score": float(pred == it.gold), "parsed": float(pred is not None)})
    return out


# --------------------------------------------------------------------------- #
# Code extraction and sandboxed execution
# --------------------------------------------------------------------------- #
_RUNNER = r'''
import contextlib, io, json, os, sys, types, unittest
def _limit(mem_gb):
    try:
        import resource
        b = int(mem_gb * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (b, b))
    except Exception:
        pass
def main():
    path, mode, mem = sys.argv[1], sys.argv[2], float(sys.argv[3])
    if mem > 0:
        _limit(mem)
    src = open(path, encoding="utf-8").read()
    res = {"passed": False, "error": ""}
    buf = io.StringIO()
    real_out = sys.stdout
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            mod = types.ModuleType("__test__")
            mod.__dict__.update({"__file__": os.path.abspath(path), "__builtins__": __builtins__})
            sys.modules["__test__"] = mod
            exec(compile(src, "__test__.py", "exec"), mod.__dict__)
            if mode == "unittest":
                suite = unittest.TestLoader().loadTestsFromTestCase(mod.TestCases)
                r = unittest.TestResult()
                suite.run(r)
                issues = r.failures + r.errors
                res["passed"] = r.testsRun > 0 and not issues
                res["error"] = "\n".join(t for _, t in issues)[-3000:]
            else:
                res["passed"] = True
    except BaseException as e:  # noqa: BLE001
        res["error"] = (type(e).__name__ + ": " + str(e))[-3000:]
    real_out.write("\n__XKV_RESULT__" + json.dumps(res) + "\n")
    real_out.flush()
    os._exit(0)
main()
'''


def run_program(program: str, mode: str = "script", timeout: float = 10.0, mem_gb: float = 4.0) -> dict:
    """Execute ``program`` in a fresh subprocess; ``mode`` is "script" or "unittest".

    Returns ``{"passed": bool, "status": "pass"|"fail"|"timeout", "error": str}``.
    """
    with tempfile.TemporaryDirectory(prefix="xkv_exec_", ignore_cleanup_errors=True) as d:
        runner, prog = os.path.join(d, "_xkv_runner.py"), os.path.join(d, "_xkv_prog.py")
        with open(runner, "w", encoding="utf-8") as f:
            f.write(_RUNNER)
        with open(prog, "w", encoding="utf-8") as f:
            f.write(program)
        env = {k: v for k, v in os.environ.items()
               if k in ("PATH", "SYSTEMROOT", "SystemRoot", "COMSPEC", "LANG", "LC_ALL", "PYTHONPATH",
                        "NLTK_DATA", "VIRTUAL_ENV", "CONDA_PREFIX", "LD_LIBRARY_PATH", "USERPROFILE",
                        "APPDATA", "LOCALAPPDATA")}
        env.update({"HOME": env.get("HOME", d), "TMPDIR": d, "TEMP": d, "TMP": d, "MPLBACKEND": "Agg",
                    "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8",
                    "TF_CPP_MIN_LOG_LEVEL": "3", "PYTHONDONTWRITEBYTECODE": "1"})
        kw: dict[str, Any] = {}
        if os.name == "posix":
            kw["start_new_session"] = True
        else:
            kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        p = subprocess.Popen([sys.executable, runner, prog, mode, str(mem_gb if os.name == "posix" else 0)],
                             cwd=d, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             stdin=subprocess.DEVNULL, **kw)
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(p)
            return {"passed": False, "status": "timeout", "error": f"timeout after {timeout}s"}
        text = out.decode("utf-8", "replace")
        marker = text.rfind("__XKV_RESULT__")
        if marker < 0:
            return {"passed": False, "status": "fail",
                    "error": ("no result (crash?) rc=%s " % p.returncode) + err.decode("utf-8", "replace")[-2000:]}
        res = json.loads(text[marker + len("__XKV_RESULT__"):].strip().splitlines()[0])
        res["status"] = "pass" if res["passed"] else "fail"
        return res


def _kill_tree(p: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            import signal

            os.killpg(p.pid, signal.SIGKILL)
        else:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
    except Exception:  # noqa: BLE001
        pass
    try:
        p.kill()
        p.communicate(timeout=5)
    except Exception:  # noqa: BLE001
        pass


def _run_many(programs: Sequence[tuple[str, str, float, float]], workers: int | None = None) -> list[dict]:
    workers = workers or max(1, min(8, (os.cpu_count() or 2) // 2))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda a: run_program(*a), programs))


_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+\-]*)[ \t]*\n(.*?)(?:```|\Z)", re.S)


def extract_code(text: str, entry_point: str | None = None, in_fence: bool = False) -> str:
    """Return the Python code in a chat response.

    ``in_fence``: the assistant prefix already opened a ```python fence, so the
    response starts with code and ends at the next fence. Otherwise the fenced
    block defining ``entry_point`` (else the longest python block) is used, and
    unfenced text is returned as-is.
    """
    if in_fence:
        return text.split("```", 1)[0]
    blocks = [(lang.lower(), body) for lang, body in _FENCE.findall(text)]
    blocks = [b for lang, b in blocks if lang in ("", "python", "py", "python3")] or [b for _, b in blocks]
    if not blocks:
        return text
    if entry_point:
        hit = [b for b in blocks if re.search(rf"^\s*(async\s+)?def\s+{re.escape(entry_point)}\s*\(", b, re.M)]
        if hit:
            return max(hit, key=len)
    return max(blocks, key=len)


def _import_header(src: str) -> str:
    return "\n".join(l for l in src.splitlines() if re.match(r"\s*(import|from)\s+\S+", l) and not l.startswith(" "))


_EVALPLUS_INSTR = ("Please provide a self-contained Python script that solves the following problem "
                   "in a markdown code block:\n```\n{problem}\n```\n")
_EVALPLUS_PREFIX = ("Below is a Python script with a self-contained function that solves the problem and "
                    "passes corresponding tests:\n```python\n")


# --------------------------------------------------------------------------- #
# Text metrics: ROUGE (QMSum) and LoCoMo token F1
# --------------------------------------------------------------------------- #
_ROUGE: Any = None


def _rouge(pred: str, ref: str) -> dict:
    """ROUGE-1/2/L F1 via ``rouge_score`` (Porter stemming); pure-python fallback without stemming."""
    global _ROUGE
    try:
        if _ROUGE is None:
            from rouge_score import rouge_scorer

            _ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        s = _ROUGE.score(ref, pred)
        return {k: s[k].fmeasure for k in ("rouge1", "rouge2", "rougeL")}
    except ImportError:
        pass
    tk = lambda x: re.findall(r"[a-z0-9]+", x.lower())  # noqa: E731
    p, r = tk(pred), tk(ref)

    def f(overlap: int, a: int, b: int) -> float:
        if not overlap or not a or not b:
            return 0.0
        pr, rc = overlap / a, overlap / b
        return 2 * pr * rc / (pr + rc)

    out = {}
    for n in (1, 2):
        pc = Counter(tuple(p[i:i + n]) for i in range(len(p) - n + 1))
        rcn = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        out[f"rouge{n}"] = f(sum((pc & rcn).values()), sum(pc.values()), sum(rcn.values()))
    prev = [0] * (len(r) + 1)
    for a in p:
        cur = [0]
        for j, b in enumerate(r):
            cur.append(prev[j] + 1 if a == b else max(prev[j + 1], cur[j]))
        prev = cur
    out["rougeL"] = f(prev[-1], len(p), len(r))
    return out


_STEM: Any = None


def _stem(w: str) -> str:
    global _STEM
    if _STEM is None:
        try:
            from nltk.stem import PorterStemmer

            _STEM = PorterStemmer().stem
        except ImportError:
            _STEM = lambda x: x  # noqa: E731
    return _STEM(w)


def _locomo_norm(s: str) -> str:
    s = s.replace(",", "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())


def _locomo_f1_single(pred: str, gold: str) -> float:
    p = [_stem(w) for w in _locomo_norm(pred).split()]
    g = [_stem(w) for w in _locomo_norm(gold).split()]
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    pr, rc = same / len(p), same / len(g)
    return 2 * pr * rc / (pr + rc)


def _locomo_f1_multi(pred: str, gold: str) -> float:
    preds = [x.strip() for x in pred.split(",")]
    golds = [x.strip() for x in gold.split(",")]
    return sum(max(_locomo_f1_single(p, g) for p in preds) for g in golds) / len(golds)


# --------------------------------------------------------------------------- #
# MATH answer checking
# --------------------------------------------------------------------------- #
def last_boxed(text: str) -> str | None:
    """Content of the last ``\\boxed{...}`` / ``\\fbox{...}`` (brace-matched)."""
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        m = re.match(r"\\boxed\s+([^\s$]+)", text[idx:])
        return m.group(1) if m else None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def _strip_math(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:mbox|mathrm|textbf|mathbf)\{([^}]*)\}", r"\1", s)
    for a, b in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\ ", ""),
                 ("dfrac", "frac"), ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""),
                 ("%", ""), ("\\$", ""), ("\\displaystyle", ""), ("\\le", "<="), ("\\ge", ">=")]:
        s = s.replace(a, b)
    s = re.sub(r"(degrees?|units?|cm|inches|inch|meters?|feet|dollars?|cents?)\s*$", "", s)
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s)  # "x = 5" -> "5"
    s = s.replace(" ", "").rstrip(".")
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac\{([^{}]*)\}(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\sqrt(\d+)", r"\\sqrt{\1}", s)
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    if s.startswith("0.") is False and re.fullmatch(r"\.\d+", s):
        s = "0" + s
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)
    return s


def _num(s: str) -> float | None:
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    try:
        if m:
            return (-1 if m.group(1) else 1) * float(m.group(2)) / float(m.group(3))
        if re.fullmatch(r"-?\d+/\d+", s):
            a, b = s.split("/")
            return float(a) / float(b)
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def math_equal(pred_text: str, gold: str) -> bool:
    """``math_verify`` if importable, else boxed-answer string/number normalization."""
    try:
        from math_verify import parse, verify

        to = None if os.name == "nt" else 5  # math_verify times out via SIGALRM (POSIX main thread)
        g = parse("$" + gold + "$", parsing_timeout=to)
        boxed = last_boxed(pred_text)
        p = parse(pred_text if boxed is None else "$" + boxed + "$", parsing_timeout=to)
        if g and p:
            return bool(verify(g, p, timeout_seconds=to))
        if not p:
            return False
    except ImportError:
        pass
    pred = last_boxed(pred_text)
    if pred is None:
        return False
    a, b = _strip_math(pred), _strip_math(gold)
    if a == b:
        return True
    x, y = _num(a), _num(b)
    return x is not None and y is not None and abs(x - y) <= 1e-6 * max(1.0, abs(y))


# --------------------------------------------------------------------------- #
# 1. ARC-Chat
# --------------------------------------------------------------------------- #
def _build_arc_chat(tok: Any, limit: int, seed: int, max_ctx: int | None, max_new: int = 32):
    docs = _load_hf("allenai/ai2_arc", "ARC-Challenge", "test")

    def build(i: int) -> Item:
        d = docs[i]
        labels, texts = list(d["choices"]["label"]), list(d["choices"]["text"])
        letters = _letters(len(texts))
        gold = letters[labels.index(d["answerKey"])]
        user = (f"Answer the following multiple-choice question.\n\nQuestion: {d['question']}\n"
                f"{_mc_block(texts, letters)}\n\nRespond with only the letter of the correct answer.")
        return _mk(tok, chat_prompt(tok, user), gold, max_new, oracle=gold, meta={"letters": letters})

    return _select(len(docs), build, limit, seed, max_ctx)


# --------------------------------------------------------------------------- #
# 2. ContractNLI
# --------------------------------------------------------------------------- #
_NLI_NAMES = {"entailment": "Entailment", "contradiction": "Contradiction", "neutral": "Not mentioned"}


def _build_contract_nli(tok: Any, limit: int, seed: int, max_ctx: int | None, max_new: int = 16):
    docs = _load_hf("tasksource/contract-nli", "contractnli_b", "test")
    names = docs.features["label"].names

    def build(i: int) -> Item:
        d = docs[i]
        gold = names[int(d["label"])]
        user = (f"{d['premise'].strip()}\n\n---\n\nThe text above is a contract (non-disclosure agreement). "
                "Decide whether the following hypothesis is entailed by the contract, contradicted by the "
                "contract, or not mentioned in the contract.\n\n"
                f"Hypothesis: {d['hypothesis']}\n\n"
                "Answer with exactly one of: Entailment, Contradiction, Not mentioned.")
        return _mk(tok, chat_prompt(tok, user), gold, max_new, groups={"label": gold},
                   oracle=_NLI_NAMES[gold])

    return _select(len(docs), build, limit, seed, max_ctx)


def parse_nli(text: str) -> str | None:
    t = text.lower()
    cands = {"neutral": [r"not\s+mentioned", r"neutral", r"not\s+addressed", r"no\s+mention"],
             "contradiction": [r"contradict"], "entailment": [r"entail"]}
    best: tuple[int, str] | None = None
    for lab, pats in cands.items():
        for p in pats:
            m = re.search(p, t)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), lab)
    return best[1] if best else None


def _score_contract_nli(items: list[Item], outputs: list[str]) -> list[dict]:
    return [{"score": float(parse_nli(o) == it.gold)} for it, o in zip(items, outputs)]


# --------------------------------------------------------------------------- #
# 3. BAMBOO PaperQA (16k)
# --------------------------------------------------------------------------- #
_BAMBOO_SYSTEM = "You are a helpful assistant."
_BAMBOO_TEMPLATE = (
    "You are given an article or a long dialogue, a multiple-choice question with four optional answer"
    "(marked by A, B, C, D). Please choose the best answer to the question based on the given content by "
    "responsing its corresponding letter (either A, B, C, or D). Do not provide any explanation.\n\n"
    "Article: ```{content}```\n\nQuestion: {question}\nOptions: {options}\n\n"
    "Answer(Only give the corresponding letter (either A, B, C, or D)): ")


def _load_bamboo() -> list[dict]:
    """paperqa_16k rows that parse and have exactly four options (84 of 100 upstream)."""
    rows = []
    for line in _fetch("bamboo_paperqa").read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line, strict=False)
        except json.JSONDecodeError:
            continue  # one upstream line is malformed JSON
        if len(r.get("options", [])) == 4 and r.get("answer") in ("A", "B", "C", "D"):
            rows.append(r)
    return rows


def _build_bamboo_paperqa(tok: Any, limit: int, seed: int, max_ctx: int | None, max_new: int = 16):
    rows = _load_bamboo()

    def build(i: int) -> Item:
        r = rows[i]
        user = _BAMBOO_TEMPLATE.format(content=r["content"], question=r["question"],
                                       options="\t".join(r["options"]))
        return _mk(tok, chat_prompt(tok, user, system=_BAMBOO_SYSTEM), r["answer"], max_new, oracle=r["answer"])

    return _select(len(rows), build, limit, seed, max_ctx)


def bamboo_qa_accuracy(pred: str, right: str) -> int:
    """Official BAMBOO ``qa_accuracy`` (first character of the first line)."""
    first = pred.strip().split("\n")[0].strip(" ")
    p = first[0] if first else ""
    for L in "ABCD":
        if f"[{L}" in p or p.lower().startswith(L.lower()):
            return int(L == right)
    return 0


def _score_bamboo(items: list[Item], outputs: list[str]) -> list[dict]:
    return [{"score": float(bamboo_qa_accuracy(o, it.gold))} for it, o in zip(items, outputs)]


# --------------------------------------------------------------------------- #
# 4. LongHealth (task 1: all documents of the patient)
# --------------------------------------------------------------------------- #
_LH_SYSTEM = """
You are a highly skilled and detail-oriented assistant, specifically trained to assist medical professionals in interpreting and extracting key information from medical documents. Your primary responsibility will be to analyze discharge letters from hospitals. When you receive one or more of these letters, you will be expected to carefully review the contents and accurately answer multiple-choice questions related to these documents.

Your answers should be:
1. Accurate: Make sure your answers are based on the information provided in the letters.
2. Concise: Provide brief and direct answers without unnecessary elaboration.
3. Contextual: Consider the context and specifics of each question to provide the most relevant information.

Remember, your job is to streamline the physician's decision-making process by providing them with accurate and relevant information from discharge summaries. Efficiency and reliability are key.
"""
_LH_TEMPLATE = """
--------------BEGIN DOCUMENTS--------------

{documents}

--------------END DOCUMENTS--------------

{question_text}
{options}

Please answer using the following format:
1. Begin your answer with the phrase "The correct answer is".
2. State the letter of the correct option (e.g., A, B, C, D, E).
3. Follow the letter with a colon and the exact text of the option you chose.
4. Make sure your answer is a single, concise sentence.

For example, if the correct answer to a question is option C, and the text for C is 'Acute Bronchitis', your answer should be:
'The correct answer is C: Acute bronchitis.'
"""
_LH_SEP = "\n\n-------------- NEW DOCUMENT --------------\n\n"


def _build_longhealth(tok: Any, limit: int, seed: int, max_ctx: int | None, max_new: int = 64):
    bench = json.loads(_fetch("longhealth").read_text(encoding="utf-8"))
    flat = [(pid, qi) for pid, p in bench.items() for qi in range(len(p["questions"]))]

    def build(i: int) -> Item:
        pid, qi = flat[i]
        p, q = bench[pid], bench[pid]["questions"][qi]
        opts = [q[f"answer_{l}"] for l in "abcde"]
        gold = "ABCDE"[opts.index(q["correct"])] if q["correct"] in opts else None
        options = "\n".join(f"{L}: {o}" for L, o in zip("ABCDE", opts))
        answer_ids = list(q.get("answer_location", {}).keys())
        docs = [(k, p["texts"][k]) for k in answer_ids if k in p["texts"]]
        docs += [(k, t) for k, t in p["texts"].items() if k not in answer_ids]
        if max_ctx is not None:  # official create_prompt: answer docs first, then cut what overflows
            fixed = len(_encode(tok, chat_prompt(tok, _LH_TEMPLATE.format(
                documents="", question_text=q["question"], options=options), system=_LH_SYSTEM)))
            budget = max_ctx - max_new - fixed
            sep = len(_encode(tok, _LH_SEP))
            kept = []
            for k, t in docs:
                ids = _encode(tok, t)
                if len(ids) <= budget:
                    kept.append((k, t))
                    budget -= len(ids) + sep
                else:
                    if budget > 32:
                        kept.append((k, _decode(tok, ids[:budget])))
                    break
            docs = kept
        random.Random(f"{seed}-{pid}-{qi}").shuffle(docs)
        user = _LH_TEMPLATE.format(documents=_LH_SEP.join(t for _, t in docs), question_text=q["question"],
                                   options=options)
        return _mk(tok, chat_prompt(tok, user, system=_LH_SYSTEM), gold, max_new,
                   groups={"diagnosis": p.get("diagnosis", "?")},
                   oracle=f"The correct answer is {gold}: {q['correct']}.",
                   meta={"n_docs": len(docs), "letters": "ABCDE"})

    return _select(len(flat), build, limit, seed, max_ctx)


def _score_longhealth(items: list[Item], outputs: list[str]) -> list[dict]:
    out = []
    for it, o in zip(items, outputs):
        m = re.search(r"correct answer is\s*:?\s*\(?\**([A-E])\b", o, re.I)
        pred = m.group(1).upper() if m else _parse_letter(o, "ABCDE")
        out.append({"score": float(pred == it.gold)})
    return out


# --------------------------------------------------------------------------- #
# 5. QMSum (query-based meeting summarization)
# --------------------------------------------------------------------------- #
def _build_qmsum(tok: Any, limit: int, seed: int, max_ctx: int | None, max_new: int = 512):
    docs = _load_hf("Ahren09/QMSum", None, "test")
    flat = []
    for mi, d in enumerate(docs):
        for kind in ("specific", "general"):
            for q in d[f"{kind}_query_list"]:
                flat.append((mi, kind, q["query"], q["answer"]))
    cache: dict[int, str] = {}

    def transcript(mi: int) -> str:
        if mi not in cache:
            turns = docs[mi]["meeting_transcripts"]
            # Drop QMSum's {disfmarker}/{vocalsound}-style annotation tokens.
            clean = lambda s: re.sub(r"\s*\{[a-z_]+\}\s*", " ", s).strip()  # noqa: E731
            cache[mi] = "\n".join(t["speaker"] + ": " + clean(t["content"]) for t in turns)
        return cache[mi]

    def build(i: int) -> Item:
        mi, kind, query, answer = flat[i]
        user = (f"Meeting transcript:\n\n{transcript(mi)}\n\n---\n\n"
                "Answer the following query about the meeting transcript above. Respond with a concise "
                f"summary of a few sentences.\n\nQuery: {query}")
        return _mk(tok, chat_prompt(tok, user), answer.strip(), max_new, groups={"query_type": kind},
                   oracle=answer.strip())

    return _select(len(flat), build, limit, seed, max_ctx)


def _score_qmsum(items: list[Item], outputs: list[str]) -> list[dict]:
    out = []
    for it, o in zip(items, outputs):
        r = _rouge(o.strip(), it.gold)
        r["rouge_geo"] = (r["rouge1"] * r["rouge2"] * r["rougeL"]) ** (1 / 3)
        r["score"] = r["rougeL"]
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
# 6-8. HumanEval, MBPP, BigCodeBench
# --------------------------------------------------------------------------- #
def _build_humaneval(tok: Any, limit: int, seed: int, max_ctx: int | None, max_new: int = 768):
    docs = _load_hf("openai/openai_humaneval", None, "test")

    def build(i: int) -> Item:
        d = docs[i]
        prompt = chat_prompt(tok, _EVALPLUS_INSTR.format(problem=d["prompt"].strip()),
                             assistant_prefix=_EVALPLUS_PREFIX)
        return _mk(tok, prompt, d["task_id"], max_new, oracle=d["prompt"] + d["canonical_solution"] + "```",
                   meta={"doc": dict(d)})

    return _select(len(docs), build, limit, seed, max_ctx)


def _humaneval_program(d: dict, code: str) -> str:
    ep = d["entry_point"]
    if not re.search(rf"^\s*def\s+{re.escape(ep)}\s*\(", code, re.M):
        code = d["prompt"] + code  # body-only completion
    else:
        code = _import_header(d["prompt"]) + "\n" + code
    return code + "\n\n" + d["test"] + f"\n\ncheck({ep})\n"


def _score_humaneval(items: list[Item], outputs: list[str]) -> list[dict]:
    progs = [(_humaneval_program(it.meta["doc"], extract_code(o, it.meta["doc"]["entry_point"], in_fence=True)),
              "script", 10.0, 4.0) for it, o in zip(items, outputs)]
    return [{"score": float(r["passed"]), "timeout": float(r["status"] == "timeout")} for r in _run_many(progs)]


def _build_mbpp(tok: Any, limit: int, seed: int, max_ctx: int | None, max_new: int = 768):
    docs = _load_hf("google-research-datasets/mbpp", "sanitized", "test")

    def build(i: int) -> Item:
        d = docs[i]
        problem = f"{d['prompt']}\nYour code should pass these tests:\n\n" + "\n".join(d["test_list"])
        prompt = chat_prompt(tok, _EVALPLUS_INSTR.format(problem=problem), assistant_prefix=_EVALPLUS_PREFIX)
        return _mk(tok, prompt, d["task_id"], max_new, oracle=d["code"] + "\n```", meta={"doc": dict(d)})

    return _select(len(docs), build, limit, seed, max_ctx)


def _mbpp_program(d: dict, code: str) -> str:
    return "\n".join(d["test_imports"]) + "\n" + code + "\n\n" + "\n".join(d["test_list"]) + "\n"


def _score_mbpp(items: list[Item], outputs: list[str]) -> list[dict]:
    progs = [(_mbpp_program(it.meta["doc"], extract_code(o, None, in_fence=True)), "script", 10.0, 4.0)
             for it, o in zip(items, outputs)]
    return [{"score": float(r["passed"]), "timeout": float(r["status"] == "timeout")} for r in _run_many(progs)]


def _build_bigcodebench(tok: Any, limit: int, seed: int, max_ctx: int | None, max_new: int = 1280,
                        subset: str = "full", docs: Any = None):
    if docs is None:
        docs = _load_hf("bigcode/bigcodebench-hard" if subset == "hard" else "bigcode/bigcodebench", None, "v0.1.4")

    def build(i: int) -> Item:
        d = docs[i]
        prompt = chat_prompt(tok, _EVALPLUS_INSTR.format(problem=d["instruct_prompt"].strip()),
                             assistant_prefix=_EVALPLUS_PREFIX)
        return _mk(tok, prompt, d["task_id"], max_new, oracle=d["code_prompt"] + d["canonical_solution"] + "\n```",
                   meta={"doc": dict(d)})

    return _select(len(docs), build, limit, seed, max_ctx)


def _bigcodebench_program(d: dict, code: str) -> str:
    # Calibrated-style: re-add the task's import header so a solution that omits imports is not
    # failed for that alone (BigCodeBench "calibrated" pass@1 does the same with code_prompt).
    return _import_header(d["code_prompt"]) + "\n" + code + "\n\n" + d["test"] + "\n"


def _score_bigcodebench(items: list[Item], outputs: list[str], timeout: float = 120.0) -> list[dict]:
    progs = [(_bigcodebench_program(it.meta["doc"], extract_code(o, "task_func", in_fence=True)), "unittest",
              timeout, 30.0) for it, o in zip(items, outputs)]
    return [{"score": float(r["passed"]), "timeout": float(r["status"] == "timeout")} for r in _run_many(progs)]


# --------------------------------------------------------------------------- #
# 9. MATH-500
# --------------------------------------------------------------------------- #
def _build_math500(tok: Any, limit: int, seed: int, max_ctx: int | None, max_new: int = 3072):
    docs = _load_hf("HuggingFaceH4/MATH-500", None, "test")

    def build(i: int) -> Item:
        d = docs[i]
        user = f"{d['problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        return _mk(tok, chat_prompt(tok, user), d["answer"], max_new,
                   groups={"subject": d["subject"], "level": str(d["level"])},
                   oracle=f"Working it out, the final answer is $\\boxed{{{d['answer']}}}$.")

    return _select(len(docs), build, limit, seed, max_ctx)


def _score_math500(items: list[Item], outputs: list[str]) -> list[dict]:
    return [{"score": float(math_equal(o, it.gold)), "boxed": float(last_boxed(o) is not None)}
            for it, o in zip(items, outputs)]


# --------------------------------------------------------------------------- #
# 10. ZebraLogic (grid mode, ZeroEval prompt)
# --------------------------------------------------------------------------- #
_ZEBRA = """
# Example Puzzle

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

{
    "reasoning": "Given Clue 1, we know Peter is in House 2. According to Clue 2, Arnold is directly left of the one who only drinks water. The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.",
    "solution": {
        "House 1": {
            "Name": "Arnold",
            "Drink": "tea"
        },
        "House 2": {
            "Name": "Peter",
            "Drink": "water"
        },
        "House 3": {
            "Name": "Eric",
            "Drink": "milk"
        }
    }
}

# Puzzle to Solve

{puzzle}


# Instruction

Now please solve the above puzzle. Present your reasoning and solution in the following json format:

{json_template}

"""


def _zebra_table(sol: dict) -> dict:
    cols = sol["header"]
    return {f"House {i + 1}": {cols[j]: row[j] for j in range(1, len(cols))} for i, row in enumerate(sol["rows"])}


def _zebra_size_bucket(size: str) -> str:
    h, m = (int(x) for x in size.split("*"))
    space = m * math.log10(math.factorial(h))
    return "small" if space < 3 else "medium" if space < 6 else "large" if space < 10 else "xl"


def _build_zebralogic(tok: Any, limit: int, seed: int, max_ctx: int | None, max_new: int = 4096):
    docs = _load_hf("WildEval/ZebraLogic", "grid_mode", "test")

    def build(i: int) -> Item:
        d = docs[i]
        sol = d["solution"]
        tmpl = {"reasoning": "___", "solution": {h: {c: "___" for c in v} for h, v in _zebra_table(sol).items()}}
        user = _ZEBRA.replace("{puzzle}", d["puzzle"]).replace("{json_template}", json.dumps(tmpl, indent=4))
        gold = _zebra_table(sol)
        return _mk(tok, chat_prompt(tok, user), gold, max_new,
                   groups={"size": d["size"], "search_space": _zebra_size_bucket(d["size"])},
                   oracle=json.dumps({"reasoning": "By elimination.", "solution": gold}, indent=4))

    return _select(len(docs), build, limit, seed, max_ctx)


def extract_last_json(text: str) -> dict | None:
    """Last complete top-level JSON object in ``text`` (ZeroEval ``extract_last_complete_json``)."""
    text = re.sub(r"```(?:json)?", "", text)
    stack, last = [], None
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            if not stack:
                try:
                    obj = json.loads(text[start:i + 1])
                    if isinstance(obj, dict):
                        last = obj
                except json.JSONDecodeError:
                    pass
    return last


def _score_zebralogic(items: list[Item], outputs: list[str]) -> list[dict]:
    out = []
    for it, o in zip(items, outputs):
        pred = extract_last_json(o)
        table = pred.get("solution") if isinstance(pred, dict) else None
        total = sum(len(v) for v in it.gold.values())
        correct = 0
        if isinstance(table, dict):
            for house, cols in it.gold.items():
                ph = table.get(house)
                if not isinstance(ph, dict):
                    continue
                for c, v in cols.items():
                    pv = ph.get(c)
                    if isinstance(pv, list):
                        pv = pv[0] if pv else None
                    if pv is not None and str(pv).lower().strip() == v.lower().strip():
                        correct += 1
        out.append({"score": float(correct == total), "cell_acc": correct / total,
                    "parsed": float(table is not None)})
    return out


# --------------------------------------------------------------------------- #
# 11. DevEval (repository-level code generation)
# --------------------------------------------------------------------------- #
def _deveval_data() -> dict[str, dict]:
    tgz = _fetch("deveval_data")
    out = {}
    with tarfile.open(tgz) as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith("data.jsonl"))
        for line in tf.extractfile(member).read().decode("utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                out[d["namespace"]] = d
    return out


def _build_deveval(tok: Any, limit: int, seed: int, max_ctx: int | None, max_new: int = 1024,
                   setting: str = "local_completion"):
    data = _deveval_data()
    prompts = [json.loads(l) for l in _fetch(f"deveval_{setting}").read_text(encoding="utf-8").splitlines() if l.strip()]

    def build(i: int) -> Item:
        p = prompts[i]
        d = data[p["namespace"]]
        return _mk(tok, chat_prompt(tok, p["prompt"]), p["namespace"], max_new,
                   groups={"type": d["type"], "topic": d["project_path"].split("/")[0]}, meta={"data": d})

    return _select(len(prompts), build, limit, seed, max_ctx)


def deveval_body(response: str, d: dict) -> str:
    """Turn an instruct-model response into the function body DevEval splices into the file."""
    code = extract_code(response)
    name = d["namespace"].split(".")[-1]
    try:
        tree = ast.parse(textwrap.dedent(code))
        lines = textwrap.dedent(code).splitlines()
        fn = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == name), None)
    except SyntaxError:
        fn = None
    if fn is None:  # assume the response is already a body
        body = code
    else:
        stmts = fn.body
        if (stmts and isinstance(stmts[0], ast.Expr) and isinstance(getattr(stmts[0], "value", None), ast.Constant)
                and isinstance(stmts[0].value.value, str) and len(stmts) > 1):
            stmts = stmts[1:]  # the prompt already contains the docstring
        body = "\n".join(lines[stmts[0].lineno - 1:fn.end_lineno]) + "\n"
    return textwrap.indent(textwrap.dedent(body), " " * int(d["indent"]))


def _deveval_exec(d: dict, body: str, root: Path, timeout: float = 60.0) -> str:
    """Official pass_k.py procedure: splice body at body_position, run `python setup.py pytest`, restore."""
    path = root / d["completion_path"]
    orig = path.read_text(encoding="utf-8")
    lines = orig.splitlines(keepends=True)
    sos, eos = d["body_position"][0] - 1, d["body_position"][1]
    path.write_text("".join(lines[:sos] + ["\n", body, "\n"] + lines[eos:]), encoding="utf-8")
    try:
        for test in d["tests"]:
            try:
                r = subprocess.run([sys.executable, "setup.py", "pytest", "--addopts", test],
                                   cwd=root / d["project_path"], capture_output=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                return "TimeOut"
            if r.returncode != 0:
                return "Error"
        return "Pass"
    finally:
        path.write_text(orig, encoding="utf-8")


def _score_deveval(items: list[Item], outputs: list[str], source_root: str | None = None) -> list[dict]:
    root = source_root or os.environ.get("DEVEVAL_SOURCE_ROOT")
    out = []
    for it, o in zip(items, outputs):
        body = deveval_body(o, it.meta["data"])
        if not root or body.strip() == "pass":
            out.append({"score": None if not root else 0.0})
            continue
        out.append({"score": float(_deveval_exec(it.meta["data"], body, Path(root)) == "Pass")})
    return out


# --------------------------------------------------------------------------- #
# 12. QuALITY
# --------------------------------------------------------------------------- #
def _build_quality(tok: Any, limit: int, seed: int, max_ctx: int | None, max_new: int = 16):
    docs = _load_hf("emozilla/quality", None, "validation")

    def build(i: int) -> Item:
        d = docs[i]
        letters = _letters(len(d["options"]))
        gold = letters[int(d["answer"])]
        user = (f"{d['article'].strip()}\n\n---\n\nAnswer the following question about the story above.\n\n"
                f"Question: {d['question']}\n{_mc_block(d['options'], letters)}\n\n"
                "Respond with only the letter of the correct answer.")
        return _mk(tok, chat_prompt(tok, user), gold, max_new,
                   groups={"difficulty": "hard" if d["hard"] else "easy"}, oracle=gold, meta={"letters": letters})

    return _select(len(docs), build, limit, seed, max_ctx)


# --------------------------------------------------------------------------- #
# 13. LongBench v2
# --------------------------------------------------------------------------- #
_LB2_TEMPLATE = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)"."""


def _build_longbench_v2(tok: Any, limit: int, seed: int, max_ctx: int | None, max_new: int = 128):
    docs = _load_hf("zai-org/LongBench-v2", None, "train")

    def build(i: int) -> Item:
        d = docs[i]
        fill = lambda ctx: (_LB2_TEMPLATE.replace("$DOC$", ctx.strip()).replace("$Q$", d["question"].strip())  # noqa: E731
                            .replace("$C_A$", d["choice_A"].strip()).replace("$C_B$", d["choice_B"].strip())
                            .replace("$C_C$", d["choice_C"].strip()).replace("$C_D$", d["choice_D"].strip()))
        ctx, truncated = d["context"], False
        if max_ctx is not None:  # official: keep the first and last halves of the token budget
            overhead = len(_encode(tok, chat_prompt(tok, fill(""))))
            budget = max_ctx - max_new - overhead - 8
            ids = _encode(tok, ctx)
            if len(ids) > budget:
                ctx = _decode(tok, ids[:budget // 2]) + _decode(tok, ids[len(ids) - budget // 2:])
                truncated = True
        it = _mk(tok, chat_prompt(tok, fill(ctx)), d["answer"], max_new,
                 groups={"difficulty": d["difficulty"], "length": d["length"], "domain": d["domain"]},
                 oracle=f"The correct answer is ({d['answer']})", meta={"truncated": truncated})
        if max_ctx is not None and not _fits(it, max_ctx):  # retokenization drift: trim a little more
            it = None if len(it.ids) + max_new > max_ctx + 64 else it
        return it

    items, skipped = _select(len(docs), build, limit, seed, None)
    if max_ctx is not None:
        keep = [it for it in items if _fits(it, max_ctx + 64)]
        skipped += len(items) - len(keep)
        items = keep
    return items, skipped


def parse_longbench_v2(text: str) -> str | None:
    """Official LongBench v2 ``extract_answer``."""
    text = text.replace("*", "")
    m = re.search(r"The correct answer is \(([A-D])\)", text)
    if m:
        return m.group(1)
    m = re.search(r"The correct answer is ([A-D])", text)
    return m.group(1) if m else None


def _score_longbench_v2(items: list[Item], outputs: list[str]) -> list[dict]:
    return [{"score": float(parse_longbench_v2(o) == it.gold), "truncated": float(it.meta["truncated"])}
            for it, o in zip(items, outputs)]


# --------------------------------------------------------------------------- #
# 14. RULER (synthetic: retrieval / multi-hop tracing / QA)
# --------------------------------------------------------------------------- #
_RULER_NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
_RULER_NIAH = ("Some special magic {t} are hidden within the following text. Make sure to memorize it. "
               "I will quiz you about the {t} afterwards.\n{context}\n"
               "What are all the special magic {t} for {query} mentioned in the provided text?")
_RULER_NIAH_PREFIX = " The special magic {t} for {query} mentioned in the provided text are"
_RULER_VT = ("Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\n"
             "Question: Find all variables that are assigned the value {query} in the text above.")
_RULER_VT_PREFIX = (" Answer: According to the chain(s) of variable assignment in the text above, {n} variables "
                    "are assgined the value {query}, they are: ")
_RULER_QA = ("Answer the question based on the given documents. Only give me the answer and do not output any "
             "other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on "
             "the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {query}")
_RULER_QA_PREFIX = " Answer:"
RULER_TASKS = {  # name -> (category, max_new)  (RULER tokens_to_generate)
    "niah_single_1": ("retrieval", 128), "niah_single_2": ("retrieval", 128),
    "niah_multikey_2": ("retrieval", 128), "niah_multikey_3": ("retrieval", 128),
    "niah_multivalue": ("retrieval", 128), "niah_multiquery": ("retrieval", 128),
    "vt": ("multi-hop", 30), "qa_squad": ("qa", 32), "qa_hotpot": ("qa", 32),
}
_WORDS_A = ("abundant adorable agreeable alive ancient angry bewildered bitter brave breezy calm careful "
            "charming cheerful clean clever cloudy clumsy colossal crooked cuddly curious damp delightful "
            "eager elegant enchanting faithful fancy fierce gentle gifted glamorous gleaming grumpy handsome "
            "helpful hilarious hollow jolly kind lazy lively lucky magnificent misty modern nervous nutty "
            "obedient polite proud quaint quick quiet rapid rich roomy rough rustic scary shiny silly smooth "
            "sparkling spicy splendid steady sticky stormy tame tender tense thoughtful tidy tiny victorious "
            "vivacious wandering wise witty wonderful zany zealous").split()
_WORDS_N = ("apple badge banana basket battery beach bicycle blanket bottle bracelet bridge bucket cabinet "
            "camera candle canyon carpet castle cellar chair chimney circus cloud coffee compass cookie cottage "
            "crayon dentist desert diamond dragon drawer engine falcon feather fiddle forest fountain garden "
            "giraffe glacier guitar hammer harbor helmet island jacket kettle kitten ladder lantern lemon "
            "library lizard magnet meadow mirror monkey needle notebook ocean orchard paddle panther parrot "
            "pebble pencil pepper piano pillow planet pocket puzzle quilt rabbit rainbow ribbon rocket saddle "
            "scarf shovel spider squirrel stadium sunflower teapot tiger tornado trumpet tunnel umbrella "
            "valley violin volcano wagon walnut whistle window wizard yacht zebra").split()
_RULER_CACHE: dict[str, Any] = {}


def _ruler_essay_sents() -> list[str]:
    """Natural-text haystack (RULER uses Paul Graham essays; we use wikitext-2, digits removed)."""
    if "essay" not in _RULER_CACHE:
        ds = _load_hf("Salesforce/wikitext", "wikitext-2-raw-v1", "train")
        sents = []
        for row in ds:
            line = row["text"].strip()
            if not line or line.startswith("="):
                continue
            line = re.sub(r"\d", "", line).replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
            sents.extend(s.strip() + "." for s in line.split(" . ") if len(s.split()) >= 5)
        _RULER_CACHE["essay"] = sents
    return _RULER_CACHE["essay"]


def _ruler_qa_pool(kind: str) -> Any:
    if kind not in _RULER_CACHE:
        if kind == "squad":
            ds = _load_hf("rajpurkar/squad", None, "validation")
            ctxs = sorted(set(ds["context"]))
            idx = {c: i for i, c in enumerate(ctxs)}
            qs, seen = [], set()
            for r in ds:  # one question per context (RULER dedups contexts too)
                if r["context"] in seen:
                    continue
                seen.add(r["context"])
                qs.append({"q": r["question"], "answers": sorted(set(r["answers"]["text"])), "gold": [idx[r["context"]]]})
            _RULER_CACHE[kind] = (ctxs, qs)
        else:
            ds = _load_hf("hotpotqa/hotpot_qa", "distractor", "validation")
            paras: dict[str, int] = {}
            texts: list[str] = []
            qs = []
            for r in ds:
                ids = []
                for title, sents in zip(r["context"]["title"], r["context"]["sentences"]):
                    t = f"{title}\n{''.join(sents)}"
                    if t not in paras:
                        paras[t] = len(texts)
                        texts.append(t)
                    ids.append(paras[t])
                qs.append({"q": r["question"], "answers": [r["answer"]], "gold": ids})
            _RULER_CACHE[kind] = (texts, qs)
    return _RULER_CACHE[kind]


def _fit_units(tok: Any, render: Callable[[int], str], target: int, hi: int) -> str:
    """Largest unit count whose rendered prompt has <= target tokens (binary search)."""
    lo, best = 0, render(0)
    while lo <= hi:
        mid = (lo + hi) // 2
        s = render(mid)
        if len(_encode(tok, s)) <= target:
            best, lo = s, mid + 1
        else:
            hi = mid - 1
    return best


def _ruler_sample(tok: Any, task: str, length: int, rng: random.Random) -> Item:
    cat, max_new = RULER_TASKS[task]
    target = length - max_new

    def word_key() -> str:
        return f"{rng.choice(_WORDS_A)}-{rng.choice(_WORDS_N)}"

    def uuid() -> str:
        h = "".join(rng.choice("0123456789abcdef") for _ in range(32))
        return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"

    if task.startswith("niah"):
        t = "numbers"
        n_v, n_q, n_k = 1, 1, 1
        key_fn, val_fn = word_key, lambda: str(rng.randint(1_000_000, 9_999_999))
        haystack = "noise" if task == "niah_single_1" else "essay"
        if task == "niah_multikey_2":
            haystack, n_k = "needle", 1
        if task == "niah_multikey_3":
            haystack, key_fn, val_fn, t = "needle", uuid, uuid, "uuids"
        if task == "niah_multivalue":
            n_v = 4
        if task == "niah_multiquery":
            n_q = 4
        keys = []
        while len(keys) < max(n_q, n_k):
            k = key_fn()
            if k not in keys:
                keys.append(k)
        values = [[val_fn() for _ in range(n_v)] for _ in keys]
        needles = [f"One of the special magic {t} for {k} is: {v}." for k, vs in zip(keys, values) for v in vs]
        rng.shuffle(needles)
        depths = sorted(rng.random() for _ in needles)
        queries = keys[:n_q]
        query = queries[0] if n_q == 1 else ", ".join(queries[:-1]) + ", and " + queries[-1]
        golds = [v for k, vs in zip(keys, values) if k in queries for v in vs]
        if haystack == "needle":  # distractor needles with other keys
            pool = []
            while len(pool) < 4000:
                k = key_fn()
                if k not in keys:
                    pool.append(f"One of the special magic {t} for {k} is: {val_fn()}.")
        elif haystack == "essay":
            sents = _ruler_essay_sents()
            start = rng.randrange(len(sents) - 3000)
            pool = sents[start:start + 3000]
        else:
            pool = [_RULER_NOISE] * 4000

        def render(n: int) -> str:
            units = list(pool[:n])
            for d, nd in sorted(zip(depths, needles), key=lambda x: -x[0]):
                units.insert(int(round(d * len(units))), nd)
            ctx = (" " if haystack != "needle" else "\n").join(units)
            return chat_prompt(tok, _RULER_NIAH.format(t=t, context=ctx, query=query),
                               assistant_prefix=_RULER_NIAH_PREFIX.format(t=t, query=query).lstrip())

        prompt = _fit_units(tok, render, target, len(pool))
        return _mk(tok, prompt, golds, max_new, groups={"task": task, "category": cat, "length": str(length)},
                   oracle=" " + ", ".join(golds) + ".", meta={"metric": "all"})

    if task == "vt":
        names = []
        while len(names) < 5:
            v = "".join(rng.choice(string.ascii_uppercase) for _ in range(5))
            if v not in names:
                names.append(v)
        value = str(rng.randint(10000, 99999))
        chain = [f"VAR {names[0]} = {value}"] + [f"VAR {names[i]} = VAR {names[i - 1]}" for i in range(1, 5)]
        depths = sorted(rng.random() for _ in chain)

        def render(n: int) -> str:
            units = [_RULER_NOISE] * n
            for d, c in sorted(zip(depths, chain), key=lambda x: -x[0]):
                units.insert(int(round(d * len(units))), c + ".")
            return chat_prompt(tok, _RULER_VT.format(context="\n".join(units), query=value),
                               assistant_prefix=_RULER_VT_PREFIX.format(n=len(names), query=value).lstrip())

        prompt = _fit_units(tok, render, target, 4000)
        return _mk(tok, prompt, names, max_new, groups={"task": task, "category": cat, "length": str(length)},
                   oracle=" ".join(names), meta={"metric": "all"})

    texts, qs = _ruler_qa_pool("squad" if task == "qa_squad" else "hotpot")
    q = qs[rng.randrange(len(qs))]
    gold_ids = set(q["gold"])
    distract = [i for i in rng.sample(range(len(texts)), min(len(texts), 3000)) if i not in gold_ids]

    def render(n: int) -> str:
        docs = list(q["gold"]) + distract[:n]
        random.Random(len(docs) * 7919 + q["gold"][0]).shuffle(docs)
        ctx = "\n\n".join(f"Document {j + 1}:\n{texts[d]}" for j, d in enumerate(docs))
        return chat_prompt(tok, _RULER_QA.format(context=ctx, query=q["q"]), assistant_prefix=_RULER_QA_PREFIX.lstrip())

    prompt = _fit_units(tok, render, target, len(distract))
    return _mk(tok, prompt, q["answers"], max_new, groups={"task": task, "category": cat, "length": str(length)},
               oracle=" " + q["answers"][0], meta={"metric": "part"})


def _build_ruler(tok: Any, limit: int, seed: int, max_ctx: int | None,
                 lengths: Sequence[int] = (4096, 8192, 16384), tasks: Sequence[str] | None = None):
    """``limit`` samples per length, round-robin over ``tasks`` (all of :data:`RULER_TASKS` by default)."""
    tasks = list(tasks or RULER_TASKS)
    items, skipped = [], 0
    for L in lengths:
        if max_ctx is not None and L > max_ctx:
            skipped += max(limit, 0)
            continue
        for j in range(max(limit, 1)):
            task = tasks[j % len(tasks)]
            items.append(_ruler_sample(tok, task, L, random.Random(f"ruler-{seed}-{L}-{j}")))
    return items, skipped


def _score_ruler(items: list[Item], outputs: list[str]) -> list[dict]:
    out = []
    for it, o in zip(items, outputs):
        p = o.lower()
        hits = [r.lower() in p for r in it.gold]
        out.append({"score": (sum(hits) / len(hits)) if it.meta["metric"] == "all" else float(any(hits))})
    return out


# --------------------------------------------------------------------------- #
# 15. LoCoMo (long conversational memory QA)
# --------------------------------------------------------------------------- #
LOCOMO_CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
_LOCOMO_START = ("Below is a conversation between two people: {} and {}. The conversation takes place over "
                 "multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n")
_LOCOMO_QA = ("Based on the above conversations, write a short answer for the following question in a few words. "
              "Do not write complete and lengthy sentences. Answer with exact words from the conversations "
              "whenever possible.\n\nQuestion: {}")


def _locomo_context(conv: dict, budget_fn: Callable[[str], bool] | None) -> str:
    """Official ``get_input_context`` formatting; drops the oldest turns when over budget."""
    nums = sorted(int(k.split("_")[-1]) for k in conv if k.startswith("session_") and "date_time" not in k)
    sessions = []
    for i in nums:
        turns = []
        for t in conv[f"session_{i}"]:
            s = t["speaker"] + ' said, "' + t["text"] + '"\n'
            if "blip_caption" in t:
                s += " and shared %s." % t["blip_caption"]
            turns.append(s + "\n")
        sessions.append((conv[f"session_{i}_date_time"], turns))

    def render(skip: int) -> str:
        out, seen = [], 0
        for date, turns in sessions:
            kept = []
            for t in turns:
                seen += 1
                if seen > skip:
                    kept.append(t)
            if kept:
                out.append("\nDATE: " + date + "\nCONVERSATION:\n" + "".join(kept))
        return _LOCOMO_START.format(conv["speaker_a"], conv["speaker_b"]) + "".join(out)

    if budget_fn is None or budget_fn(render(0)):
        return render(0)
    total = sum(len(t) for _, t in sessions)
    lo, hi = 0, total
    while lo < hi:  # smallest number of dropped (oldest) turns that fits
        mid = (lo + hi) // 2
        if budget_fn(render(mid)):
            hi = mid
        else:
            lo = mid + 1
    return render(lo)


def _build_locomo(tok: Any, limit: int, seed: int, max_ctx: int | None, max_new: int = 50):
    data = json.loads(_fetch("locomo").read_text(encoding="utf-8"))
    flat = [(si, qi) for si, s in enumerate(data) for qi in range(len(s["qa"]))]
    ctx_cache: dict[int, str] = {}

    def build(i: int) -> Item:
        si, qi = flat[i]
        qa = data[si]["qa"][qi]
        cat = int(qa["category"])
        q = qa["question"]
        gold: Any = str(qa.get("answer", ""))
        oracle = gold
        if cat == 2:
            q += " Use DATE of CONVERSATION to answer with an approximate date."
        elif cat == 5:
            adv = str(qa.get("adversarial_answer", qa.get("answer", "")))
            if random.Random(f"{seed}-{si}-{qi}").random() < 0.5:
                opts = {"a": "No information available", "b": adv}
            else:
                opts = {"a": adv, "b": "No information available"}
            q += f" (a) {opts['a']} (b) {opts['b']}. Select the correct answer by writing (a) or (b)."
            gold = opts
            oracle = "(a)" if opts["a"] == "No information available" else "(b)"
        elif cat == 3:
            gold = gold.split(";")[0].strip()
            oracle = gold
        question = _LOCOMO_QA.format(q)
        if max_ctx is None:
            if si not in ctx_cache:
                ctx_cache[si] = _locomo_context(data[si]["conversation"], None)
            ctx = ctx_cache[si]
        else:
            overhead = len(_encode(tok, chat_prompt(tok, "\n\n" + question)))
            ctx = _locomo_context(data[si]["conversation"],
                                  lambda c: len(_encode(tok, c)) + overhead + max_new <= max_ctx)
        return _mk(tok, chat_prompt(tok, ctx + "\n\n" + question), gold, max_new,
                   groups={"category": LOCOMO_CATEGORIES.get(cat, str(cat))}, oracle=oracle,
                   meta={"category": cat})

    return _select(len(flat), build, limit, seed, max_ctx)


def _score_locomo(items: list[Item], outputs: list[str]) -> list[dict]:
    out = []
    for it, o in zip(items, outputs):
        ans = o.replace('\\"', "'").strip()
        lines = [w.strip() for w in ans.split("\n") if w.strip()]
        ans = lines[0] if lines else ""
        cat = it.meta["category"]
        if cat == 5:
            a = ans.lower()
            pick = it.gold["a"] if "(a)" in a else it.gold["b"]
            low = pick.lower()
            s = float("no information available" in low or "not mentioned" in low)
        else:
            a = ans.lower().replace("(a)", "").replace("(b)", "").replace("a)", "").replace("b)", "") \
                .replace("answer:", "").strip()
            s = _locomo_f1_multi(a, it.gold) if cat == 1 else _locomo_f1_single(a, it.gold)
        out.append({"score": s})
    return out


# --------------------------------------------------------------------------- #
# 16. OpenAI MRCR
# --------------------------------------------------------------------------- #
def _len_bin(n: int) -> str:
    lo = 1 << max(12, (max(n, 1) - 1).bit_length() - 1)
    return f"{lo // 1024}k-{2 * lo // 1024}k" if n > 4096 else "<=4k"


def _build_mrcr(tok: Any, limit: int, seed: int, max_ctx: int | None, max_new: int = 2048,
                needles: int = 2, min_ctx: int = 0):
    files = [f"{needles}needle/{needles}needle_0.parquet", f"{needles}needle/{needles}needle_1.parquet"]
    docs = _load_hf("openai/mrcr", None, "train", data_files=files)
    n_chars = docs["n_chars"]

    def build(i: int) -> Item | None:
        # cheap pre-filter on characters (Qwen tokenizer: ~3-5 chars/token on this data)
        if max_ctx is not None and n_chars[i] > 6 * max_ctx:
            return None
        if n_chars[i] < 2 * min_ctx:
            return None
        d = docs[i]
        it = _mk(tok, chat_prompt(tok, messages=json.loads(d["prompt"])), d["answer"], max_new,
                 oracle=d["answer"], meta={"prefix": d["random_string_to_prepend"]})
        if len(it.ids) < min_ctx:
            return None
        it.groups = {"length": _len_bin(len(it.ids))}
        return it

    return _select(len(docs), build, limit, seed, max_ctx)


def mrcr_grade(response: str, answer: str, prefix: str) -> float:
    """Official MRCR grader."""
    if not response.startswith(prefix):
        return 0.0
    return float(difflib.SequenceMatcher(None, response.removeprefix(prefix), answer.removeprefix(prefix)).ratio())


def _score_mrcr(items: list[Item], outputs: list[str]) -> list[dict]:
    return [{"score": mrcr_grade(o, it.gold, it.meta["prefix"])} for it, o in zip(items, outputs)]


# --------------------------------------------------------------------------- #
# Registry and public entry points
# --------------------------------------------------------------------------- #
_BENCH: dict[str, Bench] = {
    "arc_chat": Bench("arc_chat", "acc", _build_arc_chat, _mc_score),
    "contract_nli": Bench("contract_nli", "acc", _build_contract_nli, _score_contract_nli),
    "bamboo_paperqa": Bench("bamboo_paperqa", "acc", _build_bamboo_paperqa, _score_bamboo),
    "longhealth": Bench("longhealth", "acc", _build_longhealth, _score_longhealth),
    "qmsum": Bench("qmsum", "rougeL", _build_qmsum, _score_qmsum),
    "humaneval": Bench("humaneval", "pass@1", _build_humaneval, _score_humaneval),
    "mbpp": Bench("mbpp", "pass@1", _build_mbpp, _score_mbpp),
    "bigcodebench": Bench("bigcodebench", "pass@1", _build_bigcodebench, _score_bigcodebench),
    "math500": Bench("math500", "acc", _build_math500, _score_math500),
    "zebralogic": Bench("zebralogic", "puzzle_acc", _build_zebralogic, _score_zebralogic),
    "deveval": Bench("deveval", "pass@1", _build_deveval, _score_deveval),
    "quality": Bench("quality", "acc", _build_quality, _mc_score),
    "longbench_v2": Bench("longbench_v2", "acc", _build_longbench_v2, _score_longbench_v2),
    "ruler": Bench("ruler", "acc", _build_ruler, _score_ruler),
    "locomo": Bench("locomo", "f1", _build_locomo, _score_locomo),
    "mrcr": Bench("mrcr", "ratio", _build_mrcr, _score_mrcr),
}


def _run(name: str, gen: GenFn, tok: Any, limit: int, seed: int, max_ctx: int | None, batch: int,
         build_kw: dict | None = None, score_kw: dict | None = None) -> dict:
    b = _BENCH[name]
    build_kw = {k: v for k, v in (build_kw or {}).items() if v is not None}
    items, skipped = b.build(tok, limit, seed, max_ctx, **build_kw)
    outputs = _generate(gen, tok, items, batch)
    scores = b.score(items, outputs, **(score_kw or {}))
    return _aggregate(name, b.metric, items, outputs, scores, skipped, max_ctx=max_ctx, seed=seed)


def run_arc_chat(gen, tok, limit=100, seed=0, max_ctx=None, batch=8, max_new=None) -> dict:
    """ARC-Challenge test answered in chat; accuracy of the parsed letter."""
    return _run("arc_chat", gen, tok, limit, seed, max_ctx, batch, {"max_new": max_new})


def run_contract_nli(gen, tok, limit=100, seed=0, max_ctx=None, batch=8, max_new=None) -> dict:
    """ContractNLI (full NDA documents, 3-way: entailment / contradiction / not mentioned)."""
    return _run("contract_nli", gen, tok, limit, seed, max_ctx, batch, {"max_new": max_new})


def run_bamboo_paperqa(gen, tok, limit=100, seed=0, max_ctx=None, batch=8, max_new=None) -> dict:
    """BAMBOO PaperQA-16k; the population is 84 items, so ``limit>=84`` evaluates all of them."""
    return _run("bamboo_paperqa", gen, tok, limit, seed, max_ctx, batch, {"max_new": max_new})


def run_longhealth(gen, tok, limit=100, seed=0, max_ctx=None, batch=8, max_new=None) -> dict:
    """LongHealth task 1 (all of a patient's letters, 5-way MC)."""
    return _run("longhealth", gen, tok, limit, seed, max_ctx, batch, {"max_new": max_new})


def run_qmsum(gen, tok, limit=100, seed=0, max_ctx=None, batch=8, max_new=None) -> dict:
    """QMSum test queries; main metric ROUGE-L F1 (also ROUGE-1/2 and their geometric mean)."""
    return _run("qmsum", gen, tok, limit, seed, max_ctx, batch, {"max_new": max_new})


def run_humaneval(gen, tok, limit=100, seed=0, max_ctx=None, batch=8, max_new=None) -> dict:
    return _run("humaneval", gen, tok, limit, seed, max_ctx, batch, {"max_new": max_new})


def run_mbpp(gen, tok, limit=100, seed=0, max_ctx=None, batch=8, max_new=None) -> dict:
    return _run("mbpp", gen, tok, limit, seed, max_ctx, batch, {"max_new": max_new})


def run_bigcodebench(gen, tok, limit=100, seed=0, max_ctx=None, batch=8, max_new=None, subset="full",
                     timeout=120.0) -> dict:
    """BigCodeBench-Instruct (``subset`` "full" or "hard"); needs :data:`BIGCODEBENCH_PIP` installed."""
    return _run("bigcodebench", gen, tok, limit, seed, max_ctx, batch, {"max_new": max_new, "subset": subset},
                {"timeout": timeout})


def run_math500(gen, tok, limit=100, seed=0, max_ctx=None, batch=8, max_new=None) -> dict:
    return _run("math500", gen, tok, limit, seed, max_ctx, batch, {"max_new": max_new})


def run_zebralogic(gen, tok, limit=100, seed=0, max_ctx=None, batch=8, max_new=None) -> dict:
    return _run("zebralogic", gen, tok, limit, seed, max_ctx, batch, {"max_new": max_new})


def run_deveval(gen, tok, limit=100, seed=0, max_ctx=None, batch=8, max_new=None, setting="local_completion",
                source_root=None) -> dict:
    """DevEval. Scoring needs the extracted ``Source_Code`` tree inside the DevEval environment
    (``source_root`` or ``$DEVEVAL_SOURCE_ROOT``); without it outputs are generated but unscored."""
    return _run("deveval", gen, tok, limit, seed, max_ctx, batch, {"max_new": max_new, "setting": setting},
                {"source_root": source_root})


def run_quality(gen, tok, limit=100, seed=0, max_ctx=None, batch=8, max_new=None) -> dict:
    return _run("quality", gen, tok, limit, seed, max_ctx, batch, {"max_new": max_new})


def run_longbench_v2(gen, tok, limit=100, seed=0, max_ctx=None, batch=8, max_new=None) -> dict:
    """LongBench v2 (official 0-shot, no CoT). ``max_ctx`` triggers official middle truncation;
    ``None`` feeds full contexts (up to ~2M tokens)."""
    return _run("longbench_v2", gen, tok, limit, seed, max_ctx, batch, {"max_new": max_new})


def run_ruler(gen, tok, limit=100, seed=0, max_ctx=None, batch=8, lengths=(4096, 8192, 16384),
              tasks=None) -> dict:
    """RULER retrieval / multi-hop (variable tracking) / QA at each length (``limit`` items per length)."""
    return _run("ruler", gen, tok, limit, seed, max_ctx, batch, {"lengths": tuple(lengths), "tasks": tasks})


def run_locomo(gen, tok, limit=100, seed=0, max_ctx=None, batch=8, max_new=None) -> dict:
    """LoCoMo QA; F1 per official evaluation, broken down by category."""
    return _run("locomo", gen, tok, limit, seed, max_ctx, batch, {"max_new": max_new})


def run_mrcr(gen, tok, limit=100, seed=0, max_ctx=None, batch=8, max_new=None, needles=2, min_ctx=0) -> dict:
    """OpenAI MRCR (``needles`` 2/4/8); items with prompt tokens in [min_ctx, max_ctx - max_new]."""
    return _run("mrcr", gen, tok, limit, seed, max_ctx, batch,
                {"max_new": max_new, "needles": needles, "min_ctx": min_ctx})


# --------------------------------------------------------------------------- #
# Self-check (CPU, tokenizer only)
# --------------------------------------------------------------------------- #
def _oracle_gen(tok: Any, items: Sequence[Item]) -> GenFn:
    table = {tuple(it.ids): _encode(tok, it.oracle or "") for it in items}

    def gen(prompts: list[list[int]], max_new: int) -> list[list[int]]:
        return [table[tuple(p)] for p in prompts]  # oracle ignores max_new on purpose

    return gen


def _dummy_gen(tok: Any) -> GenFn:
    ids = _encode(tok, "I am not sure.")
    return lambda prompts, max_new: [list(ids) for _ in prompts]


def _bcb_local_docs(n: int) -> list[dict]:
    """BigCodeBench tasks whose libraries are importable here (so gold code can be executed)."""
    import importlib.util

    docs = _load_hf("bigcode/bigcodebench", None, "v0.1.4")
    out = []
    for i in _order(len(docs), 0):
        d = docs[i]
        libs = set(ast.literal_eval(d["libs"]) if isinstance(d["libs"], str) else d["libs"])
        for node in ast.walk(ast.parse(d["test"] + "\n" + d["code_prompt"] + "    pass\n")):
            if isinstance(node, ast.Import):
                libs.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                libs.add(node.module.split(".")[0])
        if all(importlib.util.find_spec(l) is not None for l in libs) and "requests" not in libs \
                and "socket" not in libs and "urllib" not in libs:
            out.append(dict(d))
        if len(out) >= n:
            break
    return out


def _unit_checks() -> list[str]:
    msgs = []
    assert _parse_letter("The answer is (C).", "ABCD") == "C"
    assert _parse_letter("B", "ABCD") == "B" and _parse_letter("**D**. Because", "ABCD") == "D"
    assert parse_nli("Not mentioned") == "neutral" and parse_nli("Contradiction.") == "contradiction"
    assert bamboo_qa_accuracy("C", "C") == 1 and bamboo_qa_accuracy("D. foo", "C") == 0
    assert parse_longbench_v2("The correct answer is (B)") == "B"
    assert math_equal("so \\boxed{\\frac{1}{2}}", "\\frac12") and math_equal("\\boxed{0.5}", "\\frac{1}{2}")
    assert not math_equal("\\boxed{3}", "4")
    assert last_boxed("a \\boxed{x^{2}} b \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert _strip_math("\\dfrac{3}{4}") == _strip_math("\\frac34")
    assert extract_code("x\n```python\ndef f():\n    return 1\n```\n", "f").startswith("def f")
    assert mrcr_grade("abcHello", "abcHello", "abc") == 1.0 and mrcr_grade("Hello", "abcHello", "abc") == 0.0
    assert abs(_locomo_f1_single("7 May 2023", "7 May 2023") - 1) < 1e-9
    r = _rouge("the cat sat on the mat", "the cat sat on the mat")
    assert abs(r["rougeL"] - 1) < 1e-9
    assert run_program("assert 1 + 1 == 2\n")["passed"]
    assert not run_program("assert 1 + 1 == 3\n")["passed"]
    assert run_program("import time\ntime.sleep(30)\n", timeout=2)["status"] == "timeout"
    ut = "import unittest\nclass TestCases(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(f(), 1)\n"
    assert run_program("def f():\n    return 1\n" + ut, "unittest")["passed"]
    assert not run_program("def f():\n    return 2\n" + ut, "unittest")["passed"]
    d = {"namespace": "pkg.mod.f", "indent": 4}
    body = deveval_body("```python\ndef f(x):\n    \"\"\"doc\"\"\"\n    y = x + 1\n    return y\n```", d)
    assert body == "    y = x + 1\n    return y\n", repr(body)
    msgs.append("unit checks OK (parsers, math, rouge, F1, MRCR grader, sandbox, DevEval body)")
    try:
        import math_verify  # noqa: F401
        msgs.append("math_verify: available")
    except ImportError:
        msgs.append("math_verify: NOT installed (fallback normalizer used)")
    try:
        import rouge_score  # noqa: F401
        msgs.append("rouge_score: available")
    except ImportError:
        msgs.append("rouge_score: NOT installed (unstemmed fallback used)")
    return msgs


def selfcheck(names: Iterable[str] | None = None, tokenizer: str = "Qwen/Qwen3.5-4B", n: int = 2) -> dict:
    from transformers import AutoTokenizer

    for m in _unit_checks():
        print("[unit]", m, flush=True)
    tok = AutoTokenizer.from_pretrained(tokenizer)
    status: dict[str, str] = {}
    for name in names or list(_BENCH):
        b = _BENCH[name]
        try:
            kw: dict[str, Any] = {}
            if name == "ruler":
                kw = {"lengths": (4096,), "tasks": None}
                items, skipped = b.build(tok, len(RULER_TASKS), 0, None, **kw)
            elif name == "bigcodebench":
                items, skipped = b.build(tok, n, 0, None, docs=_bcb_local_docs(n))
            elif name == "mrcr":
                items, skipped = b.build(tok, n, 0, 32768)
            elif name == "longbench_v2":
                items, skipped = b.build(tok, n, 0, 32768)
            else:
                items, skipped = b.build(tok, n, 0, None)
            assert items, "no items built"
            lens = [len(it.ids) for it in items]
            if name == "deveval":
                res = _evaluate(b, _dummy_gen(tok), tok, items, skipped, 8)
                assert res["n"] == len(items) and res["n_scored"] == 0
                status[name] = (f"PARTIAL (prompts built, tokens={lens}; execution needs DevEval Source_Code + "
                                "env via DEVEVAL_SOURCE_ROOT)")
                print(f"[{name}] {status[name]}", flush=True)
                continue
            good = _evaluate(b, _oracle_gen(tok, items), tok, items, skipped, 8)
            bad = _evaluate(b, _dummy_gen(tok), tok, items, skipped, 8)
            ok = good["score"] is not None and good["score"] > 0.999 and bad["score"] < 0.999
            groups = {k: list(v) for k, v in good.items() if k.startswith("by_")}
            status[name] = ("PASS" if ok else "FAIL") + f" (oracle={good['score']:.3f} dummy={bad['score']:.3f} " \
                                                       f"n={good['n']} prompt_tokens={lens} groups={groups})"
            if not ok:
                status[name] += " outputs=" + json.dumps(good["outputs"][:1])[:1500]
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            status[name] = f"ERROR {type(e).__name__}: {str(e)[:300]}"
        print(f"[{name}] {status[name]}", flush=True)
    return status


def prompt_stats(names: Iterable[str] | None = None, tokenizer: str = "Qwen/Qwen3.5-4B", n: int = 20,
                 seed: int = 0) -> dict:
    """Prompt-length statistics (tokens) over ``n`` sampled items per benchmark (no model)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    out = {}
    for name in names or list(_BENCH):
        kw: dict[str, Any] = {"lengths": (4096, 8192, 16384)} if name == "ruler" else {}
        items, _ = _BENCH[name].build(tok, n, seed, None, **kw)
        out[name] = _len_stats([len(it.ids) for it in items])
        print(f"[{name}] {out[name]}", flush=True)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selfcheck", action="store_true", help="CPU self-check (tokenizer + datasets only)")
    ap.add_argument("--stats", type=int, default=0, help="prompt-length stats over N items per benchmark")
    ap.add_argument("--only", nargs="*", default=None, help="subset of benchmarks")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    args = ap.parse_args(argv)
    if args.stats:
        prompt_stats(args.only, args.tokenizer, args.stats)
    if args.selfcheck or not args.stats:
        st = selfcheck(args.only, args.tokenizer)
        print("\nSUMMARY")
        for k, v in st.items():
            print(f"  {k:15s} {v.split(' ')[0]}")
        return 0 if all(not v.startswith(("FAIL", "ERROR")) for v in st.values()) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
