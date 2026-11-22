"""Scorers: letter / label extraction, MATH answer check, ROUGE (SCROLLS-style geometric mean), Wilson CI.
Code items are scored by exec_code.py inside a network-less sandbox, never here."""
import math
import re
from collections import Counter


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def extract_letter(text, n_opts):
    t = strip_think(text)
    valid = "ABCDEFGHIJ"[:n_opts]
    m = re.search(r"(?:answer|option)\s*(?:is|:)?\s*\(?\**([A-J])\b", t, flags=re.I)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    m = re.match(r"\W*\(?([A-J])(?:\)|\.|:|\s|$)", t)
    if m and m.group(1) in valid:
        return m.group(1)
    m = re.findall(rf"\b([{valid}])\b", t)
    return m[0] if m else None


def extract_label(text, labels):
    t = strip_think(text).lower()
    pos = [(t.find(l.lower()), l) for l in labels]
    pos = [(p, l) for p, l in pos if p >= 0]
    if not pos and "not" in t and "mention" in t:
        return "Not mentioned"
    return min(pos)[1] if pos else None


def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
        k += 1
    return None


def _norm_math(s):
    s = s.strip().strip("$").replace("\\!", "").replace("\\left", "").replace("\\right", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").replace("^\\circ", "").replace("^{\\circ}", "")
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = s.replace(" ", "").rstrip(".")
    return s


def math_correct(text, gold):
    pred = last_boxed(strip_think(text))
    if pred is None:
        return False
    try:
        from math_verify import parse, verify
        return bool(verify(parse("$" + gold + "$"), parse("$" + pred + "$")))
    except ImportError:
        return _norm_math(pred) == _norm_math(gold)
    except Exception:
        return _norm_math(pred) == _norm_math(gold)


def _tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def _f(p, r):
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def _ngram_f(a, b, n):
    ca = Counter(tuple(a[i:i + n]) for i in range(len(a) - n + 1))
    cb = Counter(tuple(b[i:i + n]) for i in range(len(b) - n + 1))
    ov = sum((ca & cb).values())
    return _f(ov / max(1, sum(ca.values())), ov / max(1, sum(cb.values())))


def _lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge(pred, gold):
    """ROUGE-1/2/L F1 (no stemming) and their geometric mean (the SCROLLS QMSum score)."""
    a, b = _tok(strip_think(pred)), _tok(gold)
    r1, r2 = _ngram_f(a, b, 1), _ngram_f(a, b, 2)
    l = _lcs(a, b)
    rl = _f(l / max(1, len(a)), l / max(1, len(b)))
    return dict(r1=r1, r2=r2, rl=rl, geo=(r1 * r2 * rl) ** (1 / 3))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, c - h, c + h)


def mean_ci(xs, z=1.96):
    n = len(xs)
    if n == 0:
        return (0.0, 0.0)
    m = sum(xs) / n
    if n < 2:
        return (m, 0.0)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return (m, z * sd / math.sqrt(n))


def _last_json_with(text, key):
    """Last balanced {...} in text that parses as JSON and contains `key`."""
    import json
    ends = [i for i, c in enumerate(text) if c == "}"]
    for e in reversed(ends):
        depth = 0
        for b in range(e, -1, -1):
            if text[b] == "}":
                depth += 1
            elif text[b] == "{":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[b:e + 1])
                    except Exception:
                        break
                    if isinstance(obj, dict) and key in obj:
                        return obj
                    break
    return None


def _nz(x):
    return re.sub(r"\s+", " ", str(x)).strip().lower()


def zebra_score(text, gold):
    """ZebraLogic grid mode: puzzle solved iff every cell matches (case/space-insensitive); also cell accuracy."""
    obj = _last_json_with(strip_think(text), "solution")
    cells = sum(len(v) for v in gold.values())
    if not obj or not isinstance(obj.get("solution"), dict):
        return dict(pred=None, score=0.0, cell_acc=0.0)
    sol = {_nz(k): v for k, v in obj["solution"].items()}
    ok = 0
    for house, attrs in gold.items():
        got = sol.get(_nz(house)) or {}
        got = {_nz(k): v for k, v in got.items()} if isinstance(got, dict) else {}
        ok += sum(1 for a, v in attrs.items() if _nz(got.get(_nz(a), "")) == _nz(v))
    return dict(pred=None, score=float(ok == cells), cell_acc=ok / cells)


def score_item(item, text):
    k = item["kind"]
    if k == "letter":
        p = extract_letter(text, item["meta"]["n_opts"])
        return dict(pred=p, score=float(p == item["answer"]))
    if k == "label":
        p = extract_label(text, item["meta"]["labels"])
        return dict(pred=p, score=float(p == item["answer"]))
    if k == "math":
        return dict(pred=last_boxed(strip_think(text)), score=float(math_correct(text, item["answer"])))
    if k == "summary":
        r = rouge(text, item["answer"])
        return dict(pred=None, score=r["geo"], rouge=r)
    if k == "zebra":
        return zebra_score(text, item["answer"])
    return dict(pred=None, score=None)  # code: exec_code.py
