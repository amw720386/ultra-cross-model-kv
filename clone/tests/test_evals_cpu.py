"""CPU tests for kvx.evals scorers and the code sandbox runner (no GPU, no network)."""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kvx.evals import score, exec_code  # noqa: E402


def test_letters():
    assert score.extract_letter("B", 4) == "B"
    assert score.extract_letter("The answer is (C).", 4) == "C"
    assert score.extract_letter("<think>maybe A</think>\nD. Paris", 4) == "D"
    assert score.extract_letter("Answer: **E**", 5) == "E"
    assert score.extract_letter("E", 4) is None


def test_labels():
    L = ["Entailment", "Contradiction", "Not mentioned"]
    assert score.extract_label("Contradiction.", L) == "Contradiction"
    assert score.extract_label("not mentioned", L) == "Not mentioned"
    assert score.extract_label("It is not explicitly mentioned", L) == "Not mentioned"


def test_math():
    assert score.last_boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert score.math_correct(r"\boxed{\dfrac{1}{2}}", r"\frac{1}{2}")
    assert not score.math_correct(r"\boxed{3}", "4")
    assert not score.math_correct("no box", "4")


def test_rouge():
    r = score.rouge("the cat sat on the mat", "the cat sat on the mat")
    assert abs(r["geo"] - 1.0) < 1e-9
    assert score.rouge("dog", "the cat")["geo"] == 0.0
    r = score.rouge("the cat sat", "the cat sat on the mat")
    assert abs(r["r1"] - 2 * 1.0 * 0.5 / 1.5) < 1e-9
    assert 0 < r["geo"] < 1


def test_wilson():
    p, lo, hi = score.wilson(61, 100)
    assert p == 0.61 and 0.51 < lo < 0.52 and 0.69 < hi < 0.71


def test_exec_code():
    he = dict(id="h", bench="humaneval", kind="code", text="```python\ndef add(a, b):\n    return a + b\n```",
              meta=dict(prompt="def add(a, b):\n", test="def check(f):\n    assert f(1, 2) == 3\n", entry_point="add"))
    bad = dict(he, id="hb", text="```python\ndef add(a, b):\n    return a - b\n```")
    body = dict(he, id="hc", text="```python\n    return a + b\n```")
    mb = dict(id="m", bench="mbpp", kind="code", text="```python\ndef sq(x):\n    return x * x\n```",
              meta=dict(test="assert sq(3) == 9", entry_point=None))
    loop = dict(mb, id="ml", text="```python\nwhile True: pass\n```")
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "g.jsonl"), os.path.join(d, "s.jsonl")
        with open(src, "w") as f:
            for r in (he, bad, body, mb, loop):
                f.write(json.dumps(r) + "\n")
        exec_code.main(src, dst)
        got = {json.loads(l)["id"]: json.loads(l)["score"] for l in open(dst)}
    assert got == {"h": 1.0, "hb": 0.0, "hc": 1.0, "m": 1.0, "ml": 0.0}


def test_zebra():
    gold = {"House 1": {"Name": "Arnold", "Color": "red"}, "House 2": {"Name": "Bob", "Color": "blue"}}
    good = 'reasoning... {"solution": {"House 1": {"Name": "arnold", "Color": "Red"}, "House 2": {"Name": "Bob", "Color": "blue"}}}'
    half = '{"solution": {"House 1": {"Name": "Bob", "Color": "red"}, "House 2": {"Name": "Arnold", "Color": "blue"}}}'
    assert score.zebra_score(good, gold) == dict(pred=None, score=1.0, cell_acc=1.0)
    r = score.zebra_score(half, gold)
    assert r["score"] == 0.0 and r["cell_acc"] == 0.5
    assert score.zebra_score("no json", gold)["score"] == 0.0


def test_ladder_loaders_parse():
    from kvx.evals import data
    root = os.environ.get("KVX_DATA")
    if not root:
        import pytest
        pytest.skip("KVX_DATA not set")
    for b, n in dict(hellaswag=5, piqa=5, arc_c=5, quality=5, zebralogic=3, medqa=5, obqa=5, contract_nli=3,
                     longhealth=3, bamboo_paperqa=3, math500=3).items():
        items = data.load(b, root, n)
        assert len(items) == n, b
        for it in items:
            if it["kind"] == "letter":
                assert it["answer"] in "ABCDEFGHIJ"[:it["meta"]["n_opts"]], (b, it["id"])
            if it["kind"] == "zebra":
                assert all(isinstance(v, dict) and v for v in it["answer"].values())
