"""CPU end-to-end smoke of modal_fit_hybrid.fit_impl on tiny random models (no Modal, no downloads of weights).

Covers the full job flow (calibration passes, solve, maps file, sink template, KL eval, needle eval) for a pair before
spending GPU money on it. Default: tiny GLM-4.7-Flash-style source -> GLM-5.3-style (MLA + DSA) target with the
token-embedding feature and device_map="auto".

    cd clone && PYTHONPATH=<transformers>=5>:. python tests/local_fit_smoke.py [--pair glm]
"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class FakeTok:
    """Minimal tokenizer: ids are byte values mod the vocab (enough for the needle prompts)."""
    bos_token_id, eos_token_id, name_or_path = None, 0, "tiny/glm"

    def __init__(self, V):
        self.V = V

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [3 + (b % (self.V - 3)) for b in text.encode()]}

    def decode(self, ids):
        return "".join(chr(97 + (i % 26)) for i in ids)


class _Stub:  # stands in for the modal client (only object construction happens at import)
    def __getattr__(self, name):
        return _Stub()

    def __call__(self, *a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        return _Stub()


def main():
    sys.modules.setdefault("modal", _Stub())
    import modal_fit_hybrid as M
    from test_hybrid_cpu import _tiny_glm
    S, T = _tiny_glm("lite", 1, layers=3), _tiny_glm("dsa", 2, layers=4)
    root = tempfile.mkdtemp()
    for name, d in (("src", S), ("tgt", T)):
        d.model.save_pretrained(f"{root}/{name}")
    M.snapshot = lambda n: f"{root}/{'src' if n == 'tiny/src' else 'tgt'}"
    V = 101
    rng = torch.Generator().manual_seed(0)
    M.cached = lambda tok, tag, build: torch.randint(3, V, (int(tag.split("_")[1].split("x")[0]), int(tag.split("x")[1].split("_")[0])), generator=rng).tolist()
    M.hf_shared = type("V", (), {"reload": lambda self: None})()
    M._commit = lambda: None
    import transformers
    transformers.AutoTokenizer.from_pretrained = staticmethod(lambda p, **k: FakeTok(V))
    os.makedirs("/tmp/kvx-smoke-vol/runs", exist_ok=True)
    orig_makedirs = os.makedirs

    def mk(p, *a, **k):  # keep /vol writes inside a temp dir
        return orig_makedirs(p.replace("/vol/", "/tmp/kvx-smoke-vol/"), *a, **k)
    M.os.makedirs = mk
    orig_save = torch.save
    torch.save = lambda obj, f, *a, **k: orig_save(obj, f.replace("/vol/", "/tmp/kvx-smoke-vol/") if isinstance(f, str) else f, *a, **k)
    import builtins
    orig_open = builtins.open
    builtins.open = lambda f, *a, **k: orig_open(f.replace("/vol/", "/tmp/kvx-smoke-vol/") if isinstance(f, str) else f, *a, **k)
    res = M.fit_impl(src="tiny/src", tgt="tiny/tgt", run_id="smoke-glm", n_calib=8, L=64, w=0, lam_rels="1e-3",
                     n_eval=12, P=48, sinks="2", tails="8", bs=2, oracle=0, stride=1, emb_feats=1, niah=2, niah_len=400,
                     tgt_device_map="")
    print({k: res[k] for k in ("passes", "niah") if k in res})
    for k, v in sorted(res["eval"].items()):
        print(k, {a: round(b, 4) for a, b in v.items()})
    assert all(v["kl"] == v["kl"] for v in res["eval"].values()), "NaN KL"
    print("SMOKE OK")


if __name__ == "__main__":
    main()
