"""GPU debug: where does kvx/xkv_adapter (HF override path) diverge from src/xkv's own transfer_prefill?

Tiny pair (Qwen3.5-0.8B -> 2B by default): a small xkv ridge + attention-map calibration, one prompt, then per target
layer the synthesized cache from both paths: attention K (post-RoPE) / V, DeltaNet conv state + recurrent state.
Runs in xkv's image (its transformers pin) and, with --tf 5.24.0, in a newer one.

    cd clone && PYTHONPATH=../src KVX_GPU=L4 modal run tests/modal_xkv_adapter_diff.py::main
"""
import json
import os

import modal

TF = os.environ.get("KVX_TF", "5.20.2")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install("torch==2.8.0", f"transformers=={TF}", "accelerate", "safetensors", "huggingface_hub", "datasets",
                 "numpy", "flash-linear-attention==0.4.1", "fla-core==0.4.1", "kernels")
    .pip_install("https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
                 "flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl")
    .env({"HF_HOME": "/hf", "HF_HUB_CACHE": "/vol/hf/hub", "HF_DATASETS_CACHE": "/vol/hf/datasets",
          **{k: os.environ[k] for k in ("KVX_VOLUME", "KVX_WEIGHTS_VOLUME") if k in os.environ}})
    .add_local_python_source("xkv", "kvx")
)
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"), create_if_missing=True)
weights = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
app = modal.App(os.environ.get("KVX_APP", "kvx-xkv-diff"), image=image)


def run(src_key: str, tgt_key: str, n_seqs: int, seq_len: int, P: int):
    import torch
    import transformers
    if not os.path.exists("/artifacts"):
        os.makedirs("/vol/xkv-diff", exist_ok=True)
        os.symlink("/vol/xkv-diff", "/artifacts")
    from xkv import evals, experiments as E
    from xkv.attnmap import AttnStats
    from xkv.mapper import Stats, transfer_prefill
    from kvx import hybrid, xkv_adapter as XA
    sm, stok = E.get(src_key)
    tm, ttok = E.get(tgt_key)
    seqs = evals.load_heldout_text(ttok, n_seqs, seq_len, seed=0, split="calib", source="fineweb")
    st = Stats(sm, tm, radius=2)
    at = AttnStats(sm, tm, radius=2, embed=True)
    for s in seqs:
        st.add(sm, tm, E.T(s))
        at.add(sm, tm, E.T(s))
    mapper, _ = st.solve(lam=1e-3)
    am = at.solve(lam=1e-3)
    mapper.attn = am[0] if isinstance(am, tuple) else am
    mapper.max_src = max(mapper.max_src, max(mapper.attn.need))
    os.makedirs("/artifacts/maps", exist_ok=True)
    mapper.save("/artifacts/maps/dbg.pt")
    mapper.attn.save("/artifacts/maps/dbg-attn.pt")
    ids = E.T(evals.load_heldout_text(ttok, 1, P + 1, seed=7, split="eval", source="fineweb")[0])
    # xkv path: map ids[:, :P], tail = 1 native token
    xc, _ = transfer_prefill(sm, tm, mapper, ids, tail=1)
    # adapter path on the same HF modules
    S, T = hybrid.Dec(sm.hf), hybrid.Dec(tm.hf)
    xm = XA.load_xkv("/artifacts/maps/dbg.pt", "/artifacts/maps/dbg-attn.pt", device="cuda")
    g = XA.xkv_given(S, T, xm, ids[:, :P])
    hc = hybrid.mapped_cache(T, ids[:, :P], g)
    T.forward(ids[:, P:P + 1], cache=hc)  # xkv's tail=1 already ran this token natively on its cache
    out = {"transformers": transformers.__version__, "layers": {}}
    rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))
    for j, (kind, _) in enumerate(hybrid.layer_taps(T)):
        lay = hc.layers[j]
        if kind == "attn":
            hk = lay.keys.transpose(1, 2)  # HF [B, H, T, D] -> [B, T, H, D] (xkv layout)
            hv = lay.values.transpose(1, 2)
            out["layers"][j] = {"kind": kind, "K_rel": rel(hk[:, :P], xc.k[j][:, :P]), "V_rel": rel(hv[:, :P], xc.v[j][:, :P])}
        else:
            conv = lay.conv_states[0]  # HF [B, conv_dim, kernel]
            S_hf = lay.recurrent_states[0]
            out["layers"][j] = {"kind": kind, "S_rel": rel(S_hf.float(), xc.S[j].float()),
                                "S_rel_T": rel(S_hf.transpose(-1, -2), xc.S[j]),
                                "conv_rel_last3": rel(conv[..., -3:].transpose(1, 2), xc.conv[j])}
    return out


@app.function(gpu=os.environ.get("KVX_GPU", "L4"), volumes={"/vol": vol, "/hf": weights.read_only()}, timeout=1800,
              memory=32768)
def diff(src_key: str = "0.8b", tgt_key: str = "2b", n_seqs: int = 8, seq_len: int = 1024, P: int = 512):
    weights.reload()
    return run(src_key, tgt_key, n_seqs, seq_len, P)


@app.local_entrypoint()
def main(src_key: str = "0.8b", tgt_key: str = "2b", n_seqs: int = 8):
    r = diff.remote(src_key, tgt_key, n_seqs)
    print(json.dumps(r, indent=1))
