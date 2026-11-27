"""CPU Modal check (no GPU): can modal_fit_eval's image load a quantized target, and does it stay compressed?

Loads `tgt` exactly like modal_fit_eval (kvx.models.load_lm, bf16 dtype, sdpa) on CPU, reports the in-memory
parameter bytes (FP8 kept vs decompressed to bf16), the k_proj module type, and runs a 16-token forward to check
the cache K/V are finite. Use before a GPU fit on an FP8 checkpoint.

    cd clone && KVX_FIT_EXTRA_PIP="compressed-tensors==0.11.0" PYTHONPATH=. modal run tools/fit_target_check.py::main \
        --tgt RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic
"""
import json
import os

import modal

from modal_fit_eval import hf_shared, image

app = modal.App(os.environ.get("KVX_APP", "kvx-fit-target-check"), image=image.add_local_python_source("modal_fit_eval"))


@app.function(cpu=16.0, memory=100 * 1024, timeout=3000, volumes={"/hf": hf_shared.read_only()})
def check(tgt: str):
    import time
    import torch
    from kvx.models import load_lm
    t0 = time.time()
    m = load_lm(tgt).eval()
    by_dtype = {}
    for p_ in list(m.parameters()) + list(m.buffers()):
        by_dtype[str(p_.dtype)] = by_dtype.get(str(p_.dtype), 0) + p_.numel() * p_.element_size()
    kp = m.model.layers[0].self_attn.k_proj
    ids = torch.arange(100, 116)[None]
    with torch.no_grad():
        out = m(input_ids=ids, use_cache=True)
    pkv = out.past_key_values
    k0 = pkv.layers[0].keys if hasattr(pkv, "layers") else pkv[0][0]
    return dict(load_s=round(time.time() - t0), bytes_by_dtype={k: round(v / 1e9, 2) for k, v in by_dtype.items()},
                k_proj_type=type(kp).__name__, k_proj_weight_dtype=str(getattr(kp, "weight", torch.empty(0)).dtype),
                k_shape=list(k0.shape), k_finite=bool(torch.isfinite(k0).all()), k_absmean=float(k0.float().abs().mean()),
                logits_finite=bool(torch.isfinite(out.logits).all()))


@app.local_entrypoint()
def main(tgt: str = "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"):
    print(json.dumps(check.remote(tgt), indent=1))
