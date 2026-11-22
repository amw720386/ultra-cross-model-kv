"""CPU: how much does fp8 (e4m3, per-output-channel) quantization of the SOURCE model's linear weights change its KV?
Compare with the mapper's own error (sqrt(1 - R^2) ~ 0.5-0.7 relative). modal run tests/modal_fp8_source_cpu.py"""
import os
import modal

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "transformers==4.56.2", "accelerate", "numpy")
         .add_local_python_source("kvx"))
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"))
hf_shared = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
app = modal.App(os.environ.get("KVX_APP", "kvx-fp8src-cpu"), image=image)


@app.function(cpu=8.0, memory=32768, timeout=2400, volumes={"/vol": vol, "/hf": hf_shared.read_only()})
def check(src: str, n: int, docs: int):
    import glob, torch
    from kvx import core
    from kvx.models import load_lm
    torch.set_grad_enabled(False)
    ev = torch.load(sorted(glob.glob("/vol/data/eval_64x1024_bos0_*.pt"))[0])
    ids = torch.tensor([list(x[:n]) for x in ev[:docs]])
    m = load_lm(src, dtype=torch.float32).eval()
    cs, ss = core.rope_cos_sin(m, n, "cpu")
    K0, V0, _ = core.capture_kv(m, ids, cs, ss)
    nq = 0
    for name, mod in m.named_modules():
        if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
            W = mod.weight.data
            s = W.abs().amax(1, keepdim=True).clamp(min=1e-12) / 448.0
            mod.weight.data = (W / s).to(torch.float8_e4m3fn).float() * s
            nq += 1
    K1, V1, _ = core.capture_kv(m, ids, cs, ss)
    rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
    per = [(l, round(rel(K1[l], K0[l]), 4), round(rel(V1[l], V0[l]), 4)) for l in range(K0.shape[0])]
    return dict(src=src, linear_layers_quantized=nq, tokens=n * docs,
                k_rel_mean=sum(p[1] for p in per) / len(per), v_rel_mean=sum(p[2] for p in per) / len(per),
                worst=sorted(per, key=lambda p: -max(p[1], p[2]))[:5])


@app.local_entrypoint()
def main(src: str = "Qwen/Qwen2.5-3B", n: int = 512, docs: int = 4):
    import json
    print(json.dumps(check.remote(src, n, docs), indent=1))
