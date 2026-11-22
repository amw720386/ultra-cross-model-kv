"""Full-map KL distillation (kvx/distill_full.py) of a saved ridge maps file, one GPU boot: lr sweep by held-out KL,
then the full run at the best lr. Output: /vol/runs/<run_id>/maps_k<k>.pt (same format, drops into the connector).
KVXQ_GPU=A100-80GB modal run modal_distill.py --maps /vol/runs/pairs-q17-q8/maps_k24.pt --run-id distill-q17-q8-a
"""
import json
import os
import time

import modal

GPU = os.environ.get("KVXQ_GPU", "A100-80GB").split(",")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.9.0", "transformers==4.57.1", "accelerate", "huggingface_hub")
         .env({"HF_HUB_CACHE": "/hf/hub", "HF_HUB_OFFLINE": "1"})
         .add_local_python_source("kvx"))
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"))
hf_shared = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
app = modal.App(os.environ.get("KVX_APP", "kvx-quality-distill"), image=image)


@app.function(gpu=GPU, volumes={"/vol": vol, "/hf": hf_shared.read_only()}, timeout=3300)
def run(maps: str, run_id: str, src: str, tgt: str, docs: str, lrs: str, sweep_steps: int, steps: int,
        bs: int, accum: int, n_val: int, budget_s: int):
    import torch
    from kvx import core, distill_full
    from kvx.models import load_lm
    t_start = time.time()
    hf_shared.reload()
    vol.reload()
    torch.backends.cuda.matmul.allow_tf32 = True
    S, T = load_lm(src).cuda().eval(), load_lm(tgt).cuda().eval()
    for p in list(S.parameters()) + list(T.parameters()):
        p.requires_grad_(False)
    d = torch.load(maps, map_location="cpu")
    mk = [{k: (v.cuda() if torch.is_tensor(v) else v) for k, v in m.items()} for m in d["k"]]
    mv = [{k: (v.cuda() if torch.is_tensor(v) else v) for k, v in m.items()} for m in d["v"]]
    all_docs = torch.load(docs).cuda()
    val, train = all_docs[-n_val:], all_docs[:-n_val]
    L = all_docs.shape[1]
    P = int(docs.rsplit("x", 1)[1].split("+")[0])
    cs, ss = core.rope_cos_sin(S, L, "cuda")
    ct, st = core.rope_cos_sin(T, L, "cuda")
    kw = torch.stack([l.self_attn.k_norm.weight for l in T.model.layers]).float()
    log = lambda s: print(s, flush=True)
    res = dict(maps=maps, src=src, tgt=tgt, docs=docs, P=P, L=L, n_train=len(train), n_val=n_val, sweep={})
    deadline = t_start + budget_s
    best = None
    for lr in [float(x) for x in lrs.split(",")]:
        t0 = time.time()
        _, h = distill_full.train(S, T, mk, mv, train, val, P, cs, ss, ct, st, kw, lr=lr, steps=sweep_steps, bs=bs,
                                  accum=accum, log=log, deadline=deadline - 900)
        res["sweep"][lr] = dict(val=h["val"], s=round(time.time() - t0), diverged=h.get("diverged"))
        if best is None or h["val"][-1][1] < res["sweep"][best]["val"][-1][1]:
            best = lr
        torch.cuda.empty_cache()
    res["best_lr"] = best
    tm, h = distill_full.train(S, T, mk, mv, train, val, P, cs, ss, ct, st, kw, lr=best, steps=steps, bs=bs,
                               accum=accum, eval_every=max(1, steps // 5), log=log, deadline=deadline)
    res["final"] = dict(val=h["val"], stopped=h.get("stopped"), diverged=h.get("diverged"))
    ek, ev = tm.export()
    os.makedirs(f"/vol/runs/{run_id}", exist_ok=True)
    out = dict(d, k=ek, v=ev, distill=dict(lr=best, steps=len(h["train"]), docs=docs, from_maps=maps))
    torch.save(out, f"/vol/runs/{run_id}/maps_k{d.get('k_layers', 'x')}.pt")
    json.dump(res, open(f"/vol/runs/{run_id}/result.json", "w"), indent=1, default=str)
    vol.commit()
    res["wall_s"] = round(time.time() - t_start)
    return res


@app.local_entrypoint()
def main(maps: str, run_id: str, src: str = "Qwen/Qwen3-1.7B", tgt: str = "Qwen/Qwen3-8B",
         docs: str = "/vol/data/distill/pg19chat_800x3072+512.pt", lrs: str = "1e-6,1e-5,1e-4", sweep_steps: int = 60,
         steps: int = 600, bs: int = 1, accum: int = 8, n_val: int = 32, budget_s: int = 3000):
    r = run.remote(maps, run_id, src, tgt, docs, lrs, sweep_steps, steps, bs, accum, n_val, budget_s)
    print(json.dumps(r, indent=1, default=str)[:6000])
