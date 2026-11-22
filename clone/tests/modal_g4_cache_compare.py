"""GPU check: a serving connector's written cache vs the HF mapped cache for the same prompt (Gemma-4 E4B -> 31B).

Input dump (written by the serving side): {token_ids: [T], kv: {target layer: (K post-RoPE [T', H, D], V [T', H, D])}}
(sliding layers may hold only the last window). The HF side builds the same mapped cache: split-K/V maps without
recolor, rms restore (k_norm weight; V unit RMS), template sink rows at 0..A-1, every other position mapped.

    cd clone && KVX_GPU=H200 modal run tests/modal_g4_cache_compare.py::main --dump runs/kvx-speed-g4-dump/e4b_2048_tp1.pt
"""
import json
import os

import modal

from modal_fit_hybrid import hf_shared, image, snapshot, vol  # same image/volumes as the fits

app = modal.App(os.environ.get("KVX_APP", "kvx-g4-compare"), image=image.add_local_python_source("modal_fit_hybrid"))


_G = os.environ.get("KVX_GPU", "H200")


@app.function(gpu=_G.split(",") if "," in _G else _G, volumes={"/vol": vol, "/hf": hf_shared.read_only()},
              timeout=1800, memory=131072)
def compare(dump: str, maps_run: str, src: str, tgt: str):
    import torch
    from kvx import hybrid
    hf_shared.reload()
    vol.reload()
    d = torch.load(f"/vol/{dump}", map_location="cpu")
    ids = torch.tensor([d["token_ids"]], device="cuda")
    T = ids.shape[1]
    S = hybrid.load(src, path=snapshot(src))
    Tm = hybrid.load(tgt, path=snapshot(tgt))
    S.model.to("cuda").requires_grad_(False)
    Tm.model.to("cuda").requires_grad_(False)
    u = torch.load(f"/vol/runs/{maps_run}/hybrid_units.pt", map_location="cuda")
    us, ut = [tuple(x) for x in u["units_s"]], [tuple(x) for x in u["units_t"]]
    maps = {int(j): {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in m.items() if k != "rc"}
            for j, m in u["maps"].items()}
    for m in maps.values():
        m["W"] = m["W"].to(torch.bfloat16)
    tmpl = torch.load(f"/vol/runs/{maps_run}/sink_template.pt", map_location="cuda")
    A = next(iter(next(iter(tmpl.values())).values())).shape[0]
    feats = hybrid.source_features(S, ids)
    g = hybrid.restore_rms(hybrid.apply_units(maps, feats, 1, T, Tm, us, ut, src=S, out_dtype=torch.bfloat16), Tm)
    g = {t: {k: torch.cat([tmpl[t][k][:A].to(v)[None], v[:, A:]], 1) for k, v in dd.items()} for t, dd in g.items()}
    cache = hybrid.mapped_cache(Tm, ids, g)
    out = {}
    rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))
    for t, (k_s, v_s) in d["kv"].items():
        t = int(t)
        lay = cache.layers[t]
        hk, hv = lay.keys[0].transpose(0, 1), lay.values[0].transpose(0, 1)  # [T'', H, D]
        n = min(hk.shape[0], k_s.shape[0])
        k_s, v_s = k_s[-n:].to(hk.device), v_s[-n:].to(hv.device)
        out[t] = {"rows": n, "K_rel": rel(hk[-n:], k_s), "V_rel": rel(hv[-n:], v_s),
                  "K_rel_first4": rel(hk[:4], k_s[:4]) if n == T else None}
    worst = sorted(out.items(), key=lambda kv: -kv[1]["K_rel"])[:5]
    return {"T": T, "layers": out, "worst_K": worst}


@app.local_entrypoint()
def main(dump: str, maps_run: str = "pairs-g4e4bB-g4-31bB", src: str = "google/gemma-4-E4B",
         tgt: str = "google/gemma-4-31B"):
    r = compare.remote(dump, maps_run, src, tgt)
    print(json.dumps({"T": r["T"], "worst_K": r["worst_K"]}, indent=1))
    os.makedirs("out", exist_ok=True)
    json.dump(r, open("out/g4_cache_compare.json", "w"), indent=1)
