"""CPU: per-layer / per-KV-head R^2 and bias statistics of a saved map file. modal run tests/modal_maps_inspect_cpu.py --run q25-3b-72b --k 16"""
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.8.0", "numpy")
vol = modal.Volume.from_name("kvx")
app = modal.App("kvx-mapsinspect-cpu", image=image)


@app.function(cpu=2.0, memory=16384, timeout=900, volumes={"/vol": vol})
def inspect(run: str, k: int):
    import torch
    d = torch.load(f"/vol/runs/{run}/maps_k{k}.pt", map_location="cpu")
    out = {}
    for kind in ("k", "v"):
        r2 = torch.stack([m["r2"] for m in d[kind]])  # [Lt, H]
        mu = torch.stack([m["mu"] for m in d[kind]])  # [Lt, H*D]
        H, D = d[kind][0]["H"], d[kind][0]["D"]
        vn = torch.stack([m["vnat"] for m in d[kind]])  # [Lt, H] native centered variance per head
        mun = mu.view(len(mu), H, D).norm(dim=-1)
        chmax = mu.view(len(mu), H, D).abs().amax(-1)
        out[kind] = dict(
            r2_layer_mean=[round(x, 3) for x in r2.mean(1).tolist()],
            r2_head_min=[round(x, 3) for x in r2.min(1).values.tolist()],
            worst_heads=[(int(i // H), int(i % H), round(float(r2.view(-1)[i]), 3)) for i in r2.view(-1).argsort()[:12]],
            heads_below_0_2=int((r2 < 0.2).sum()), n_heads=r2.numel(),
            mean_norm_over_std_layer=[round(x, 2) for x in (mun / vn.clamp(min=1e-12).sqrt()).mean(1).tolist()],
            mean_absmax_channel_layer=[round(x, 2) for x in chmax.max(1).values.tolist()],
            r2_vs_layer_corr_with_varnat=float(torch.corrcoef(torch.stack([r2.view(-1), vn.view(-1).log()]))[0, 1]))
    return out


@app.local_entrypoint()
def main(run: str = "q25-3b-72b", k: int = 16):
    import json
    print(json.dumps(inspect.remote(run, k)))
