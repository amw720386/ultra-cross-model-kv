"""CPU, vLLM serving image: the connector's mapped K/V (dense bf16 path) vs the fit_eval reference path, on real maps.
No target model needed. modal run tests/modal_connector_vs_ref_cpu.py --run q25-3b-72b --tmpl q25-3b-72b-tmpl"""
import modal

image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
         .pip_install("vllm==0.34.0", "hf_transfer")
         .env({"TRITON_INTERPRET": "1", "HF_HUB_CACHE": "/hf/hub", "HF_HUB_OFFLINE": "1"})
         .add_local_python_source("kvx"))
vol = modal.Volume.from_name("kvx")
hf_shared = modal.Volume.from_name("gap-push-models-xl-hf")
app = modal.App("kvx-connref-cpu", image=image)


@app.function(cpu=8.0, memory=49152, timeout=3000, volumes={"/vol": vol, "/hf": hf_shared.read_only()})
def check(run: str, tmpl: str, src: str, tgt: str, n: int):
    import glob, json, torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from kvx import core
    from kvx.vllm_connector import CrossModelConnector, target_cos_sin
    maps_path = glob.glob(f"/vol/runs/{run}/maps_k*.pt")[0]
    ids = list(torch.load(sorted(glob.glob("/vol/data/eval_64x1024_bos0_*.pt"))[0])[0][:n])
    out = {}
    # ---- connector path (dense bf16, CPU)
    c = CrossModelConnector.__new__(CrossModelConnector)
    c.src_name, c.maps_path, c.restore, c.src_device, c.fp8 = src, maps_path, "none", None, False
    c.sink_path = f"/vol/runs/{tmpl}/sink_template.pt" if tmpl else None
    c.tgt_name, c.max_len, c.chunk = tgt, 2048, 20
    c._init_worker(torch.device("cpu"))
    got = {l: (K.float(), V.float()) for l, K, V in c._map_layers(ids)}
    # ---- reference path (fit_eval): HF source capture, apply_maps, no restore, template, RoPE with target cos/sin
    S = c.S
    cs, ss = core.rope_cos_sin(S, n, "cpu")
    x = torch.tensor([ids])
    Ks, Vs, _ = core.capture_kv(S, x, cs, ss)
    mp = torch.load(maps_path, map_location="cpu")
    Km = core.apply_maps(mp["k"], Ks.float())[:, 0]  # [Lt, H, T, D]
    Vm = core.apply_maps(mp["v"], Vs.float())[:, 0]
    if tmpl:
        st = torch.load(f"/vol/runs/{tmpl}/sink_template.pt")
        A = st["K"].shape[2]
        Km[:, :, :A], Vm[:, :, :A] = st["K"], st["V"]
    tcfg = AutoConfig.from_pretrained(tgt)
    ct, stt = target_cos_sin(tcfg, n, "cpu")
    Kr = core.rope(Km, ct, stt)
    rows = []
    for l in range(Km.shape[0]):
        Kc, Vc = got[l]  # [T, H, D]
        Kref, Vref = Kr[l].permute(1, 0, 2), Vm[l].permute(1, 0, 2)
        rel = lambda a, b: ((a - b).norm() / b.norm()).item()
        rows.append(dict(l=l, k_rel=round(rel(Kc, Kref), 4), v_rel=round(rel(Vc, Vref), 4),
                         k0_rel=round(rel(Kc[:4], Kref[:4]), 4), v0_rel=round(rel(Vc[:4], Vref[:4]), 4)))
    out["worst"] = sorted(rows, key=lambda r: -max(r["k_rel"], r["v_rel"]))[:6]
    out["mean_k_rel"] = sum(r["k_rel"] for r in rows) / len(rows)
    out["mean_v_rel"] = sum(r["v_rel"] for r in rows) / len(rows)
    out["first4_mean"] = (sum(r["k0_rel"] for r in rows) / len(rows), sum(r["v0_rel"] for r in rows) / len(rows))
    # also: does the source capture agree with transformers' own cache layout? (cos/sin sanity)
    out["src_cos_dev"] = (cs - c.cs[:n].float()).abs().max().item()
    out["tgt_cos_dev"] = (ct - c.ct[:n].float()).abs().max().item()
    return out


@app.local_entrypoint()
def main(run: str = "q25-3b-72b", tmpl: str = "q25-3b-72b-tmpl", src: str = "Qwen/Qwen2.5-3B",
         tgt: str = "Qwen/Qwen2.5-72B", n: int = 704):
    import json
    print(json.dumps(check.remote(run, tmpl, src, tgt, n), indent=1))
