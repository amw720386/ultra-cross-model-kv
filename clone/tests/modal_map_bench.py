"""Profile the connector's KV-map stage alone (no target model): per configuration (dense / sparse maps, fp8, TP rank
emulation), the wall time of _map_chunks minus the source prefill, plus the top CUDA kernels from torch.profiler.
KVX_GPU=H100 modal run tests/modal_map_bench.py --cfgs cfg.json
cfg.json: [{"name":..., "maps": "/vol/runs/...pt", "tgt": "Qwen/Qwen3-32B", "mapper": "dense", "fp8": 1, "tp": 2,
            "lens": "32768,65536", "yarn": 4}]"""
import json
import os

import modal

GPU = os.environ.get("KVX_GPU", "H100")
from modal_vllm import image, vol, hf_shared

app = modal.App(os.environ.get("KVX_APP", "kvx-speed-mapbench"), image=image)


@app.function(gpu=GPU, timeout=1800, volumes={"/vol": vol, "/hf": hf_shared.read_only()})
def bench(cfgs: list):
    try:
        return _bench(cfgs)
    except BaseException:
        import traceback
        return {"error": traceback.format_exc()[-4000:]}


def _bench(cfgs):
    import time
    import torch
    from torch.profiler import ProfilerActivity, profile
    from kvx.vllm_connector import CrossModelConnector
    torch.set_grad_enabled(False)
    dev = torch.device("cuda:0")
    out = []
    for cfg in cfgs:
        c = CrossModelConnector.__new__(CrossModelConnector)
        c.src_name, c.tgt_name, c.maps_path = cfg.get("src", "Qwen/Qwen3-1.7B"), cfg["tgt"], cfg["maps"]
        c.restore, c.src_device, c.fp8, c.sink_path, c.chunk = cfg.get("restore", "rms"), None, bool(cfg.get("fp8", 1)), None, 20
        c.max_len = max(int(x) for x in cfg["lens"].split(",")) + 64
        c.yarn, c.orig_ctx, c.fp8_source, c.src_sdpa = float(cfg.get("yarn", 0)), 32768, False, "cudnn"
        c.tp_override, c.src_tp, c.mapper = (0, int(cfg.get("tp", 1))), False, cfg.get("mapper", "dense")
        c.group = int(cfg.get("group", 4))
        c.tok_chunk, c.block, c.mem_fraction = int(cfg.get("tok_chunk", 65536)), 16, 0
        c.src_sdpa, c.src_fp8_attn, c.src_chunk = cfg.get("src_sdpa", "cudnn"), bool(cfg.get("src_fp8_attn", 0)), int(cfg.get("src_chunk", 0))
        c._init_worker(dev)
        c.fused_build = bool(cfg.get("fused_build", 1))
        res = dict(name=cfg["name"], rows=[])
        if cfg.get("verify"):  # fused Triton build vs the torch build+quant8 on real source KV (the e4m3 cast on GPU)
            ids4 = torch.randint(1000, 30000, (4096,)).tolist()
            outs = []
            for fb in (True, False):
                c.fused_build = fb
                outs.append({l: (K.float().clone(), V.float().clone()) for l, K, V in c._map_layers(ids4)})
            rel = lambda a, b: ((a - b).norm() / b.norm()).item()
            res["verify_rel_k"] = max(rel(outs[0][l][0], outs[1][l][0]) for l in outs[0])
            res["verify_rel_v"] = max(rel(outs[0][l][1], outs[1][l][1]) for l in outs[0])
            print("verify", res["verify_rel_k"], res["verify_rel_v"], flush=True)
            c.fused_build = bool(cfg.get("fused_build", 1))
        for n in [int(x) for x in cfg["lens"].split(",")]:
            ids = torch.randint(1000, 30000, (n,)).tolist()
            x = torch.tensor([ids], device=dev)

            def sync():
                torch.cuda.synchronize()
                return time.perf_counter()
            ts = []
            for rep in range(3):
                t0 = sync()
                o = c.S(input_ids=x, use_cache=True, logits_to_keep=1)
                t1 = sync()
                ts.append(t1 - t0)
            src = min(ts[1:])
            S_real = c.S
            c.S = lambda **kw: o  # map stage alone: the source forward is replayed from its cached output
            tm = []
            for rep in range(3):
                t1 = sync()
                for _ in c._map_chunks(ids):
                    pass
                tm.append(sync() - t1)
            tot = src + min(tm[1:])
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in c._map_chunks(ids):
                    pass
                torch.cuda.synchronize()
            c.S = S_real
            del o
            ka = prof.key_averages()
            top = sorted(ka, key=lambda e: -e.device_time_total)[:14]
            res["rows"].append(dict(len=n, src_s=round(src, 4), src_plus_map_s=round(tot, 4), map_s=round(tot - src, 4),
                                    peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1),
                                    kernels=[(e.key[:80], round(e.device_time_total / 1e3, 2), e.count) for e in top]))
            print(json.dumps(res["rows"][-1])[:1500], flush=True)
        out.append(res)
        del c
        torch.cuda.empty_cache()
    return out


@app.local_entrypoint()
def main(cfgs: str):
    r = bench.remote(json.load(open(cfgs)))
    os.makedirs("out/speed", exist_ok=True)
    json.dump(r, open(f"out/speed/mapbench_{os.path.basename(cfgs)}", "w"), indent=1)
    print(json.dumps(r, indent=1)[:12000])
