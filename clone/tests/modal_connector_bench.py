"""Time the connector's stages at long prompts WITHOUT the target model: source prefill, features, mapper GEMMs +
epilogue (rows=1 vs rows=32), and the paged KV write into a fake FlashAttention cache. One small GPU, a few minutes.
KVX_GPU=L40S modal run tests/modal_connector_bench.py --lens 16384,32512"""
import os
import modal

GPU = os.environ.get("KVX_GPU", "L40S")
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
         .pip_install("vllm==0.34.0", "hf_transfer")
         .env({"HF_HUB_CACHE": "/hf/hub", "HF_HUB_OFFLINE": "1"})
         .add_local_python_source("kvx"))
vol = modal.Volume.from_name("kvx")
hf_shared = modal.Volume.from_name("gap-push-models-xl-hf")
app = modal.App("kvx-connbench", image=image)


@app.function(gpu=GPU, timeout=1200, volumes={"/vol": vol, "/hf": hf_shared.read_only()})
def bench(run: str, tmpl: str, src: str, tgt: str, lens: str, fp8: int, yarn: float = 0.0, maxlen: int = 33024,
          restore: str = "none", extras: int = 1):
    import glob, time, torch
    torch.set_grad_enabled(False)  # the direct source forward below kept autograd activations (78 GB at 16K)
    from kvx import triton_kernels as tk
    from kvx.vllm_connector import CrossModelConnector, write_kv_fa
    c = CrossModelConnector.__new__(CrossModelConnector)
    c.src_name, c.restore, c.src_device, c.fp8 = src, restore, None, bool(fp8)
    c.yarn, c.orig_ctx, c.fp8_source, c.src_sdpa = yarn, 32768, False, "cudnn"
    c.maps_path = glob.glob(f"/vol/runs/{run}/maps_k*.pt")[0]
    c.sink_path = f"/vol/runs/{tmpl}/sink_template.pt" if tmpl else None
    c.tgt_name, c.max_len, c.chunk = tgt, maxlen, 20
    dev = torch.device("cuda:0")
    c._init_worker(dev)
    H, D = c.H, c.D
    Lt = c.bk.numel() // (H * D)
    out = dict(gpu=torch.cuda.get_device_name(0), rows=[], src_attn=getattr(c.S.config, "_attn_implementation", None))
    torch.cuda.reset_peak_memory_stats()
    orig = tk.restore_rope_

    def sync():
        torch.cuda.synchronize()
        return time.perf_counter()

    for n in [int(x) for x in lens.split(",")]:
        ids = torch.randint(1000, 30000, (n,)).tolist()
        bs = 64
        nb = n // bs + 1
        kv = torch.zeros(nb, bs, H, 2 * D, device=dev, dtype=torch.bfloat16).permute(0, 2, 1, 3)  # vLLM FA view
        slots = torch.arange(n, device=dev)
        bi, off = slots // bs, slots % bs
        row = dict(len=n)
        for rows in ((1, 32) if extras else (32,)):
            tk.restore_rope_ = (lambda Y, kw, cs, sn, _r=rows: orig(Y, kw, cs, sn, rows=_r))
            for rep in range(3):
                t0 = sync()
                x = torch.tensor([ids], device=dev)
                from torch.nn.attention import SDPBackend, sdpa_kernel
                with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                                  SDPBackend.MATH], set_priority=True):
                    o = c.S(input_ids=x, use_cache=True, logits_to_keep=1)
                t1 = sync()
                del o
                tw = 0.0
                tm0 = sync()
                for l, K, V in c._map_layers(ids):  # includes a second source prefill; subtract below
                    a = sync()
                    write_kv_fa(kv, bi, off, K, V, bs)
                    tw += sync() - a
                tm1 = sync()
            row[f"src_prefill_s"] = round(t1 - t0, 4)
            row["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
            row[f"map_total_rows{rows}_s"] = round(tm1 - tm0 - tw, 4)
            row[f"map_minus_src_rows{rows}_s"] = round(tm1 - tm0 - tw - (t1 - t0), 4)
            row["write_s"] = round(tw, 4)
        tk.restore_rope_ = orig
        if not extras:
            out["rows"].append(row); print(row, flush=True); continue
        # epilogue alone on a [n, Lt, H, D] tensor
        Y = torch.randn(n, Lt, H, D, device=dev, dtype=torch.bfloat16)
        ct, st = c.ct[:n].contiguous(), c.st[:n].contiguous()
        for rows in (1, 32):
            orig(Y, None, ct, st, rows=rows); a = sync()
            for _ in range(3):
                orig(Y, None, ct, st, rows=rows)
            row[f"epilogue_only_rows{rows}_s"] = round((sync() - a) / 3, 4)
        out["rows"].append(row)
        print(row, flush=True)
    if not extras:
        return out
    # ---- fp8 source: prefill time per length and KV error at 4096 tokens vs bf16
    from kvx import core
    from kvx.fp8 import quantize_linears
    ids4 = torch.tensor([torch.randint(1000, 30000, (4096,)).tolist()], device=dev)
    cs, ss = core.rope_cos_sin(c.S, 4096, dev)
    K0, V0, _ = core.capture_kv(c.S, ids4, cs, ss)
    nq = quantize_linears(c.S)
    K1, V1, _ = core.capture_kv(c.S, ids4, cs, ss)
    rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
    out["fp8_source"] = dict(linears=nq, k_rel=round(rel(K1, K0), 4), v_rel=round(rel(V1, V0), 4))
    del K0, V0, K1, V1
    for n in [int(x) for x in lens.split(",")]:
        x = torch.tensor([torch.randint(1000, 30000, (n,)).tolist()], device=dev)
        ts = []
        for rep in range(3):
            t0 = sync(); o = c.S(input_ids=x, use_cache=True, logits_to_keep=1); ts.append(sync() - t0); del o
        out["fp8_source"][f"src_prefill_{n}_s"] = round(min(ts), 4)
    print(out["fp8_source"], flush=True)
    return out


@app.local_entrypoint()
def main(run: str = "q25-3b-72b", tmpl: str = "q25-3b-72b-tmpl", src: str = "Qwen/Qwen2.5-3B",
         tgt: str = "Qwen/Qwen2.5-72B", lens: str = "16384,32512", fp8: int = 1, yarn: float = 0.0, maxlen: int = 33024,
         restore: str = "none", extras: int = 1):
    import json
    print(json.dumps(bench.remote(run, tmpl, src, tgt, lens, fp8, yarn, maxlen, restore, extras), indent=1))
