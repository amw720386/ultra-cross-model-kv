"""vLLM serving test: native vs CrossModelConnector TTFT and output agreement.

KVX_GPU=H200,B200 modal run modal_vllm.py --mode native|mapped --lens 4096,8192,16384,32768
"""
import json, os, time
try:
    import modal
except ImportError:
    from modal_fit_eval import modal  # no-op stand-in on rented boxes

GPU = os.environ.get("KVX_GPU", "H200,B200")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.34.0", "hf_transfer")
    # flash-attn backend: predictable paged layout [2, blocks, block_size, H, D] for the connector
    .env({"HF_HOME": "/vol/hf", "HF_HUB_CACHE": "/hf/hub", "HF_HUB_OFFLINE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_ATTENTION_BACKEND_UNUSED": "set via attention_backend"})
    # resource names from the launching shell (kvx-env), so the container's module import resolves the same objects
    .env({k: v for k, v in os.environ.items() if k.startswith("KVX_") and k.endswith(("_VOLUME", "_SECRET"))})
    .add_local_python_source("kvx", "modal_vllm")
)
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"), create_if_missing=True)
hf_shared = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))  # shared weights volume (KVX_WEIGHTS_VOLUME), read-only
app = modal.App(os.environ.get("KVX_APP", "kvx-vllm"), image=image)


def serve_impl(mode: str, src: str, tgt: str, maps: str, lens: str, tail: int, reps: int = 3, fp8: int = -1, tag: str = "",
               util: float = 0.72, restore: str = "rms", sink: str = "", backend: str = "FLASH_ATTN",
               yarn: float = 0.0, orig_ctx: int = 32768, quality_long: int = 0, maxlen: int = 0, src_sdpa: str = "cudnn", tp: int = 1, src_tp: int = 1, lens_quality: int = 1,
               long_from: int = 0, reps_long: int = 1, extra: dict = None, long_n: int = 4, needle_n: int = 0,
               max_num_seqs: int = 0):
    try:
        import torch
        from vllm import LLM, SamplingParams
        from vllm.config import KVTransferConfig
        kw = dict(model=tgt, dtype="bfloat16", max_model_len=maxlen or (max(int(x) for x in lens.split(",")) + 256),
                  gpu_memory_utilization=util, enable_prefix_caching=False, enforce_eager=False,
                  **({"attention_backend": backend} if backend else {}), tensor_parallel_size=tp)
        if yarn > 0:  # YaRN on the target via HF config overrides (rope_parameters in transformers 5)
            from transformers import AutoConfig
            from kvx.models import apply_yarn
            tc = apply_yarn(AutoConfig.from_pretrained(tgt), yarn, orig_ctx)
            kw["hf_overrides"] = {"rope_parameters": tc.rope_parameters, "rope_scaling": tc.rope_scaling,
                                  "max_position_embeddings": tc.max_position_embeddings}
        if max_num_seqs:  # hybrid (GatedDeltaNet) targets: vLLM refuses the default 256 state slots on 80 GB
            kw["max_num_seqs"] = max_num_seqs
        if mode == "mapped":
            extra = dict(extra or {})
            # extra["connector"] = "hybrid": the GatedDeltaNet-target connector (kvx/vllm_hybrid_connector.py)
            ckind = extra.pop("connector", "")
            cname, cmod = {"hybrid": ("HybridCrossModelConnector", "kvx.vllm_hybrid_connector"),
                           "gemma": ("GemmaCrossModelConnector", "kvx.vllm_gemma_connector")}.get(
                               ckind, ("CrossModelConnector", "kvx.vllm_connector"))
            kw["kv_transfer_config"] = KVTransferConfig(
                kv_connector=cname, kv_connector_module_path=cmod, kv_role="kv_both",
                kv_connector_extra_config={"src": src, "maps": maps, "tail": tail, "min_len": 512, "restore": restore,
                                           **({"sink_template": sink} if sink else {}),
                                           **({"yarn": yarn, "orig_ctx": orig_ctx} if yarn > 0 else {}),
                                           "src_sdpa": src_sdpa, "src_tp": src_tp, **(extra or {}),
                                           **({"src_device": os.environ["KVX_SRC_DEVICE"]} if os.environ.get("KVX_SRC_DEVICE") else {}),
                                           **({"fp8": True} if (fp8 == 1 or (fp8 < 0 and os.environ.get("KVX_FP8") == "1")) else {})})
        llm = LLM(**kw)
        tok = llm.get_tokenizer()
        sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=5)
        sp32 = SamplingParams(max_tokens=32, temperature=0.0)
        rows = []
        g = torch.Generator().manual_seed(0)
        for n in [int(x) for x in lens.split(",")]:
            ts = []
            lng = long_from and n >= long_from  # 256K-1M: a native request takes minutes; the engine is warm from the
            nrep = reps_long if lng else reps      # shorter lengths, so no warm-up request and fewer reps
            for r in range(nrep + (0 if lng else 1)):
                ids = torch.randint(1000, 30000, (n,), generator=g).tolist()
                t = time.perf_counter()
                llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)
                if r > 0 or lng:
                    ts.append(time.perf_counter() - t)
            ts.sort()
            rows.append(dict(len=n, ttft_s=ts[len(ts) // 2], reps=len(ts), all_s=[round(x, 4) for x in ts]))
            print(json.dumps(rows[-1]), flush=True)
        # quality: first-token top-20 logprobs on 16 real FineWeb-Edu prefixes (eval token cache), compared offline
        import glob
        out = {}
        fs = sorted(glob.glob("/vol/data/eval_64x1024_bos0_*.pt"))
        if fs and lens_quality:
            ev = torch.load(fs[0])[:16]
            spq = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
            outs = llm.generate([{"prompt_token_ids": list(x[:1024 - 64 + 1])} for x in ev], spq, use_tqdm=False)
            out["first_token_top20"] = [{int(t): float(lp.logprob) for t, lp in o.outputs[0].logprobs[0].items()} for o in outs]
            g = llm.generate([{"prompt_token_ids": list(x[:961])} for x in ev[:8]], sp32, use_tqdm=False)
            out["greedy32_ids"] = [list(o.outputs[0].token_ids) for o in g]
        if quality_long > 0:  # 4 real long prompts: consecutive 4096-token PG19 chunks, first-token top-20
            fl = sorted(glob.glob("/vol/data/pg19train_*x4096_bos0_*.pt"))
            if fl:
                pg = torch.load(fl[0])
                per = -(-quality_long // 4096)
                n_pg = min(long_n, 16) if long_n > 4 else long_n  # beyond 4: half PG19 books, half concatenated FineWeb docs
                prompts = [sum((list(pg[j]) for j in range(i * per, (i + 1) * per)), [])[:quality_long] for i in range(n_pg)]
                fc = sorted(glob.glob("/vol/data/calib_4000x1024_bos0_*.pt"))
                if long_n > n_pg and fc:
                    fw = torch.load(fc[0])
                    dper = -(-quality_long // len(fw[0]))  # token caches are tensors or lists of lists
                    prompts += [sum((list(fw[j]) for j in range(i * dper, (i + 1) * dper)), [])[:quality_long]
                                for i in range(long_n - n_pg)]
                spq = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
                outs = llm.generate([{"prompt_token_ids": p} for p in prompts], spq, use_tqdm=False)
                out["long_len"] = quality_long
                out["long_first_token_top20"] = [{int(t): float(lp.logprob) for t, lp in o.outputs[0].logprobs[0].items()}
                                                 for o in outs]
                out["long_n"] = len(prompts)
                if needle_n > 0:  # RULER-style niah_single: a 7-digit magic number at depth 10-90%, asked at the end
                  try:
                    import random
                    rng = random.Random(0)
                    hay_chunks = list(range(len(pg) - 1, -1, -1))  # haystack from the far end of the PG19 cache
                    items, nprompts = [], []
                    for i in range(needle_n):
                        key = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(8))
                        val = str(rng.randrange(1000000, 9999999))
                        needle = tok.encode(f" The special magic number for {key} is: {val}. ", add_special_tokens=False)
                        q = tok.encode(f"\nWhat is the special magic number for {key} mentioned in the provided text? "
                                       f"The special magic number for {key} is:", add_special_tokens=False)
                        nh = quality_long - len(needle) - len(q)
                        c0 = (i * per) % max(1, len(pg) - per)
                        hay = sum((list(pg[hay_chunks[(c0 + j) % len(hay_chunks)]]) for j in range(per)), [])[:nh]
                        depth = 0.1 + 0.8 * (i % 9) / 8
                        at = int(depth * nh)
                        nprompts.append(hay[:at] + needle + hay[at:] + q)
                        items.append(dict(depth=round(depth, 2), val=val))
                    spn = SamplingParams(max_tokens=12, temperature=0.0)
                    outs = llm.generate([{"prompt_token_ids": p} for p in nprompts], spn, use_tqdm=False)
                    for it, o in zip(items, outs):
                        it["text"] = o.outputs[0].text
                        it["correct"] = it["val"] in it["text"]
                    out["needle"] = items
                    out["needle_acc"] = sum(it["correct"] for it in items) / len(items)
                  except Exception as e:  # keep the timing/quality results if the needle set breaks
                    out["needle_error"] = repr(e)[:500]
        res = dict(mode=mode, src=src, tgt=tgt, maps=maps, tag=tag, yarn=yarn, tp_=tp, gpu=torch.cuda.get_device_name(0), tail=tail, rows=rows, concurrency=1, backend=backend, src_sdpa=src_sdpa, tp=tp, src_tp=src_tp,
                   fp8=fp8, restore=restore, sink=sink, util=util, **out)
        os.makedirs("/vol/runs/vllm", exist_ok=True)
        json.dump(res, open(f"/vol/runs/vllm/{mode}_{tgt.split('/')[-1]}{tag or os.environ.get('KVX_TAG', '')}.json", "w"), indent=1)
        try:
            vol.commit()
        except Exception:
            pass
        return res
    except Exception:
        import traceback
        part = json.dumps(locals().get("rows", []))  # rows measured before the failure are still results
        raise RuntimeError(traceback.format_exc()[-3000:] + "\nPARTIAL_ROWS=" + part) from None


serve = app.function(gpu=GPU.split(",") if "," in GPU else GPU, volumes={"/vol": vol, "/hf": hf_shared.read_only()}, timeout=3000)(serve_impl)


@app.local_entrypoint()
def main(mode: str = "native", src: str = "Qwen/Qwen3-1.7B", tgt: str = "Qwen/Qwen3-32B",
         maps: str = "/vol/runs/q17-q32/maps_k16.pt", lens: str = "4096,8192,16384,32768", tail: int = 64,
         fp8: int = 0, tag: str = "", util: float = 0.72, restore: str = "rms", sink: str = "", backend: str = "FLASH_ATTN",
         yarn: float = 0.0, orig_ctx: int = 32768, quality_long: int = 0, maxlen: int = 0, reps: int = 3,
         src_sdpa: str = "cudnn", tp: int = 1):
    r = serve.remote(mode, src, tgt, maps, lens, tail, reps, fp8, tag, util, restore, sink, backend, yarn, orig_ctx,
                     quality_long, maxlen, src_sdpa, tp)
    os.makedirs("out/vllm", exist_ok=True)
    json.dump(r, open(f"out/vllm/{mode}_{tgt.split('/')[-1]}{tag}.json", "w"), indent=1)
    print(json.dumps(r, indent=1))
