"""Modal runner for research/mc128.py (OneTriangle notes 001 / 016 / 019 replication).

CPU (outside the team GPU cap):  modal run research/modal_mc128.py --pairs Qwen/Qwen3-0.6B:Qwen/Qwen3-1.7B --cpu 1
GPU:  KVX_GPU=H200 modal run research/modal_mc128.py --pairs a:b,c:d --cpu 0
Results: volume kvx-research /runs/<run_id>/<src>__<tgt>.json, also written locally to ./out/<run_id>/.
"""
import json, os

import modal

GPU = os.environ.get("KVX_GPU", "H200")
# KVX_T5=1: transformers 5 image (Mistral3 / Ministral-3 need it); default 4.57.1 (every earlier run)
TRANSFORMERS = "transformers==5.24.0" if os.environ.get("KVX_T5") == "1" else "transformers==4.57.1"
CPUS = float(os.environ.get("KVX_CPUS", "32"))
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.0", TRANSFORMERS, "accelerate", "datasets==5.0.1", "numpy", "hf_transfer",
                 extra_index_url="https://download.pytorch.org/whl/cpu" if os.environ.get("KVX_CPU_IMAGE") else None)
    # local env does not reach containers: copy the volume names into the image env (module-level reads run in both)
    .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          **{k: os.environ[k] for k in ("KVX_VOLUME", "KVX_WEIGHTS_VOLUME", "KVX_RESEARCH_VOLUME") if k in os.environ}})
    .add_local_python_source("kvx", "research")
)
vol = modal.Volume.from_name(os.environ.get("KVX_RESEARCH_VOLUME", "kvx-research"), create_if_missing=True)
hf_shared = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
app = modal.App(os.environ.get("KVX_APP", "kvx-research-mc128"), image=image)
kvx_vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"))  # lead / kvx-pairs maps (read-only here)
VOLS = {"/vol": vol, "/hf": hf_shared.read_only(), "/kvx": kvx_vol.read_only()}


def _impl(src, tgt, run_id, kw):
    hf_shared.reload()
    from research import mc128
    out = f"/vol/runs/{run_id}/{src.split('/')[-1]}__{tgt.split('/')[-1]}.json"
    try:
        r = mc128.run_pair(src, tgt, out=out, **kw)
    finally:
        vol.commit()
    items = json.load(open(out.replace(".json", "_items.json")))
    return r, items


@app.function(cpu=CPUS, memory=65536, timeout=3300, volumes=VOLS)
def mc_cpu(src, tgt, run_id, kw):
    import torch
    torch.set_num_threads(int(CPUS))
    return _impl(src, tgt, run_id, kw)


@app.function(gpu=GPU, timeout=3300, volumes=VOLS)
def mc_gpu(src, tgt, run_id, kw):
    return _impl(src, tgt, run_id, kw)


@app.function(gpu=GPU, timeout=3300, volumes=VOLS)
def mc_gpu_seq(pairs, run_id, kw, harness="mc128"):
    """All directions in one container; each model loaded once (kept while a later pair needs it)."""
    import gc, time, torch
    hf_shared.reload()
    from research import mc128, kl32, arena, xfit, recall015, debugkv, qa018, ttft007
    mod = {"kl32": kl32, "arena": arena, "xfit": xfit, "recall015": recall015, "debugkv": debugkv, "qa018": qa018, "ttft007": ttft007}.get(harness, mc128)
    cache, outs = {}, []
    for i, (s, t) in enumerate(pairs):
        later = {m for p in pairs[i:] for m in p}
        for m in list(cache):
            if m not in later:
                del cache[m]
        gc.collect(); torch.cuda.empty_cache()
        for m in (s, t):
            if m not in cache:
                t0 = time.time()
                # xfit with 2 GPUs: source on cuda:0, target on cuda:1 (its run_pair hooks the target's I/O)
                d_ = "cuda:1" if (harness == "xfit" and torch.cuda.device_count() > 1 and m == t) else "cuda:0"
                cache[m] = mc128.load(m, d_, torch.bfloat16)
                print(f"loaded {m} in {time.time()-t0:.0f}s", flush=True)
        out = f"/vol/runs/{run_id}/{s.split('/')[-1]}__{t.split('/')[-1]}.json"
        try:
            r = mod.run_pair(s, t, out=out, models=(cache[s], cache[t]), **kw)
            ip = out.replace(".json", "_items.json")
            outs.append((r, json.load(open(ip)) if os.path.exists(ip) else {}))
        except Exception as e:  # keep going: one failed direction must not cost the others
            import traceback
            traceback.print_exc()
            outs.append(f"{type(e).__name__}: {str(e)[:500]}")  # a string: the traceback pins GPU tensors, torch-less client
            del e
            gc.collect(); torch.cuda.empty_cache()
        finally:
            vol.commit()
    return outs


# native vLLM prefill TTFT (strict baseline for TTFT rows): own image, same GPU class
vllm_image = (modal.Image.debian_slim(python_version="3.12")
              .pip_install("vllm", "datasets", "hf_transfer")
              .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1",
                    **{k: os.environ[k] for k in ("KVX_VOLUME", "KVX_WEIGHTS_VOLUME", "KVX_RESEARCH_VOLUME") if k in os.environ}}))


@app.function(gpu=GPU, timeout=3300, volumes=VOLS, image=vllm_image)
def vllm_native_ttft(model, lens, warm=5, reps=20, max_model_len=0):
    """Single-request prefill TTFT in vLLM (prefix caching off, max_tokens=1): p50 wall time over `reps` real FineWeb
    prompts of each length (fresh tokens per rep so no cache can hit)."""
    import time, statistics
    hf_shared.reload()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from huggingface_hub import snapshot_download
    path = snapshot_download(model, cache_dir="/hf/hub", local_files_only=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    mlen = max_model_len or (max(lens) + 64)
    llm = LLM(model=path, max_model_len=mlen, enable_prefix_caching=False, gpu_memory_utilization=0.85, seed=0)
    from datasets import load_dataset
    ds = iter(load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True))
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    import vllm as _v
    out = dict(model=model, engine=f"vllm {_v.__version__}", prefix_caching=False)
    for L in lens:
        prompts = []
        while len(prompts) < warm + reps:
            ids = tok(next(ds)["text"], add_special_tokens=False)["input_ids"]
            if len(ids) >= L - 1:
                prompts.append(([tok.bos_token_id] if tok.bos_token_id is not None else []) + ids[:L - 1])
        ts = []
        for i, p in enumerate(prompts):
            t0 = time.perf_counter()
            llm.generate([TokensPrompt(prompt_token_ids=p[:L])], sp, use_tqdm=False)
            if i >= warm:
                ts.append(1000 * (time.perf_counter() - t0))
        out[str(L)] = dict(p50_ms=statistics.median(ts), min_ms=min(ts), max_ms=max(ts), reps=reps, warmups=warm)
        print(L, out[str(L)], flush=True)
    return out


@app.local_entrypoint()
def main(pairs: str, run_id: str = "mc128", cpu: int = 0, n_calib: int = 500, length: int = 512, ks: str = "1,4,8,16",
         n_heldout: int = 32, per_task: int = 32, hs_n: int = 0, piqa_n: int = 0, one_pass: int = 1, dtype: str = "bfloat16",
         calib_src: str = "fineweb", recipes: str = "", harness: str = "mc128", extra: str = ""):
    if harness in ("kl32", "arena", "xfit", "recall015", "debugkv", "qa018", "ttft007"):  # kl32.run_pair keywords; extra = JSON overrides (windows, prefix, steps, mc, forced)
        kw = dict(n_calib=n_calib, L=length, ks=ks, calib_src=calib_src, recipes=recipes, one_pass=one_pass)
    else:
        kw = dict(n_calib=n_calib, L=length, ks=ks, n_heldout=n_heldout, per_task=per_task, hs_n=hs_n, piqa_n=piqa_n,
                  one_pass=one_pass, dtype=dtype, calib_src=calib_src, recipes=recipes)
    kw.update(json.loads(extra) if extra else {})
    if harness == "vllm_native":  # pairs = model ids; extra {"lens": [...]} ; result -> out/<run_id>/vllm_native.json
        ex = json.loads(extra) if extra else {}
        os.makedirs(f"out/{run_id}", exist_ok=True)
        r = vllm_native_ttft.remote(pairs, ex.get("lens", [8192]), ex.get("warm", 5), ex.get("reps", 20))
        json.dump(r, open(f"out/{run_id}/vllm_native.json", "w"), indent=1)
        print(json.dumps(r))
        return
    plist = [p.split(":") for p in pairs.split(",")]
    os.makedirs(f"out/{run_id}", exist_ok=True)
    if cpu:
        results = mc_cpu.starmap([(s, t, run_id, kw) for s, t in plist], return_exceptions=True)
    else:
        results = mc_gpu_seq.remote(plist, run_id, kw, harness)
    for (s, t), res in zip(plist, results):
        name = f"{s.split('/')[-1]}__{t.split('/')[-1]}"
        if isinstance(res, (Exception, str)):
            print(f"FAILED {name}: {res!r}")
            continue
        r, items = res
        json.dump(r, open(f"out/{run_id}/{name}.json", "w"), indent=1)
        json.dump(items, open(f"out/{run_id}/{name}_items.json", "w"))
        mc = r.get("mc128", {})
        print(name, "k", r.get("k_selected"), {a: round(v["acc"], 4) for a, v in mc.items()})
        for key in ("heldout", "kl32_greedy", "kl32_text"):
            if key in r:
                print("  ", key, {k: (round(v.get("agree", 0), 3), round(v.get("ppl_ratio", v.get("kl32", 0)), 5)) for k, v in r[key].items()})
