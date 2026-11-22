"""GPU oracle for the hybrid vLLM connector: identity kvx-hybrid-v1 maps (source = target = Qwen3.5-4B, drop_q=False,
restore none) must make the mapped engine reproduce native vLLM: the connector writes the target's own taps back
as attention KV + GatedDeltaNet states. Checks vLLM's conv channel order, recurrent-state orientation, per-group
block ids and the HMA path. KVXQ_GPU=L40S modal run tests/modal_hybrid_oracle.py
"""
import json
import os
import subprocess
import sys

import modal

GPU = os.environ.get("KVXQ_GPU", "L40S").split(",")
# self-contained (importing modal_evals failed inside the container): same image as modal_evals
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.34.0", "hf_transfer", "pyarrow")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_CACHE": "/hf/hub", "HF_HUB_OFFLINE": "1"})
    .add_local_python_source("kvx")
)
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"))
hf_shared = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
app = modal.App(os.environ.get("KVX_APP", "kvx-quality-hybrid-oracle"), image=image)
MODEL = "Qwen/Qwen3.5-4B"


def oracle_maps(cfg):
    """Identity maps from the text config alone: every target layer reads the same source layer's full taps."""
    import torch
    g = cfg.get
    kd = g("linear_key_head_dim") * g("linear_num_key_heads")
    vd = g("linear_value_head_dim") * g("linear_num_value_heads")
    Hv = g("linear_num_value_heads")
    hd = g("head_dim")
    kv = g("num_key_value_heads") * hd
    taps, maps, kinds = [], {}, []
    for l, t in enumerate(g("layer_types")):
        if t == "linear_attention":
            tp = [("qkv", 2 * kd + vd), ("a", Hv), ("b", Hv)]
            kinds.append("lin")
        else:
            tp = [("k", kv), ("v", kv)]
            kinds.append("attn")
        w = sum(x for _, x in tp)
        taps.append(dict(kind=kinds[-1], taps=tp))
        maps[l] = dict(sel=[l], W=torch.eye(w, dtype=torch.bfloat16), b=torch.zeros(w), r2=None)
    return dict(format="kvx-hybrid-v1", src=MODEL, tgt=MODEL, lam_rel=0.0, w=1, drop_q=False,
                src_feats=[dict(kind=k) for k in kinds], src_widths=[sum(x for _, x in t["taps"]) for t in taps],
                tgt_taps=taps, lin_dims=None, src_lin_dims=None, maps=maps, restore="none", sinks=0, sink_template={})


RUN = r'''
import json, sys, torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
mode, maps, out = sys.argv[1], sys.argv[2], sys.argv[3]
kw = dict(model="%s", max_model_len=8192, gpu_memory_utilization=0.55, enable_prefix_caching=False,
          limit_mm_per_prompt={"image": 0, "video": 0}, seed=0, enforce_eager=True, max_num_seqs=32)
if mode == "mapped":
    from vllm.config import KVTransferConfig
    kw["kv_transfer_config"] = KVTransferConfig(kv_connector="HybridCrossModelConnector",
        kv_connector_module_path="kvx.vllm_hybrid_connector", kv_role="kv_both",
        kv_connector_extra_config={"src": "%s", "maps": maps, "tail": 64, "min_len": 256})
llm = LLM(**kw)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("%s")
import pyarrow.parquet as pq
rows = pq.read_table("/vol/data/evals/quality_val.parquet").to_pylist()[:16]
prompts = []
for r in rows:
    ids = tok(r["article"], add_special_tokens=False)["input_ids"][:1500 + 37 * len(prompts)]
    prompts.append(TokensPrompt(prompt_token_ids=ids))
sp = SamplingParams(temperature=0.0, max_tokens=32, logprobs=20)
res = llm.generate(prompts, sp)
o = dict(first=[{str(k): v.logprob for k, v in r.outputs[0].logprobs[0].items()} for r in res],
         greedy=[list(r.outputs[0].token_ids) for r in res], n=[len(p["prompt_token_ids"]) for p in prompts])
json.dump(o, open(out, "w"))
''' % (MODEL, MODEL, MODEL)


@app.function(gpu=GPU, volumes={"/vol": vol, "/hf": hf_shared.read_only()}, timeout=2400)
def oracle():
    import math
    import torch
    from transformers import AutoConfig
    hf_shared.reload()
    vol.reload()
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg = (getattr(cfg, "text_config", None) or cfg).to_dict()
    torch.save(oracle_maps(cfg), "/tmp/oracle_maps.pt")
    open("/tmp/run.py", "w").write(RUN)
    logs = {}
    for mode in ("native", "mapped"):
        p = subprocess.run([sys.executable, "/tmp/run.py", mode, "/tmp/oracle_maps.pt", f"/tmp/{mode}.json"],
                           capture_output=True, text=True)
        logs[mode] = (p.returncode, (p.stdout + p.stderr)[-5000:])
        if p.returncode:
            return dict(error=mode, logs=logs)
    a, b = json.load(open("/tmp/native.json")), json.load(open("/tmp/mapped.json"))
    kls, agree, pref = [], 0, []
    for p, q, x, y in zip(a["first"], b["first"], a["greedy"], b["greedy"]):
        floor = min(q.values()) - 1.0
        kls.append(sum(math.exp(lp) * (lp - q.get(t, floor)) for t, lp in p.items()))
        agree += max(p, key=p.get) == max(q, key=q.get)
        pref.append(next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), len(x)))
    hint = [l for l in logs["mapped"][1].splitlines() if "kvx hybrid" in l][-5:]
    return dict(n=len(kls), approx_kl_top20=sum(kls) / len(kls), top1=agree / len(kls), greedy32_prefix=sum(pref) / len(pref),
                lens=a["n"], connector_log=hint)


@app.local_entrypoint()
def main():
    print(json.dumps(oracle.remote(), indent=1)[:8000])
