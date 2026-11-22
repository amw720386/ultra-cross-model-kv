"""Hybrid connector identity oracle without Modal (any box with vLLM >= 0.34, CUDA or ROCm).
Identity kvx-hybrid-v1 maps (source = target, drop_q False, restore none, no sinks) must make the mapped engine
reproduce the native engine: the connector writes the target's own taps back as attention KV + GatedDeltaNet states.

python tools/hybrid_oracle_local.py maps Qwen/Qwen3.5-4B /tmp/oracle_maps.pt
python tools/hybrid_oracle_local.py run native|mapped Qwen/Qwen3.5-4B /tmp/oracle_maps.pt <quality parquet> out.json [tp] [extra json]
python tools/hybrid_oracle_local.py compare native.json mapped.json
"""
import json
import math
import sys


def oracle_maps(model, out):
    import torch
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model)
    g = (getattr(cfg, "text_config", None) or cfg).to_dict().get
    kd = g("linear_key_head_dim") * g("linear_num_key_heads")
    vd = g("linear_value_head_dim") * g("linear_num_value_heads")
    Hv = g("linear_num_value_heads")
    kv = g("num_key_value_heads") * g("head_dim")
    taps, maps, kinds = [], {}, []
    for l, t in enumerate(g("layer_types")):
        tp = [("qkv", 2 * kd + vd), ("a", Hv), ("b", Hv)] if t == "linear_attention" else [("k", kv), ("v", kv)]
        kinds.append("lin" if t == "linear_attention" else "attn")
        w = sum(x for _, x in tp)
        taps.append(dict(kind=kinds[-1], taps=tp))
        maps[l] = dict(sel=[l], W=torch.eye(w, dtype=torch.bfloat16), b=torch.zeros(w), r2=None)
    torch.save(dict(format="kvx-hybrid-v1", src=model, tgt=model, lam_rel=0.0, w=1, drop_q=False,
                    src_feats=[dict(kind=k) for k in kinds], src_widths=[sum(x for _, x in t["taps"]) for t in taps],
                    tgt_taps=taps, lin_dims=None, src_lin_dims=None, maps=maps, restore="none", sinks=0,
                    sink_template={}), out)


def run(mode, model, maps, parquet, out, tp="1", extra="{}"):
    """tp: tensor parallel size; extra: JSON merged into the connector's extra_config (e.g. {"stream_feats": 0,
    "fast_gdn": 0}) to A/B one connector path at a time."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    kw = dict(model=model, max_model_len=8192, gpu_memory_utilization=0.45, enable_prefix_caching=False,
              limit_mm_per_prompt={"image": 0, "video": 0}, seed=0, enforce_eager=True, max_num_seqs=32,
              tensor_parallel_size=int(tp))
    if mode == "mapped":
        from vllm.config import KVTransferConfig
        kw["kv_transfer_config"] = KVTransferConfig(
            kv_connector="HybridCrossModelConnector", kv_connector_module_path="kvx.vllm_hybrid_connector",
            kv_role="kv_both", kv_connector_extra_config=dict({"src": model, "maps": maps, "tail": 64, "min_len": 256,
                                                               "recolor": 0}, **json.loads(extra)))
    llm = LLM(**kw)
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    tok = AutoTokenizer.from_pretrained(model)
    rows = pq.read_table(parquet).to_pylist()[:16]
    prompts = [TokensPrompt(prompt_token_ids=tok(r["article"], add_special_tokens=False)["input_ids"][:1500 + 37 * i])
               for i, r in enumerate(rows)]
    res = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=32, logprobs=20))
    json.dump(dict(first=[{str(k): v.logprob for k, v in r.outputs[0].logprobs[0].items()} for r in res],
                   greedy=[list(r.outputs[0].token_ids) for r in res],
                   n=[len(p["prompt_token_ids"]) for p in prompts]), open(out, "w"))


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    kls, agree, pref = [], 0, []
    for p, q, x, y in zip(a["first"], b["first"], a["greedy"], b["greedy"]):
        floor = min(q.values()) - 1.0
        kls.append(sum(math.exp(lp) * (lp - q.get(t, floor)) for t, lp in p.items()))
        agree += max(p, key=p.get) == max(q, key=q.get)
        pref.append(next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), len(x)))
    print(json.dumps(dict(n=len(kls), approx_kl_top20=sum(kls) / len(kls), top1=agree / len(kls),
                          greedy32_prefix=sum(pref) / len(pref), lens=a["n"])))


if __name__ == "__main__":  # vLLM spawns its engine core: nothing may start on import
    c = sys.argv[1]
    if c == "maps":
        oracle_maps(sys.argv[2], sys.argv[3])
    elif c == "run":
        run(*sys.argv[2:9])
    else:
        compare(sys.argv[2], sys.argv[3])
