"""Do the connector's target RoPE tables (target_cos_sin on the apply_yarn config) equal the cos/sin cache vLLM builds
for the same hf_overrides? CPU container, vLLM image. modal run tests/modal_yarn_vllm_check.py"""
import os
import modal

image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
         .pip_install("vllm==0.34.0")
         .env({k: v for k, v in os.environ.items() if k.startswith("KVX_") and k.endswith(("_VOLUME", "_SECRET"))})
         .add_local_python_source("kvx"))
app = modal.App(os.environ.get("KVX_APP", "kvx-speed-yarncheck"), image=image)
hf = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))


@app.function(cpu=4.0, memory=16384, timeout=900, volumes={"/hf": hf.read_only()}, env={"HF_HUB_CACHE": "/hf/hub", "HF_HUB_OFFLINE": "1"})
def run(cases):
    import inspect, torch
    from transformers import AutoConfig
    from kvx.models import apply_yarn
    from kvx.vllm_connector import target_cos_sin
    from vllm.model_executor.layers import rotary_embedding as re_
    out = []
    for name, factor, orig in cases:
        cfg = apply_yarn(AutoConfig.from_pretrained(name), factor, orig)
        D = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        T = 32
        pos = torch.tensor([0, 1000, 30000, 100000, 130000, 500000, 1000000])
        pos = pos[pos < factor * orig]
        ct, st = target_cos_sin(cfg, int(pos.max()) + 1, "cpu")
        rs = getattr(cfg, "rope_scaling", None) or getattr(cfg, "rope_parameters", None)
        theta = (rs or {}).get("rope_theta") or getattr(cfg, "rope_theta", 10000)
        sig = inspect.signature(re_.get_rope)
        kw = dict(head_size=D, max_position=int(factor * orig), is_neox_style=True)
        if "rotary_dim" in sig.parameters:
            kw["rotary_dim"] = D
        if "base" in sig.parameters:
            kw["base"] = theta
        if "rope_scaling" in sig.parameters:
            kw["rope_scaling"] = dict(rs)
        if "rope_parameters" in sig.parameters:
            kw["rope_parameters"] = dict(rs, rope_theta=theta)
        if "dtype" in sig.parameters:
            kw["dtype"] = torch.float32
        try:
            from vllm.config import VllmConfig, set_current_vllm_config
            with set_current_vllm_config(VllmConfig()):
                rope = re_.get_rope(**kw)
            cache = rope.cos_sin_cache.float()  # [max_pos, D] = cat(cos[:D/2], sin[:D/2])
            vc, vs = cache[pos, :D // 2], cache[pos, D // 2:]
            hc, hs = ct[pos, :D // 2], st[pos, :D // 2]
            out.append(dict(name=name, factor=factor, rope=type(rope).__name__, rs=str(rs)[:200],
                            max_abs_cos=float((vc - hc).abs().max()), max_abs_sin=float((vs - hs).abs().max()),
                            v_norm0=float((vc[0] ** 2 + vs[0] ** 2).mean()), h_norm0=float((hc[0] ** 2 + hs[0] ** 2).mean())))
        except Exception as e:
            out.append(dict(name=name, factor=factor, error=repr(e)[:600], sig=str(sig)[:400]))
    return out


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(run.remote([("Qwen/Qwen2.5-72B", 4.0, 32768), ("Qwen/Qwen2.5-72B", 16.0, 32768),
                                 ("Qwen/Qwen3-32B", 4.0, 32768), ("Qwen/Qwen3-32B", 32.0, 32768)]), indent=1))
