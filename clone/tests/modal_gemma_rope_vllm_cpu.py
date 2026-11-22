"""CPU check (vLLM image): the Gemma connector's per-type RoPE tables (gemma_rope_tables) vs vLLM's get_rope rows for
Gemma-4-31B's sliding (default, theta 1e4, D 256) and global (proportional partial 0.25, theta 1e6, D 512) layers, at
positions up to 1M (beyond the 262144 card: plain RoPE). modal run tests/modal_gemma_rope_vllm_cpu.py"""
import os

import modal

image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
         .pip_install("vllm==0.34.0").add_local_python_source("kvx"))
app = modal.App(os.environ.get("KVX_APP", "kvx-speed-g4rope"), image=image)
hf = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))


@app.function(cpu=4.0, memory=32768, timeout=900, volumes={"/hf": hf.read_only()},
              env={"HF_HUB_CACHE": "/hf/hub", "HF_HUB_OFFLINE": "1"})
def run():
    import traceback
    import torch
    from transformers import AutoConfig
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from kvx.vllm_gemma_connector import gemma_rope_tables
    tc = AutoConfig.from_pretrained("google/gemma-4-31B").text_config
    N = 1048576
    pos = torch.tensor([0, 1, 1000, 262143, 600000, 1048000])
    out = []
    for ty, D in (("sliding_attention", 256), ("full_attention", 512)):  # 31B per-layer head dims (config.json)
        r = dict(type=ty, D=D, rp=str(tc.rope_parameters[ty]))
        try:
            with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
                rope = get_rope(head_size=D, max_position=N, is_neox_style=True,
                                rope_parameters=dict(tc.rope_parameters[ty]), dtype=torch.float32)
            cache = rope.cos_sin_cache.float()
            rd = cache.shape[1]
            c, s = gemma_rope_tables(tc.rope_parameters[ty], D, int(pos.max()) + 1, "cpu")
            vc, vs = cache[pos, :rd // 2], cache[pos, rd // 2:]
            r.update(cls=type(rope).__name__, rd_vllm=rd, cache_rows=cache.shape[0],
                     per_pos_cos=[float((vc[i] - c[pos[i], :rd // 2]).abs().max()) for i in range(len(pos))],
                     per_pos_sin=[float((vs[i] - s[pos[i], :rd // 2]).abs().max()) for i in range(len(pos))],
                     conn_rot_dims_nonzero=int((c[1] != 1).sum()))
        except Exception:
            r["error"] = traceback.format_exc()[-1500:]
        out.append(r)
    return out


@app.local_entrypoint()
def main():
    import json
    for r in run.remote():
        print(json.dumps(r))
