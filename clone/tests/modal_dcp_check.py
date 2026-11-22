"""CPU check (vLLM image, config only): does vLLM 0.34 offer decode/prefill context parallel for GLM-5.3 (MLA KV
sharded across TP ranks instead of replicated), and what does a 1M max_model_len config resolve to with it.
modal run tests/modal_dcp_check.py"""
import os

import modal
from modal_vllm import glm_image as image, hf_shared

app = modal.App(os.environ.get("KVX_APP", "kvx-speed-dcp-check"), image=image)


@app.function(cpu=4.0, memory=32768, timeout=900, volumes={"/hf": hf_shared.read_only()})
def run():
    import dataclasses
    import traceback
    from vllm.engine.arg_utils import EngineArgs
    names = [f.name for f in dataclasses.fields(EngineArgs)]
    out = {"cp_fields": [n for n in names if "context_parallel" in n or "dcp" in n or "pcp" in n]}
    for kw in ({"decode_context_parallel_size": 8}, {"prefill_context_parallel_size": 8}):
        k = list(kw)[0]
        if k not in names:
            out[k] = "absent"
            continue
        try:
            ea = EngineArgs(model="zai-org/GLM-5.3", max_model_len=1048576, tensor_parallel_size=8, **kw)
            mc = ea.create_model_config()
            out[k] = f"model config ok ({mc.architectures})"
            try:
                vc = ea.create_engine_config()
                out[k] += f"; engine config ok: {str(vc.parallel_config)[:300]}"
            except Exception:
                out[k] += "; engine config: " + traceback.format_exc()[-600:]
        except Exception:
            out[k] = traceback.format_exc()[-800:]
    import vllm, inspect
    src = inspect.getsource(__import__("vllm.model_executor.layers.attention.mla_attention", fromlist=["x"]))
    out["mla_mentions_dcp"] = src.count("dcp") + src.count("decode_context_parallel")
    return out


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(run.remote(), indent=1)[:5000])
