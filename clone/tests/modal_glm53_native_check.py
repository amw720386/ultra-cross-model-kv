"""CPU check for tests/modal_glm53_native_probe.py (no GPU): vLLM resolves GLM-5.3's config at 1M with the probe's
settings, and kvx-pairs' gate command starts (--help). modal run tests/modal_glm53_native_check.py [--gate-cmd '<json>']"""
import json
import os
import subprocess
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modal_vllm import image, vol, hf_shared

app = modal.App(os.environ.get("KVX_APP", "kvx-speed-glm53-check"), image=image)
TGT = "zai-org/GLM-5.3"
MAXLEN = 1048576


@app.function(cpu=4.0, memory=32768, timeout=900, volumes={"/vol": vol, "/hf": hf_shared.read_only()})
def dry(gate_cmd: str = ""):
    """vLLM resolves GLM-5.3's config at 1M with the probe's settings; the gate command at least starts (--help)."""
    import traceback
    out = {}
    try:
        from vllm.engine.arg_utils import EngineArgs
        from modal_vllm import target_overrides
        ov, allow = target_overrides(TGT, 0.0, 32768, MAXLEN)
        if allow:
            os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        mc = EngineArgs(model=TGT, max_model_len=MAXLEN, dtype="bfloat16", tensor_parallel_size=8,
                        **({"hf_overrides": ov} if ov else {})).create_model_config()
        tc = mc.hf_text_config
        out.update(arch=mc.architectures, max_model_len=mc.max_model_len, allow_long=allow,
                   max_pos=getattr(tc, "max_position_embeddings", None), index_topk=getattr(tc, "index_topk", None),
                   kv_lora_rank=getattr(tc, "kv_lora_rank", None), quant=str(getattr(tc, "quantization_config", None))[:200])
    except Exception:
        out["config_error"] = traceback.format_exc()[-2000:]
    if gate_cmd:
        argv = json.loads(gate_cmd)
        p = subprocess.run(argv + ["--help"], capture_output=True, text=True)
        out.update(gate_help_rc=p.returncode, gate_help=(p.stdout + p.stderr)[-1500:])
    return out


@app.local_entrypoint()
def main(gate_cmd: str = ""):
    print(json.dumps(dry.remote(gate_cmd), indent=1)[:6000])
