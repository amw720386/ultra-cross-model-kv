"""CPU-only check (no GPU): connector target_cos_sin == HF rotary embedding, incl. YaRN and llama3 scaling.
modal run tests/modal_rope_check.py"""
import ast, modal

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "transformers==4.56.2", "numpy")
         .add_local_file("kvx/vllm_connector.py", "/root/vllm_connector.py"))
app = modal.App("kvx-ropecheck-cpu", image=image)


@app.function(cpu=4.0, memory=16384, timeout=900)
def check():
    import torch
    from transformers import AutoConfig
    src = open("/root/vllm_connector.py").read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "target_cos_sin")
    ns = {"torch": torch}
    exec(compile(ast.Module([fn], []), "target_cos_sin", "exec"), ns)
    out = {}
    cases = [("Qwen/Qwen3-32B", {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}, 131072),
             ("Qwen/Qwen3-32B", None, 32768),
             ("unsloth/Meta-Llama-3.1-8B", "keep", 131072)]
    for name, rs, T in cases:
        cfg = AutoConfig.from_pretrained(name)
        if rs != "keep":
            cfg.rope_scaling = rs
            if rs is not None:
                cfg.max_position_embeddings = T
        mt = cfg.model_type
        mod = __import__(f"transformers.models.{mt}.modeling_{mt}", fromlist=["x"])
        Rot = getattr(mod, [k for k in dir(mod) if k.endswith("RotaryEmbedding")][0])
        rot = Rot(config=cfg)
        pos = torch.arange(T)[None]
        c_ref, s_ref = rot(torch.zeros(1, T, 1), pos)
        c, s = ns["target_cos_sin"](cfg, T, "cpu")
        out[f"{name} {rs if rs != 'keep' else cfg.rope_scaling}"] = dict(
            max_abs_cos=float((c - c_ref[0]).abs().max()), max_abs_sin=float((s - s_ref[0]).abs().max()))
    return out


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(check.remote(), indent=1))
