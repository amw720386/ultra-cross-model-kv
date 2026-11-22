import os
import modal
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12").pip_install("vllm==0.34.0", "hf_transfer"))
app = modal.App(os.environ.get("KVX_APP", "kvx-vllminspect-cpu"), image=image)
@app.function(cpu=2.0, memory=8192, timeout=600)
def run():
    import inspect, subprocess
    out = []
    cmd = r"""python - <<'PY'
import inspect
try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    print("SIG", inspect.signature(flash_attn_varlen_func))
except Exception as e:
    print("ERR vllm_flash_attn", repr(e)[:300])
try:
    from vllm.v1.attention.backends import fa_utils
    print("FA_UTILS", [n for n in dir(fa_utils) if not n.startswith("_")][:40])
    print(inspect.getsource(fa_utils.get_flash_attn_version)[:2500])
except Exception as e:
    print("ERR fa_utils", repr(e)[:300])
try:
    from transformers import AttentionInterface
    print("ATTN_IFACE ok", AttentionInterface)
except Exception as e:
    print("ERR iface", repr(e)[:200])
PY"""
    g = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out.append(g.stdout[-12000:] + g.stderr[-1000:])
    return "\n".join(out)
@app.local_entrypoint()
def main():
    print(run.remote())
