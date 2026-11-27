"""CPU check (runs in the vLLM image): the connector's fp8 KV-cache write round-trips, and how vLLM maps "fp8"."""
import torch
from kvx.vllm_connector import quant_for_cache, write_kv_fa

res = {}
try:
    import vllm.utils.torch_utils as tu
    res["vllm_fp8_cache_dtype"] = str(tu.STR_DTYPE_TO_TORCH_DTYPE.get("fp8"))
except Exception as e:  # location moved across versions
    try:
        from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
        res["vllm_fp8_cache_dtype"] = str(STR_DTYPE_TO_TORCH_DTYPE.get("fp8"))
    except Exception as e2:
        res["vllm_fp8_cache_dtype"] = f"lookup failed: {e!r} / {e2!r}"
torch.manual_seed(0)
blocks, H, block, D = 8, 4, 16, 128
for cache_dtype in (torch.uint8, torch.float8_e4m3fn, torch.bfloat16):
    for ks, vs in ((1.0, 1.0), (0.5, 2.0)):
        kv = torch.zeros(blocks, H, block, 2 * D, dtype=cache_dtype)
        n = 40
        K, V = torch.randn(n, H, D, dtype=torch.bfloat16) * 3, torch.randn(n, H, D, dtype=torch.bfloat16)
        slots = torch.randperm(blocks * block)[:n]
        bi, off = slots // block, slots % block
        Kq, Vq = quant_for_cache(K, V, kv.dtype, ks, vs)
        write_kv_fa(kv, bi, off, Kq, Vq, block)
        rd = kv[bi, :, off]
        if cache_dtype == torch.uint8:
            rd = rd.view(torch.float8_e4m3fn)
        Kr, Vr = rd[..., :D].float(), rd[..., D:].float()
        if cache_dtype != torch.bfloat16:
            Kr, Vr = Kr * ks, Vr * vs
        rel = lambda a, b: float((a - b.float()).norm() / b.float().norm())
        res[f"{cache_dtype}_ks{ks}"] = {"K_rel": round(rel(Kr, K), 4), "V_rel": round(rel(Vr, V), 4)}
        assert rel(Kr, K) < 0.06 and rel(Vr, V) < 0.06, res
# the old path (.to(uint8) of bf16) is garbage: prove the check would catch it
bad = K.to(torch.uint8).view(torch.float8_e4m3fn).float()
res["old_path_rel_err"] = round(float((bad - K.float()).norm() / K.float().norm()), 2)
print(res)
print("FP8_CACHE_WRITE_OK")
