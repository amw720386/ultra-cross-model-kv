"""CPU check (vLLM image, config only, no weights loaded) of modal_vllm.target_overrides for the 1M window targets:
vLLM's resolved text config (rope, max positions, max_model_len accepted) for Qwen3.8-27B YaRN x4 from 262144 (rope
keys partial_rotary_factor / mrope must survive), Gemma-4-31B at 1M on plain RoPE (allow_long), Qwen3-32B YaRN x32
(regression). For Qwen3.8 also: vLLM's rotary cos/sin vs the hybrid connector's table (hybrid_state.rope_cos_sin) at
far positions. modal run tests/modal_target_overrides_cpu.py"""
import os

import modal

image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
         .pip_install("vllm==0.34.0")
         .add_local_python_source("kvx").add_local_file("modal_vllm.py", "/root/modal_vllm.py"))
app = modal.App(os.environ.get("KVX_APP", "kvx-speed-tgt-overrides"), image=image)
hf = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
CASES = [("Qwen/Qwen3.8-27B", 4.0, 262144, 1048576), ("google/gemma-4-31B", 0.0, 32768, 1048576),
         ("Qwen/Qwen3-32B", 32.0, 32768, 1048576)]


@app.function(cpu=4.0, memory=16384, timeout=900, volumes={"/hf": hf.read_only()},
              env={"HF_HUB_CACHE": "/hf/hub", "HF_HUB_OFFLINE": "1"})
def run(keep_maxpos: int = 0):
    if keep_maxpos:
        os.environ["KVX_KEEP_MAXPOS"] = "1"
    import inspect
    import sys
    import traceback
    import torch
    sys.path.insert(0, "/root")
    from modal_vllm import target_overrides
    from vllm.engine.arg_utils import EngineArgs
    out = []
    for name, yarn, orig, L in CASES:
        r = dict(name=name, yarn=yarn)
        try:
            ov, allow = target_overrides(name, yarn, orig, L)
            if os.environ.get("KVX_KEEP_MAXPOS") and ov and "text_config" in ov:  # variant: card max_position kept
                ov["text_config"]["max_position_embeddings"] = orig
            r.update(ov=str(ov)[:400], allow_long=allow)
            if allow:
                os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
            else:
                os.environ.pop("VLLM_ALLOW_LONG_MAX_MODEL_LEN", None)
            mc = EngineArgs(model=name, max_model_len=L, dtype="bfloat16", **({"hf_overrides": ov} if ov else {})).create_model_config()
            tc = mc.hf_text_config
            rp = getattr(tc, "rope_parameters", None)
            r.update(max_model_len=mc.max_model_len, max_pos=getattr(tc, "max_position_embeddings", None),
                     rope=str(rp)[:400])
            if name.startswith("Qwen/Qwen3.8") and rp:
                from vllm.model_executor.layers import rotary_embedding as re_
                from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
                from kvx import hybrid_state as hs
                sig = inspect.signature(re_.get_rope)
                D = tc.head_dim
                kw = dict(head_size=D, max_position=int(tc.max_position_embeddings), is_neox_style=True)
                if "rope_parameters" in sig.parameters:
                    kw["rope_parameters"] = dict(rp)
                if "partial_rotary_factor" in sig.parameters:
                    kw["partial_rotary_factor"] = rp.get("partial_rotary_factor", 1.0)
                if "rotary_dim" in sig.parameters:
                    kw["rotary_dim"] = int(D * rp.get("partial_rotary_factor", 1.0))
                if "dtype" in sig.parameters:
                    kw["dtype"] = torch.float32
                with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
                    rope = re_.get_rope(**kw)
                cache = rope.cos_sin_cache.float()
                rd = cache.shape[1]
                p = hs.TargetParams(tc, None, D) if "lin" in inspect.signature(hs.TargetParams).parameters else hs.TargetParams(tc)
                pos = torch.tensor([0, 1000, 262143, 600000, 1048000])
                hc, hsn, hrd = hs.rope_cos_sin(p, int(pos.max()) + 1, "cpu")
                vc, vs = cache[pos, :rd // 2], cache[pos, rd // 2:]
                r.update(rope_cls=type(rope).__name__, rd_vllm=rd, rd_conn=hrd, sig=str(sig)[:300],
                         max_abs_cos=float((vc - hc[pos, :hrd // 2]).abs().max()),
                         max_abs_sin=float((vs - hsn[pos, :hrd // 2]).abs().max()))
                try:
                    vinv = rope._compute_inv_freq(rope.base).float()
                except Exception as e:
                    vinv = None
                    r["inv_err"] = repr(e)[:200]
                hinv, _ = hs._yarn_inv_freq(64, p.theta, p.yarn, "cpu")
                if vinv is not None:
                    r.update(inv_v=[float("%.4g" % x) for x in vinv], inv_h=[float("%.4g" % x) for x in hinv],
                             yarn_conn=str(p.yarn))
                ang = torch.atan2(cache[1, rd // 2:], cache[1, :rd // 2])
                r["inv_from_cache_pos1"] = [float("%.4g" % x) for x in ang]
                import copy
                pt = copy.copy(p)
                hs.vllm_rope_table(pt, rp, int(pos.max()) + 1, "cpu")
                tc_, ts_, trd = hs.rope_cos_sin(pt, int(pos.max()) + 1, "cpu")
                r.update(table_max_abs_cos=float((vc - tc_[pos, :trd // 2]).abs().max()),
                         table_max_abs_sin=float((vs - ts_[pos, :trd // 2]).abs().max()), table_rd=trd)
                tc2, ts2, _ = hs.rope_cos_sin(pt, 5, "cpu", offset=600000)  # offset path
                r["table_offset_ok"] = bool(torch.equal(tc2, tc_[600000:600005]))
                p0 = copy.copy(p); p0.yarn = None  # plain RoPE table: is vLLM ignoring YaRN?
                pc, ps_, _ = hs.rope_cos_sin(p0, int(pos.max()) + 1, "cpu")
                r.update(per_pos_cos_yarn=[float((vc[i] - hc[pos[i], :hrd // 2]).abs().max()) for i in range(len(pos))],
                         per_pos_cos_plain=[float((vc[i] - pc[pos[i], :hrd // 2]).abs().max()) for i in range(len(pos))],
                         vllm_pos0=[round(float(x), 4) for x in vc[0, :4]] + [round(float(x), 4) for x in vc[0, -4:]],
                         conn_pos0=[round(float(x), 4) for x in hc[0, :4]] + [round(float(x), 4) for x in hc[0, hrd // 2 - 4:hrd // 2]],
                         vllm_p1000=[round(float(x), 4) for x in vc[1, -4:]], conn_p1000=[round(float(x), 4) for x in hc[1000, hrd // 2 - 4:hrd // 2]],
                         plain_p1000=[round(float(x), 4) for x in pc[1000, hrd // 2 - 4:hrd // 2]],
                         cache_shape=list(cache.shape), rope_attrs={k: str(v)[:80] for k, v in vars(rope).items()
                                                                    if not k.startswith("_") and not torch.is_tensor(v)})
        except Exception:
            r["error"] = traceback.format_exc()[-1500:]
        out.append(r)
    return out


@app.local_entrypoint()
def main():
    import json
    for r in run.remote(int(os.environ.get("KEEP", "0"))):
        print(json.dumps(r, indent=1))
