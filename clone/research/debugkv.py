"""Debug of run mc128-gpu-f (B200): native-K/V arms were at chance (garbage native cache) while the mapped arm was sane.
Checks, per prompt length: target's own captured cache -> build_cache -> scored continuation == native forward
(oracle roundtrip); capture with an explicit DynamicCache vs the default cache; K/V stats. No fit."""
import json, os, time

import torch

from kvx import core
from research import mc128


@torch.no_grad()
def run_pair(src, tgt, out="", models=None, **_):
    from transformers import AutoTokenizer
    from kvx.models import model_path
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(model_path(tgt))
    S, T = models if models else (mc128.load(src, dev, torch.bfloat16), mc128.load(tgt, dev, torch.bfloat16))
    P = mc128.Pair(S, T, tok, dev, Lmax=4096)
    res = dict(src=src, tgt=tgt, gpu=torch.cuda.get_device_name(), torch=str(torch.__version__))
    items = mc128.mc_items(tok, 4)
    rows = []
    for it in items[:8]:
        c = tok(it["ctx"], add_special_tokens=False)["input_ids"]
        es = [tok(e, add_special_tokens=False)["input_ids"] for e in it["conts"]]
        pre = torch.tensor([c[:-1]], device=dev)
        Kn, Vn = P.native(pre)
        Ks, Vs = P.src_kv(pre)
        out_d = T(input_ids=pre, use_cache=True)
        kd, vd = core.cache_layers(out_d.past_key_values)[0]
        c0, s0 = core._cs(P.ct, P.st, 0, pre.shape[1])
        k0 = core.unrope(kd.float(), c0, s0)
        nat = mc128.score_choices(P, c, es, None)
        orc = mc128.score_choices(P, c, es, (Kn, Vn))
        Km, Vm = P.mapped(Ks, Vs, list(P.maps)[0][1], "none") if P.maps else (None, None)
        rows.append(dict(T=pre.shape[1], native=[round(x, 3) for x in nat], oracle=[round(x, 3) for x in orc],
                         k0_vs_default=float((k0 - Kn[0]).abs().max()), v0_vs_default=float((vd.float() - Vn[0]).abs().max()),
                         Kn_absmax=float(Kn.abs().max()), Vn_absmax=float(Vn.abs().max()), Kn_nan=bool(torch.isnan(Kn).any()),
                         Ks_absmax=float(Ks.float().abs().max())))
        print(json.dumps(rows[-1]), flush=True)
    res["rows"] = rows
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        json.dump(res, open(out, "w"), indent=1)
    return res
