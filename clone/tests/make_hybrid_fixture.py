"""Write tests/fixtures/hybrid_fixture.pt: tiny Qwen3.5-style source/target (configs + weights), a kvx-hybrid-v1 maps
file fit on random tokens, prompt ids, the mapped taps, and HF's reference cache after the mapped prefix.
For kvx-quality's vLLM state writer: from (maps, source taps) it must reproduce ref_cache exactly.
Run: PYTHONPATH=<transformers 5>:. python tests/make_hybrid_fixture.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_hybrid_cpu import _tiny  # noqa: E402
from kvx import hybrid  # noqa: E402


def cache_dump(dec, cache):
    out = {}
    for l, (kind, _) in enumerate(hybrid.layer_taps(dec)):
        lay = cache.layers[l]
        if kind == "attn":
            out[l] = dict(kind="attn", K_post_rope=lay.keys.clone(), V=lay.values.clone())
        else:
            out[l] = dict(kind="lin", conv_state=lay.conv_states[0].clone(), recurrent_state=lay.recurrent_states[0].clone())
    return out


def main(path):
    S, T = _tiny(4, layers=4, hidden=48, vh=2), _tiny(5, layers=8, hidden=64, vh=4)
    torch.manual_seed(0)
    batches = [torch.randint(0, 101, (4, 48)) for _ in range(8)]
    rr = hybrid.calibrate(S, T, batches, w=2, device="cpu")
    widths = hybrid.tap_widths(T)
    maps = rr.solve(lam_rel=1e-3, groups=widths)
    A = 2
    tmpl = {}
    got = {}
    with hybrid.Taps(T).record(lambda l, d: got.__setitem__(l, d)):
        T.forward(torch.cat(batches)[:, :A], keep=1, use_cache=False)
    tmpl = {l: {k: v.float().mean(0) for k, v in d.items()} for l, d in got.items()}
    mf = hybrid.maps_file(S, T, maps, src="tiny-src", tgt="tiny-tgt", lam_rel=1e-3, w=2, drop_q=True, sinks=A,
                          sink_template=tmpl)
    ids = torch.randint(0, 101, (1, 40))
    P = 30
    feats = hybrid.source_features(S, ids[:, :P])
    fmaps = {t: dict(sel=m["sel"], W=m["W"].float(), b=m["b"]) for t, m in mf["maps"].items()}  # as stored (bf16 W)
    given = hybrid.restore_rms(hybrid.apply(fmaps, feats, 1, P, widths), T)
    given = {t: {k: torch.cat([tmpl[t][k][:A][None], v[:, A:]], 1) for k, v in d.items()} for t, d in given.items()}
    cache = hybrid.mapped_cache(T, ids[:, :P], given)
    ref = cache_dump(T, cache)
    logits, _ = T.forward(ids[:, P:], cache=cache)
    torch.save(dict(
        note="mapped taps = restore_rms(apply(maps, source_features(src, ids[:, :P]))), positions < sinks from "
             "sink_template; ref_cache = HF cache after the target prefix pass with those taps; cont_logits = target "
             "logits for ids[:, P:] on that cache",
        src_config=S.cfg.to_dict(), tgt_config=T.cfg.to_dict(),
        src_state=S.model.state_dict(), tgt_state=T.model.state_dict(),
        maps_file=mf, ids=ids, P=P, source_feats=[f.clone() for f in feats], mapped_taps=given,
        ref_cache=ref, cont_logits=logits), path)
    print("wrote", path, os.path.getsize(path) // 1024, "KB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "fixtures", "hybrid_fixture.pt"))
