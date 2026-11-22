"""Use src/xkv map files (Mapper + AttnMap) with HF models and the kvx-hybrid-v1 tap interface.

xkv predicts the same per-token target quantities as kvx-hybrid-v1 (attention: k post k_norm / pre-RoPE and v;
GatedDeltaNet: pre-conv q|k|v, a, b), so anything that consumes kvx-hybrid-v1 taps (hybrid.mapped_cache, a vLLM
state writer) can run on xkv maps. Only the source side differs:

  feature x_i  = the source layer i's input_layernorm output (normalized residual stream entering layer i), bias row
                 appended ([x_i, 1] @ W, bias in W's last row)
  DeltaNet j:   y = [x_i, 1] @ W[i][:, slices[i][j]] -> k | v | a | b;  q = [x_i, 1] @ Wq[j] (needed only for the
                 conv state, i.e. the last kernel-1 rows), with i = src_of[j]
  attention j:  if an AttnMap is given: y = b + sum_{i in layers[j]} x_i @ parts[j][i] + emb @ e + emb_prev @ p
                 (+ optional residual MLP), emb = source token embeddings, emb_prev = previous token's (0 at pos 0);
                 else from W like DeltaNet layers. y -> k | v.
"""
import torch

from . import hybrid


def load_xkv(mapper_path, attn_path=None, device="cpu"):
    m = torch.load(mapper_path, map_location=device)
    out = dict(W=m["W"], Wq=m["Wq"], slices=m["slices"], src_of={int(j): int(i) for j, i in m["src_of"].items()})
    if attn_path:
        a = torch.load(attn_path, map_location=device)
        out["attn"] = dict(layers={int(j): list(ls) for j, ls in a["layers"].items()}, embed=a["embed"],
                           parts={int(j): {(int(k) if str(k).lstrip("-").isdigit() else k): v for k, v in d.items()}
                                  for j, d in a["parts"].items()},
                           mlp=a.get("mlp"))
    return out


def needed_layers(xm):
    need = set(xm["src_of"].values())
    if "attn" in xm:
        need |= {i for ls in xm["attn"]["layers"].values() for i in ls}
    return sorted(need)


@torch.no_grad()
def source_inputs(src, ids, layers, on_feat=None):
    """{i: [B, T, h]} input_layernorm outputs of source layers `layers`, plus (emb, emb_prev) [B, T, h].
    on_feat(i, x): streaming; each layer's output goes to the callback as the forward reaches it and is not kept
    (feats is returned empty), so long prompts never hold every needed layer at once. emb/emb_prev come first."""
    emb = src.text.get_input_embeddings()(ids)
    emb_prev = torch.cat([torch.zeros_like(emb[:, :1]), emb[:, :-1]], 1)
    feats, hs = {}, []
    put = on_feat if on_feat is not None else feats.__setitem__
    for i in layers:
        hs.append(src.layers[i].input_layernorm.register_forward_hook(
            lambda mod, inp, out, i=i: put(i, out.detach())))
    try:
        src.forward(ids, keep=1, use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return feats, emb, emb_prev


def _mm(x, W):
    W = W.to(device=x.device, dtype=x.dtype)
    return torch.addmm(W[-1], x.reshape(-1, x.shape[-1]), W[:-1]).view(*x.shape[:-1], W.shape[1])


@torch.no_grad()
def mapped_taps(xm, feats, emb, emb_prev, tgt, out_dtype=torch.bfloat16):
    """kvx-hybrid-v1 taps {target layer: {tap: [B, T, w]}} from xkv maps and source inputs."""
    kinds = [k for k, _ in hybrid.layer_taps(tgt)]
    c = tgt.cfg
    out = {}
    attn = xm.get("attn")
    for i, sl in xm["slices"].items():
        i = int(i)
        Y = _mm(feats[i], xm["W"][i])
        for j, (a0, a1) in sl.items():
            j = int(j)
            y = Y[..., a0:a1]
            if kinds[j] == "lin":
                kd = c.linear_key_head_dim * c.linear_num_key_heads
                vd = c.linear_value_head_dim * c.linear_num_value_heads
                nv = c.linear_num_value_heads
                q = _mm(feats[i], xm["Wq"][j])
                out[j] = {"qkv": torch.cat([q, y[..., :kd + vd]], -1).to(out_dtype),
                          "a": y[..., kd + vd:kd + vd + nv].to(out_dtype),
                          "b": y[..., kd + vd + nv:kd + vd + 2 * nv].to(out_dtype)}
            elif attn is None:
                h = y.shape[-1] // 2
                out[j] = {"k": y[..., :h].to(out_dtype), "v": y[..., h:].to(out_dtype)}
    if attn is not None:
        for j, ls in attn["layers"].items():
            d = attn["parts"][j]
            dt = emb.dtype
            y = d["b"].to(device=emb.device, dtype=dt).expand(*emb.shape[:-1], -1).clone()
            for i in ls:
                y = y + feats[i].to(dt) @ d[i].to(device=emb.device, dtype=dt)
            if attn["embed"]:
                y = y + emb @ d["e"].to(device=emb.device, dtype=dt) + emb_prev @ d["p"].to(device=emb.device, dtype=dt)
            mlp = attn.get("mlp")
            if mlp and j in mlp:
                m = mlp[j]
                f = torch.cat([feats[i].to(dt) for i in ls] + ([emb, emb_prev] if attn["embed"] else []), -1)
                y = y + torch.nn.functional.gelu(f @ m["w1"].to(f) + m["b1"].to(f)) @ m["w2"].to(f)
            h = y.shape[-1] // 2
            out[j] = {"k": y[..., :h].to(out_dtype), "v": y[..., h:].to(out_dtype)}
    return out


def xkv_given(src, tgt, xm, ids, out_dtype=torch.bfloat16):
    """One call for a mapped prefix: source forward -> kvx-hybrid-v1 taps for hybrid.mapped_cache."""
    feats, emb, emb_prev = source_inputs(src, ids, needed_layers(xm))
    return mapped_taps(xm, feats, emb, emb_prev, tgt, out_dtype=out_dtype)
