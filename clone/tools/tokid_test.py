"""Token-identity feature test (lead decision kvx-tokid-test): plain k16 vs k16 + token-embedding rows, one harness.

One calibration pass over the frozen long-cal set (Qwen3-1.7B -> Qwen3-8B, 1000 x 8,192 decontaminated, stride 4)
collects Gram stats for the source KV of every layer PLUS n_emb pseudo-layers holding [E(x_t) | E(x_{t-1})] (the
source token embeddings, scaled to the KV features' RMS, split into ds-wide blocks). From the SAME stats:
  plain: top-16 source layers by R^2 over the real layers only (= the frozen long-cal k16 recipe; checked against
         the saved maps_k16.pt)
  emb:   the same 16 layers + the forced embedding pseudo-layers
Both arms, same harness: FineWeb 1K KL (64 docs, P=960, rms restore = the modal_fit_eval eval set) and the h2h 30K
DEV needles (/vol/h2h/prompts-dev/needles_30720.jsonl, chat template, tail 1: every prompt token but the last mapped,
greedy 16 tokens). Digit miss = right code word, wrong digits. Pre-stated bar: emb >= 23/24 AND digit misses <= half
of plain's AND FineWeb KL <= plain * 1.05.

    cd clone && KVX_GPU='H100!' PYTHONPATH=. modal run tools/tokid_test.py::main
"""
import json
import os
import time

import modal

from modal_fit_eval import _cache_path, cached, hf_shared, image, load_texts, vol

GPU = os.environ.get("KVX_GPU", "H100!")
app = modal.App(os.environ.get("KVX_APP", "kvx-tokid-test"), image=image.add_local_python_source("modal_fit_eval"))
SRC, TGT = "Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B"


@app.function(gpu=GPU, timeout=3300, memory=98304, volumes={"/vol": vol, "/hf": hf_shared.read_only()})
def run(n_calib: int, k: int, stride: int, n_eval: int, needles: str, frozen_maps: str, limit: int = 0, out_name: str = "pairs-tokid-test"):
    import torch
    from transformers import AutoTokenizer
    from kvx import core
    from kvx.models import load_lm, model_path
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    dev = "cuda"
    res = dict(gpu=torch.cuda.get_device_name(), src=SRC, tgt=TGT, k=k, n_calib=n_calib, stride=stride)
    tok = AutoTokenizer.from_pretrained(model_path(TGT))
    tok.name_or_path = TGT
    S = load_lm(SRC).to(dev).eval()
    T = load_lm(TGT).to(dev).eval()
    S.requires_grad_(False); T.requires_grad_(False)
    L = 8192
    half = n_calib // 2
    calib = torch.cat([torch.tensor(torch.load(_cache_path(tok, f"calib_500x{L}_bos0"))[:half]),
                       torch.tensor(torch.load(_cache_path(tok, f"pg19train_500x{L}_bos0"))[:half])])
    hd = lambda c: getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    Hs, Ds, Ht, Dt = S.config.num_key_value_heads, hd(S.config), T.config.num_key_value_heads, hd(T.config)
    Ls, Lt = S.config.num_hidden_layers, T.config.num_hidden_layers
    ds = Hs * Ds
    hid = S.config.hidden_size
    n_emb = -(-2 * hid // ds)
    cs, ss = core.rope_cos_sin(S, 32768, dev)
    ct, st_ = core.rope_cos_sin(T, 32768, dev)
    emb_layer = S.get_input_embeddings()
    scale = {}

    def emb_rows(ids, kind, ref):
        """[n_emb, B, Hs, T, Ds] pseudo-layers of [E(x_t) | E(x_{t-1})], scaled to the KV feature RMS of `kind`."""
        E = emb_layer(ids).float()
        Ep = torch.cat([torch.zeros_like(E[:, :1]), E[:, :-1]], 1)
        X = torch.cat([E, Ep], -1)
        pad = n_emb * ds - X.shape[-1]
        if pad:
            X = torch.cat([X, X.new_zeros(*X.shape[:-1], pad)], -1)
        if kind not in scale:  # fixed once from the first calibration batch, stored with the maps
            scale[kind] = float(ref.float().pow(2).mean().sqrt() / X.pow(2).mean().sqrt().clamp(min=1e-12))
        X = X * scale[kind]
        B, Tn = ids.shape
        return X.view(B, Tn, n_emb, Hs, Ds).permute(2, 0, 3, 1, 4).to(torch.bfloat16)

    # ---- one calibration pass: stats over Ls + n_emb source "layers"
    sts = {kk: core.GramStats(Ls + n_emb, ds, Lt, Ht * Dt, dev) for kk in ("k", "v")}
    cbs = 1
    for i in range(0, len(calib), cbs):
        ids = calib[i:i + cbs].to(dev)
        with torch.no_grad():
            Ks, Vs, _ = core.capture_kv(S, ids, cs, ss)
            Kt, Vt, _ = core.capture_kv(T, ids, ct, st_)
            for kk, Xs, Yt in (("k", Ks, Kt), ("v", Vs, Vt)):
                Xe = torch.cat([Xs, emb_rows(ids, kk, Xs)], 0)[:, :, :, ::stride]
                X = core.flat_tokens(Xe).reshape(-1, (Ls + n_emb) * ds).float()
                Y = core.flat_tokens(Yt[:, :, :, ::stride]).reshape(-1, Lt * Ht * Dt).float()
                sts[kk].add(X, Y)
                del X, Y, Xe
        del Ks, Vs, Kt, Vt
    res["t_stats"] = time.time() - t0
    maps = {}
    force = list(range(Ls, Ls + n_emb))
    for kk in ("k", "v"):
        st = sts[kk]
        st.finalize()
        R2 = core.layer_r2(st, 0.01, Ht, 1e-3)
        maps[("plain", kk)] = core.fit_maps(st, R2, k, 0.01, Ht, 1e-3, n_real=Ls)
        maps[("emb", kk)] = core.fit_maps(st, R2, k, 0.01, Ht, 1e-3, force=force, n_real=Ls)
        res[f"r2_{kk}"] = {a: torch.stack([m["r2"] for m in maps[(a, kk)]]).mean().item() for a in ("plain", "emb")}
        del st
    sts.clear()
    torch.cuda.empty_cache()
    res.update(n_emb=n_emb, emb_scale=scale, t_fit=time.time() - t0)
    # sanity: plain arm == the frozen long-cal k16 maps (same data, stride, lambda, selection)
    if frozen_maps:
        fz = torch.load(f"/vol/{frozen_maps}", map_location="cpu")
        chk = {}
        for kk in ("k", "v"):
            same_sel = sum(list(a["sel"]) == list(b["sel"]) for a, b in zip(maps[("plain", kk)], fz[kk]))
            rel = max(float((a["W"].cpu() - b["W"]).norm() / b["W"].norm()) for a, b in zip(maps[("plain", kk)], fz[kk])
                      if list(a["sel"]) == list(b["sel"]))
            chk[kk] = dict(same_sel_layers=same_sel, of=len(fz[kk]), max_rel_W_diff=rel)
        res["plain_vs_frozen"] = chk
    kw = [l.self_attn.k_norm.weight.detach().to(dev) for l in T.model.layers]

    def mapped(arm, ids):
        Ks, Vs, _ = core.capture_kv(S, ids, cs, ss)
        if arm == "emb":
            Ks = torch.cat([Ks, emb_rows(ids, "k", Ks)], 0)
            Vs = torch.cat([Vs, emb_rows(ids, "v", Vs)], 0)
        Km = core.restore_scale(core.apply_maps(maps[(arm, "k")], Ks), maps[(arm, "k")], "rms", Ht, kw)
        Vm = core.apply_maps(maps[(arm, "v")], Vs)
        return Km.to(torch.bfloat16), Vm.to(torch.bfloat16)

    # ---- FineWeb 1K KL (modal_fit_eval's eval set: 64 x 1024 from doc 20000 on, P = 960)
    evals = torch.tensor(cached(tok, "eval_64x1024_bos0", lambda: load_texts(tok, 64, 1024, skip_docs=20000)))[:n_eval]
    P = 960
    kl = {a: [] for a in ("plain", "emb")}
    with torch.no_grad():
        for i in range(0, len(evals), 4):
            ids = evals[i:i + 4].to(dev)
            lp = torch.log_softmax(T(input_ids=ids, logits_to_keep=1024 - P).logits.float(), -1)
            for a in kl:
                Km, Vm = mapped(a, ids[:, :P])
                lg = T(input_ids=ids[:, P:], past_key_values=core.build_cache(Km.float(), Vm, ct, st_)).logits.float()
                kl[a].append(core.kl_rows(lp, lg).mean(1).cpu())
    res["fineweb_kl"] = {a: float(torch.cat(v).mean()) for a, v in kl.items()}
    # ---- 30K dev needles, tail 1
    rows = [json.loads(l) for l in open(f"/vol/{needles}")]
    rows = rows[:limit] if limit else rows
    out = {a: [] for a in ("plain", "emb")}
    with torch.no_grad():
        for r in rows:
            text = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]], device=dev)
            for a in out:
                Km, Vm = mapped(a, ids[:, :-1])
                cache = core.build_cache(Km.float(), Vm, ct, st_)
                o = T(input_ids=ids[:, -1:], past_key_values=cache, use_cache=True)
                gen = []
                for _ in range(16):
                    nxt = o.logits[:, -1].argmax(-1)
                    if int(nxt) in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")):
                        break
                    gen.append(int(nxt))
                    o = T(input_ids=nxt[:, None], past_key_values=o.past_key_values, use_cache=True)
                reply = tok.decode(gen)
                word = r["answer"].split("-")[0]
                ok = r["answer"] in reply
                out[a].append(dict(id=r["id"], answer=r["answer"], reply=reply, correct=ok,
                                   digit_miss=(not ok) and (word in reply), prompt_tokens=int(ids.shape[1])))
                del cache, o
    res["needles"] = {a: dict(correct=sum(x["correct"] for x in v), n=len(v), digit_miss=sum(x["digit_miss"] for x in v))
                      for a, v in out.items()}
    p, e = res["needles"]["plain"], res["needles"]["emb"]
    res["bar"] = dict(rule="emb >= 23/24 AND emb digit misses <= plain/2 AND emb FineWeb KL <= plain * 1.05",
                      needles_ok=e["correct"] >= 23, digits_ok=e["digit_miss"] <= p["digit_miss"] / 2,
                      kl_ok=res["fineweb_kl"]["emb"] <= res["fineweb_kl"]["plain"] * 1.05)
    res["bar"]["PASS"] = all(v for k_, v in res["bar"].items() if k_.endswith("_ok"))
    res["t_total"] = time.time() - t0
    os.makedirs(f"/vol/runs/{out_name}", exist_ok=True)
    json.dump(dict(res, replies=out), open(f"/vol/runs/{out_name}/result.json", "w"), indent=1)
    vol.commit()
    return res


@app.local_entrypoint()
def main(n_calib: int = 1000, k: int = 16, stride: int = 4, n_eval: int = 64,
         needles: str = "h2h/prompts-dev/needles_30720.jsonl",
         frozen_maps: str = "runs/pairs-q17-q8-cal8k/maps_all/maps_k16.pt", limit: int = 0, out_name: str = "pairs-tokid-test"):
    print(json.dumps(run.remote(n_calib, k, stride, n_eval, needles, frozen_maps, limit, out_name), indent=1, default=str))
