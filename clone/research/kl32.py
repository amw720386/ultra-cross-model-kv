"""KL32 replication for OneTriangle notes 005 / 006 / 007 (sibling and cross-model transfer).

Their metric: KL32 = mean KL(native target || mapped-cache target) over 32 oracle-forced target steps after the
cached prefix, plus 32-step top-1 agreement. 006: 1,024-token raw-token prefix, 28 frozen WikiText windows;
007: 256-token prefix, 64 untouched WikiText-103 prompts. Ours: same windows from wikitext-103-raw-v1 test,
forced tokens = the native target's own greedy continuation (the "oracle"), and also the text's own tokens
(teacher forcing) as a second reading. Arms: direct injection (nearest-depth copy), ridge k (each fitted k),
ridge + native sink/tail recipe, native K + mapped V, mapped K + native V.
"""
import json, math, os, time

import torch

from kvx import core
from research import mc128


def direct_maps(P):
    """Identity maps from the nearest-depth source layer (= direct injection) in the kvx map format, so distillation
    can start from direct reuse (their note 006/007 base) instead of ridge. Needs equal KV head shapes."""
    assert (P.Hs, P.Ds) == (P.Ht, P.Dt)
    d = P.Hs * P.Ds
    out = {}
    for kind in ("k", "v"):
        ms = []
        for t in range(P.Lt):
            s_ = min(P.Ls - 1, round(t * (P.Ls - 1) / max(1, P.Lt - 1)))
            ms.append(dict(sel=[s_], W=torch.eye(d, device=P.dev), b=torch.zeros(d, device=P.dev), H=P.Ht, D=P.Dt,
                           mu=torch.zeros(d, device=P.dev), r2=torch.ones(P.Ht, device=P.dev), vnat=torch.ones(P.Ht, device=P.dev)))
        out[kind] = ms
    return out


def wikitext_windows(tok, n, L, seed_skip=0):
    """n windows of L tokens from wikitext-103-raw-v1 test (concatenated article text, non-overlapping)."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    text = "".join(ds["text"])
    ids = tok(text, add_special_tokens=False)["input_ids"]
    out = [ids[i:i + L] for i in range(seed_skip, len(ids) - L, L)][:n]
    return torch.tensor(out)


@torch.no_grad()
def kl32(P, wins, prefix, steps=32, ks=(), recipes=(), restore="rms", forced="greedy"):
    acc = {}

    def add(name, lg, lp_ref, top):
        lp = torch.log_softmax(lg.float(), -1)
        kl = (lp_ref.exp() * (lp_ref - lp)).sum(-1)  # [1, steps]
        a = acc.setdefault(name, dict(kl=[], agree=[]))
        a["kl"].append(kl.mean().item())
        a["agree"].append((lg.argmax(-1) == top).float().mean().item())

    for i in range(len(wins)):
        ids = wins[i:i + 1, :prefix].to(P.dev)
        if forced == "greedy":  # native target greedy continuation = the oracle-forced tokens
            g = P.T.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=steps, do_sample=False,
                             pad_token_id=P.tok.pad_token_id or 0)
            cont = g[:, prefix:prefix + steps]
        else:
            cont = wins[i:i + 1, prefix:prefix + steps].to(P.dev)
        # logits for the `steps` forced positions: last prefix token + cont[:-1] predict cont
        full = torch.cat([ids, cont], 1)
        lg_ref = P.T(input_ids=full).logits[:, prefix - 1:prefix - 1 + steps].float()
        lp_ref = torch.log_softmax(lg_ref, -1)
        top = lg_ref.argmax(-1)
        add("native", lg_ref, lp_ref, top)
        tail_in = torch.cat([ids[:, -1:], cont[:, :-1]], 1)  # target runs the last prefix token natively
        pre = ids[:, :-1]
        Ks, Vs = P.src_kv(pre)
        run = lambda K, V: P.T(input_ids=tail_in, past_key_values=core.build_cache(K, V, P.ct, P.st)).logits
        if (P.Hs, P.Ds) == (P.Ht, P.Dt):
            add("direct", run(*P.direct(Ks, Vs)), lp_ref, top)
        Kn, Vn = P.native(pre)
        for k in ks:
            Km, Vm = P.mapped(Ks, Vs, k, restore)
            add(f"ridge_k{k}", run(Km, Vm), lp_ref, top)
            if k == ks[-1]:
                add(f"nk_mv_k{k}", run(Kn, Vm), lp_ref, top)
                add(f"mk_nv_k{k}", run(Km, Vn), lp_ref, top)
            for r in recipes:  # native sink A + native tail N (tail recomputed by the target), mapped middle
                A_, N_ = (int(x) for x in r[1:].split("t"))
                Kr, Vr = Km[:, :, :, :pre.shape[1] - N_ + 1].clone(), Vm[:, :, :, :pre.shape[1] - N_ + 1].clone()
                if A_:
                    Kr[:, :, :, :A_], Vr[:, :, :, :A_] = Kn[:, :, :, :A_], Vn[:, :, :, :A_]
                tin = torch.cat([ids[:, -N_:], cont[:, :-1]], 1)
                lg = P.T(input_ids=tin, past_key_values=core.build_cache(Kr, Vr, P.ct, P.st)).logits[:, N_ - 1:]
                add(f"ridge_k{k}_{r}", lg, lp_ref, top)
    out = {}
    for n_, a in acc.items():
        kl = torch.tensor(a["kl"])
        out[n_] = dict(kl32=kl.mean().item(), kl32_se=(kl.std() / math.sqrt(len(kl))).item() if len(kl) > 1 else None,
                       agree=sum(a["agree"]) / len(a["agree"]), n=len(kl), per_window=[round(x, 6) for x in a["kl"]])
    return out


def run_pair(src, tgt, n_calib=500, L=1024, ks="1,8,16,24", windows=28, prefix=1024, steps=32, recipes="s4t64",
             calib_src="mix", restore="rms", lam_rel=1e-3, stride=4, out="", models=None, one_pass=0, mc=0, dev=None,
             forced="greedy,text", distill_steps=0, distill_lrs="1e-5,3e-5,1e-4", distill_k=0, distill_n=512,
             distill_init="ridge"):
    from transformers import AutoTokenizer
    from kvx.models import model_path
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_path(tgt))
    S, T = models if models else (mc128.load(src, dev, torch.bfloat16), mc128.load(tgt, dev, torch.bfloat16))
    P = mc128.Pair(S, T, tok, dev, Lmax=max(L, prefix + steps + 64))
    klist = [min(int(k), P.Ls) for k in str(ks).split(",")]
    calib = mc128.calibration(tok, n_calib, L, calib_src)
    bos = tok.bos_token_id if (tok.bos_token_id is not None and "qwen" not in tgt.lower()) else None
    if bos is not None:  # Llama/Mistral: position 0 is the attention sink; calibrate and evaluate BOS-first
        calib[:, 0] = bos
    res = dict(src=src, tgt=tgt, n_calib=n_calib, L=L, ks=klist, calib_src=calib_src, restore=restore,
               windows=windows, prefix=prefix, steps=steps, t_load=time.time() - t0)
    t1 = time.time()
    res["r2"] = P.fit(calib, klist, lam_rel, stride, one_pass=bool(one_pass))
    res["t_fit"] = time.time() - t1
    print(f"[{time.time()-t0:.0f}s] fit {res['r2']}", flush=True)
    if distill_steps:  # KV-Lingo-style KL distillation of the k map (research/kvdistill.py); evaluated as k "d"
        from research import kvdistill
        dk = distill_k or klist[-1]
        if distill_init == "direct":  # start from direct injection (identity, nearest depth); no key restore on top
            dm = direct_maps(P)
            P.maps[("k", "dir")], P.maps[("v", "dir")] = dm["k"], dm["v"]
            dk = "dir"
        C = 64
        tr = mc128.fineweb(tok, distill_n, prefix + C, skip=0, stop=mc128.EVAL_SKIP)
        va = mc128.fineweb(tok, 16, prefix + C, skip=mc128.EVAL_SKIP + 5000)
        if bos is not None:
            tr[:, 0] = bos; va[:, 0] = bos
        mk, mv, rep = kvdistill.distill(P, dk, tr, va, prefix, [float(x) for x in distill_lrs.split(",")],
                                         distill_steps, 4, "none" if dk == "dir" else restore, log=lambda m: print(m, flush=True))
        P.maps[("k", "d")], P.maps[("v", "d")] = mk, mv
        if dk == "dir":
            P.norestore = {"dir", "d"}
        res["distill"] = rep
        res["distill_init"] = distill_init
        klist = klist + ["d"]
    wins = wikitext_windows(tok, windows, prefix + steps)
    if bos is not None:
        wins[:, 0] = bos
    res["bos"] = bos
    for f in forced.split(","):
        res[f"kl32_{f}"] = kl32(P, wins, prefix, steps, klist, [r for r in recipes.split(",") if r], restore, f)
        print(f"[{time.time()-t0:.0f}s] kl32 {f} " + json.dumps({a: (round(v['kl32'], 5), round(v['agree'], 4)) for a, v in res[f'kl32_{f}'].items()}), flush=True)
    if mc:
        items = mc128.mc_items(tok, 32)
        kbest = min(klist, key=lambda k: res["kl32_greedy"][f"ridge_k{k}"]["kl32"])
        res["mc_k"] = kbest
        arms = ["native", "mapped", "nk_mv", "mk_nv", "wrong", "source"] + (["direct"] if (P.Hs, P.Ds) == (P.Ht, P.Dt) else [])
        res["mc128"], _ = mc128.run_mc(P, items, kbest, arms, restore=restore)
    res["t_total"] = time.time() - t0
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        json.dump(res, open(out, "w"), indent=1)
    return res
