"""Fit tap maps (kvx/hybrid.py) for pairs the KV-only pipeline cannot handle (hybrid GatedDeltaNet targets, transformers-5
architectures) and measure next-token KL vs native prefill.  kvx-pairs seat.

KVX_GPU=B200 modal run modal_fit_hybrid.py::main --src Qwen/Qwen3.5-4B --tgt Qwen/Qwen3.8-27B --run-id pairs-q35-4b-q38-27b
"""
import json, os, time

import modal

GPU = os.environ.get("KVX_GPU", "B200")
GPU_SPEC = GPU.split(",") if "," in GPU else GPU
APP = os.environ.get("KVX_APP", "kvx-pairs-hybrid")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.0", "transformers==5.24.0", "accelerate", "datasets==3.6.0", "numpy", "hf_transfer",
                 "flash-linear-attention", "einops", "kernels")
    .env({"HF_HOME": "/vol/hf", **{k: os.environ[k] for k in ("KVX_VOLUME", "KVX_WEIGHTS_VOLUME") if k in os.environ}})  # datasets cache writable on the seat volume; weights via snapshot(cache_dir=/hf/hub)
    .add_local_python_source("kvx")
)
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"), create_if_missing=True)
hf_shared = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
app = modal.App(APP, image=image)
EVAL_DOCS = (20000, 30000)  # same reserved eval range as modal_fit_eval.py


def _commit():
    try:
        vol.commit()
    except Exception:
        pass


def _bos(tok):
    return tok.bos_token_id is not None and "qwen" not in tok.name_or_path.lower()


def load_texts(tok, n, L, skip_docs=0, exclude=None):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    out = []
    for i, ex in enumerate(ds):
        if i < skip_docs or (exclude and exclude[0] <= i < exclude[1]):
            continue
        ids = tok(ex["text"], add_special_tokens=False)["input_ids"]
        if _bos(tok):
            ids = [tok.bos_token_id] + ids
        if len(ids) >= L:
            out.append(ids[:L])
            if len(out) >= n:
                break
    return out


def load_pg19_train(tok, n, L, per_book=32):
    from datasets import load_dataset
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    out = []
    for ex in ds:
        ids = tok(ex["text"][: L * per_book * 6], add_special_tokens=False)["input_ids"]
        if _bos(tok):
            ids = [tok.bos_token_id] + ids
        for c in range(min(per_book, len(ids) // L)):
            out.append(ids[c * L:(c + 1) * L])
            if len(out) >= n:
                return out
    return out


def cached(tok, tag, build):
    import hashlib, torch
    h = hashlib.sha1((tok.name_or_path.split("/")[-1].split("-")[0] + str(len(tok))).encode()).hexdigest()[:8]
    p = f"/vol/data/{tag}_{h}.pt"
    if os.path.exists(p):
        return torch.load(p)
    out = build()
    os.makedirs("/vol/data", exist_ok=True)
    torch.save(out, p)
    _commit()
    return out


def snapshot(name):
    """Local snapshot dir on the shared volume. Resolved from refs/ directly: huggingface_hub >= 1.x refuses offline
    snapshots missing repo metadata files (.gitattributes, README), which some staged copies lack."""
    rd = f"/hf/hub/models--{name.replace('/', '--')}"
    ref = f"{rd}/refs/main"
    if os.path.exists(ref):
        d = f"{rd}/snapshots/{open(ref).read().strip()}"
        if os.path.exists(f"{d}/config.json"):
            return d
    snaps = sorted(os.listdir(f"{rd}/snapshots")) if os.path.isdir(f"{rd}/snapshots") else []
    for h in snaps:
        if os.path.exists(f"{rd}/snapshots/{h}/config.json"):
            return f"{rd}/snapshots/{h}"
    raise RuntimeError(f"{name} not on the shared weights volume (stage it with ~/rsi-weights/stage.py)")


def fit_impl(src: str, tgt: str, run_id: str, n_calib: int = 4000, L: int = 1024, w: int = 3, drop_q: int = 1,
             lam_rels: str = "1e-3", n_eval: int = 64, P: int = 960, sinks: str = "4", tails: str = "64,256",
             bs: int = 4, mem_frac: float = 0.80, save: int = 1, oracle: int = 1, stride: int = 4, tf32: int = 0,
             split_kv: int = 0, maps_from: str = "", ebs: int = 2, eval_l: int = 0, tgt_device_map: str = "",
             gpu0_gib: int = 60, per_gpu_gib: int = 130):
    import torch
    from transformers import AutoTokenizer
    from kvx import hybrid
    t0 = time.time()
    # tf32=1 makes the Gram GEMMs ~8x faster but its product rounding is not PSD-preserving: the 4B->27B Gram was not
    # positive definite at lam_rel 1e-3 (2026-11-22). Default fp32.
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    hf_shared.reload()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(snapshot(tgt))
    tok.name_or_path = tgt
    S = hybrid.load(src, path=snapshot(src))
    if tgt_device_map:  # target too big for one GPU (GLM-5.3): shard it; GPU 0 keeps room for the source + stats
        n = torch.cuda.device_count()
        mm = {i: f"{gpu0_gib if i == 0 else per_gpu_gib}GiB" for i in range(n)}
        T = hybrid.load(tgt, path=snapshot(tgt), device_map=tgt_device_map, max_memory=mm)
        T.model.requires_grad_(False)
    else:
        T = hybrid.load(tgt, path=snapshot(tgt))
        T.model.to(dev).requires_grad_(False)
    S.model.to(dev).requires_grad_(False)
    bos = int(_bos(tok))
    half = n_calib // 2
    calib = torch.cat([torch.tensor(cached(tok, f"calib_{half}x{L}_bos{bos}", lambda: load_texts(tok, half, L, exclude=EVAL_DOCS))),
                       torch.tensor(cached(tok, f"pg19train_{half}x{L}_bos{bos}", lambda: load_pg19_train(tok, half, L)))])
    EL = eval_l or L  # eval length (prefix P + continuation); calibration stays at L
    evals = torch.tensor(cached(tok, f"eval_{n_eval}x{EL}_bos{bos}", lambda: load_texts(tok, n_eval, EL, skip_docs=20000)))
    res = dict(src=src, tgt=tgt, run_id=run_id, n_calib=len(calib), L=L, eval_l=EL, w=w, drop_q=drop_q, P=P, n_eval=len(evals), stride=stride, tf32=tf32,
               gpu=torch.cuda.get_device_name() if dev == "cuda" else "cpu", t_load=time.time() - t0)
    print(f"[{time.time()-t0:.0f}s] loaded; calib {tuple(calib.shape)}", flush=True)

    kinds_s = [k for k, _ in hybrid.layer_taps(S)]
    kinds_t = [k for k, _ in hybrid.layer_taps(T)]
    widths = hybrid.tap_widths(T)
    Ls, Lt = len(kinds_s), len(kinds_t)
    us, ut = hybrid.units(S, bool(split_kv)), hybrid.units(T, bool(split_kv))
    win = hybrid.unit_windows(us, ut, Ls, Lt, w)
    ugroups = hybrid.unit_groups(T, ut)
    # source unit widths from one tiny forward
    f0 = hybrid.unit_feats(S, hybrid.source_features(S, calib[:1, :8].to(dev), drop_q=bool(drop_q)), us, bool(drop_q))
    ds = [0 if f is None else f.shape[1] for f in f0]
    dt = [sum(x for _, x in g) for g in ugroups]
    din = lambda j: sum(ds[i] for i in win[j])
    # passes: group target LAYERS so that xy (per target unit) + xx (per distinct window) fit the free memory budget
    free = (torch.cuda.mem_get_info()[0] * mem_frac - 12e9) if dev == "cuda" else 16e9  # headroom for activations / solves
    groups, cur, cur_w, cur_b = [], [], set(), 0.0
    for t in range(Lt):
        js = [j for j, u in enumerate(ut) if u[0] == t and u[1] != "none"]
        nw = {win[j] for j in js} - cur_w
        b = sum(4 * din(j) * dt[j] for j in js) + sum(4 * sum(ds[i] for i in wv) ** 2 for wv in nw)
        if cur and cur_b + b > free:
            groups.append(cur); cur, cur_w, cur_b = [], set(), 0.0
            b = sum(4 * din(j) * dt[j] for j in js) + sum(4 * sum(ds[i] for i in wv) ** 2 for wv in {win[j] for j in js})
        cur.append(t); cur_w |= {win[j] for j in js}; cur_b += b
    groups.append(cur)
    res.update(Ls=Ls, Lt=Lt, kinds_s=kinds_s, kinds_t=kinds_t, split_kv=split_kv, units_t=ut,
               windows={j: list(win[j]) for j in range(len(ut))}, ds=ds, dt=dt, passes=len(groups), free_gb=free / 1e9)
    print(f"[{time.time()-t0:.0f}s] {Lt} target layers ({len(ut)} units) in {len(groups)} passes; din max "
          f"{max(din(j) for j in range(len(ut)))}", flush=True)

    lams = [float(x) for x in lam_rels.split(",")]
    maps_by_lam = {lam: {} for lam in lams}
    A_ = max([int(x) for x in sinks.split(",")] + [0]) if sinks else 0
    tmpl = {}
    if maps_from:  # eval only: reuse a saved fit (no calibration); one file per lambda
        groups = []
        for lam in lams:
            tag = "" if lam == lams[0] else f"_lam{lam:g}"
            fn = f"/vol/runs/{maps_from}/{'hybrid_units' if split_kv else 'hybrid_maps'}{tag}.pt"
            d = torch.load(fn, map_location="cpu")
            maps_by_lam[lam] = {int(j): {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in m.items()}
                                for j, m in d["maps"].items()}
            if not tmpl and d.get("sink_template"):
                tmpl = d["sink_template"]
        save = 0
        res["maps_from"] = maps_from
    for gi, grp in enumerate(groups):
        tg = time.time()
        batches = (calib[i:i + bs].to(dev) for i in range(0, len(calib), bs))
        rr = hybrid.calibrate(S, T, batches, w=w, targets=grp, drop_q=bool(drop_q), device=dev, stride=stride,
                              split_kv=bool(split_kv))
        rr.offload()  # stats to host RAM; solve streams one window at a time (the in-GPU solve OOMed on 4B->27B)
        for i_l, lam in enumerate(lams):
            maps_by_lam[lam].update(rr.solve(lam_rel=lam, groups=ugroups, recolor=True, device=dev,
                                             free=(i_l == len(lams) - 1)))
        del rr
        torch.cuda.empty_cache()
        print(f"[{time.time()-t0:.0f}s] pass {gi+1}/{len(groups)} ({len(grp)} layers) {time.time()-tg:.0f}s", flush=True)
    res["t_fit"] = time.time() - t0 - res["t_load"]
    lam0 = lams[0]
    res["r2"] = {lam: {t: m["r2"] for t, m in maps_by_lam[lam].items()} for lam in lams}

    for lam in lams:  # keep maps on the GPU in bf16 (as saved and deployed): fp32 x 2 lambdas OOMed the 27B eval
        for m in maps_by_lam[lam].values():
            m["W"] = m["W"].to(dev, torch.bfloat16)
            for k in ("b", "mu", "rc"):
                if torch.is_tensor(m.get(k)):
                    m[k] = m[k].to(dev)
    torch.cuda.empty_cache() if dev == "cuda" else None
    if A_ and not tmpl:  # sink template: calibration mean of the target taps at positions < A
        acc, n = {}, 0
        for i in range(0, min(len(calib), 512), 16):
            got = {}
            with hybrid.Taps(T).record(lambda l, d: got.__setitem__(l, d)):
                T.forward(calib[i:i + 16, :A_].to(dev), keep=1, use_cache=False)
            for l, d in got.items():
                for k, v in d.items():
                    acc.setdefault(l, {}).setdefault(k, 0)
                    acc[l][k] = acc[l][k] + v.float().sum(0)
            n += min(16, len(calib) - i)
        tmpl = {l: {k: (v / n).cpu() for k, v in d.items()} for l, d in acc.items()}

    if save:
        os.makedirs(f"/vol/runs/{run_id}", exist_ok=True)
        for lam in lams:
            tag = "" if lam == lam0 else f"_lam{lam:g}"
            if split_kv:  # unit-keyed maps; attention-only targets also get modal_fit_eval's maps_k format
                torch.save(dict(format="kvx-hybrid-units-v1", src=src, tgt=tgt, lam_rel=lam, w=w, units_s=us, units_t=ut,
                                maps={j: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in m.items()}
                                      for j, m in maps_by_lam[lam].items()}, sinks=A_, sink_template=tmpl),
                           f"/vol/runs/{run_id}/hybrid_units{tag}.pt")
                if set(kinds_t) <= {"attn"}:
                    cm = hybrid.to_core_maps(maps_by_lam[lam], us, ut, T)
                    cm.update(src=src, tgt=tgt, k_layers=w, feat="kv", restore="rms" if hybrid.layer_taps(T)[0][1][0][2] else "none")
                    torch.save(cm, f"/vol/runs/{run_id}/maps_core{tag}.pt")
            else:
                out = hybrid.maps_file(S, T, maps_by_lam[lam], src=src, tgt=tgt, lam_rel=lam, w=w, drop_q=bool(drop_q),
                                       sinks=A_, sink_template=tmpl, n_calib=len(calib),
                                       calib="mix fineweb-edu+pg19-train 1024")
                torch.save(out, f"/vol/runs/{run_id}/hybrid_maps{tag}.pt")
        _commit()
        print(f"[{time.time()-t0:.0f}s] saved maps", flush=True)

    # ---- eval: KL(native || mapped) on the 64-token continuation (same protocol as modal_fit_eval.py)
    acc = {}

    def add(name, logits, lp, top):
        logits = logits.to(lp.device)
        lq = torch.log_softmax(logits.float(), -1)
        kl = (lp.exp() * (lp - lq)).sum(-1)
        a = acc.setdefault(name, dict(kl=0.0, kl_first=0.0, agree=0.0, n=0))
        a["kl"] += kl.mean(1).sum().item(); a["kl_first"] += kl[:, 0].sum().item()
        a["agree"] += (logits.argmax(-1) == top).float().mean(1).sum().item(); a["n"] += kl.shape[0]

    tail_list = [int(x) for x in tails.split(",")] if tails else []
    sink_list = [int(x) for x in sinks.split(",")] if sinks else []
    C = EL - P
    for i in range(0, len(evals), ebs):  # small eval batches: mapped taps for all target layers are ~2 GB/doc
        ids = evals[i:i + ebs].to(dev)
        B = ids.shape[0]
        lg_t, _ = T.forward(ids, keep=C, use_cache=False)
        lp = torch.log_softmax(lg_t.float(), -1)
        top = lg_t.argmax(-1)
        sc = lambda name, lg: add(name, lg, lp, top)
        sc("ref_nocontext", T.forward(ids[:, P:], use_cache=False)[0])
        _, c_nat = T.forward(ids[:, :P], keep=1)  # floor: native prefix cache + continuation (DSA is not two-step exact)
        sc("ref_native_twostep", T.forward(ids[:, P:], cache=c_nat)[0])
        del c_nat
        sc("ref_source_model", S.forward(ids, keep=C, use_cache=False)[0])
        feats = hybrid.source_features(S, ids[:, :P], drop_q=bool(drop_q))
        for lam in lams:
            lt = "" if lam == lam0 else f"_lam{lam:g}"
            given = hybrid.apply_units(maps_by_lam[lam], feats, B, P, T, us, ut, drop_q=bool(drop_q), src=S,
                                       out_dtype=torch.bfloat16)
            for rest in ("none", "rms"):
                g = hybrid.restore_rms({t: dict(d) for t, d in given.items()}, T) if rest == "rms" else given
                c = hybrid.mapped_cache(T, ids[:, :P], g)
                sc(f"map_{rest}{lt}", T.forward(ids[:, P:], cache=c)[0])
                del c
                if rest != "rms" or lam != lam0:
                    continue
                for A in sink_list:
                    for N in [0] + tail_list:
                        c = hybrid.mapped_cache(T, ids[:, :P - N], g, sinks=A)
                        sc(f"map_rms_sink{A}_tail{N}", T.forward(ids[:, P - N:], cache=c)[0][:, N:])
                        del c
                        if tmpl:  # deployable: sink positions from the calibration template, no target prefill
                            gt = {t: {k: torch.cat([tmpl[t][k][:A].to(dev)[None].expand(B, -1, -1), v[:, A:]], 1)
                                      for k, v in d.items()} for t, d in g.items()}
                            c = hybrid.mapped_cache(T, ids[:, :P - N], gt, sinks=0)
                            sc(f"map_rms_tmpl{A}_tail{N}", T.forward(ids[:, P - N:], cache=c)[0][:, N:])
                            del c, gt
                for N in tail_list:
                    c = hybrid.mapped_cache(T, ids[:, :P - N], g)
                    sc(f"map_rms_tail{N}", T.forward(ids[:, P - N:], cache=c)[0][:, N:])
                    del c
                # recolor (whitened-Procrustes-style covariance match per 128-channel block; kvx-research lever)
                grc = hybrid.restore_rms({t: dict(d) for t, d in
                                          hybrid.apply_units(maps_by_lam[lam], feats, B, P, T, us, ut, recolor=True,
                                                             drop_q=bool(drop_q), src=S, out_dtype=torch.bfloat16).items()}, T)
                sc("map_rc_rms", T.forward(ids[:, P:], cache=hybrid.mapped_cache(T, ids[:, :P], grc))[0])
                if tmpl and sink_list:
                    A = sink_list[0]
                    for N in tail_list:
                        gt = {t: {k: torch.cat([tmpl[t][k][:A].to(dev)[None].expand(B, -1, -1), v[:, A:]], 1)
                                  for k, v in d.items()} for t, d in grc.items()}
                        c = hybrid.mapped_cache(T, ids[:, :P - N], gt)
                        sc(f"map_rc_rms_tmpl{A}_tail{N}", T.forward(ids[:, P - N:], cache=c)[0][:, N:])
                        del c, gt
                del grc
                if oracle:  # which layer kind carries the error: exact taps for one kind, mapped for the other
                    exact = {}
                    with hybrid.Taps(T).record(lambda l, d: exact.__setitem__(l, {k: v.clone() for k, v in d.items()})):
                        T.forward(ids[:, :P], keep=1, use_cache=False)
                    sc("oracle_roundtrip", T.forward(ids[:, P:], cache=hybrid.mapped_cache(T, ids[:, :P], exact))[0])
                    mix_a = {t: (exact[t] if kinds_t[t] == "attn" else g[t]) for t in g}
                    sc("oracle_exactattn_maplin", T.forward(ids[:, P:], cache=hybrid.mapped_cache(T, ids[:, :P], mix_a))[0])
                    mix_l = {t: (exact[t] if kinds_t[t] == "lin" else g[t]) for t in g}
                    sc("oracle_exactlin_mapattn", T.forward(ids[:, P:], cache=hybrid.mapped_cache(T, ids[:, :P], mix_l))[0])
                    del exact, mix_a, mix_l
            del given
        del feats
    res["eval"] = {k: {m: v[m] / v["n"] for m in ("kl", "kl_first", "agree")} for k, v in acc.items()}
    res["t_total"] = time.time() - t0
    os.makedirs(f"/vol/runs/{run_id}", exist_ok=True)
    json.dump(res, open(f"/vol/runs/{run_id}/result.json", "w"), indent=1, default=str)
    _commit()
    return json.loads(json.dumps(res, default=str))


def fit_safe(**kw):
    try:
        return fit_impl(**kw)
    except Exception:
        import traceback
        raise RuntimeError(traceback.format_exc()[-3000:]) from None


fit = app.function(gpu=GPU_SPEC, volumes={"/vol": vol, "/hf": hf_shared.read_only()}, timeout=3300, memory=131072)(fit_safe)


@app.local_entrypoint()
def main(src: str = "Qwen/Qwen3.5-4B", tgt: str = "Qwen/Qwen3.8-27B", run_id: str = "", n_calib: int = 4000,
         w: int = 3, drop_q: int = 1, lam_rels: str = "1e-3", n_eval: int = 64, sinks: str = "4", tails: str = "64,256",
         bs: int = 4, oracle: int = 1, save: int = 1, stride: int = 4, tf32: int = 0, split_kv: int = 0,
         maps_from: str = "", prefix: int = 960, eval_l: int = 0, ebs: int = 2, tgt_device_map: str = "",
         gpu0_gib: int = 60, per_gpu_gib: int = 130):
    run_id = run_id or time.strftime("pairs-%Y%m%dT%H%M%SZ", time.gmtime())
    r = fit.remote(src=src, tgt=tgt, run_id=run_id, n_calib=n_calib, w=w, drop_q=drop_q, lam_rels=lam_rels,
                   n_eval=n_eval, sinks=sinks, tails=tails, bs=bs, oracle=oracle, save=save, stride=stride, tf32=tf32, split_kv=split_kv,
                   maps_from=maps_from, P=prefix, eval_l=eval_l, ebs=ebs, tgt_device_map=tgt_device_map,
                   gpu0_gib=gpu0_gib, per_gpu_gib=per_gpu_gib)
    os.makedirs(f"out/{run_id}", exist_ok=True)
    json.dump(r, open(f"out/{run_id}/result.json", "w"), indent=1)
    print(json.dumps(r.get("eval", r), indent=1))

