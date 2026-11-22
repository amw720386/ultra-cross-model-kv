"""Windowed-source quality gate for GLM-4.7-Flash -> GLM-5.3 at long context (HF, one 8-GPU box).

Two arms, same maps, same prompts, same boot:
  default: source prefill with full causal attention (our default config);
  window:  source prefill with a sliding window of --src-window tokens (kvx/fa.py vllm_fa, window_size passed).
Both arms: mapped cache = native sinks (first --sinks tokens run natively) + mapped rows for the rest of the prefix,
written straight into the HF cache (kvx.glm_rows: kv_a_layernorm latent + RoPE key + DSA indexer key), so the target
never prefills the long prefix; the target then runs the last --tail prompt tokens natively and is scored.

Prompts (--prompts, built by tools/build_glm_gate_prompts.py): a list of dicts
  {"kind": "long", "ids": [...], "cont": [64 real next tokens]}         -> teacher-forced NLL and top-1 on cont
  {"kind": "needle", "ids": [...], "answer": "1234567", "depth": 0.35} -> greedy answer contains the number
Pass rule (lead, fixed before any numbers): window is within noise of default on all of
  needle hits: window >= default - 1 (of the needles run); mean top-1: window >= default - .05; NLL: window <= default + .05.
--budget-min stops early (complete=false, no verdict) and still exits 0.

    python -m kvx.glm_gate --maps-from pairs-glm47f-glm53 --prompts runs/glm-gate/prompts.pt --src-window 8192 \
        --out runs/glm-gate/result.json --budget-min 55
    python -m kvx.glm_gate --selfcheck 1      (CPU: tiny random GLM models, whole flow + writer-vs-override oracle)
"""
import argparse
import json
import os
import sys
import time

PASS = dict(needle_hits_drop=1, top1_drop=0.05, nll_rise=0.05)
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(f"[{time.strftime('%H:%M:%S', time.gmtime())}Z] {s}")
    print(LOG[-1], flush=True)


def mem(stage):
    """Log host memory (RSS, MemAvailable, page cache, cgroup usage) and /tmp disk free at a stage."""
    import shutil
    info = {}
    try:
        st = open("/proc/self/status").read()
        info["rss_gb"] = int(st.split("VmRSS:")[1].split()[0]) / 1e6
        mi = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if ":" in l}
        info.update(avail_gb=mi.get("MemAvailable", 0) / 1e6, cached_gb=mi.get("Cached", 0) / 1e6)
        for f in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
            if os.path.exists(f):
                info["cgroup_gb"] = int(open(f).read()) / 1e9
                break
        info["tmp_free_gb"] = shutil.disk_usage(SPILL if os.path.isdir(SPILL) else "/tmp").free / 1e9
    except Exception as e:
        info["err"] = repr(e)[:100]
    log("mem", stage, json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in info.items()}))
    return info


SPILL = "/tmp/kvx_gate_feats"


def ensure_pkgs():
    """accelerate (device_map) and kernels (HF FP8) may be missing in the serving image: install into /tmp, no deps."""
    import importlib.util
    import subprocess
    miss = [p for p in ("accelerate", "kernels") if importlib.util.find_spec(p) is None]
    if miss:
        d = "/tmp/kvx_gate_pkgs"
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", d] + miss, check=True)
        sys.path.insert(0, d)
        log("installed", miss, "->", d)
    return miss


def snapshot(name, root="/hf/hub"):
    rd = f"{root}/models--{name.replace('/', '--')}"
    ref = f"{rd}/refs/main"
    if os.path.exists(ref):
        d = f"{rd}/snapshots/{open(ref).read().strip()}"
        if os.path.exists(f"{d}/config.json"):
            return d
    for h in sorted(os.listdir(f"{rd}/snapshots")):
        if os.path.exists(f"{rd}/snapshots/{h}/config.json"):
            return f"{rd}/snapshots/{h}"
    raise FileNotFoundError(rd)


# ---------------------------------------------------------------- source side

class FAProbe:
    """Wraps kvx.fa's kernel lookup and records the window_size each source attention call really passed."""

    def __init__(self):
        from kvx import fa
        self.fa = fa
        self.reset()
        orig = fa._fa_func

        def wrapped():
            fn, ver = orig()

            def f(*a, **k):
                self.calls += 1
                w = tuple(k.get("window_size") or (-1, -1))
                self.windows[w] = self.windows.get(w, 0) + 1
                W = self.fa.WINDOW
                if W and int(k.get("max_seqlen_k", 0)) > W:
                    self.over += 1
                if w != (-1, -1):
                    self.windowed += 1
                return fn(*a, **k)
            return f, ver
        fa._fa_func = wrapped

    def reset(self):
        self.calls, self.windows, self.over, self.windowed = 0, {}, 0, 0
        if hasattr(self.fa, "reset_stats"):
            self.fa.reset_stats()

    def summary(self):
        return dict(attn_impl=self.fa.NAME, fa_WINDOW=self.fa.WINDOW, calls=self.calls, windowed_calls=self.windowed,
                    calls_over_window=self.over, window_size_counts={str(k): v for k, v in self.windows.items()},
                    fa_STATS=dict(getattr(self.fa, "STATS", {})))

    def proof(self, W):
        """Criterion from kvx/fa.py STATS (agreed): WINDOW arm: windowed_calls == calls_over_window, calls_over_window > 0,
        window_sizes == [[W-1, 0]]; DEFAULT arm: windowed_calls == 0 and sliding_layer_calls == 0 (GLM-4.7-Flash has no
        sliding layers of its own). The wrapper's own counts must agree with STATS (cross-check)."""
        st = getattr(self.fa, "STATS", None)
        if st is None or "calls_over_window" not in st:
            return dict(ok=False, rule="kvx/fa.py STATS with calls_over_window required", stats=st)
        if W:
            ok = (st["windowed_calls"] == st["calls_over_window"] and st["calls_over_window"] > 0
                  and st["window_sizes"] == [[W - 1, 0]])
            rule = "windowed_calls == calls_over_window AND calls_over_window > 0 AND window_sizes == [[W-1, 0]]"
        else:
            ok = st["windowed_calls"] == 0 and st["sliding_layer_calls"] == 0
            rule = "windowed_calls == 0 AND sliding_layer_calls == 0"
        cross = dict(calls=self.calls == st["calls"], windowed_calls=self.windowed == st["windowed_calls"],
                     calls_over_window=(self.over == st["calls_over_window"]) if W else True)
        return dict(ok=bool(ok and all(cross.values())), rule=rule, stats=dict(st), wrapper_agrees=cross)


def source_feats(S, ids, chunk, emb=True):
    """Chunked source prefill with a cache; returns per-layer features [T, d] (bf16, on CPU) + the emb unit."""
    import torch
    from kvx import hybrid
    from transformers import DynamicCache
    n = len(hybrid.layer_taps(S))
    parts = [[] for _ in range(n)]
    taps_s = hybrid.layer_taps(S)

    def on_layer(l, d):
        x = torch.cat([d[nm] for nm, _, _ in taps_s[l][1]], -1)
        parts[l].append(x.reshape(-1, x.shape[-1]).to("cpu", torch.bfloat16))
    cache = DynamicCache(config=S.cfg)
    T = ids.shape[1]
    with torch.no_grad(), hybrid.Taps(S).record(on_layer):
        for a in range(0, T, chunk):
            S.forward(ids[:, a:a + chunk].to(S.device0), cache=cache, keep=1, use_cache=True)
    del cache
    out = [torch.cat(p) for p in parts]
    if emb:
        with torch.no_grad():
            E = S.text.get_input_embeddings()(ids.to(S.device0))
            Ep = torch.cat([torch.zeros_like(E[:, :1]), E[:, :-1]], 1)
            out.append(torch.cat([E, Ep], -1).reshape(-1, 2 * E.shape[-1]).to("cpu", torch.bfloat16))
    return out


# ---------------------------------------------------------------- target side

def write_cache(Tm, ids, feats, maps, ut, widths, sinks, chunk, dev0):
    """HF cache for ids[:, :P]: native forward over the first `sinks` tokens, then mapped rows for positions sinks..P-1
    appended layer by layer (no target prefill over the prefix)."""
    import torch
    from kvx import glm_rows as G
    P = ids.shape[1]
    _, cache = Tm.forward(ids[:, :sinks].to(Tm.device0), keep=1)
    for a in range(sinks, P, chunk):
        b = min(P, a + chunk)
        cos, sin = G.cos_sin(Tm, torch.arange(a, b, device=dev0))
        fs = [None if f is None else f[a:b].to(dev0, non_blocking=True) for f in feats]
        for j, (t, kind, _) in enumerate(ut):
            if kind != "mla" or j not in maps:
                continue
            m = maps[j]
            X = torch.cat([fs[s] for s in m["sel"]], 1)
            Y = (X @ m["W"]).float() + m["b"].float()
            d, o = {}, 0
            for name, wd in widths[t]:
                d[name] = Y[:, o:o + wd]
                o += wd
            attn = Tm.layers[t].self_attn
            ldev = next(attn.parameters()).device
            cdt = attn.kv_a_layernorm.weight.dtype  # the model's own activation / cache dtype (bf16 on GLM-5.3)
            with torch.no_grad():
                r = G.layer_rows(attn, d["ckv"].to(ldev, cdt), None if "ik" not in d else d["ik"].to(ldev, cdt),
                                 cos.to(ldev), sin.to(ldev), layout="hf")
            R = attn.kv_lora_rank
            cache.update(r["mla"][None, None, :, :R].contiguous(), r["mla"][None, None, :, R:].contiguous(), t)
            if r["idx"] is not None:
                cache.update_indexer(r["idx"][None].contiguous(), t)
            del X, Y, d, r
    return cache


def run_tail(Tm, cache, ids, step):
    """Run ids (the native tail, plus scored tokens) on the cache in chunks; returns logits [1, n, V] on CPU (fp32)."""
    import torch
    out = []
    with torch.no_grad():
        for a in range(0, ids.shape[1], step):
            lg, cache = Tm.forward(ids[:, a:a + step].to(Tm.device0), cache=cache)
            out.append(lg.float().cpu())
    return torch.cat(out, 1), cache


def greedy(Tm, cache, last_logits, n):
    import torch
    toks, lg = [], last_logits
    with torch.no_grad():
        for _ in range(n):
            nxt = lg[:, -1:].argmax(-1)
            toks.append(int(nxt))
            lg, cache = Tm.forward(nxt.to(Tm.device0), cache=cache)
    return toks


def verdict(arms, n_needles):
    d, w = arms["default"], arms["window"]
    checks = {
        "needles": dict(default=d["needle_hits"], window=w["needle_hits"], rule=f"window >= default - {PASS['needle_hits_drop']}",
                        ok=w["needle_hits"] >= d["needle_hits"] - PASS["needle_hits_drop"]),
        "top1": dict(default=d["top1"], window=w["top1"], rule=f"window >= default - {PASS['top1_drop']}",
                     ok=w["top1"] is not None and d["top1"] is not None and w["top1"] >= d["top1"] - PASS["top1_drop"]),
        "nll": dict(default=d["nll"], window=w["nll"], rule=f"window <= default + {PASS['nll_rise']}",
                    ok=w["nll"] is not None and d["nll"] is not None and w["nll"] <= d["nll"] + PASS["nll_rise"]),
    }
    return dict(passed=all(c["ok"] for c in checks.values()), checks=checks, n_needles=n_needles)


def gate(a):
    import torch
    from kvx import hybrid
    t0 = time.time()
    budget = a.budget_min * 60
    vol_root = a.vol_root
    out_path = f"{vol_root}/{a.out}"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    res = dict(args=vars(a), pass_rule=PASS, complete=False, arms={}, prompts=[])

    def save():
        res["t_total_s"] = time.time() - t0
        res["log"] = LOG[-400:]
        json.dump(res, open(out_path, "w"), indent=1, default=str)
        open(out_path.replace(".json", "") + ".log.txt", "w").write("\n".join(LOG) + "\n")
        if a.commit:
            try:
                import modal
                modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx")).commit()
            except Exception as e:  # inside a Modal container the mounted volume also commits on exit
                log("volume commit skipped:", repr(e)[:200])

    prompts = torch.load(f"{vol_root}/{a.prompts}") if not a.selfcheck else a._prompts
    if a.max_long or a.max_needles:
        L = [p for p in prompts if p["kind"] == "long"][:a.max_long or None]
        N = [p for p in prompts if p["kind"] == "needle"][:a.max_needles or None]
        prompts = [x for pair in zip(L, N) for x in pair] + L[len(N):] + N[len(L):]
    log(f"{len(prompts)} prompts:", [(p['kind'], len(p['ids'])) for p in prompts])
    dev0 = "cuda:0" if torch.cuda.is_available() else "cpu"
    tok = None

    # ---- 1. source features, both arms, then free the source
    ts = time.time()
    S = hybrid.load(a.src, path=a._src_path or snapshot(a.src), attn="sdpa")
    S.model.to(dev0).requires_grad_(False)
    probe = None
    if not a.selfcheck:
        from kvx import fa
        probe = FAProbe()
    log(f"source loaded {time.time()-ts:.0f}s")
    feats = {}  # (arm, prompt) -> spill file on local disk (one prompt-arm in RAM at a time; ~13 GB each at 202K)
    os.makedirs(SPILL, exist_ok=True)
    mem("source loaded")
    arms = [("default", 0), ("window", a.src_window)]
    for arm, W in arms:
        if probe is not None:
            fa.use_vllm_fa(S.model, window=W)
            probe.reset()
        ta = time.time()
        for i, p in enumerate(prompts):
            Pm = len(p["ids"]) - a.tail
            ids = torch.tensor([p["ids"][:Pm]])
            f = source_feats(S, ids, a.src_chunk if (probe is not None) else 10 ** 9, emb=True)
            feats[(arm, i)] = f"{SPILL}/{arm}_{i}.pt"
            torch.save(f, feats[(arm, i)])
            del f
            if arm == "default" and i == 0:  # disk check from the first real file: all prompt-arms must fit
                import shutil
                need = os.path.getsize(feats[(arm, i)]) * (2 * len(prompts) - 1) * 1.05
                free = shutil.disk_usage(SPILL).free
                log(f"spill: {os.path.getsize(feats[(arm, i)]) / 1e9:.1f} GB per prompt-arm, need {need / 1e9:.0f} GB more, free {free / 1e9:.0f} GB")
                if free < need:
                    save()
                    raise SystemExit(f"local disk too small for the feature spill: free {free / 1e9:.0f} GB < {need / 1e9:.0f} GB")
        mem(f"{arm} source features spilled")
        src_info = dict(probe.summary(), proof=probe.proof(W)) if probe is not None else dict(attn_impl="sdpa (selfcheck, no window)")
        res["arms"].setdefault(arm, {})["source_attention"] = dict(src_window_arg=W, **src_info)
        res["arms"][arm]["t_source_s"] = time.time() - ta
        log(arm, "source", f"{time.time()-ta:.0f}s", json.dumps(src_info))
        if probe is not None and not src_info["proof"]["ok"]:
            save()
            raise SystemExit(f"{arm} arm source attention proof failed: {json.dumps(src_info)}")
    del S
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    save()

    # ---- 2. target + maps
    tt = time.time()
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        Tm = hybrid.load(a.tgt, path=snapshot(a.tgt), device_map="balanced",
                         max_memory={i: f"{a.per_gpu_gib}GiB" for i in range(n)})
    else:
        Tm = hybrid.load(a.tgt, path=a._tgt_path, attn="sdpa")
    Tm.model.requires_grad_(False)
    res["t_target_load_s"] = time.time() - tt
    if res["t_target_load_s"] > 20 * 60:
        res["note_target_load"] = f"target load took {res['t_target_load_s'] / 60:.1f} min (> 20): page cache likely evicted"
        log(res["note_target_load"])
    mem("target loaded")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a._tgt_path or snapshot(a.tgt))
    d = torch.load(f"{vol_root}/runs/{a.maps_from}/hybrid_maps.pt", map_location="cpu")
    maps = {int(j): dict(sel=list(m["sel"]), W=m["W"].to(dev0, torch.bfloat16), b=m["b"].to(dev0))
            for j, m in d["maps"].items()}
    ut = hybrid.units(Tm, False)
    widths = hybrid.tap_widths(Tm)
    log(f"target + maps loaded {time.time()-tt:.0f}s ({len(maps)} units)")
    save()

    # ---- 3. score
    acc = {arm: dict(needle_hits=0, needles=0, nll_sum=0.0, top1_sum=0, n_tok=0, preds={}) for arm, _ in arms}
    for i, p in enumerate(prompts):
        if time.time() - t0 > budget:
            log(f"budget {a.budget_min} min reached before prompt {i}: stopping (complete=false)")
            break
        Pm = len(p["ids"]) - a.tail
        full = torch.tensor([p["ids"]])
        rec = dict(i=i, kind=p["kind"], len=len(p["ids"]), depth=p.get("depth"))
        for arm, _ in arms:
            tp = time.time()
            fx = torch.load(feats[(arm, i)])
            cache = write_cache(Tm, full[:, :Pm], fx, maps, ut, widths, a.sinks, a.map_chunk, dev0)
            del fx
            os.remove(feats[(arm, i)])
            if p["kind"] == "long":
                cont = torch.tensor([p["cont"]])
                lg, cache = run_tail(Tm, cache, torch.cat([full[:, Pm:], cont[:, :-1]], 1), a.tail_step)
                lg = lg[0, a.tail - 1:]  # predicts cont[0..]
                lp = torch.log_softmax(lg, -1)
                y = cont[0]
                nll = float(-lp.gather(1, y[:, None]).mean())
                pred = lg.argmax(-1)
                top1 = float((pred == y).float().mean())
                A = acc[arm]
                A["nll_sum"] += nll * len(y); A["top1_sum"] += float((pred == y).sum()); A["n_tok"] += len(y)
                A["preds"][i] = pred.tolist()
                rec[arm] = dict(nll=nll, top1=top1)
            else:
                lg, cache = run_tail(Tm, cache, full[:, Pm:], a.tail_step)
                toks = greedy(Tm, cache, lg[:, -1:].to(Tm.device0), a.gen)
                text = tok.decode(toks)
                hit = p["answer"] in text
                acc[arm]["needle_hits"] += int(hit); acc[arm]["needles"] += 1
                rec[arm] = dict(hit=hit, text=text)
            rec[arm]["t_s"] = time.time() - tp
            del cache
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if p["kind"] == "long":
            pd, pw = acc["default"]["preds"][i], acc["window"]["preds"][i]
            rec["agree_window_default"] = sum(x == y for x, y in zip(pd, pw)) / len(pd)
        res["prompts"].append(rec)
        log("prompt", i, json.dumps({k: v for k, v in rec.items() if k != "i"})[:400])
        mem(f"after prompt {i}")
        save()
    done = len(res["prompts"]) == len(prompts)
    for arm, _ in arms:
        A = acc[arm]
        res["arms"][arm].update(needle_hits=A["needle_hits"], needles=A["needles"],
                                nll=A["nll_sum"] / A["n_tok"] if A["n_tok"] else None,
                                top1=A["top1_sum"] / A["n_tok"] if A["n_tok"] else None, n_tok=A["n_tok"])
    ag = [r["agree_window_default"] for r in res["prompts"] if "agree_window_default" in r]
    res["agree_window_default"] = sum(ag) / len(ag) if ag else None
    res["complete"] = done
    res["verdict"] = verdict(res["arms"], acc["default"]["needles"]) if done else "none (incomplete: budget stop)"
    log("verdict", json.dumps(res["verdict"], default=str)[:600])
    save()
    return res


def _tiny_glm(kind, seed, layers=4):
    """Tiny random GLM-MoE-DSA (target) / GLM-4.7-Flash-style (source) decoder (same as tests/test_hybrid_cpu.py)."""
    import torch
    from kvx import hybrid
    torch.manual_seed(seed)
    common = dict(vocab_size=101, intermediate_size=128, num_hidden_layers=layers, num_attention_heads=4,
                  num_key_value_heads=4, qk_rope_head_dim=8, qk_nope_head_dim=16, v_head_dim=16, n_routed_experts=4,
                  num_experts_per_tok=2, n_shared_experts=1, first_k_dense_replace=1, max_position_embeddings=256)
    if kind == "dsa":
        from transformers.models.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
        m = GlmMoeDsaForCausalLM(GlmMoeDsaConfig(hidden_size=64, moe_intermediate_size=32, kv_lora_rank=32, q_lora_rank=48,
                                                 index_n_heads=2, index_head_dim=16, index_topk=8, **common)).eval()
    else:
        from transformers.models.glm4_moe_lite import Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM
        common.update(intermediate_size=96)
        m = Glm4MoeLiteForCausalLM(Glm4MoeLiteConfig(hidden_size=48, moe_intermediate_size=24, kv_lora_rank=24,
                                                     q_lora_rank=32, **common)).eval()
    m.config._attn_implementation = "eager"
    return hybrid.Dec(m)


def selfcheck():
    """CPU: tiny random GLM-4.7-Flash-like source and GLM-5.3-like (MLA + DSA) target, random maps; checks the direct
    cache writer against the override-forward path (kvx.hybrid.mapped_cache) and runs the whole gate flow."""
    import tempfile
    import torch
    from kvx import hybrid
    import transformers.models.glm_moe_dsa  # noqa: F401  (the serving image's transformers must have it)
    S, T = _tiny_glm("lite", 1, layers=3), _tiny_glm("dsa", 2, layers=4)
    root = tempfile.mkdtemp()
    S.model.save_pretrained(f"{root}/src"); T.model.save_pretrained(f"{root}/tgt")
    batches = [torch.randint(3, 101, (2, 32)) for _ in range(4)]
    us, ut = hybrid.units(S, False, emb=True), hybrid.units(T, False)
    rr = hybrid.calibrate(S, T, batches, w=0, device="cpu", emb=True)
    maps = rr.solve(lam_rel=1e-3, groups=hybrid.unit_groups(T, ut))
    # oracle: direct writer == override forward (same mapped taps, same native sinks)
    ids = torch.randint(3, 101, (1, 40))
    P, A = 30, 4
    feats = source_feats(S, ids[:, :P], chunk=7, emb=True)
    f_ref = hybrid.source_features(S, ids[:, :P], emb=True)
    assert all(torch.allclose(x.float(), y.float(), atol=2e-2) for x, y in zip(feats, f_ref) if y is not None), "chunked source feats"
    m16 = {j: dict(sel=m["sel"], W=m["W"].to(torch.bfloat16).float(), b=m["b"].float()) for j, m in maps.items()}
    widths = hybrid.tap_widths(T)
    c1 = write_cache(T, ids[:, :P], [f.float() for f in feats], m16, ut, widths, A, 9, "cpu")
    l1, _ = T.forward(ids[:, P:], cache=c1)
    given = hybrid.apply_units(m16, [f.float() for f in feats], 1, P, T, us, ut, src=S)
    c2 = hybrid.mapped_cache(T, ids[:, :P], given, sinks=A)
    l2, _ = T.forward(ids[:, P:], cache=c2)
    err = float((l1 - l2).abs().max())
    cerr = {}
    for l in range(len(c1.layers)):
        x1, x2 = c1.layers[l], c2.layers[l]
        for nm in ("keys", "values", "indexer_keys"):
            u, v = getattr(x1, nm, None), getattr(x2, nm, None)
            if u is not None or v is not None:
                assert u is not None and v is not None and u.shape == v.shape, (l, nm)
                cerr[nm] = max(cerr.get(nm, 0.0), float((u - v).abs().max()))
    log("selfcheck writer vs override forward: cache max |d|", json.dumps({k: f"{v:.1e}" for k, v in cerr.items()}),
        "| max |dlogits|", f"{err:.2e}")
    assert max(cerr.values()) < 1e-4, cerr  # the rows themselves (DSA top-k can amplify tiny row noise in the logits)
    assert err < 5e-3, err
    # window path through kvx/fa.py with a reference varlen kernel (CPU): window features == banded-mask features,
    # and the STATS proof holds for both arms
    from kvx import fa

    def ref_kernel(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal,
                   window_size=None, **kw):
        Tq, Sk = q.shape[0], k.shape[0]
        sc = torch.einsum("thd,shd->hts", q.float(), k.float()) * softmax_scale
        i = torch.arange(Tq)[:, None] + (Sk - Tq)
        j = torch.arange(Sk)[None]
        ok = j <= i
        if window_size is not None and window_size[0] >= 0:
            ok &= (i - j) <= window_size[0]
        sc = sc.masked_fill(~ok[None], float("-inf"))
        return torch.einsum("hts,shd->thd", sc.softmax(-1), v.float()).to(q.dtype)
    fa._fa_func = lambda: (ref_kernel, None)
    probe = FAProbe()
    Wn = 8
    ids_w = torch.randint(3, 101, (1, 30))
    fa.use_vllm_fa(S.model, window=Wn)
    probe.reset()
    fw = source_feats(S, ids_w, chunk=7, emb=False)
    pw = probe.proof(Wn)
    fa.use_vllm_fa(S.model, window=0)
    probe.reset()
    source_feats(S, ids_w, chunk=7, emb=False)
    pd = probe.proof(0)
    log("selfcheck window proof", json.dumps(pw["stats"]), pw["ok"], "| default", json.dumps(pd["stats"]), pd["ok"])
    assert pw["ok"] and pd["ok"], (pw, pd)
    S.model.set_attn_implementation("eager") if hasattr(S.model, "set_attn_implementation") else None
    S.model.config._attn_implementation = "eager"
    band = torch.zeros(30, 30).masked_fill(~((torch.arange(30)[None] <= torch.arange(30)[:, None]) &
                                             (torch.arange(30)[:, None] - torch.arange(30)[None] < Wn)), float("-inf"))
    got = {}
    with torch.no_grad(), hybrid.Taps(S).record(lambda l, d: got.__setitem__(l, torch.cat([d[n] for n, _, _ in hybrid.layer_taps(S)[l][1]], -1)[0])):
        S.text(input_ids=ids_w, attention_mask=band[None, None], use_cache=False)
    werr = max(float((fw[l].float() - got[l].float()).abs().max()) for l in got)
    log("selfcheck window feats vs banded-mask reference: max |d|", f"{werr:.2e}")
    assert werr < 2e-2, werr
    run = f"{root}/vol/runs/selfmaps"
    os.makedirs(run)
    torch.save(dict(maps={j: dict(sel=m["sel"], W=m["W"].to(torch.bfloat16), b=m["b"]) for j, m in maps.items()}),
               f"{run}/hybrid_maps.pt")

    class Tok:
        def decode(self, t):
            return " ".join(map(str, t))
    import transformers
    transformers.AutoTokenizer.from_pretrained = staticmethod(lambda *x, **k: Tok())
    prompts = [dict(kind="long", ids=torch.randint(3, 101, (60,)).tolist(), cont=torch.randint(3, 101, (8,)).tolist()),
               dict(kind="needle", ids=torch.randint(3, 101, (60,)).tolist(), answer="7", depth=0.5)]
    ns = argparse.Namespace(src="tiny/src", tgt="tiny/tgt", maps_from="selfmaps", prompts="", src_window=16,
                            out="runs/selfgate/result.json", budget_min=30, sinks=A, tail=12, tail_step=5, gen=4,
                            map_chunk=11, src_chunk=10 ** 9, per_gpu_gib=1, vol_root=f"{root}/vol", commit=0,
                            max_long=0, max_needles=0, selfcheck=1, _prompts=prompts, _src_path=f"{root}/src",
                            _tgt_path=f"{root}/tgt")
    r = gate(ns)
    assert r["complete"] and isinstance(r["verdict"], dict), r.get("verdict")
    log("SELFCHECK OK", json.dumps(r["verdict"]["checks"], default=str)[:300])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--src", default="zai-org/GLM-4.7-Flash")
    ap.add_argument("--tgt", default="zai-org/GLM-5.3")
    ap.add_argument("--maps-from", default="pairs-glm47f-glm53", help="run dir on the volume: runs/<id>/hybrid_maps.pt")
    ap.add_argument("--prompts", default="runs/glm-gate/prompts.pt", help="prompt file, relative to --vol-root")
    ap.add_argument("--src-window", type=int, default=8192)
    ap.add_argument("--out", default="runs/glm-gate/result.json", help="relative to --vol-root")
    ap.add_argument("--budget-min", type=float, default=55)
    ap.add_argument("--sinks", type=int, default=4)
    ap.add_argument("--tail", type=int, default=256, help="native tail (last prompt tokens run by the target)")
    ap.add_argument("--tail-step", type=int, default=64, help="target query chunk over the long cache")
    ap.add_argument("--gen", type=int, default=12, help="greedy tokens for needle answers")
    ap.add_argument("--map-chunk", type=int, default=16384, help="positions per map GEMM / cache append")
    ap.add_argument("--src-chunk", type=int, default=32768, help="source prefill chunk (FA, cache-exact)")
    ap.add_argument("--per-gpu-gib", type=int, default=120)
    ap.add_argument("--max-long", type=int, default=0)
    ap.add_argument("--max-needles", type=int, default=0)
    ap.add_argument("--vol-root", default="/vol")
    ap.add_argument("--commit", type=int, default=1)
    ap.add_argument("--selfcheck", type=int, default=0)
    a = ap.parse_args(argv)
    a._prompts = a._src_path = a._tgt_path = None
    if a.selfcheck:
        selfcheck()
        return 0
    ensure_pkgs()
    gate(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
