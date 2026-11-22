"""Turn research/mc128 + kl32 result files into the lead's ours.json rows (keyed by scoreboard id).

python research/build_ours.py RESULTS_DIR OURS_JSON
Rule for which run feeds a row (declared before the mcmix results were seen, 2026-11-22 03:35Z): per direction, the
newest run with calib_src mcmix, else mix, else fineweb. The headline value is always the STRICT arm (target runs only
the last prompt token, their protocol); recipe arms (native sink/tail) go into "note".
"""
import glob, json, os, random, sys

SIZE = {"Qwen3-0.6B": "0.6", "Qwen3-1.7B": "1.7", "Qwen3-4B": "4", "Qwen3-8B": "8", "Qwen3-14B": "14", "Qwen3-32B": "32"}
PREF = {"mcmix": 0, "mix": 1, "fineweb": 2}
RIG = {"mc128-cpu-a": "Modal CPU 32 cores, HF fp32", "mc128-gpu-b": "Modal H200, HF bf16", "mc128-gpu-a2": "Modal A100-80GB, HF bf16",
       "mc128-gpu-f": "Modal B200, HF bf16", "mc128-gpu-g": "Modal A100-80GB, HF bf16"}


# runs whose mixed K/V arms are under investigation (B200 run f: native-K / native-V arms at chance while mapped and
# native are sane) -> their diagnostic rows are withheld until the cause is found
SUSPECT_MIXED = {"mc128-gpu-f"}
WITHHELD = {"mc128-gpu-f"}  # lead 04:35Z: whole run withheld until the mixed-arm bug is found and the run repeated


def paired_ci(a, b, iters=2000, seed=0):
    """95% bootstrap CI of mean(a - b) in pp over paired items (task-stratified like theirs)."""
    rng = random.Random(seed)
    by = {}
    for x, y in zip(a, b):
        by.setdefault(x["task"], []).append(x["correct"] - y["correct"])
    ds = []
    for _ in range(iters):
        tot = n = 0
        for v in by.values():
            for _ in range(len(v)):
                tot += v[rng.randrange(len(v))]; n += 1
        ds.append(100 * tot / n)
    ds.sort()
    return ds[int(0.025 * iters)], ds[int(0.975 * iters)]


def load_runs(root):
    best = {}
    for f in glob.glob(f"{root}/*/*.json"):
        if f.endswith("_items.json"):
            continue
        r = json.load(open(f))
        if "mc128" not in r:
            continue
        s, t = r["src"].split("/")[-1], r["tgt"].split("/")[-1]
        if s not in SIZE or t not in SIZE:
            continue
        key = (SIZE[s], SIZE[t])
        rank = (PREF.get(r.get("calib_src", "fineweb"), 3), -os.path.getmtime(f))
        if key not in best or rank < best[key][0]:
            best[key] = (rank, f, r)
    return {k: (f, r) for k, (_, f, r) in best.items()}


KL32_RIG = {"kl32-gpu-c": "Modal H200, HF bf16", "kl32-gpu-d": "Modal A100-80GB, HF bf16", "kl32-gpu-e": "Modal B200, HF bf16"}
# scoreboard id -> which of our arms answers it: "ridge" = best ridge k, "best" = best strict arm (ridge/direct/distilled)
KL32_IDS = {
    ("Qwen3-4B-Instruct-2507", "Qwen3-4B-Thinking-2507"): {
        "x-af-kl32-ridge": "ridge", "x-af-kl32-base-ridge": "ridge", "x-af-fresh-kl32-mapped_ridge": "ridge",
        "x-af-kl32-ridge-resid": "best", "x-af-fresh-kl32-mapped_ridge+rank2_quadratic": "best",
        "x-af-kl32-r8-direct-after": "best", "x-af-kl32-quad-direct": "best"},
    ("Qwen3-30B-A3B-Instruct-2507", "Qwen3-30B-A3B-Thinking-2507"): {
        "x-af-30b-kl32-ridge": "ridge", "x-30b-kl-ridge": "ridge", "x-30b-top1-ridge": "ridge:agree",
        "x-30b-kl-direct_plus_residual": "best", "x-30b-top1-direct_plus_residual": "best:agree",
        "x-af-30b-kl32-ridge-resid": "best", "x-30b-kl-ridge_plus_residual": "best", "x-30b-top1-ridge_plus_residual": "best:agree"},
    ("Llama-3.1-Minitron-4B-Width-Base", "Meta-Llama-3.1-8B"): {
        "x-ml-kl32-mapped_direct+": "best", "x-ml-top1-mapped_direct+": "best:agree",
        "x-ml-kl32-mapped_ridge": "ridge", "x-ml-top1-mapped_ridge": "ridge:agree"},
}


def arena_rows(root, ours):
    for f in glob.glob(f"{root}/arena-*/*.json"):
        if f.endswith("_items.json"):
            continue
        r = json.load(open(f))
        if "mapped" not in r or "native8b" not in r:  # judge_tally.json etc.
            continue
        run_id = os.path.basename(os.path.dirname(f))
        m = r["mapped"]
        note = (f"ours: Qwen3-1.7B -> 8B strict mapped arm (target runs only the last prompt token), ridge k={r['k_selected']} "
                f"{r['calib_src']} calib; their 100 prompts, chat template non-thinking, sampled T0.7/top-p .8/top-k 20 seed 0, "
                f"2048-token budget; loop detector research/loops.py (reproduces their flags 499/500); our native 8B loop "
                f"{100 * r['native8b']['loop_rate']:.0f}%, recipe s4t64 {100 * r.get('recipe_s4t64', {}).get('loop_rate', float('nan')):.0f}%")
        for sid in ("x-ar-loop-mapped_kvlingo", "x-ar-loop-mapped_ridge", "x-ar-loop-mapped_heterofold"):
            ours[sid] = dict(value=100 * m["loop_rate"], n=100, rig="Modal A100-80GB, HF bf16", run_id=run_id, note=note)
        jp = os.path.join(os.path.dirname(f), "judge_tally.json")
        if os.path.exists(jp):  # GLM pairwise judge vs THEIR native-8B replies, same judge for their KV-Lingo replies
            t = json.load(open(jp))
            a, th = t.get("annot36|ours_mapped"), t.get("annot36|theirs_KVLingo")
            if a:
                g = t.get("glm_half", {})
                jn = (f"ours mapped vs THEIR native-8B reply on their 36 annotated prompts, blind A/B judged by Claude subagents "
                      f"(4 x 18 pairs, research/arena_judge/verdicts_claude_*.jsonl; theirs were hand labels 6/8/22). Same judge on "
                      f"THEIR KV-Lingo replies: {th}. Second opinion GLM-5.3 on half the prompts: ours {g.get('annot36|ours_mapped')}, "
                      f"theirs {g.get('annot36|theirs_KVLingo')}")
                for k_, v_ in (("better", a["better"]), ("same", a["same"]), ("worse", a["worse"])):
                    ours[f"x-ar-vs8b-{k_}"] = dict(value=v_, n=36, run_id=run_id + "+glm-judge", note=jn)
                ours["x-ar-match-or-exceed"] = dict(value=a["better"] + a["same"], n=36, run_id=run_id + "+glm-judge", note=jn)


def ttft007_rows(root, ours, their_kl=0.18138, their_top1=82.52):
    """x-ml-ttft-8k-speedup: speedup vs native vLLM 8K prefill (strict), value_hf_baseline vs HF prefill (theirs)."""
    fs = glob.glob(f"{root}/ttft007-*/*.json")
    vs = glob.glob(f"{root}/vllm-llama8b-*/vllm_native.json")
    if not fs:
        return
    r = json.load(open(fs[0]))
    tm, q = r["timing"], r["kl32_greedy"]
    arms = {"handoff_direct": "direct", "handoff_distilled": "ridge_kd"}
    best = None
    for a, qa in arms.items():
        if a not in tm or qa not in q:
            continue
        ok = q[qa]["kl32"] <= their_kl + 2 * (q[qa].get("kl32_se") or 0)  # not worse than their learned arm beyond noise
        cand = (tm[a]["p50_ms"], a, qa, ok)
        if ok and (best is None or cand[0] < best[0]):
            best = cand
    if best is None:  # neither arm meets the quality bar: report the faster one, flagged
        a = min((x for x in arms if x in tm), key=lambda x: tm[x]["p50_ms"])
        best = (tm[a]["p50_ms"], a, arms[a], False)
    ms, a, qa, ok = best
    vn = json.load(open(vs[0]))["8192"]["p50_ms"] if vs else None
    run_id = os.path.basename(os.path.dirname(fs[0]))
    note = (f"handoff with the source cache RESIDENT (source prefill not counted, as theirs): arm {a}, target TTFT p50 {ms:.2f} ms "
            f"(their learned arm 38.25 ms); native HF prefill p50 {tm['native_prefill']['p50_ms']:.1f} ms (theirs 302.69); "
            f"native vLLM 8K prefill p50 {vn if vn is None else round(vn, 1)} ms; {tm['tokens']} real FineWeb tokens, "
            f"p50 of {tm[a]['reps']} reps after {tm[a]['warmups']} warmups, CUDA-synchronized, {r['gpu']}; quality of this arm: "
            f"KL32 {q[qa]['kl32']:.5f} / top-1 {100 * q[qa]['agree']:.2f}% on 64 WikiText-103 windows x 256 prefix (their learned "
            f"{their_kl} / {their_top1}); quality_ok={ok}")
    ours["x-ml-ttft-8k-speedup"] = dict(gpu_measured=r["gpu"], value=(vn / ms) if vn else None, value_hf_baseline=tm["native_prefill"]["p50_ms"] / ms,
                                       baseline_vllm=vn, rig=f"Modal {r['gpu']}, HF bf16 handoff; vLLM native baseline",
                                       run_id=run_id, quality_ok=ok, note=note)
    ours["x-ml-ttft-8k-learned"] = dict(gpu_measured=r["gpu"], value=ms, rig=f"Modal {r['gpu']}, HF bf16", run_id=run_id, quality_ok=ok, note=note)


def kl32_rows(root, ours):
    for f in glob.glob(f"{root}/kl32-*/*.json"):
        r = json.load(open(f))
        key = (r["src"].split("/")[-1], r["tgt"].split("/")[-1])
        if key not in KL32_IDS or "kl32_greedy" not in r:
            continue
        run_id = os.path.basename(os.path.dirname(f))
        g, t = r["kl32_greedy"], r.get("kl32_text", {})
        strict = {a: v for a, v in g.items() if a != "native" and not a.startswith(("nk_", "mk_")) and "_s4" not in a}
        ridge = min((a for a in strict if a.startswith("ridge_k")), key=lambda a: strict[a]["kl32"])
        best = min(strict, key=lambda a: strict[a]["kl32"])
        rec = {a: round(v["kl32"], 5) for a, v in g.items() if "_s4" in a}
        for sid, what in KL32_IDS[key].items():
            arm = ridge if what.startswith("ridge") else best
            v = 100 * g[arm]["agree"] if what.endswith(":agree") else g[arm]["kl32"]
            ours[sid] = dict(value=round(v, 5), n=g[arm]["n"], rig=KL32_RIG.get(run_id, run_id), run_id=run_id,
                             note=f"ours arm {arm} (strict: target runs only the last prefix token natively); "
                                  f"{r['windows']} WikiText-103 test windows, {r['prefix']}-token prefix, 32 steps forced on the "
                                  f"native target's greedy tokens (text-forced: {round(t.get(arm, {}).get('kl32', float('nan')), 5)}); "
                                  f"calib {r['calib_src']} {r['n_calib']}x{r['L']}; recipe arms with native sink+tail (not their protocol) {rec}")


def main(root, out_path):
    ours = {}  # rebuilt from the result files every time (this script is the only writer of the seat's ours.json)
    runs = load_runs(root)
    means = {}
    for (s, t), (f, r) in sorted(runs.items()):
        run_id = os.path.basename(os.path.dirname(f))
        if run_id in WITHHELD:
            continue
        items = json.load(open(f.replace(".json", "_items.json")))["mc128"]
        mc = r["mc128"]
        acc = lambda a: 100 * mc[a]["acc"]
        rec = {a: round(acc(a), 2) for a in mc if a.startswith("recipe")}
        base = dict(n=mc["mapped"]["n"], rig=RIG.get(run_id, run_id), run_id=run_id, cost_usd=None,
                    note=f"ours: ridge top-k={r.get('k_selected')} ({r.get('calib_src')} calib {r.get('n_calib')}x{r.get('L')}, rms key restore); "
                         f"strict protocol (target runs only the last prompt token); prompt format 'Question/A./Answer:' (theirs unpublished); "
                         f"our native {acc('native'):.2f}, retention mapped/native {acc('mapped') / acc('native'):.3f} "
                         f"(native-K arm {acc('nk_mv') / acc('native'):.3f}, native-V arm {acc('mk_nv') / acc('native'):.3f}); recipe arms (native sink+tail) {rec}")
        ci = lambda a: round(100 * mc[a]["ci95"], 2)
        D = f"{s}-{t}"
        lo, hi = paired_ci(items["mapped"], items["native"])
        vlo, vhi = paired_ci(items["mk_nv"], items["nk_mv"])  # value-minus-key harm = (nat - nk_mv) - (nat - mk_nv)
        vals = {
            f"x-twj-{D}-mapped": (acc("mapped"), ci("mapped")),
            f"x-twj-{D}-target_native": (acc("native"), ci("native")),
            f"x-twj-{D}-wrong_cache": (acc("wrong"), ci("wrong")),
            f"x-twj-{D}-delta": (acc("mapped") - acc("native"), None),
            f"x-twj-{D}-ci-low": (lo, None), f"x-twj-{D}-ci-high": (hi, None),
            f"x-kvm-{D}-acc-diagnostic_native_V": (acc("mk_nv"), ci("mk_nv")),
            f"x-kvm-{D}-acc-diagnostic_native_K": (acc("nk_mv"), ci("nk_mv")),
            f"x-kvm-{D}-kl-mapped": (mc["mapped"]["choice_kl"], None),
            f"x-kvm-{D}-kl-diagnostic_native_V": (mc["mk_nv"]["choice_kl"], None),
            f"x-kvm-{D}-kl-diagnostic_native_K": (mc["nk_mv"]["choice_kl"], None),
            f"x-kvm-{D}-vk-harm": (acc("mk_nv") - acc("nk_mv"), None),
            f"x-kvm-{D}-vk-harm-cilo": (vlo, None), f"x-kvm-{D}-vk-harm-cihi": (vhi, None),
        }
        if "gold_nll" in mc["mapped"]:
            vals.update({f"x-kvm-{D}-nll-mapped": (mc["mapped"]["gold_nll"], None),
                         f"x-kvm-{D}-nll-native_target": (mc["native"]["gold_nll"], None),
                         f"x-kvm-{D}-nll-diagnostic_native_V": (mc["mk_nv"]["gold_nll"], None),
                         f"x-kvm-{D}-nll-diagnostic_native_K": (mc["nk_mv"]["gold_nll"], None)})
        tq = {("4", "8"): "4to8", ("8", "4"): "8to4", ("8", "14"): "8to14", ("14", "32"): "14to32"}.get((s, t))
        if tq:
            vals[f"x-twq-{tq}-mapped"] = (acc("mapped"), ci("mapped"))
            vals[f"x-twq-{tq}-native" + ({"4to8": "8", "8to4": "4"}.get(tq, ""))] = (acc("native"), ci("native"))
            vals[f"x-twq-{tq}-gap"] = (acc("mapped") - acc("native"), None)
            if tq == "4to8":
                vals["x-twq-4to8-wrongcache"] = (acc("wrong"), ci("wrong"))
        if run_id in SUSPECT_MIXED:
            vals = {k: v for k, v in vals.items() if "diagnostic" not in k and "vk-harm" not in k}
        for k, (v, c) in vals.items():
            row = dict(base, value=round(v, 4))
            if c is not None:
                row["ci"] = c
            ours[k] = row
        for a in ("mapped", "nk_mv", "mk_nv", "native"):
            means.setdefault(a, []).append(acc(a) if not (run_id in SUSPECT_MIXED and a in ("nk_mv", "mk_nv")) else None)
    v0 = runs.get(("0.6", "1.7"))  # note 001 rows (v0): held-out agreement / ppl per k, HellaSwag / PIQA at the selected k
    if v0 and "hellaswag" in v0[1]:
        f, r = v0
        run_id = os.path.basename(os.path.dirname(f))
        h, hs, pq = r["heldout"], r["hellaswag"], r["piqa"]
        ksel = r.get("k_mc", r.get("k_selected"))
        base = dict(rig=RIG.get(run_id, run_id), run_id=run_id,
                    note=f"ours: Qwen3-0.6B->1.7B ridge ({r.get('calib_src')} calib {r.get('n_calib')}x{r.get('L')}, rms restore); "
                         f"held-out = 32 FineWeb-Edu docs (eval range), 448-token mapped prefix + 64 native continuation tokens "
                         f"(their held-out text unpublished); HellaSwag/PIQA n={hs['mapped']['n']} at k={ksel}, acc_norm/sum-logprob, "
                         f"our native HS {100*hs['native']['acc']:.1f} PIQA {100*pq['native']['acc']:.1f} (theirs 59.4 / 74.0)")
        cn = lambda m, nat, ch: (m - ch) / (nat - ch)
        ret = 100 * (cn(hs["mapped"]["acc"], hs["native"]["acc"], .25) + cn(pq["mapped"]["acc"], pq["native"]["acc"], .5)) / 2
        put = lambda k_, v_, n_: ours.__setitem__(k_, dict(base, value=round(v_, 4), n=n_))
        for kk in ("1", "4"):
            if f"map_k{kk}" in h:
                put(f"x-v0-k{kk}-nta", 100 * h[f"map_k{kk}"]["agree"], h[f"map_k{kk}"]["n"] * 64)
                put(f"x-v0-k{kk}-ppl", h[f"map_k{kk}"]["ppl"], h[f"map_k{kk}"]["n"] * 64)
            put(f"x-v0-k{kk}-hellaswag", 100 * hs["mapped"]["acc"], hs["mapped"]["n"])
            put(f"x-v0-k{kk}-piqa", 100 * pq["mapped"]["acc"], pq["mapped"]["n"])
            put(f"x-v0-retention-k{kk}", ret, hs["mapped"]["n"] + pq["mapped"]["n"])
        if "map_k1" in h:
            put("x-v0-ppl-ratio", h["map_k1"]["ppl_ratio"], h["map_k1"]["n"] * 64)
            if "direct" in h:
                put("x-v0-nta-gain", h["map_k1"]["agree"] / max(h["direct"]["agree"], 1e-9), h["map_k1"]["n"] * 64)
    kl32_rows(root, ours)
    ttft007_rows(root, ours)
    arena_rows(root, ours)
    if len(means.get("mapped", [])) == 10:  # the ten-direction means (their 43.67 / 49.14 / 56.02 / 72.11)
        m = {a: (sum(v) / 10 if None not in v else None) for a, v in means.items()}
        note = "mean over the 10 directions (0.6<->1.7, 1.7<->4, 4<->8, 8<->14, 14<->32), 128 Q each, strict arm"
        print("ten-direction means", m)
        for k, a in (("x-wgl-both-translated", "mapped"), ("x-wgl-native-k", "nk_mv"), ("x-wgl-native-v", "mk_nv")):
            if m[a] is None:
                ours.pop(k, None); continue
            ours[k] = dict(value=round(m[a], 3), n=1280, rig="Modal CPU/A100/H200/B200, HF", run_id="mc128 runs", note=note)
        if m["mk_nv"] is not None and m["nk_mv"] is not None:
          ours["x-wgl-vk-diff"] = dict(value=round(m["mk_nv"] - m["nk_mv"], 3), n=1280, run_id="mc128 runs", note=note)
    for v in ours.values():  # lead rule rig-h200-hold: every row carries the GPU it was measured on
        v.setdefault("gpu_measured", (v.get("rig") or "unknown").replace("Modal ", "").split(",")[0])
    sb = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(out_path)))), "scoreboard.json")
    if os.path.exists(sb):  # only ids the scoreboard has
        d = json.load(open(sb))
        ids = {r["id"] for r in (d["rows"] if isinstance(d, dict) else d)}
        extra = sorted(k for k in ours if k not in ids)
        if extra:
            print("dropping ids not on the scoreboard:", extra)
        ours = {k: v for k, v in ours.items() if k in ids}
    json.dump(ours, open(out_path, "w"), indent=1, sort_keys=True)
    print(f"{len(runs)} directions, {len(ours)} rows -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
