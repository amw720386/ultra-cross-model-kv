"""Turn a modal_speed result (out/speed/<plan>.json) into scoreboard rows (seats/kvx-speed/ours.json) and ledger rows.

python tools/speed_report.py out/speed/q17-q32-h100x2.json --rig "2xH100 SXM, vLLM 0.34 TP=2" --cost 4.1 [--write]

Per context: the fastest mapped config's median TTFT (and its tag), native vLLM median on the same rig, speedup =
native / mapped, components from the connector's 'kvx components:' lines (median per mapped length; first target
token = mapped TTFT - connector total). Prints everything; --write merges into ours.json and appends ledger rows.
"""
import argparse
import datetime
import json
import os
import re
import statistics

DATA = os.path.expanduser(os.environ.get("KVX_DATA", "~/kvx-data"))  # scoreboard + seat files (set KVX_DATA)
SB = os.path.join(DATA, "scoreboard.json")
OURS = os.path.join(DATA, "seats/kvx-speed/ours.json")
LEDGER = os.path.join(DATA, "seats/kvx-speed/LEDGER.jsonl")

# target model -> (home-row pair, component-row pair, context labels by prompt length)
PAIRS = {
    "Qwen/Qwen3-32B": ("Qwen3-1.7B -> Qwen3-32B", "Qwen3-1.7B -> Qwen3-4B–32B",
                       {8192: "8K", 16384: "16K", 32768: "32K", 65536: "64K", 130816: "128K", 131072: "128K",
                        261888: "256K", 262144: "256K", 523776: "512K", 1048320: "1M"}),
    "Qwen/Qwen3-8B": ("Qwen3-1.7B -> Qwen3-8B", "Qwen3-1.7B -> Qwen3-8B",
                      {8192: "8K", 16384: "16K", 32768: "32K", 65536: "64K", 130816: "128K", 261888: "256K",
                       523776: "512K", 1048320: "1M"}),
    "mistralai/Mistral-Medium-3.5-128B": ("Ministral-3-3B -> Mistral-Medium-3.5-128B", "Ministral-3-3B -> Mistral-Medium-3.5-128B",
                                          {8192: "8K", 16384: "16K", 32768: "32K", 65536: "64K"}),
    "Qwen/Qwen2.5-72B": ("Qwen2.5-3B -> Qwen2.5-72B", "Qwen2.5-3B -> Qwen2.5-72B",
                         {32768: "32K", 65536: "64K", 130816: "127K", 261888: "262K", 523776: "524K", 1048320: "1M"}),
}
# their headline peak context per pair (Tier 1 "peak speedup x (headline)" rows) and the documented max context
PEAK = {"Qwen3-1.7B -> Qwen3-32B": ("1M", 131072), "Qwen3-1.7B -> Qwen3-8B": ("1M", 131072),
        "Qwen2.5-3B -> Qwen2.5-72B": ("127K", 131072), "Ministral-3-3B -> Mistral-Medium-3.5-128B": ("32K", 262144)}
COMP = {"src_ms": "source-model prefill", "map_ms": "KV map", "move_ms": "cache move", "first_ms": "first target token"}
LINE = re.compile(r"kvx components: n=(\d+) src_ms=([\d.]+) map_ms=([\d.]+) move_ms=([\d.]+) total_ms=([\d.]+)")


def components(log, n_mapped):
    xs = [tuple(float(v) for v in m.groups()[1:]) for m in map(LINE.search, log) if m and int(m.group(1)) == n_mapped]
    if len(xs) > 1:
        xs = xs[1:]  # first one at this length is the warm-up request (or the only one for 256K+)
    if not xs:
        return None
    med = lambda i: statistics.median(x[i] for x in xs)
    return dict(src_ms=med(0), map_ms=med(1), move_ms=med(2), total_ms=med(3), n_req=len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result", nargs="+")
    ap.add_argument("--tgt", default="")
    ap.add_argument("--rig", required=True)
    ap.add_argument("--cost", type=float, default=0.0)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    res = sum((json.load(open(f)) for f in a.result), [])
    run_id = os.path.basename(a.result[-1]).rsplit(".", 1)[0]
    ok = [dict(x["result"], tag=x["result"].get("tag") or x["cfg"].get("tag"), maps=x["result"].get("maps") or x["cfg"].get("maps"),
               extra_cfg=x["cfg"].get("extra"))
          for x in res if "result" in x]
    for x in res:  # a config that died later (e.g. OOM at 128K) still measured its shorter lengths
        if "error" in x and x.get("partial_rows"):
            ok.append(dict(x["cfg"], rows=x["partial_rows"], connector_log=x.get("connector_log", []), partial=True))
    for x in res:
        if "error" in x:
            print("FAILED config", x["cfg"].get("tag"), x["error"][-800:])
    if a.tgt:
        ok = [r for r in ok if r.get("tgt") == a.tgt]
    native = [r for r in ok if r["mode"] == "native"]
    mapped = [r for r in ok if r["mode"] == "mapped"]
    if not native or not mapped:
        print("need native and mapped results"); return
    tgt = native[0]["tgt"]
    home_pair, comp_pair, labels = PAIRS[tgt]
    nat = {row["len"]: row["ttft_s"] for row in native[0]["rows"]}
    rows = json.load(open(SB))["rows"]
    byid = {r["id"]: r for r in rows}
    out, ledger = {}, []
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for L in sorted(nat):
        lab = labels.get(L)
        cands = [(row["ttft_s"], r) for r in mapped for row in r["rows"] if row["len"] == L]
        if not cands or lab is None:
            continue
        t, best = min(cands, key=lambda c: c[0])
        tag = best.get("tag") or ""
        sp = nat[L] / t
        hid = "home-" + re.sub(r"[^a-z0-9]+", "-", home_pair.lower().replace("->", " ")).strip("-")
        hid = re.sub(r"-+", "-", hid)
        n_m = ((L - best["tail"]) // 16) * 16
        comp = components(best.get("connector_log", []), n_m)
        qn = ""
        try:  # this config's own quality vs native vLLM (lead's rule for configs that change the source computation)
            from speed_quality import cmp as _cmp
            nf = next((r.get("quality_files") for r in native if r.get("quality_files")), None)
            bf = best.get("quality_files")
            same = nf is not None and bf is not None and all(nf.get(k) == bf.get(k) for k in ("fw", "long", "long_fw"))
            if not same:
                raise ValueError("prompt files differ or unrecorded")  # lead's rule: no quality note across prompt sets
            qn += f"; prompt files {bf}"
            for key, lab_ in (("first_token_top20", "FW 1K"), ("long_first_token_top20", "PG19 long")):
                nq = next((r.get(key) for r in native if r.get(key)), None)
                if nq and best.get(key):
                    ag, nn, kl = _cmp(nq, best[key])
                    qn += f"; quality {lab_}: top1 {ag}/{nn}, approx first-token KL {kl:.3f} vs native"
        except Exception as e:
            qn = f"; quality NOT compared ({e})"
        note = (f"mapped {t:.3f} s vs native vLLM {nat[L]:.3f} s ({tag}); src_tp={best.get('src_tp')}, tail={best['tail']}, "
                f"fp8 mapper={best.get('fp8')}, YaRN-served; maps {best.get('maps')}; extra {best.get('extra_cfg')}" + qn)
        for rid, val in ((f"{hid}-{lab}-ttft", t), (f"{hid}-{lab}-speedup", sp)):
            if rid not in byid:
                print("no scoreboard row", rid); continue
            out[rid] = dict(value=round(val, 4), rig=a.rig, run_id=f"{run_id}{tag}", baseline_vllm=round(nat[L], 4),
                            cost_usd=a.cost, note=note)
            hf = byid.get(rid.replace("-speedup", "-ttft"), {}).get("their_baseline_native_s")
            if rid.endswith("-speedup") and hf:  # ours against THEIR HF-forwards native number (their own baseline)
                out[rid]["value_hf_baseline"] = round(float(hf) / t, 3)
            print(f"{rid:45s} ours {val:8.3f}  theirs {byid[rid]['their_value']}")
        pk = PEAK.get(home_pair)
        if pk and lab == pk[0] and f"{hid}-peak" in byid:  # the headline peak row: our speedup at THEIR peak context
            beyond = L > pk[1]
            base = out.get(f"{hid}-{lab}-speedup") or dict(rig=a.rig, run_id=f"{run_id}{tag}", baseline_vllm=round(nat[L], 4),
                                                         cost_usd=a.cost, note=note)
            out[f"{hid}-peak"] = dict(base, value=round(sp, 4), note=(("BEYOND MODEL SPEC (card max %d tokens; RoPE/YaRN "
                                      "extended, same for native and mapped); " % pk[1]) if beyond else "") + f"measured at {L} tokens; " + note)
            print(f"{hid + '-peak':45s} ours {sp:8.3f}  theirs {byid[hid + '-peak']['their_value']}")
        ledger.append(dict(app_or_label="kvx-speed", baseline=round(nat[L], 4), chip=a.rig, cost_usd=a.cost,
                           delta_pct=round(100 * (t / nat[L] - 1), 1), lane="kvx", lever="cross-model KV transfer (" + tag + ")",
                           metric=f"TTFT s @{lab} c=1 vs native vLLM", model=f"{home_pair}", provider="modal",
                           run_id=f"{run_id}{tag}", ts=ts, value=round(t, 4)))
        if comp:
            comp["first_ms"] = max(0.0, 1e3 * t - comp["total_ms"])
            for key, phrase in COMP.items():
                cr = [r for r in rows if r["pair"] == comp_pair and phrase in r.get("metric", "")
                      and r.get("context", "").startswith(f"context {lab};")]
                if not cr:
                    print("no component row", comp_pair, phrase, lab); continue
                r0 = cr[0]
                out[r0["id"]] = dict(value=round(comp[key], 1), rig=a.rig, run_id=f"{run_id}{tag}", cost_usd=a.cost,
                                     note=f"connector log median of {comp['n_req']} requests at n={n_m} mapped tokens; "
                                          f"TTFT {t:.3f} s; components synced at boundaries")
                print(f"{r0['id']:45s} ours {comp[key]:8.1f}  theirs {r0['their_value']}")
    if a.write:
        os.makedirs(os.path.dirname(OURS), exist_ok=True)
        cur = json.load(open(OURS)) if os.path.exists(OURS) else {}
        kept = 0
        for rid, v in out.items():  # keep the better measurement per row (rows carry their own rig and run_id)
            old = cur.get(rid)
            higher = byid[rid].get("direction") == "higher"
            if old and old.get("quality_ok") is not False and old.get("run_id") != v["run_id"] and (
                    old["value"] > v["value"] if higher else old["value"] < v["value"]):
                kept += 1
                continue
            cur[rid] = v
        print(f"kept {kept} better existing rows")
        json.dump(cur, open(OURS, "w"), indent=1, sort_keys=True)
        key = lambda r: (r["run_id"], r["metric"], r["model"])  # re-running the report replaces, never duplicates
        old = [json.loads(l) for l in open(LEDGER)] if os.path.exists(LEDGER) else []
        new = {key(r): r for r in old + ledger}
        with open(LEDGER, "w") as f:
            for l in new.values():
                f.write(json.dumps(l, sort_keys=True) + "\n")
        print(f"wrote {len(out)} rows to {OURS}, {len(ledger)} ledger rows")


if __name__ == "__main__":
    main()
