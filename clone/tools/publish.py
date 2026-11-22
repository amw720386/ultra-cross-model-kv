"""Append ledger rows from a fit_eval result.json and push lane data to traceDB.

python tools/publish.py out/<run_id>/result.json --cost 0.42 --chip L40S --lever ridge-kv-map
"""
import argparse, json, os, subprocess, time

DATA = os.path.expanduser("~/kvx-data")


def short(m):
    return m.split("/")[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--cost", type=float, required=True)
    ap.add_argument("--chip", required=True)
    ap.add_argument("--lever", default="ridge-kv-map")
    ap.add_argument("--keys", default="", help="comma list of eval keys to log; default all map_* and refs")
    a = ap.parse_args()
    r = json.load(open(a.result))
    run_id = os.path.basename(os.path.dirname(os.path.abspath(a.result)))
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    model = f"{short(r['src'])}->{short(r['tgt'])}"
    base = r["eval"]["ref_nocontext"]["kl_first"]
    keys = a.keys.split(",") if a.keys else list(r["eval"])
    os.makedirs(f"{DATA}/runs/{run_id}", exist_ok=True)
    json.dump(r, open(f"{DATA}/runs/{run_id}/result.json", "w"), indent=1)
    with open(f"{DATA}/LEDGER.jsonl", "a") as f:
        for k in keys:
            e = r["eval"][k]
            for metric in ("kl_first", "kl"):
                f.write(json.dumps(dict(
                    app_or_label="kvx-fit", baseline=round(base, 5), chip=a.chip,
                    cost_usd=round(a.cost / len(keys), 4), delta_pct=round(100 * (e[metric] - base) / base, 2),
                    lane="kvx", lever=f"{a.lever}:{k}", metric=f"{metric}_nats(P={r['P']},{r.get('eval_src', 'fineweb-edu')})",
                    model=model, provider="modal", run_id=run_id, ts=ts, value=round(e[metric], 5)),
                    sort_keys=True) + "\n")
    for i in range(6):  # traceDB sqlite is shared with other seats: retry on lock
        # optional shared lock for hosts where several writers share one traceDB catalog (KVX_INGEST_LOCK)
        lock = os.environ.get("KVX_INGEST_LOCK")
        cmd = ["tracedb", "put", DATA, "--prefix", os.environ.get("KVX_TRACEDB_PREFIX", "traces/kvx/")]
        if subprocess.run((["flock", "-w", "3500", lock] if lock else []) + cmd).returncode == 0:
            return
        time.sleep(5)
    raise SystemExit("tracedb put failed 6x")


if __name__ == "__main__":
    main()
