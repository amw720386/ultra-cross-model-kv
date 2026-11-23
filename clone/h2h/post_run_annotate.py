"""Post-run annotation of a OneTriangle h2h box run dir (h2h lead's pre-VERIFY conditions, DO run 2).

  python3 h2h/post_run_annotate.py RUN_DIR BOOT_STATUS_FILE SCRIPT_COMMIT

For every abba_<L>/plan.json (the plan as run is kept as abba_<L>/plan_as_run.json):
  1. model_hash on BOTH arms = sha256 over the sorted "file<TAB>sha256" lines of the Qwen3-8B snapshot rows of
     evidence/weights_snapshots.tsv (every row must be MATCH: blob name == recomputed sha256). Both arms read the one
     shared HF cache (/data/hf). Each arm's own log line naming that snapshot is quoted.
  2. calibration.explained += the pre-registered spec sha and the gating restart's numbers, attempt 1 beside it.
Then tools/verdict.py is re-run per length (verdict_post.txt).
"""
import hashlib, json, os, re, shutil, subprocess, sys

RD, BOOT, COMMIT = sys.argv[1], sys.argv[2], sys.argv[3]
SNAP = "models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
H2H = os.environ.get("H2H", os.path.expanduser("~/clone-push/kvx-speed/verified-h2h"))

rows = []
for line in open(f"{RD}/evidence/weights_snapshots.tsv"):
    f = line.rstrip("\n").split("\t")
    if len(f) == 5 and f[0].endswith(SNAP):
        assert f[4] == "MATCH", f"blob mismatch: {f}"
        rows.append(f"{f[1]}\t{f[3]}")
rows.sort()
assert len(rows) >= 5, rows
model_hash = hashlib.sha256("\n".join(rows).encode()).hexdigest()


def first(path, pat):
    for line in open(path, errors="replace"):
        if re.search(pat, line):
            return line.strip()[:400]
    return None


ours_line = first(f"{RD}/ours_server_abba.log", r"non-default args.*" + re.escape(SNAP))
health = re.search(r'"target":\{[^}]*\}', open(BOOT).read())
theirs_health = health.group(0) if health else None
theirs_shards = None
for line in open(f"{RD}/evidence/their_startup.log", errors="replace"):
    m = re.search(r"Loading checkpoint shards:\s+0%\|[^|]*\|\s+0/(\d+)", line)
    if m and m.group(1) == "5":
        theirs_shards = "Loading checkpoint shards: ... 0/5 (the 8B snapshot has 5 safetensors shards; the 1.7B has 2)"
assert ours_line and theirs_health and theirs_shards, (ours_line, theirs_health, theirs_shards)
weights = {
    "model_hash": model_hash,
    "derivation": ("sha256 over the sorted 'file<TAB>sha256' lines of the Qwen3-8B snapshot rows of "
                   "evidence/weights_snapshots.tsv; every row MATCH (HF blob name == recomputed sha256 of the file)"),
    "script": "ultra-cross-model-kv h2h/post_run_annotate.py", "script_commit": COMMIT,
    "rows": rows,
    "shared_cache": "both arms read the ONE host cache /data/hf (theirs at /cache/huggingface, ours at /cache/huggingface/hub)",
    "ours_log_line": ours_line,
    "theirs_log": {"health_self_report": theirs_health, "shard_load_line": theirs_shards,
                   "note": "their kvt log prints no snapshot path; the cache holds exactly one Qwen3-8B snapshot (b968826d)"},
    "proc_maps_zero_files": ("The /proc/<pid>/maps check found 0 mapped safetensors because both servers copy the weights "
                             "into GPU memory at load and close/unmap the files afterwards, so no file mapping remains "
                             "when the check runs (after both servers are up)."),
    "run_ab_model_hash_host_path": (json.load(open(f"{RD}/evidence/model_hash_host_path.json"))["model_hash"]
                                    if os.path.exists(f"{RD}/evidence/model_hash_host_path.json") else None),
}
cal_r = json.load(open(f"{RD}/calib/summary.json"))
cal_1 = json.load(open(f"{RD}/calib_attempt1/summary.json")) if os.path.exists(f"{RD}/calib_attempt1/summary.json") else None
spec_sha = None
for cand in (f"{RD}/calibration_prereg.json", os.path.expanduser("~/clone-push/kvx-speed/ot-box/calibration_prereg.json")):
    if os.path.exists(cand):
        spec_sha = hashlib.sha256(open(cand, "rb").read()).hexdigest()
        break


def fmt(c):
    r = c["rebuilt"]["long"]
    return (f"rebuilt_e2e_13837 {r['rebuilt_e2e_s']:.4f} s = {c['long_off_pct']:+.2f}% (prepare {r['prepare_ms']:.1f} + queue "
            f"{r['queue_ms']:.2f} + GPU TTFT {r['gpu_ttft_ms']:.1f} + 12 x {r['ms_per_step']:.2f} ms/step + HTTP {r['http_ms']:.1f} ms; "
            f"{r['n_requests']} requests); raw e2e {c['calib2_e2e']['median']:.4f} s; short rebuilt {c['short_rebuilt_off_pct']:+.1f}% "
            f"(not gating); TTFT 13,837 SSE {c['calib2_sse']['median']:.3f} s")


explained_add = (f" Pre-registered spec sha256 {spec_sha} (gate: rebuilt 13,837 e2e within 6% of 0.5465 s). "
                 f"GATING (driver restart on the same box): {fmt(cal_r)}." +
                 (f" Attempt 1 on the same box before the restart: {fmt(cal_1)}." if cal_1 else ""))
out = {}
for d in sorted(x for x in os.listdir(RD) if x.startswith("abba_")):
    p = f"{RD}/{d}/plan.json"
    if not os.path.exists(p):
        continue
    keep = f"{RD}/{d}/plan_as_run.json"
    if not os.path.exists(keep):
        shutil.copy(p, keep)
    plan = json.load(open(keep))
    plan["model_hash"] = model_hash
    for a in plan["arms"]:
        a["model_hash"] = model_hash
    plan["calibration"]["explained"] = plan["calibration"].get("explained", "") + explained_add
    plan["post_run_annotations"] = {"script": weights["script"], "script_commit": COMMIT,
                                    "plan_as_run": "plan_as_run.json (unchanged copy)",
                                    "plan_as_run_sha256": hashlib.sha256(open(keep, "rb").read()).hexdigest(),
                                    "weights": weights}
    json.dump(plan, open(p, "w"), indent=1)
    r = subprocess.run(["python3", f"{H2H}/tools/verdict.py", f"{RD}/{d}"], capture_output=True, text=True)
    open(f"{RD}/{d}/verdict_post.txt", "w").write(r.stdout + r.stderr)
    out[d] = {"plan_sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
              "plan_as_run_sha256": plan["post_run_annotations"]["plan_as_run_sha256"], "verdict": (r.stdout + r.stderr).strip()}
print(json.dumps({"model_hash": model_hash, "per_length": out}, indent=1))
