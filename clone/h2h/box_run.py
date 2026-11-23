"""OneTriangle h2h on a bare-metal (non-gVisor) H100 box: their image replayed from their Dockerfile, our stack in a
separate venv, both servers resident on GPU0, ABBA via the shared harness (decision ot-h2h-vast).

  python3 box_run.py setup | calib | devpick | abba | native      (one driver runs them in order; see box_driver.sh)

Layout on the box (/root): ot-repo/ (their repo @8f05e5f1), replay.sh + replay_map.tsv, ours/ (our clone snapshot),
h2h/ (verified-h2h snapshot), prompts/, prompts-dev/, maps_*.pt. Everything is written under OUT=/root/out/<run_id>/.
"""
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = "/root"
RUN_ID = os.environ.get("RUN_ID", "ot-h2h")
OUT = f"{ROOT}/out/{RUN_ID}"
HF = "/cache/huggingface/hub"
QWEN8 = f"{HF}/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
QWEN17 = f"{HF}/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
THEIRS = ("http://127.0.0.1:8000", "kvt-qwen3-1.7b-to-8b")
OURS = ("http://127.0.0.1:8001", "qwen3-8b-kvx")
CLIENT = f"python3 {ROOT}/h2h/bench/sse_ttft.py"
QCLIENT = f"python3 {ROOT}/h2h/bench/quality_chat_client.py"
EXTRA = json.dumps({"chat_template_kwargs": {"enable_thinking": False}})
P, D = f"{ROOT}/prompts", f"{ROOT}/prompts-dev"
LENGTHS = (8192, 16384, 30720)
# lead: verdict-critical order (the length most likely to win first); native context runs last
ABBA_LENGTHS = tuple(int(x) for x in os.environ.get("ABBA_LENGTHS", ",".join(map(str, LENGTHS))).split(","))
DOCKER = os.environ.get("OT_MODE") == "docker"   # EC2: their runtime is the container ot-runtime (started at boot)
OURS_IMAGE = "vllm/vllm-openai:v0.34.0"
THEIR_IMAGE = "onetriangle/qwen3-1p7b-to-8b-prefill:2026-11-12"
PUBLISHED = {"short_e2e_s": 0.1877, "long_e2e_s": 0.5465}
SRC_URL = ("https://huggingface.co/turtler12/onetriangle-qwen3-1.7b-to-8b-prefill-docker (VALIDATION_REPORT.md, "
           "ACCEPTANCE_RESULTS.json) @ 8f05e5f1d1d30bc60be81a6e23ae743cc31f6f7c")


def sh(cmd, log=None, timeout=None, check=False):
    p = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    if log:
        with open(log, "a") as f:
            f.write(f"\n### {time.strftime('%FT%TZ', time.gmtime())} $ {cmd}\n{p.stdout}{p.stderr}")
    if check and p.returncode:
        raise RuntimeError(f"rc={p.returncode}: {cmd}\n{(p.stdout + p.stderr)[-2000:]}")
    return p.returncode, p.stdout + p.stderr


def mark(name, data=None):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/progress.log", "a") as f:
        f.write(f"{time.strftime('%FT%TZ', time.gmtime())} {name} {json.dumps(data) if data is not None else ''}\n")


def wait(url, timeout, proc=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited rc={proc.returncode} before {url} was healthy")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return round(time.time() - t0, 1)
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"{url} not healthy after {timeout}s")


def their_env():
    env = dict(os.environ)
    for ln in open("/etc/ot-replay.env"):  # their Dockerfile ENV/ARG, as replayed
        if "=" in ln:
            k, v = ln.rstrip("\n").split("=", 1)
            env[k] = v
    env.update(LOCAL_FILES_ONLY=os.environ.get("OT_LOCAL_FILES_ONLY", "0"), CUDA_VISIBLE_DEVICES="0")  # compose values
    env.pop("PYTHONPATH", None)
    return env


class _Docker:
    """Stand-in for a Popen: a named container."""
    def __init__(self, name):
        self.name, self.pid, self.returncode = name, None, None

    def poll(self):
        rc, out = sh(f"docker inspect -f '{{{{.State.Running}}}}' {self.name}")
        return None if out.strip() == "true" else 1


def start_theirs(tag):
    if DOCKER:
        hs = wait(THEIRS[0] + "/health", 3600)
        return _Docker("ot-runtime"), hs
    log = open(f"{OUT}/theirs_server_{tag}.log", "ab")
    p = subprocess.Popen(["/usr/bin/tini", "--", "/usr/local/bin/runtime-entrypoint"], env=their_env(), stdout=log,
                         stderr=subprocess.STDOUT, start_new_session=True)
    open(f"{OUT}/theirs.pid", "w").write(str(p.pid))
    return p, wait(THEIRS[0] + "/health", 3600, p)


def ours_args(cfg, mapped=True):
    kv = {"kv_connector": "CrossModelConnector", "kv_connector_module_path": "kvx.vllm_connector", "kv_role": "kv_both",
          "kv_connector_extra_config": dict({"src": QWEN17, "maps": cfg["maps"], "tail": cfg["tail"], "min_len": 512,
                                             "restore": "rms", "src_sdpa": cfg.get("src_sdpa", "vllm_fa"), "src_tp": 1},
                                            **({"fp8": True} if cfg.get("fp8", 1) else {}), **cfg.get("extra", {}))}
    args = ["/opt/ours/bin/vllm", "serve", QWEN8, "--served-model-name", OURS[1], "--port", "8001",
            "--max-model-len", "32768", "--gpu-memory-utilization", str(cfg.get("util", 0.33)), "--dtype", "bfloat16",
            "--no-enable-prefix-caching", "--max-num-seqs", "2"]
    args += [str(a) for a in cfg.get("vllm_args", [])]  # screened vLLM flags (part of the frozen config)
    if mapped:
        args += ["--kv-transfer-config", json.dumps(kv)]
    return args


def start_ours(cfg, tag, mapped=True):
    if DOCKER:
        cfg = dict(cfg, maps=cfg["maps"].replace(ROOT, "/in")) if cfg.get("maps") else cfg
        args = ["serve"] + ours_args(cfg, mapped)[2:]  # run with --entrypoint vllm: independent of the image default
        sh("docker rm -f ours")
        cmd = ("docker run -d --name ours --gpus device=0 --ipc=host -v /data/hf:/cache/huggingface/hub:ro "
               f"-v {ROOT}/ours:/ours:ro -v {ROOT}:/in:ro -e PYTHONPATH=/ours -e HF_HUB_OFFLINE=1 "
               f"-p 127.0.0.1:8001:8001 --entrypoint vllm {OURS_IMAGE} " + " ".join(shlex.quote(a) for a in args))
        json.dump({"docker_cmd": cmd, "args": args}, open(f"{OUT}/ours_cmd_{tag}.json", "w"), indent=1)
        sh(cmd, log=f"{OUT}/ours_docker_{tag}.log", check=True)
        p = _Docker("ours")
        try:
            return p, wait(OURS[0] + "/health", 1800, p)
        finally:
            sh(f"docker logs ours > {OUT}/ours_server_{tag}.log 2>&1")
    env = dict(os.environ, PYTHONPATH=f"{ROOT}/ours", HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="0")
    args = ours_args(cfg, mapped)
    json.dump(args, open(f"{OUT}/ours_cmd_{tag}.json", "w"), indent=1)
    log = open(f"{OUT}/ours_server_{tag}.log", "ab")
    p = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    open(f"{OUT}/ours.pid", "w").write(str(p.pid))
    return p, wait(OURS[0] + "/health", 1800, p)


def stop(p):
    if isinstance(p, _Docker):
        if p.name == "ours":
            sh(f"docker logs ours >> {OUT}/ours_server_last.log 2>&1; docker rm -f ours")
        return  # their container stays as the boot driver left it; the box is terminated at the end
    if p is not None and p.poll() is None:
        os.killpg(p.pid, 15)
        try:
            p.wait(90)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 9)


def client(arm, prompts, out, max_tokens=16, stream=True, warmup=1):
    url, model = THEIRS if arm == "theirs" else OURS
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = (f"{CLIENT} --base-url {url} --model {model} --prompts {prompts} --max-tokens {max_tokens} --out {out} "
           f"--warmup {warmup} --extra-json '{EXTRA}'" + ("" if stream else " --no-stream"))
    rc, txt = sh(cmd, log=f"{OUT}/clients.log", timeout=3600)
    return json.load(open(out)) if rc == 0 and os.path.exists(out) else {"error": txt[-1500:]}


def quality(arm, files, out_dir):
    url, model = THEIRS if arm == "theirs" else OURS
    os.makedirs(out_dir, exist_ok=True)
    rc, txt = sh(f"{QCLIENT} --base-url {url} --model {model} --out-dir {out_dir} {files}", log=f"{OUT}/clients.log",
                 timeout=5400)
    q = f"{out_dir}/quality.json"
    return json.load(open(q)) if os.path.exists(q) else {"error": txt[-1500:]}


def weights_evidence(tag, pids_cmd):
    """h2h review #4: same target weights on both arms. Every safetensors file of the pinned 8B / 1.7B snapshots with
    its blob name (= the LFS sha256) and its recomputed sha256, plus the safetensors this arm's processes have
    memory-mapped right now (/proc/<host pid>/maps): which files each server actually serves from."""
    ev = f"{OUT}/evidence"
    hf = "/data/hf" if DOCKER else HF  # their container's cache has no hub/ level
    if not os.path.exists(f"{ev}/weights_snapshots.tsv"):
        rows = []
        for snap in (QWEN8, QWEN17):
            d = snap.replace(HF, hf)
            for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                if fn.endswith(".safetensors"):
                    real = os.path.realpath(os.path.join(d, fn))
                    _, h = sh(f"sha256sum {shlex.quote(real)} | cut -d' ' -f1")
                    rows.append(f"{snap}\t{fn}\t{os.path.basename(real)}\t{h.strip()}\t{'MATCH' if h.strip() == os.path.basename(real) else 'MISMATCH'}")
        open(f"{ev}/weights_snapshots.tsv", "w").write("snapshot\tfile\tblob\tsha256\tblob_check\n" + "\n".join(rows) + "\n")
    _, pids = sh(pids_cmd)
    out = []
    for pid in pids.split():
        try:
            for line in open(f"/proc/{pid}/maps"):
                if ".safetensors" in line or line.rstrip().endswith(".pt"):
                    out.append(f"{pid}\t{line.split()[-1]}")
        except OSError:
            pass
    open(f"{ev}/weights_mapped_{tag}.tsv", "w").write("\n".join(sorted(set(out))) + "\n")
    return len(set(o.split("\t")[1] for o in out))


def phase_setup_docker():
    ev = f"{OUT}/evidence"
    os.makedirs(ev, exist_ok=True)
    sh(f"cp /data/boot.status /data/ot_repo_head.txt /data/ot_sha256sums_check.txt /data/their_build.log /data/their_run.txt "
       f"/var/log/ot-boot.log {ev}/ 2>/dev/null; docker images --digests --no-trunc > {ev}/docker_images.txt; "
       f"docker inspect {THEIR_IMAGE} > {ev}/their_image_inspect.json; docker inspect {OURS_IMAGE} > {ev}/ours_image_inspect.json; "
       f"docker inspect ot-runtime > {ev}/their_container_inspect.json; "
       f"docker exec ot-runtime python3 -m pip freeze > {ev}/pip_freeze_theirs.txt 2>&1; "
       f"docker exec ot-runtime sh -c 'cd /app/kvt && find . -type f | LC_ALL=C sort | xargs sha256sum' > {ev}/app_kvt_files.sha256; "
       f"sha256sum {ev}/app_kvt_files.sha256 > {ev}/app_kvt_tree.sha256; "
       f"docker exec ot-runtime sha256sum /usr/local/bin/runtime-entrypoint > {ev}/their_entrypoint.sha256; "
       "docker exec ot-runtime dpkg-query -W -f " + shlex.quote("${Package}\t${Version}\n") +
       f" ca-certificates curl gosu python3 python3-pip python3-venv tini > {ev}/apt_their_line.tsv 2>&1; "
       f"docker run --rm --entrypoint pip {OURS_IMAGE} freeze > {ev}/pip_freeze_ours.txt 2>&1; "
       f"cp {ROOT}/ours/SNAPSHOT_SHA {ev}/ours_sha.txt; cp {ROOT}/h2h/SNAPSHOT_SHA {ev}/h2h_sha.txt; "
       f"(cd {ROOT} && sha256sum prompts/* prompts-dev/* calib/* maps_*.pt) > {ev}/inputs.sha256; "
       f"lscpu > {ev}/lscpu.txt; grep -m1 'model name' /proc/cpuinfo > {ev}/cpu_model.txt; "
       f"cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c > {ev}/cpu_governor.txt; "
       f"[ -s {ev}/cpu_governor.txt ] || echo 'no cpufreq governor exposed' > {ev}/cpu_governor.txt; "
       f"nvidia-smi > {ev}/nvidia-smi.txt; nvidia-smi -q > {ev}/nvidia-smi-q.txt; "
       f"curl -s -m 3 http://169.254.169.254/latest/meta-data/instance-type > {ev}/ec2_instance_type.txt; "
       f"curl -s -m 3 http://169.254.169.254/latest/meta-data/placement/availability-zone > {ev}/ec2_az.txt")
    sh(f"bash {ROOT}/h2h/bench/evidence.sh {ev}/start")
    mark("SETUP_DONE")


def phase_setup():
    if DOCKER:
        return phase_setup_docker()
    os.makedirs(f"{OUT}/evidence", exist_ok=True)
    ev = f"{OUT}/evidence"
    sh(f"cp {ROOT}/replay.sh {ROOT}/replay_map.tsv {ROOT}/replay.sh.meta.json {ROOT}/replay.log {ev}/")
    rc, txt = sh(f"tail -1 {ROOT}/replay.log")  # box_driver.sh ran their replayed Dockerfile before this python started
    if "REPLAY_DONE" not in txt:
        raise RuntimeError("their Dockerfile replay did not complete: see replay.log")
    sh(f"python3 -m pip freeze > {ev}/pip_freeze_theirs.txt; "
       "dpkg-query -W -f='${Package}\\t${Version}\\n' ca-certificates curl gosu python3 python3-pip python3-venv tini "
       f"> {ev}/apt_their_line.tsv; dpkg-query -W -f='${{Package}}\\t${{Version}}\\n' > {ev}/dpkg_all_after_replay.tsv; "
       f"(cd /app/kvt && find . -type f | LC_ALL=C sort | xargs sha256sum) > {ev}/app_kvt_files.sha256; "
       f"sha256sum {ev}/app_kvt_files.sha256 > {ev}/app_kvt_tree.sha256; "
       f"sha256sum /usr/local/bin/runtime-entrypoint > {ev}/their_entrypoint.sha256; cat /etc/os-release > {ev}/os-release.txt")
    # OUR side only below (their freeze/apt/sha256 recorded above): a C compiler for vLLM's compile + our venv
    sh("apt-get update -qq && apt-get install -y -qq --no-install-recommends gcc g++ python3-dev > /dev/null", log=f"{ev}/ours_setup.log",
       check=True)
    sh("python3 -m venv /opt/ours && /opt/ours/bin/pip install -q --no-cache-dir vllm==0.34.0",
       log=f"{ev}/ours_setup.log", check=True)
    sh(f"/opt/ours/bin/pip freeze > {ev}/pip_freeze_ours.txt; cp {ROOT}/ours/SNAPSHOT_SHA {ev}/ours_sha.txt; "
       f"cp {ROOT}/h2h/SNAPSHOT_SHA {ev}/h2h_sha.txt; (cd {ROOT} && sha256sum prompts/* prompts-dev/* calib/* maps_*.pt) > {ev}/inputs.sha256")
    sh(f"bash {ROOT}/h2h/bench/evidence.sh {ev}/start")
    sh(f"lscpu > {ev}/lscpu.txt; grep -m1 'model name' /proc/cpuinfo > {ev}/cpu_model.txt; "
       f"cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c > {ev}/cpu_governor.txt; "
       f"[ -s {ev}/cpu_governor.txt ] || echo 'no cpufreq governor exposed in the container' > {ev}/cpu_governor.txt; "
       f"nvidia-smi > {ev}/nvidia-smi.txt; nvidia-smi -q > {ev}/nvidia-smi-q.txt")
    mark("SETUP_DONE")


PUB_TOKENS = {"long": 13, "short": 8}  # published replies, Qwen3 tokenizer incl. EOS: "The retrieval code is AMBER-7421." / "...is Paris."
CAL_TOL_PCT = 6.0  # h2h lead's ruling (re-specified calibration): rebuilt 13,837 e2e within ~6% of 0.5465 s


def rebuilt_e2e(res):
    """h2h lead's re-specified calibration: e2e at the PUBLISHED reply length, from THEIR OWN completion_timing log:
    prepare + queue + GPU TTFT + (published_tokens - 1) x their measured ms/step + the measured HTTP overhead
    (client e2e - their server-side prepare+queue+gpu_section_total on the same calibration runs)."""
    import collections
    txt = sh("docker logs ot-runtime 2>&1")[1] if DOCKER else open(f"{OUT}/theirs_server_calib.log", errors="replace").read()
    open(f"{OUT}/calib/their_completion_timing.log", "w").write(
        "\n".join(l for l in txt.splitlines() if "completion_timing" in l) + "\n")
    rows = []
    for line in txt.splitlines():
        i = line.find('{"completion_tokens"')
        if i >= 0:
            try:
                d = json.loads(line[i:])
                rows.append(d)
            except ValueError:
                pass
    out = {}
    for kind, (lo, hi), client_key in (("long", (13000, 14700), "calib2_e2e"), ("short", (1, 64), "short_e2e")):
        sel = [d for d in rows if lo <= d["prompt_tokens"] <= hi and d["completion_tokens"] > 1]
        if not sel or not (res.get(client_key) or {}).get("median"):
            out[kind] = None
            continue
        med = lambda f: statistics.median(f(d) for d in sel)
        t = lambda d, k: d["timings_ms"].get(k, 0.0)
        prep, queue, ttft = med(lambda d: t(d, "prepare")), med(lambda d: t(d, "queue")), med(lambda d: t(d, "ttft_after_queue"))
        step = med(lambda d: t(d, "decode") / (d["completion_tokens"] - 1))
        server_total = med(lambda d: t(d, "prepare") + t(d, "queue") + t(d, "gpu_section_total"))
        http = 1e3 * res[client_key]["median"] - server_total
        n = PUB_TOKENS[kind]
        e2e = (prep + queue + ttft + (n - 1) * step + http) / 1e3
        out[kind] = {"rebuilt_e2e_s": e2e, "published_tokens": n, "n_requests": len(sel),
                     "measured_completion_tokens": sorted(set(d["completion_tokens"] for d in sel)),
                     "prepare_ms": prep, "queue_ms": queue, "gpu_ttft_ms": ttft, "ms_per_step": step, "http_ms": http,
                     "raw_client_e2e_s": res[client_key]["median"]}
    return out


def phase_calib():
    p, hs = start_theirs("calib")
    mark("THEIRS_UP", {"health_s": hs})
    res = {"health_s": hs}
    for kind, prompts, stream in (("short_e2e", f"{P}/short_france.jsonl", False),
                                  ("long_e2e", f"{P}/long_retrieval_13837.jsonl", False),
                                  ("long_sse", f"{P}/long_retrieval_13837.jsonl", True)) + \
            ():
        vals = []
        for k in range(5):
            r = client("theirs", prompts, f"{OUT}/calib/theirs/{kind}_{k}.json", max_tokens=32, stream=stream,
                       warmup=1 if k == 0 else 0)
            key = "median_ttft_s" if kind.endswith("sse") else "median_e2e_s"
            if "error" not in r:
                vals.append(r[key])
                res.setdefault(kind + "_text", r["records"][0]["text"][:120])
        res[kind] = {"values": vals, "median": statistics.median(vals) if vals else None}
    # calibration-only retrieval prompts (kvx-pairs, /vol/h2h/calib; never in the gate/ABBA). Rule fixed BEFORE any
    # output: probe all four once; calibrate e2e on the LOWEST-depth variant their model retrieves (AMBER-7421 in reply)
    probe = {}
    for dep in ("d075", "d090", "d097", "d100"):
        f = f"{ROOT}/calib/retrieval13837_{dep}.jsonl"
        r = client("theirs", f, f"{OUT}/calib/theirs/probe_{dep}.json", max_tokens=32, stream=False, warmup=0)
        txt = r["records"][0]["text"] if "records" in r else ""
        probe[dep] = {"retrieved": "AMBER-7421" in txt, "text": txt[:120], "e2e_s": r.get("median_e2e_s")}
    res["retrieval_vs_depth"] = probe
    chosen = next((d for d in ("d075", "d090", "d097", "d100") if probe[d]["retrieved"]), None)
    res["calib_prompt"] = chosen
    if chosen:
        f = f"{ROOT}/calib/retrieval13837_{chosen}.jsonl"
        for kind, stream in (("calib2_e2e", False), ("calib2_sse", True)):
            vals = []
            for k in range(5):
                r = client("theirs", f, f"{OUT}/calib/theirs/{kind}_{k}.json", max_tokens=32, stream=stream,
                           warmup=1 if k == 0 else 0)
                if "error" not in r:
                    vals.append(r["median_ttft_s" if stream else "median_e2e_s"])
                    res.setdefault(kind + "_text", r["records"][0]["text"][:120])
            res[kind] = {"values": vals, "median": statistics.median(vals) if vals else None}
        if res["calib2_e2e"]["median"]:
            res["long_off_pct"] = 100 * (res["calib2_e2e"]["median"] / PUBLISHED["long_e2e_s"] - 1)
    sh(f"nvidia-smi --query-gpu=memory.used --format=csv > {OUT}/calib/theirs_mem.txt")
    short = res["short_e2e"]["median"]
    res["short_off_pct"] = 100 * (short / PUBLISHED["short_e2e_s"] - 1) if short else None  # raw; reported, not gating
    res["long_raw_off_pct"] = res.get("long_off_pct")
    rb = rebuilt_e2e(res)
    res["rebuilt"] = rb
    res["long_off_pct"] = (100 * (rb["long"]["rebuilt_e2e_s"] / PUBLISHED["long_e2e_s"] - 1)) if rb.get("long") else None
    res["short_rebuilt_off_pct"] = (100 * (rb["short"]["rebuilt_e2e_s"] / PUBLISHED["short_e2e_s"] - 1)) if rb.get("short") else None
    json.dump(res, open(f"{OUT}/calib/summary.json", "w"), indent=1)
    mark("CALIB_DONE", {k: (v.get("median") if isinstance(v, dict) and "median" in v else v) for k, v in res.items()
                        if k != "retrieval_vs_depth"})  # retrieval_vs_depth has no median (DO run 1 crashed here)
    return p, res


def phase_devpick(theirs_proc, tails, maps):
    devfiles = f"{D}/needles_*.jsonl {D}/hellaswag*.jsonl"
    qt = quality("theirs", devfiles, f"{OUT}/dev/theirs")
    mark("DEV_THEIRS", {k: qt.get(k) for k in ("correct", "n")})
    import math
    pick = None
    tried = []
    for t in tails:
        cfg = {"maps": maps, "tail": t, "util": 0.33, "fp8": 1, "src_sdpa": "vllm_fa"}
        try:
            p, hs = start_ours(cfg, f"dev_tail{t}")
        except Exception as e:
            tried.append({"tail": t, "error": repr(e)[:500]})
            mark("DEV_OURS_FAIL", {"tail": t})
            continue
        qo = quality("ours", devfiles, f"{OUT}/dev/ours_tail{t}")
        stop(p)
        ok = None
        if "correct" in qo and "correct" in qt:
            pt, po = qt["correct"] / qt["n"], qo["correct"] / qo["n"]
            ci = 1.96 * math.sqrt(pt * (1 - pt) / qt["n"] + po * (1 - po) / qo["n"])
            ok = po >= pt - ci
        tried.append({"tail": t, "health_s": hs, "correct": qo.get("correct"), "n": qo.get("n"), "pass": ok})
        mark("DEV_OURS", tried[-1])
        if ok and pick is None:
            pick = cfg
            break  # smallest passing tail (tails are tried smallest first)
    frozen = {"pick": pick, "tried": tried, "theirs_dev": {k: qt.get(k) for k in ("correct", "n")},
              "rule": "smallest tail with dev quality >= theirs - CI (h2h-tail-frozen), dev files only"}
    s = json.dumps(pick, sort_keys=True)
    import hashlib
    frozen["pick_sha256"] = hashlib.sha256(s.encode()).hexdigest()
    json.dump(frozen, open(f"{OUT}/frozen_config.json", "w"), indent=1)
    mark("FROZEN", {"pick": pick, "sha": frozen["pick_sha256"]})
    return frozen


def phase_abba(cfg, calib, rounds=4):
    ours, hs = start_ours(cfg, "abba")
    mark("OURS_UP_ABBA", {"health_s": hs})
    sh(f"nvidia-smi --query-gpu=memory.used --format=csv > {OUT}/both_resident_mem.txt")
    # lead: log GPU util AND each arm's process CPU% during every client run (two resident servers share the host CPUs)
    arms = (["--arm", "theirs=docker:ot-runtime", "--arm", "ours=docker:ours"] if DOCKER else
            ["--arm", f"theirs=pidfile:{OUT}/theirs.pid", "--arm", f"ours=pidfile:{OUT}/ours.pid"])
    sampler = subprocess.Popen(["python3", f"{ROOT}/ours/h2h/arm_sampler.py", f"{OUT}/arm_samples.jsonl"] + arms,
                               start_new_session=True)
    # h2h lead: their side must not be memory-starved by co-residency. Their startup log (KV/context budget lines) and
    # the 13,837-token retrieval again WITH ours resident, to compare with the theirs-only calibration phase.
    sh(f"docker logs ot-runtime > {OUT}/evidence/their_startup.log 2>&1" if DOCKER else
       f"cp {OUT}/theirs_server_calib.log {OUT}/evidence/their_startup.log")
    calib_prompt = calib.get("calib_prompt")
    if calib_prompt:
        f = f"{ROOT}/calib/retrieval13837_{calib_prompt}.jsonl"
        vals = []
        for k in range(5):
            r = client("theirs", f, f"{OUT}/calib/theirs_coresident/calib2_e2e_{k}.json", max_tokens=32, stream=False,
                       warmup=1 if k == 0 else 0)
            if "error" not in r:
                vals.append(r["median_e2e_s"])
        mark("COResident_CALIB", {"calib2_e2e_values": vals, "median": statistics.median(vals) if vals else None,
                                  "theirs_alone_median": (calib.get("calib2_e2e") or {}).get("median")})
    co_med = statistics.median(vals) if calib_prompt and vals else None
    t_pids = "docker top ot-runtime -eo pid | tail -n +2" if DOCKER else f"pgrep -g $(ps -o pgid= -p $(cat {OUT}/theirs.pid))"
    o_pids = "docker top ours -eo pid | tail -n +2" if DOCKER else f"pgrep -g $(ps -o pgid= -p $(cat {OUT}/ours.pid))"
    mark("WEIGHTS_EVIDENCE", {"theirs_mapped_files": weights_evidence("theirs", t_pids),
                              "ours_mapped_files": weights_evidence("ours", o_pids)})
    # h2h review #1/#2: calibration as numbers in the units verdict.py compares (e2e s, non-streaming retrieval)
    long_alone = ((calib.get("rebuilt") or {}).get("long") or {}).get("rebuilt_e2e_s")
    co_med = co_med if co_med is None else co_med  # raw co-resident e2e (max_tokens 32): compare raw with raw below
    raw_alone = (calib.get("calib2_e2e") or calib["long_e2e"])["median"]
    probe = lambda c: (lambda r: r[1].strip() if r[0] == 0 else "")(sh(c + " 2>/dev/null"))
    machine = " | ".join(x for x in (probe(f"cat {OUT}/evidence/ec2_instance_type.txt"),
                                     probe("nvidia-smi --query-gpu=name --format=csv,noheader | head -1"),
                                     probe("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2")) if x)
    maps_sha = cfg.get("maps_sha256") or sh(f"sha256sum {shlex.quote(cfg['maps'])} | cut -d' ' -f1")[1].strip()
    pf = f"{ROOT}/calibration_prereg.json"  # pre-registered before the box (h2h lead's ruling, ot-box-do2)
    prereg = json.load(open(pf)) if os.path.exists(pf) else None
    prereg_sha = sh(f"sha256sum {pf} | cut -d' ' -f1")[1].strip() if prereg else None
    calibration = {
        "published": PUBLISHED["long_e2e_s"], "theirs_measured": long_alone,
        "metric": ("RE-SPECIFIED (h2h lead's ruling, pre-registered before the box): e2e s at the PUBLISHED reply length (13 tokens) "
                   "on the 13,837-token retrieval (" + (f"calib variant {calib_prompt}, which their model retrieves from"
                   if calib.get("calib2_e2e") else "the published prompt") + "), their side ALONE, rebuilt from THEIR OWN "
                   "completion_timing log: prepare + queue + GPU TTFT + 12 x their ms/step + measured HTTP overhead; "
                   f"gate within {CAL_TOL_PCT}% of 0.5465 s"),
        "rebuilt_components": (calib.get("rebuilt") or {}).get("long"),
        "raw": {"e2e_s_max_tokens_32": raw_alone, "off_pct": calib.get("long_raw_off_pct"),
                "why_not_gating": "their reply on this prompt runs to max_tokens (32) instead of the published 13-token reply"},
        "ttft_path_stability": {"this_box_sse_ttft_s": (calib.get("calib2_sse") or {}).get("median"),
                                "do_run1_20261123T1312Z_s": 0.330, "modal_20261123T0731Z_s": 0.331},
        "source_url": SRC_URL, "machine": machine,
        "short": {"published": PUBLISHED["short_e2e_s"], "measured_raw": calib["short_e2e"]["median"],
                  "rebuilt_8_tokens": (calib.get("rebuilt") or {}).get("short"), "gating": False,
                  "disclosure": ("reported, not gating (h2h lead): the residual gap is their CPU-bound HF decode speed "
                                 "(ms/step on this host CPU vs ~18.5 implied by the published number), outside the TTFT headline")},
        "coresident": {"theirs_measured_raw": co_med, "theirs_alone_raw": raw_alone,
                       "diff_pct": (100 * (co_med / raw_alone - 1)) if co_med and raw_alone else None,
                       "flag_over_5pct": bool(co_med and raw_alone and abs(co_med / raw_alone - 1) > 0.05),
                       "why": "their side re-timed with ours resident: memory/CPU starvation check (h2h lead)"},
        "ttft": "their package publishes no TTFT; TTFT is measured on this box in the ABBA rounds",
        "explained": ("Not an environment excuse: their reply length differs from their own published replies "
                      "(generation behaviour), and decode speed is not part of the headline metric; the GPU-bound TTFT path, "
                      "which IS the headline, already matches across two independent boxes (DO 0.330 s vs Modal 0.331 s)."),
        "prereg": prereg, "prereg_sha256": prereg_sha}
    fz = json.load(open(f"{OUT}/frozen_config.json"))
    assert fz.get("headline_length") in (8192, 16384), "pre-register the headline length in the frozen config before the box"
    for L in ABBA_LENGTHS:
        rd = f"{OUT}/abba_{L}"
        plan = {"run_id": f"{RUN_ID}-{L}", "metric": {"tool": "json", "key": "median_ttft_s", "direction": "lower", "unit": "s"},
                "client_cmd": (f"{CLIENT} --base-url http://127.0.0.1:{{port}} --model {{served_model}} --prompts {{prompts_file}} "
                               f"--max-tokens 16 --out {{round_dir}}/bench.json --warmup 1 --extra-json '{EXTRA}'"),
                "prompts_file": f"{P}/needles_{L}.jsonl", "model_dir": QWEN8, "rounds": rounds,
                "calibration": calibration,
                "quality_note": ("quality runs in the LAST ABBA round with BOTH servers resident (co-resident, as intended); "
                                 "quality.json per arm = needles at this length only (the per-length map gate); HellaSwag-100 "
                                 "in {arm_dir}/hellaswag is informational: ours serves prompts < 512 tokens natively by config "
                                 "(min_len 512), so it compares THEIR mapped path with OUR native short-prompt path"),
                "frozen_config": cfg, "frozen_config_sha256": fz["pick_sha256"],
                "preregistration": {
                    "headline_length": fz["headline_length"], "frozen_before_box": True,
                    "config_rule": "ONE frozen config for all lengths, picked on DEV prompts only (h2h-tail-frozen, ot-margin-first)",
                    "dev_needles_of_frozen_config": fz.get("dev_needles"),
                    "timing_and_gate_prompts": "held-out /vol/h2h/prompts (built 2026-11-23T07:12Z, before any pick), disjoint from the dev set",
                    "disclosure": ("Modal calibration boot 20261123T0731Z-onetriangle-calib ran 3 TTFT screens of OURS on these gate "
                                   "files; recorded as screens only, never used for a pick")},
                "resident": True,
                "arms": [
                    {"name": "theirs", "artifact": ("OneTriangle public Docker release @8f05e5f1, " + ("real docker build of their Dockerfile (image digest in evidence/docker_images.txt)" if DOCKER else "Dockerfile replayed") + ", kvt 3efa8e6e"),
                     "port": 8000, "served_model": THEIRS[1], "start": "true", "stop": "",
                     "health": THEIRS[0] + "/health", "pids_cmd": t_pids,
                     "memory": "as packaged: their runtime has no GPU-memory flag (see evidence/their_startup.log)",
                     "quality": (f"{QCLIENT} --base-url {THEIRS[0]} --model {THEIRS[1]} --out-dir {{arm_dir}} {P}/needles_{L}.jsonl; "
                                 f"{QCLIENT} --base-url {THEIRS[0]} --model {THEIRS[1]} --out-dir {{arm_dir}}/hellaswag {P}/hellaswag100.jsonl")},
                    {"name": "ours", "artifact": (f"ultra-cross-model-kv @{open(ROOT + '/ours/SNAPSHOT_SHA').read().strip()} vLLM 0.34 + "
                                              f"CrossModelConnector; target {QWEN8.rsplit('/', 3)[1]}@{QWEN8.rsplit('/', 1)[1][:8]}; "
                                              f"source {QWEN17.rsplit('/', 3)[1]}@{QWEN17.rsplit('/', 1)[1][:8]}; maps {os.path.basename(cfg['maps'])} "
                                              f"sha256 {maps_sha}; per-file weight shas in evidence/weights_snapshots.tsv"),
                     "port": 8001, "served_model": OURS[1], "start": "true", "stop": "",
                     "health": OURS[0] + "/health", "pids_cmd": o_pids,
                     "memory": f"--gpu-memory-utilization {cfg.get('util', 0.5)}",
                     "quality": (f"{QCLIENT} --base-url {OURS[0]} --model {OURS[1]} --out-dir {{arm_dir}} {P}/needles_{L}.jsonl; "
                                 f"{QCLIENT} --base-url {OURS[0]} --model {OURS[1]} --out-dir {{arm_dir}}/hellaswag {P}/hellaswag100.jsonl")}]}
        os.makedirs(rd, exist_ok=True)
        json.dump(plan, open(f"{rd}.plan.json", "w"), indent=1)
        rc, txt = sh(f"python3 {ROOT}/h2h/bench/run_ab.py {rd}.plan.json {rd}", log=f"{OUT}/abba.log", timeout=7200)
        vr, vt = sh(f"python3 {ROOT}/h2h/tools/verdict.py {rd}")
        open(f"{rd}/verdict_local.txt", "w").write(vt)
        mark("ABBA_DONE", {"L": L, "rc": rc, "verdict": vt[:600]})
    sampler.terminate()
    stop(ours)


def phase_native():
    p, hs = start_ours({"maps": "", "tail": 0, "util": 0.33}, "native", mapped=False)
    for L in LENGTHS:
        client("ours", f"{P}/needles_{L}.jsonl", f"{OUT}/native_context/ttft_{L}.json")
    stop(p)
    mark("NATIVE_DONE")


def main():
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1]
    if what == "all":
        phase_setup()
        tp, calib = phase_calib()
        off0, loff0 = calib.get("short_off_pct"), calib.get("long_off_pct")
        if os.environ.get("CALIB_HARD_STOP") == "1" and (loff0 is None or abs(loff0) > CAL_TOL_PCT):
            # lead (Czechia 1.8 GHz try): hard stop right after calibration if EITHER calibration is > 5% off
            mark("CALIB_FAIL_STOP", {"short_off_pct": off0, "long_off_pct": loff0, "calib_prompt": calib.get("calib_prompt")})
            stop(tp)
            sh(f"bash {ROOT}/h2h/bench/evidence.sh {OUT}/evidence/end")
            mark("ALL_DONE")
            return
        if os.environ.get("FROZEN_CONFIG"):
            # lead (on-demand box): config frozen on the Modal dev pick before launch; no paid dev pick on the box
            frozen = json.load(open(os.environ["FROZEN_CONFIG"]))
            json.dump(frozen, open(f"{OUT}/frozen_config.json", "w"), indent=1)
            mark("FROZEN", {"pick": frozen["pick"], "sha": frozen["pick_sha256"], "from": os.environ["FROZEN_CONFIG"]})
        else:
            frozen = phase_devpick(tp, [int(x) for x in os.environ.get("TAILS", "1,64,256").split(",")],
                                   os.environ.get("MAPS", f"{ROOT}/maps_k24.pt"))
        off, loff = calib.get("short_off_pct"), calib.get("long_off_pct")
        # lead: ABBA starts on its own only if BOTH calibrations are within 5% (short e2e vs 0.1877 s AND long e2e on
        # the retrieving calib variant vs 0.5465 s); otherwise wait for the lead's go with the numbers sent
        if loff is None or abs(loff) > CAL_TOL_PCT:
            mark("WAIT_GO", {"short_off_pct": off, "long_off_pct": loff, "calib_prompt": calib.get("calib_prompt"),
                             "why": "a calibration is > 5% off (or missing) -> lead's go before ABBA"})
            while not os.path.exists(f"{ROOT}/GO_ABBA") and not os.path.exists(f"{ROOT}/STOP"):
                time.sleep(20)
            if os.path.exists(f"{ROOT}/STOP"):
                stop(tp)
                mark("STOPPED")
                return
        if frozen["pick"] is None:
            mark("NO_PASSING_CONFIG")
            stop(tp)
            return
        phase_abba(frozen["pick"], calib)
        phase_native()
        stop(tp)
        sh(f"bash {ROOT}/h2h/bench/evidence.sh {OUT}/evidence/end")
        mark("ALL_DONE")


if __name__ == "__main__":
    main()
