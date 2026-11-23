"""Sample, once a second, GPU utilization/memory (nvidia-smi) and each arm's total process CPU% while an ABBA run goes,
so a resident (idle) arm can be shown to do no GPU work and little CPU while the other arm is measured.

  python3 arm_sampler.py OUT.jsonl --arm theirs=docker:ot-runtime --arm ours=docker:ours [--arm x=pidfile:/tmp/x.pid]
Each line: {"t", "gpu_util", "gpu_mem_mib", "cpu": {arm: percent over all its processes}}. Stops on SIGTERM.
"""
import json
import os
import signal
import subprocess
import sys
import time

HZ = os.sysconf("SC_CLK_TCK")


def pids(spec):
    kind, _, ref = spec.partition(":")
    if kind == "docker":
        out = subprocess.run(["docker", "top", ref, "-eo", "pid"], capture_output=True, text=True).stdout.split()[1:]
        return [int(x) for x in out if x.isdigit()]
    if kind == "pidfile":
        try:
            root = int(open(ref).read().strip())
        except (OSError, ValueError):
            return []
        out = subprocess.run(["ps", "-eo", "pid,pgid"], capture_output=True, text=True).stdout.split("\n")[1:]
        pg = os.getpgid(root) if os.path.exists(f"/proc/{root}") else None
        return [int(a) for a, b in (ln.split() for ln in out if ln.strip()) if pg is not None and int(b) == pg]
    return []


def ticks(pid):
    try:
        f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        return int(f[11]) + int(f[12])  # utime + stime
    except (OSError, IndexError, ValueError):
        return 0


def main():
    out = sys.argv[1]
    arms = dict(a.split("=", 1) for a in sys.argv[3::2]) if len(sys.argv) > 2 else {}
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    prev = {}
    with open(out, "a") as fh:
        while not stop:
            t = time.time()
            g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout.strip().split(",")
            cpu = {}
            for arm, spec in arms.items():
                tot = sum(ticks(p) for p in pids(spec))
                if arm in prev:
                    dt = t - prev[arm][1]
                    cpu[arm] = round(100.0 * (tot - prev[arm][0]) / HZ / dt, 1) if dt > 0 else None
                prev[arm] = (tot, t)
            fh.write(json.dumps({"t": round(t, 2), "gpu_util": int(g[0]) if g and g[0].strip().isdigit() else None,
                                 "gpu_mem_mib": int(g[1]) if len(g) > 1 and g[1].strip().isdigit() else None,
                                 "cpu": cpu}) + "\n")
            fh.flush()
            time.sleep(max(0.0, 1.0 - (time.time() - t)))


if __name__ == "__main__":
    main()
