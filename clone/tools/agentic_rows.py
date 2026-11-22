"""Multi-turn bundle rows (tau2, Terminal-Bench 2.0) from map_once runs: pool halves / batches, Wilson CIs, and both
mapped fractions from the connector's "kvx map_once" log lines (first requests, all request tokens).

python tools/agentic_rows.py tau2 <dir with <domain>-<half>/{summary.json,vllm.log}>
python tools/agentic_rows.py tb2 <harbor jobs root with xkvdistill27mo2-b*/> <dir with b*/vllm.log>
Prints JSON per row (value, n, ci, passed, mapped fractions, quality_ok); the caller writes ours.json and must not
write a row with quality_ok false (it carries "invalid").
"""
import glob
import json
import math
import os
import re
import sys

PAT = re.compile(r"kvx map_once: first requests (\d+) mapped ([0-9.]+) of their prompts; later requests (\d+) mapped "
                 r"([0-9.]+); all ([0-9.]+)")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fracs(logs):
    """Token-weighted over runs: the last map_once line of each log is cumulative for that server."""
    tot = dict(first_n=0, later_n=0, first=[], all=[])
    for p in logs:
        last = None
        for line in open(p, errors="replace"):
            m = PAT.search(line)
            if m:
                last = m
        if last:
            fn, ff, ln, lf, af = last.groups()
            tot["first_n"] += int(fn)
            tot["later_n"] += int(ln)
            tot["first"].append((int(fn), float(ff)))
            tot["all"].append((int(fn) + int(ln), float(af)))
    w = lambda xs: round(sum(n * f for n, f in xs) / max(1, sum(n for n, _ in xs)), 4) if xs else None
    return dict(first_requests=tot["first_n"], later_requests=tot["later_n"], mapped_frac_first=w(tot["first"]),
                mapped_frac_all_approx=w(tot["all"]))


def row(k, n, extra, conversations):
    """Guard (lead): a silent map_once fallback (later turns re-mapped as new prompts) must never become a row. Later
    requests expected = all requests - conversations; quality_ok only if >= 90% of them reused the stored prefix."""
    lo, hi = wilson(k, n)
    total = extra.get("first_requests", 0) + extra.get("later_requests", 0)
    expected = max(0, total - conversations)
    ok = expected > 0 and extra.get("later_requests", 0) >= 0.9 * expected
    return dict(passed=k, n=n, value=round(k / n, 4), ci=round((hi - lo) / 2, 4), lo=round(lo, 4), hi=round(hi, 4),
                conversations=conversations, later_expected=expected, quality_ok=ok,
                **({} if ok else {"invalid": f"map_once reused the stored prefix on {extra.get('later_requests', 0)} "
                                             f"of {expected} later requests (< 90%): silent re-map fallback"}),
                **extra)


def tau2(d):
    out, K, N = {}, 0, 0
    for dom in ("airline", "retail", "telecom"):
        k = n = 0
        logs = []
        for half in ("a", "b"):
            s = json.load(open(f"{d}/{dom}-{half}/summary.json"))
            s = s[0] if isinstance(s, list) else s
            k += s["passed"]
            n += s["scored"]
            logs += glob.glob(f"{d}/{dom}-{half}/vllm.log")
        out[dom] = row(k, n, fracs(logs), n)  # one conversation per tau2 simulation
        K, N = K + k, N + n
    out["overall"] = row(K, N, fracs(glob.glob(f"{d}/*/vllm.log")), N)
    bad = [dom for dom in ("airline", "retail", "telecom") if not out[dom]["quality_ok"]]
    if bad:  # one silently re-mapped domain invalidates the pooled row too
        out["overall"]["quality_ok"] = False
        out["overall"]["invalid"] = f"domain(s) {', '.join(bad)} failed the map_once guard"
    return out


def tb2(jobs, logdir):
    k = n = 0
    names = []
    for f in glob.glob(f"{jobs}/xkvdistill27mo2-b*/*/*/result.json"):
        r = json.load(open(f))
        rew = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        if rew is None:
            continue
        n += 1
        k += rew >= 1.0
        if rew >= 1.0:
            names.append(r.get("task_name"))
    return dict(tb2=row(k, n, dict(fracs(glob.glob(f"{logdir}/*/vllm.log")), solved=sorted(names)), n))


if __name__ == "__main__":
    print(json.dumps(tau2(sys.argv[2]) if sys.argv[1] == "tau2" else tb2(sys.argv[2], sys.argv[3]), indent=1))
