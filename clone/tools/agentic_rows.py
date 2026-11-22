"""Multi-turn bundle rows (tau2, Terminal-Bench 2.0) from map_once runs: pool halves / batches, Wilson CIs, and both
mapped fractions from the connector's "kvx map_once" log lines (first requests, all request tokens).

python tools/agentic_rows.py tau2 <dir with <domain>-<half>/{summary.json,vllm.log}>
python tools/agentic_rows.py tb2 <harbor jobs root with xkvcode27mo2-b*/> <dir with b*/vllm.log>
Prints JSON per row (value, n, ci, passed, mapped fractions, quality_ok); the caller writes ours.json and must not
write a row with quality_ok false (it carries "invalid").
"""
import glob
import json
import math
import os
import re
import sys

PAT = re.compile(r"kvx\]? ?map_once: first requests (\d+) mapped ([0-9.]+) of their prompts; later requests (\d+) mapped "
                 r"([0-9.]+); all ([0-9.]+)")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


HIT = re.compile(r"External prefix cache hit rate: ([0-9.]+)%")


def fracs(logs):
    """Per server log: requests = access-log POSTs to /v1/chat/completions (exact); the last map_once counter line
    (cumulative; exact only if the connector printed one per request) and vLLM's cumulative connector hit rate
    (= mapped share of all prompt tokens). Summed / request-weighted over logs."""
    tot = dict(requests=0, first_n=0, later_n=0, lines=0, first=[], all=[], hit=[])
    for p in logs:
        last, lines, posts, hit = None, 0, 0, None
        for line in open(p, errors="replace"):
            if '"POST /v1/chat/completions HTTP' in line:
                posts += 1
            m = PAT.search(line)
            if m:
                last, lines = m, lines + 1
            h = HIT.search(line)
            if h:
                hit = float(h.group(1)) / 100
        tot["requests"] += posts
        tot["lines"] += lines
        if hit is not None:
            tot["hit"].append((posts, hit))
        if last:
            fn, ff, ln, lf, af = last.groups()
            tot["first_n"] += int(fn)
            tot["later_n"] += int(ln)
            tot["first"].append((int(fn), float(ff)))
            tot["all"].append((int(fn) + int(ln), float(af)))
    w = lambda xs: round(sum(n * f for n, f in xs) / max(1, sum(n for n, _ in xs)), 4) if xs else None
    return dict(requests=tot["requests"], counter_lines=tot["lines"], first_requests=tot["first_n"],
                later_requests=tot["later_n"], mapped_frac_first=w(tot["first"]),
                mapped_frac_all=w(tot["hit"]) or w(tot["all"]))


def row(k, n, extra, conversations):
    """Guard (lead): a silent map_once fallback (later turns re-mapped as new prompts) must never become a row.
    Later requests expected = requests - conversations; quality_ok only if >= 90% of them reused the stored prefix.
    The counters must be complete (one counter line per request, as the connector prints since 17d5177): otherwise
    quality_ok is None (unverified) and the row is not written either."""
    lo, hi = wilson(k, n)
    expected = max(0, extra["requests"] - conversations)
    complete = extra["counter_lines"] >= 0.9 * extra["requests"] > 0
    ok = (expected > 0 and extra["later_requests"] >= 0.9 * expected) if complete else None
    why = {}
    if ok is None:
        why["unverified"] = (f"map_once counters incomplete ({extra['counter_lines']} lines for {extra['requests']} "
                             f"requests); connector hit rate {extra['mapped_frac_all']} (re-map runs ~0.955)")
    elif not ok:
        why["invalid"] = (f"map_once reused the stored prefix on {extra['later_requests']} of {expected} later "
                          f"requests (< 90%): silent re-map fallback")
    return dict(passed=k, n=n, value=round(k / n, 4), ci=round((hi - lo) / 2, 4), lo=round(lo, 4), hi=round(hi, 4),
                conversations=conversations, later_expected=expected, quality_ok=ok, **why, **extra)


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
    bad = [dom for dom in ("airline", "retail", "telecom") if out[dom]["quality_ok"] is not True]
    if bad:  # one silently re-mapped domain invalidates the pooled row too
        out["overall"]["quality_ok"] = False if any(out[x]["quality_ok"] is False for x in bad) else None
        out["overall"]["invalid"] = f"domain(s) {', '.join(bad)} failed or did not verify the map_once guard"
    return out


def tb2(jobs, logdir, fit_file=None):
    """fit_file (decision tb2-long-excluded): the row counts only tasks listed in it (agent timeout <= 1800 s); the
    worst case counts every other task as a failure over all TB2 tasks; retention on the batch with a native arm."""
    fit = set(json.load(open(fit_file))["tasks"]) if fit_file else None
    total_tasks = len(fit) + len(json.load(open(fit_file))["excluded_long"]) if fit_file else None
    k = n = k_all = 0
    names, per_batch = [], {}
    for f in glob.glob(f"{jobs}/xkvcode27mo2-b*/*/*/result.json"):
        r = json.load(open(f))
        rew = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        if rew is None:
            continue
        t = r.get("task_name")
        b = f.split("xkvcode27mo2-")[1].split("/")[0]
        k_all += rew >= 1.0
        if fit is not None and t not in fit:
            continue
        n += 1
        k += rew >= 1.0
        pb = per_batch.setdefault(b, [0, 0])
        pb[0] += rew >= 1.0
        pb[1] += 1
        if rew >= 1.0:
            names.append(t)
    out = row(k, n, dict(fracs(glob.glob(f"{logdir}/*/vllm.log")), solved=sorted(names), per_batch=per_batch), n)
    if fit is not None:
        lo, hi = wilson(k_all, total_tasks)
        out["worst_case"] = dict(passed=k_all, n=total_tasks, value=round(k_all / total_tasks, 4), lo=round(lo, 4),
                                 hi=round(hi, 4))
        out["fit_tasks"], out["unscored_fit"] = len(fit), len(fit) - n
    return dict(tb2=out)


if __name__ == "__main__":
    print(json.dumps(tau2(sys.argv[2]) if sys.argv[1] == "tau2" else
                     tb2(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None), indent=1))
