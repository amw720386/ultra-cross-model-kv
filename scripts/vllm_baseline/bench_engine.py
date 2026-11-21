"""Paired TTFT of this repo's engine: native Qwen3.8-27B vs Qwen3.5-4B -> 27B transfer in fast mode. The calls are the repo's own perf.task_ttft_repair (repair 0): native_prefill(chunk=8192) and
transfer_prefill(adaptive tail, window 8192, full_until 32768, chunk 8192), timed with a sync on each side.

Arms are interleaved (native, transfer, native, ...) after one warm-up each, so clock drift hits both alike.

  --mode random : random token ids (1000..100000, as the repo) at --lengths
  --mode real   : prompt --index from --ids (make_prompts.py's output); one prompt per process, since each new prompt
                  length captures another graphed tail and the engine keeps ~4 GiB per transfer

    PYTHONPATH=src python scripts/vllm_baseline/bench_engine.py --mode random --out engine-random.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [xkv] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["random", "real"], required=True)
    ap.add_argument("--lengths", default="8192,16384,32768")
    ap.add_argument("--ids", default="prompts.json")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--mapper", default="4b-27b-fid+4b-27b-fid-attn")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from xkv.experiments import get, load_mapper
    from xkv.mapper import adaptive_tail, native_prefill, transfer_prefill

    torch.manual_seed(0)
    sm, _ = get("4b")
    tm, _ = get("27b")
    mapper = load_mapper(a.mapper, sm, tm)
    log(f"gpu={torch.cuda.get_device_name()} torch={torch.__version__} mapper={a.mapper} "
        f"mem={torch.cuda.memory_allocated() / 1e9:.1f} GB")

    def native(ids):
        return native_prefill(tm, ids, chunk=8192)

    def transfer(ids):
        return transfer_prefill(sm, tm, mapper, ids, tail=adaptive_tail(ids.shape[1]), window=8192, chunk=8192,
                                full_until=32768)

    def timed(fn, ids):
        torch.cuda.synchronize()
        t = time.perf_counter()
        _, logits = fn(ids)
        first = int(logits[0, -1].argmax())
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        del logits
        return dt, first

    if a.mode == "random":
        prompts = [{"name": f"random-{L}", "ids": torch.randint(1000, 100000, (1, L), device="cuda")}
                   for L in (int(x) for x in a.lengths.split(","))]
    else:
        rows = json.loads(Path(a.ids).read_text())
        r = rows[a.index]
        prompts = [{"name": f"longhealth-{r['index']}-{r['pid']}-q{r['qi']}",
                    "ids": torch.tensor([r["ids"]], device="cuda")}]

    out = {"engine": "ultra-cross-model-kv", "gpu": torch.cuda.get_device_name(),
           "torch": torch.__version__, "mapper": a.mapper, "mode": a.mode, "reps": a.reps, "rows": []}
    for p in prompts:
        ids, T = p["ids"], p["ids"].shape[1]
        warm = {"native": timed(native, ids)[0], "transfer": timed(transfer, ids)[0]}
        ts = {"native": [], "transfer": []}
        first = {}
        for _ in range(a.reps):
            for arm, fn in (("native", native), ("transfer", transfer)):
                dt, tok = timed(fn, ids)
                ts[arm].append(dt)
                first[arm] = tok
        torch.cuda.empty_cache()
        row = {"prompt": p["name"], "tokens": T, "tail": adaptive_tail(T), "warmup_s": warm,
               "native_s": ts["native"], "transfer_s": ts["transfer"],
               "native_median_s": statistics.median(ts["native"]),
               "transfer_median_s": statistics.median(ts["transfer"]),
               "first_token": first}
        row["speedup"] = row["native_median_s"] / row["transfer_median_s"]
        out["rows"].append(row)
        log(f"{p['name']} T={T}: native {row['native_median_s']:.3f}s transfer {row['transfer_median_s']:.3f}s "
            f"-> {row['speedup']:.2f}x  first tokens {first}  peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    Path(a.out).write_text(json.dumps(out, indent=1))
    log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
