"""Native Qwen3.8-27B time to first token in vLLM (newest release), batch 1, one request at a time.

The same prompts as bench_engine.py: random token ids at --lengths, and the real prompts in --ids. Prefix caching is
off, so every request prefills its whole prompt. TTFT = wall time of generate(max_tokens=1) from the token ids:
scheduling, chunked prefill, the first token's sampling and output processing (a few ms over the bare prefill).
One warm-up per prompt, then --reps timed requests. Run it in its own environment (vLLM pins its own torch):

    uv venv vllm-env && VIRTUAL_ENV=vllm-env uv pip install vllm==0.34.0
    vllm-env/bin/python scripts/vllm_baseline/bench_vllm.py --model <Qwen3.8-27B snapshot> --ids prompts.json --out vllm.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [vllm] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="8192,16384,32768")
    ap.add_argument("--ids", default="prompts.json")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-num-batched-tokens", type=int, default=0, help="0 = vLLM's default")
    ap.add_argument("--compile-mode", type=int, default=-1, help="-1 = vLLM's default; 3 = VLLM_COMPILE")
    ap.add_argument("--max-num-seqs", type=int, default=128,
                    help="vLLM's default (1024) exceeds the DeltaNet state slots that fit beside the 27B on 80 GB")
    ap.add_argument("--decode-tokens", type=int, default=64, help="decode speed on the first real prompt (0 = skip)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = dict(model=a.model, dtype="bfloat16", max_model_len=40960, enable_prefix_caching=False,
              gpu_memory_utilization=0.90, seed=0, limit_mm_per_prompt={"image": 0, "video": 0},
              max_num_seqs=a.max_num_seqs)
    if a.max_num_batched_tokens:
        kw["max_num_batched_tokens"] = a.max_num_batched_tokens
    if a.compile_mode >= 0:
        kw["compilation_config"] = {"mode": a.compile_mode}
    t0 = time.perf_counter()
    llm = LLM(**kw)
    load_s = time.perf_counter() - t0
    try:  # the attribute path has moved between releases; a missing one must not cost the run
        sc = llm.llm_engine.vllm_config.scheduler_config
        sched = {"max_num_batched_tokens": sc.max_num_batched_tokens, "max_num_seqs": sc.max_num_seqs,
                 "chunked_prefill": getattr(sc, "enable_chunked_prefill", None)}
    except Exception as e:  # noqa: BLE001
        sched = {"unread": repr(e), "requested_max_num_batched_tokens": a.max_num_batched_tokens or "default"}
    log(f"vllm {vllm.__version__} torch {torch.__version__} loaded in {load_s:.0f}s; scheduler {sched}")

    sp = SamplingParams(max_tokens=1, temperature=0.0)
    gen = __import__("random").Random(0)
    prompts = [{"name": f"random-{L}", "ids": [gen.randint(1000, 99999) for _ in range(L)]}
               for L in (int(x) for x in a.lengths.split(","))]
    if a.ids and Path(a.ids).exists():
        for r in json.loads(Path(a.ids).read_text()):
            prompts.append({"name": f"longhealth-{r['index']}-{r['pid']}-q{r['qi']}", "ids": r["ids"]})

    def once(ids):
        t = time.perf_counter()
        o = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
        return time.perf_counter() - t, o[0].outputs[0].token_ids[0]

    out = {"engine": f"vllm {vllm.__version__}", "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
           "load_s": load_s, "scheduler": sched, "compile_mode": a.compile_mode, "reps": a.reps, "rows": []}
    for p in prompts:
        warm, _ = once(p["ids"])
        ts, first = [], None
        for _ in range(a.reps):
            dt, first = once(p["ids"])
            ts.append(dt)
        row = {"prompt": p["name"], "tokens": len(p["ids"]), "warmup_s": warm, "ttft_s": ts,
               "ttft_median_s": statistics.median(ts), "first_token": first}
        out["rows"].append(row)
        log(f"{p['name']} T={row['tokens']}: ttft median {row['ttft_median_s']:.3f}s "
            f"(reps {', '.join(f'{x:.3f}' for x in ts)}; warm-up {warm:.3f}s) first token {first}")
    real = [r for r in out["rows"] if r["prompt"].startswith("longhealth")]
    if a.decode_tokens and real:  # time per output token at ~32K context, batch 1: (wall of 1+N tokens - TTFT) / N
        r0 = real[0]
        ids = next(p["ids"] for p in prompts if p["name"] == r0["prompt"])
        sp_dec = SamplingParams(max_tokens=a.decode_tokens + 1, temperature=0.0, ignore_eos=True)
        walls = []
        for _ in range(3):
            t = time.perf_counter()
            llm.generate([TokensPrompt(prompt_token_ids=ids)], sp_dec, use_tqdm=False)
            walls.append(time.perf_counter() - t)
        tpot = (statistics.median(walls) - r0["ttft_median_s"]) / a.decode_tokens
        out["decode"] = {"prompt": r0["prompt"], "tokens": a.decode_tokens, "wall_s": walls, "tpot_ms": 1000 * tpot}
        log(f"decode on {r0['prompt']}: {1000 * tpot:.1f} ms/token over {a.decode_tokens} tokens (walls {walls})")
    Path(a.out).write_text(json.dumps(out, indent=1))
    log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
