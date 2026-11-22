"""Run OneTriangle's bundle benchmarks on one vLLM arm (native, or mapped through kvx.vllm_connector).

python -m kvx.evals.run --model Qwen/Qwen3.8-27B --arm native27b --benches arc_chat,contract_nli --n 100 \
    --out /w/out/native27b [--connector '{"src": ..., "maps": ..., "tail": 256}'] [--thinking]

Prompt = chat template (non-thinking unless --thinking) over one user turn: context + question.
Prompts longer than max_model_len - max_tokens get the MIDDLE of the context cut (recorded per item as `cut`).
Writes gens.jsonl (one row per item: text, score) and summary.json (per bench: n, score, 95% CI).
Code items (HumanEval/MBPP) get score null here; score them with exec_code.py in a sandbox.
"""
import argparse
import json
import os
import time

from . import data, score


def gen_budget(bench, thinking):
    return data.MAX_TOKENS[bench] + (data.THINK_TOKENS if thinking else 0)


def sampling(SamplingParams, bench, thinking, seed=0):
    if thinking:  # Qwen3 thinking-mode recommendation (greedy loops); fixed seed
        return SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=gen_budget(bench, True), seed=seed)
    return SamplingParams(temperature=0.0, max_tokens=gen_budget(bench, False))


def build_prompts(tok, items, max_len, thinking):
    out = []
    for it in items:
        budget = max_len - gen_budget(it["bench"], thinking) - 8
        q_ids = None
        ctx = it["context"]
        msgs = [{"role": "user", "content": ctx + it["question"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        cut = 0
        if len(ids) > budget and ctx:
            c_ids = tok(ctx, add_special_tokens=False)["input_ids"]
            over = len(ids) - budget + 16
            keep = max(0, len(c_ids) - over)
            head, tail_ = keep // 2, keep - keep // 2
            ctx2 = tok.decode(c_ids[:head]) + "\n...\n" + tok.decode(c_ids[len(c_ids) - tail_:]) if keep else ""
            msgs = [{"role": "user", "content": ctx2 + it["question"]}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            cut = len(c_ids) - keep
        q_text = tok.apply_chat_template([{"role": "user", "content": "\x00" + it["question"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=thinking)
        q_tail = len(tok(q_text.split("\x00", 1)[1], add_special_tokens=False)["input_ids"]) + 2
        out.append((ids, cut, q_tail))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--benches", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--ns", default=None, help='JSON per-bench n overrides, e.g. {"longbench_v2": 97}')
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=os.environ.get("KVX_DATA", "/w/data"))
    ap.add_argument("--max-model-len", type=int, default=65536)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--connector", default=None, help="JSON kv_connector_extra_config for kvx.vllm_connector")
    ap.add_argument("--attn-backend", default=None)
    ap.add_argument("--dry", action="store_true", help="build prompts only (CPU check), no vLLM")
    ap.add_argument("--question-tail", action="store_true", help="mapped arm: native tail = question tokens (+extra)")
    ap.add_argument("--question-tail-extra", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    items = []
    ns = json.loads(a.ns) if a.ns else {}
    for b in a.benches.split(","):
        items += data.load(b, a.data_root, int(ns.get(b, a.n)))
    prompts = build_prompts(tok, items, a.max_model_len, a.thinking)
    lens = [len(p) for p, _, _ in prompts]
    print(json.dumps(dict(items=len(items), max_len=max(lens), mean_len=sum(lens) / len(lens),
                          cut=sum(1 for _, c, _ in prompts if c))), flush=True)
    if a.dry:
        with open(os.path.join(a.out, "prompts_dry.jsonl"), "w") as f:
            for it, (p, c, qt) in zip(items, prompts):
                f.write(json.dumps(dict(id=it["id"], bench=it["bench"], n_tok=len(p), cut=c, q_tail=qt,
                                        head=tok.decode(p[:40]), tail=tok.decode(p[-60:]))) + "\n")
        return

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoConfig
    kw = dict(model=a.model, max_model_len=a.max_model_len, tensor_parallel_size=a.tp,
              gpu_memory_utilization=a.gpu_util, seed=0)
    if getattr(AutoConfig.from_pretrained(a.model), "vision_config", None) is not None:
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}  # text-only use of a VL checkpoint (Qwen3.5/3.8)
    if a.attn_backend:
        kw["attention_backend"] = a.attn_backend
    if a.connector:
        from vllm.config import KVTransferConfig
        cc = json.loads(a.connector)
        cls = cc.pop("_class", "CrossModelConnector")
        mod = cc.pop("_module", "kvx.vllm_connector")
        if cls == "HybridCrossModelConnector":
            kw["enable_prefix_caching"] = False  # one GatedDeltaNet state slot per request
        kw["kv_transfer_config"] = KVTransferConfig(kv_connector=cls, kv_connector_module_path=mod, kv_role="kv_both",
                                                    kv_connector_extra_config=cc)
    t0 = time.time()
    llm = LLM(**kw)
    t_load = time.time() - t0
    sps = [sampling(SamplingParams, it["bench"], a.thinking, seed=i) for i, it in enumerate(items)]
    if a.connector and a.question_tail:  # map the document, read the question natively (per-request tail)
        for sp, (_, _, qt) in zip(sps, prompts):
            sp.extra_args = {"kv_transfer_params": {"kvx_tail": qt + a.question_tail_extra}}
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p, _, _ in prompts], sps)
    t_gen = time.time() - t0

    by = {}
    with open(os.path.join(a.out, "gens.jsonl"), "w") as f:
        for it, (p, cut, qt), o in zip(items, prompts, outs):
            text = o.outputs[0].text
            s = score.score_item(it, text)
            row = dict(id=it["id"], bench=it["bench"], arm=a.arm, kind=it["kind"], answer=it["answer"], text=text,
                       n_prompt=len(p), n_gen=len(o.outputs[0].token_ids), cut=cut, q_tail=qt,
                       finish=o.outputs[0].finish_reason, **s)
            if it["kind"] == "code":
                row["meta"] = it["meta"]
            f.write(json.dumps(row) + "\n")
            by.setdefault(it["bench"], []).append(row)
    summ = dict(arm=a.arm, model=a.model, connector=json.loads(a.connector) if a.connector else None,
                thinking=a.thinking, max_model_len=a.max_model_len, t_load_s=t_load, t_gen_s=t_gen, benches={})
    for b, rows in by.items():
        sc = [r["score"] for r in rows if r["score"] is not None]
        ent = dict(n=len(rows), cut=sum(1 for r in rows if r["cut"]), mean_prompt=sum(r["n_prompt"] for r in rows) / len(rows))
        if a.connector:  # tokens the connector maps (kvx.vllm_connector._n_mapped, block 16) / prompt tokens
            cc = json.loads(a.connector)
            t, ml = int(cc.get("tail", 64)), int(cc.get("min_len", 1024))
            tt = [(r["q_tail"] + a.question_tail_extra) if a.question_tail else t for r in rows]
            mp = [0 if r["n_prompt"] < ml else max(0, (r["n_prompt"] - x) // 16 * 16) for r, x in zip(rows, tt)]
            ent["mapped_frac"] = sum(mp) / max(1, sum(r["n_prompt"] for r in rows))
        if sc and rows[0]["kind"] in ("letter", "label", "math", "zebra"):
            k = int(sum(sc))
            p, lo, hi = score.wilson(k, len(sc))
            ent.update(score=p, k=k, ci_lo=lo, ci_hi=hi)
        elif sc:
            m, h = score.mean_ci(sc)
            ent.update(score=m, ci_half=h)
        summ["benches"][b] = ent
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
