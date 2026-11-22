"""CPU side of tools/gen_native_answers.py (no GPU in this file).

--mode check:    renders 5 prompts with the target tokenizer and replays xkv task_distill's agent-sample parsing on a
                 fake answer (a sample xkv would silently skip fails loudly here); also imports the GPU file's deps.
--mode finalize: drops answers that share a 13-gram with any MBPP / HumanEval / BigCodeBench test item, writes the
                 xkv agent log /vol/xkv/agent_logs/<name>-native.jsonl (TRAIN split only; dev never) and stats.

    cd clone && PYTHONPATH=../src modal run tools/gen_native_answers_check.py::main --mode check
"""
import json
import os

import modal

from build_codemix import image as cimage, vol, weights  # CPU image with transformers + datasets

image = cimage.add_local_python_source("codeinstr_common", "build_codemix", "xkv")
app = modal.App(os.environ.get("KVX_APP", "kvx-gen-native-check"), image=image)


def snapshot(name):
    rd = f"/hf/hub/models--{name.replace('/', '--')}"
    return f"{rd}/snapshots/{open(f'{rd}/refs/main').read().strip()}"


@app.function(cpu=8.0, memory=32768, timeout=2400, volumes={"/vol": vol, "/hf": weights.read_only()})
def run(mode: str, name: str, tgt: str, cont: int):
    from transformers import AutoTokenizer
    from codeinstr_common import agent_line, check_agent_sample, eval_prompt
    tok = AutoTokenizer.from_pretrained(snapshot(tgt))
    items = [json.loads(l) for l in open(f"/vol/xkv/data/{name}.jsonl")]
    if mode == "check":
        rows = []
        for it in items[:: max(1, len(items) // 5)][:5]:
            p = eval_prompt(tok, it["user"])
            pi, ci = check_agent_sample(tok, agent_line(it["user"], "def f(x):\n    return x + 1\n```"), cont=cont)
            rows.append(dict(id=it["id"], eval_prompt_tokens=len(tok(p, add_special_tokens=False)["input_ids"]),
                             agent_prompt_tokens=len(pi), agent_cont_tokens=len(ci),
                             agent_cont_head=tok.decode(ci[:24])))
        return rows
    from build_codemix import grams, test_grams
    tg, _ = test_grams()
    ans = {a["id"]: a for a in (json.loads(l) for l in open(f"/vol/xkv/data/{name}.native.jsonl"))}
    os.makedirs("/vol/xkv/agent_logs", exist_ok=True)
    st = dict(train=0, dev=0, dropped_test_overlap=0, dropped_truncated=0, skipped_by_xkv=0)
    with open(f"/vol/xkv/agent_logs/{name}-native.jsonl", "w") as f, \
            open(f"/vol/xkv/agent_logs/{name}-native-dev.jsonl", "w") as fd:
        for it in items:
            a = ans[it["id"]]
            if it["split"] != "train":  # held-out dev: its own log (kvx.xkv_extra agent_kl), never in training
                if a["finish"] != "length":
                    fd.write(agent_line(it["user"], a["answer"]) + "\n")
                    st["dev"] += 1
                continue
            if a["finish"] == "length":
                st["dropped_truncated"] += 1
                continue
            if grams(a["answer"]) & tg:
                st["dropped_test_overlap"] += 1
                continue
            line = agent_line(it["user"], a["answer"])
            try:
                check_agent_sample(tok, line, cont=cont)
            except ValueError:
                st["skipped_by_xkv"] += 1
                continue
            f.write(line + "\n")
            st["train"] += 1
    json.dump(st, open(f"/vol/xkv/agent_logs/{name}-native.stats.json", "w"), indent=1)
    vol.commit()
    return st


@app.local_entrypoint()
def main(mode: str = "check", name: str = "codeinstr-v1", tgt: str = "Qwen/Qwen3.8-27B", cont: int = 256):
    print(json.dumps(run.remote(mode, name, tgt, cont), indent=1))
