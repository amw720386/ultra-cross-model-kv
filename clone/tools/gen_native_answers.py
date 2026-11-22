"""GPU job: the native target's greedy answers to the code-instruction prompts (on-policy targets for code map v2).

Prompts are rendered exactly like the bench (tools/codeinstr_common.eval_prompt: chat template, thinking off, EvalPlus
assistant prefix); greedy, max_new 768 (the bench's default), one engine on one GPU.
Reads /vol/xkv/data/<name>.jsonl (tools/build_codeinstr.py), writes /vol/xkv/data/<name>.native.jsonl
(id, kind, split, answer, finish, n_tokens). Filtering + the xkv agent log are a CPU step
(tools/gen_native_answers_check.py --finalize), dry-run that file's --check first.

    cd clone && KVX_GPU=H200 modal run tools/gen_native_answers.py::main --name codeinstr-v1
"""
import json
import os
import time

import modal

from modal_vllm import image as vimage, vol, hf_shared

GPU = os.environ.get("KVX_GPU", "H200")
image = vimage.add_local_python_source("codeinstr_common")
app = modal.App(os.environ.get("KVX_APP", "kvx-gen-native"), image=image)


@app.function(gpu=GPU, timeout=3000, volumes={"/vol": vol, "/hf": hf_shared.read_only()})
def gen(name: str, tgt: str, max_new: int, max_num_seqs: int):
    from vllm import LLM, SamplingParams
    from codeinstr_common import eval_prompt
    t0 = time.time()
    items = [json.loads(l) for l in open(f"/vol/xkv/data/{name}.jsonl")]
    llm = LLM(model=tgt, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.9,
              enable_prefix_caching=True, max_num_seqs=max_num_seqs)
    tok = llm.get_tokenizer()
    prompts = [eval_prompt(tok, it["user"]) for it in items]
    t1 = time.time()
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_new))
    t2 = time.time()
    n_tok = 0
    with open(f"/vol/xkv/data/{name}.native.jsonl", "w") as f:
        for it, o in zip(items, outs):
            c = o.outputs[0]
            n_tok += len(c.token_ids)
            f.write(json.dumps(dict(id=it["id"], kind=it["kind"], split=it["split"], answer=c.text,
                                    finish=c.finish_reason, n_tokens=len(c.token_ids))) + "\n")
    vol.commit()
    return dict(items=len(items), gen_tokens=n_tok, t_load_s=t1 - t0, t_gen_s=t2 - t1,
                truncated=sum(1 for o in outs if o.outputs[0].finish_reason == "length"))


@app.local_entrypoint()
def main(name: str = "codeinstr-v1", tgt: str = "Qwen/Qwen3.8-27B", max_new: int = 768, max_num_seqs: int = 64):
    print(json.dumps(gen.remote(name, tgt, max_new, max_num_seqs), indent=1))
