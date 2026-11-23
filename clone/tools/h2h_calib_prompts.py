"""CPU Modal job: CALIBRATION-ONLY 13,837-token retrieval prompts for the reference side of the head-to-head.

The reference package's acceptance test reports a 13,837-token retrieval prompt answered "The retrieval code is
AMBER-7421." in 0.5465 s, but ships no generator. On our first such prompt (code at 50% depth) their model did not
retrieve, so this builds variants with the code nearer the end; whichever their model retrieves is used to calibrate
their e2e latency. These files are NEVER part of the quality gate or the timing ABBA (those stay frozen in
/vol/h2h/prompts/); tuning them toward retrieval is calibration of the reference side only.

Writes /vol/h2h/calib/retrieval13837_d<depth>.jsonl (one prompt each, exactly 13,837 rendered tokens with the
Qwen3-8B@b968826d chat template, thinking off) + manifest.json (sha256, depth, token count).

    cd clone && modal run tools/h2h_calib_prompts.py::main --depths 0.75,0.9,0.97,1.0
"""
import hashlib
import json
import os

import modal

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("transformers==5.24.0", "tokenizers", "datasets>=4.0", "huggingface_hub", "jinja2")
         .env({"HF_HOME": "/tmp/hf", **{k: os.environ[k] for k in ("KVX_VOLUME", "KVX_WEIGHTS_VOLUME") if k in os.environ}}))
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"), create_if_missing=True)
weights = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
app = modal.App(os.environ.get("KVX_APP", "kvx-h2h-calib-prompts"), image=image)

TGT, TGT_REV = "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"
TOTAL = 13837
NEEDLE = " The retrieval code is AMBER-7421."
QUESTION = "\n\nWhat is the retrieval code?"


@app.function(cpu=8.0, memory=16384, timeout=1800, volumes={"/vol": vol, "/hf": weights.read_only()})
def build(depths: list, skip_rows: int, out_dir: str):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"/hf/hub/models--{TGT.replace('/', '--')}/snapshots/{TGT_REV}")

    def render(content):
        text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        return len(tok(text, add_special_tokens=False)["input_ids"])
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    it = (ex["text"] for k, ex in enumerate(fw) if k >= skip_rows)
    ids = []
    while len(ids) < TOTAL + 500:
        ids += tok(next(it) + "\n\n", add_special_tokens=False)["input_ids"]
    os.makedirs(f"/vol/{out_dir}", exist_ok=True)
    man = dict(tokenizer=f"{TGT}@{TGT_REV}", total_tokens=TOTAL, needle=NEEDLE, question=QUESTION,
               haystack=f"HuggingFaceFW/fineweb-edu sample-10BT from row {skip_rows}", use="calibration of the reference side only",
               files={})
    for d in depths:
        budget = TOTAL - render(NEEDLE + QUESTION)
        for _ in range(60):  # decode/encode boundaries drift by a few tokens: step the budget to the exact count
            hay = tok.decode(ids[:budget])
            if d >= 1.0:
                pos = len(hay)  # code right before the question
            else:
                cut = hay.rfind(". ", 0, int(len(hay) * d))
                pos = cut + 1 if cut > 0 else int(len(hay) * d)
            content = hay[:pos] + NEEDLE + hay[pos:] + QUESTION
            n = render(content)
            if n == TOTAL:
                break
            budget += TOTAL - n
        assert n == TOTAL, (d, n)
        name = f"retrieval13837_d{int(round(d * 100)):03d}.jsonl"
        row = dict(id=f"calib/retrieval13837_d{int(round(d * 100)):03d}", kind="needle", prompt_tokens=n, input_tokens=n,
                   depth=d, messages=[{"role": "user", "content": content}], answer="AMBER-7421", max_tokens=16,
                   published=dict(latency_s=0.5465, reply="The retrieval code is AMBER-7421.", prompt_tokens=TOTAL))
        p = f"/vol/{out_dir}/{name}"
        open(p, "w").write(json.dumps(row) + "\n")
        man["files"][name] = dict(sha256=hashlib.sha256(open(p, "rb").read()).hexdigest(), depth=d, prompt_tokens=n,
                                  chars_after_code=len(content) - content.find("AMBER-7421") - len("AMBER-7421"))
    json.dump(man, open(f"/vol/{out_dir}/manifest.json", "w"), indent=1)
    vol.commit()
    return man


@app.local_entrypoint()
def main(depths: str = "0.75,0.9,0.97,1.0", skip_rows: int = 250000, out_dir: str = "h2h/calib"):
    print(json.dumps(build.remote([float(x) for x in depths.split(",")], skip_rows, out_dir), indent=1))
