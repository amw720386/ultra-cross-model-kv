"""CPU Modal job: prompt file for kvx.glm_gate (GLM-5.3 tokenizer), so the gate boot needs no network.

  long:   PG19 test books concatenated (EOS between books) to --len tokens, plus the real next 64 tokens (scored)
  needle: PG19 validation text to --len tokens with " The secret number is N. Remember it." at depth d and the
          question at the end (inside the native tail), answer N (7 digits)
Output on KVX_VOLUME: runs/glm-gate/prompts.pt (list of dicts, see kvx/glm_gate.py) + prompts.json (stats only).

    cd clone && modal run tools/build_glm_gate_prompts.py::main --n-long 4 --depths 0.1,0.35,0.6,0.85
"""
import json
import os
import random

import modal

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("transformers==5.24.0", "tokenizers", "datasets>=4.0", "huggingface_hub", "torch==2.8.0", "jinja2")
         .env({"HF_HOME": "/tmp/hf", **{k: os.environ[k] for k in ("KVX_VOLUME", "KVX_WEIGHTS_VOLUME") if k in os.environ}}))
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"), create_if_missing=True)
weights = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
app = modal.App(os.environ.get("KVX_APP", "kvx-glm-gate-prompts"), image=image)

NEEDLE = " The secret number is {}. Remember it."
QUESTION = "\nWhat is the secret number mentioned above? The secret number is"


def snapshot(name):
    rd = f"/hf/hub/models--{name.replace('/', '--')}"
    return f"{rd}/snapshots/{open(f'{rd}/refs/main').read().strip()}"


@app.function(cpu=8.0, memory=32768, timeout=3000, volumes={"/vol": vol, "/hf": weights.read_only()})
def build(tok_name: str, L: int, n_long: int, depths: list, cont: int, out: str):
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot(tok_name))
    eos = tok.eos_token_id
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []  # same rule as the fit (modal_fit_hybrid._bos)

    def stream(split, skip=0):
        for i, ex in enumerate(load_dataset("emozilla/pg19", split=split, streaming=True)):
            if i >= skip:
                yield ex["text"]

    def take(it, n):
        buf, books = list(bos), 0
        while len(buf) < n:
            buf += tok(next(it), add_special_tokens=False)["input_ids"] + [eos]
            books += 1
        return buf[:n], books

    prompts, stats = [], []
    it = stream("test")
    for k in range(n_long):
        ids, books = take(it, L + cont)
        prompts.append(dict(kind="long", ids=ids[:L], cont=ids[L:L + cont], source=f"pg19 test, {books} books"))
        stats.append(dict(kind="long", len=L, cont=cont, books=books))
    rng = random.Random(0)
    q = tok(QUESTION, add_special_tokens=False)["input_ids"]
    itv = stream("validation")
    for d in depths:
        num = str(rng.randrange(1000000, 9999999))
        nd = tok(NEEDLE.format(num), add_special_tokens=False)["input_ids"]
        body, books = take(itv, L - len(nd) - len(q))
        pos = len(bos) + int(d * (len(body) - len(bos)))
        ids = body[:pos] + nd + body[pos:] + q
        assert len(ids) == L, len(ids)
        prompts.append(dict(kind="needle", ids=ids, answer=num, depth=d, needle_pos=pos, source=f"pg19 validation, {books} books"))
        stats.append(dict(kind="needle", len=L, depth=d, needle_pos=pos, answer=num, books=books))
    os.makedirs(os.path.dirname(f"/vol/{out}"), exist_ok=True)
    torch.save(prompts, f"/vol/{out}")
    json.dump(dict(tokenizer=tok_name, L=L, cont=cont, bos=bos, prompts=stats), open(f"/vol/{out[:-3]}.json", "w"), indent=1)
    vol.commit()
    return stats


@app.local_entrypoint()
def main(tok_name: str = "zai-org/GLM-5.3", length: int = 202752, n_long: int = 4, depths: str = "0.1,0.35,0.6,0.85",
         cont: int = 64, out: str = "runs/glm-gate/prompts.pt"):
    print(json.dumps(build.remote(tok_name, length, n_long, [float(x) for x in depths.split(",")], cont, out), indent=1))
