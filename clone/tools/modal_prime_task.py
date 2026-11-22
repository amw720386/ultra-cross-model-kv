"""CPU job: task-specific token caches for kvx.distill_full (no GPU in this file). Training docs are disjoint from
the eval items: QuALITY TRAIN articles, ContractNLI (SCROLLS) TRAIN contracts, ZebraLogic puzzles 200.. (eval = first 200).
modal run tools/modal_prime_task.py --task quality --n 209 --prefix 3072 --cont 512
Saved: /vol/data/distill/{task}chat_{n}x{prefix}+{cont}.pt; sequences shorter than prefix+cont are skipped (zebra pads
nothing: use a short prefix, e.g. --prefix 640 --cont 256)."""
import io
import json
import os
import zipfile

import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("transformers", "torch", "huggingface_hub", "pyarrow")
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"))
hf_shared = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
app = modal.App(os.environ.get("KVX_APP", "kvx-quality-prime-task"), image=image)
URL = {"quality": "https://huggingface.co/api/datasets/emozilla/quality/parquet/default/train/0.parquet",
       "contract_nli": "https://huggingface.co/datasets/tau/scrolls/resolve/main/contract_nli.zip"}


def texts(task, root):
    import urllib.request
    import pyarrow.parquet as pq
    if task == "quality":
        p = f"{root}/quality_train.parquet"
        if not os.path.exists(p):
            urllib.request.urlretrieve(URL["quality"], p)
        seen, out = set(), []
        for r in pq.read_table(p).to_pylist():
            if r["article"] not in seen:
                seen.add(r["article"])
                out.append(("Story:\n", r["article"]))
        return out
    if task == "contract_nli":
        p = f"{root}/contract_nli.zip"
        with zipfile.ZipFile(p) as z:
            rows = [json.loads(l) for l in io.TextIOWrapper(z.open("contract_nli/train.jsonl"), encoding="utf-8")]
        seen, out = set(), []
        for r in rows:
            doc = r["input"].partition("\n\n")[2]
            if doc not in seen:
                seen.add(doc)
                out.append(("Contract:\n", doc))
        return out
    if task == "zebralogic":
        rows = pq.read_table(f"{root}/zebra_grid_wildeval.parquet").to_pylist()[200:]  # puzzles only (no answers)
        return [("", r["puzzle"]) for r in rows]
    raise ValueError(task)


@app.function(cpu=4, memory=8192, volumes={"/vol": vol, "/hf": hf_shared.read_only()}, timeout=1800)
def prime(task: str, n: int, prefix: int, cont: int, model: str = "Qwen/Qwen3-8B", concat: bool = False):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot_download(model, cache_dir="/hf/hub", local_files_only=True))
    head = tok.apply_chat_template([{"role": "user", "content": "\x00"}], tokenize=False, add_generation_prompt=False).split("\x00")[0]
    L = prefix + cont
    out = []
    if concat:  # ridge calibration: chat-wrapped task docs packed back to back into L-token rows (n rows)
        buf = []
        for lab, t in texts(task, "/vol/data/evals"):
            buf += tok(head + lab + t, add_special_tokens=False)["input_ids"]
            while len(buf) >= L and len(out) < n:
                out.append(buf[:L])
                buf = buf[L:]
            if len(out) >= n:
                break
    else:
        for lab, t in texts(task, "/vol/data/evals"):
            ids = tok(head + lab + t, add_special_tokens=False)["input_ids"]
            if len(ids) >= L:
                out.append(ids[:L])
            if len(out) >= n:
                break
    t = torch.tensor(out, dtype=torch.long)
    path = (f"/vol/data/taskcal/{task}_{len(out)}x{L}.pt" if concat
            else f"/vol/data/distill/{task}chat_{len(out)}x{prefix}+{cont}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.makedirs("/vol/data/distill", exist_ok=True)
    torch.save(t, path)
    vol.commit()
    return dict(path=path, shape=tuple(t.shape))


@app.local_entrypoint()
def main(task: str, n: int, prefix: int = 3072, cont: int = 512, concat: bool = False):
    print(prime.remote(task, n, prefix, cont, concat=concat))
