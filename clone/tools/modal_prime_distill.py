"""CPU job: token cache of long chat-wrapped PG19-TRAIN chunks for kvx.distill_full (no GPU in this file).
modal run tools/modal_prime_distill.py --n 800 --prefix 3072 --cont 512
Each sequence = chat template prefix + "Story:\\n" + book text, cut to prefix + cont tokens (Qwen3 tokenizer, no BOS).
Saved: /vol/data/distill/pg19chat_{n}x{prefix}+{cont}.pt  (LongTensor [n, prefix + cont]); PG19 test is never read."""
import os
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
image = modal.Image.debian_slim(python_version="3.12").pip_install("transformers", "datasets", "torch", "huggingface_hub")
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"))
hf_shared = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
app = modal.App(os.environ.get("KVX_APP", "kvx-quality-prime"), image=image)


@app.function(cpu=8, memory=16384, volumes={"/vol": vol, "/hf": hf_shared.read_only()}, timeout=3000)
def prime(n: int, prefix: int, cont: int, model: str = "Qwen/Qwen3-8B", per_book: int = 6):
    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot_download(model, cache_dir="/hf/hub", local_files_only=True))
    head = tok.apply_chat_template([{"role": "user", "content": "\x00"}], tokenize=False, add_generation_prompt=False).split("\x00")[0]
    hid = tok(head + "Story:\n", add_special_tokens=False)["input_ids"]
    L = prefix + cont
    out = []
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    for ex in ds:
        ids = tok(ex["text"][: L * per_book * 6], add_special_tokens=False)["input_ids"]
        body = L - len(hid)
        for c in range(min(per_book, len(ids) // body)):
            out.append(hid + ids[c * body:(c + 1) * body])
            if len(out) >= n:
                break
        if len(out) >= n:
            break
    t = torch.tensor(out, dtype=torch.long)
    os.makedirs("/vol/data/distill", exist_ok=True)
    path = f"/vol/data/distill/pg19chat_{n}x{prefix}+{cont}.pt"
    torch.save(t, path)
    vol.commit()
    return dict(path=path, shape=tuple(t.shape), head=tok.decode(hid))


@app.local_entrypoint()
def main(n: int = 800, prefix: int = 3072, cont: int = 512):
    print(prime.remote(n, prefix, cont))
