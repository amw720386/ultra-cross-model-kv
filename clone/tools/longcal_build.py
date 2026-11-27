"""CPU Modal job: long-context calibration set for a 1.7B -> 8B refit, decontaminated against the head-to-head prompts.

Builds n_calib sequences of L tokens: half concatenated FineWeb-Edu docs (EOS between docs, streamed from row 0 and
asserted to stay below the reserved eval rows 20,000+ and far below the h2h haystack rows 50,000+), half PG19-train
book chunks. Any sequence sharing a 13-gram (whitespace words) with ANY h2h prompt file on the volume (needles +
HellaSwag + calibration prompts under h2h/prompts, h2h/prompts-dev, h2h/calib) is dropped and replaced; the build
asserts 0 overlaps after filtering. The sequences are written to modal_fit_eval's token cache under the exact tags its
calib_src="mix" path reads, so the fit uses them without re-streaming.

    cd clone && PYTHONPATH=. modal run tools/longcal_build.py::main --n-calib 1000 --seq-len 8192
"""
import json
import os
import re

import modal

from modal_fit_eval import _cache_path, hf_shared, image, vol

app = modal.App(os.environ.get("KVX_APP", "kvx-longcal-build"), image=image.add_local_python_source("modal_fit_eval"))
WORD = re.compile(r"\S+")
H2H_DIRS = ("h2h/prompts", "h2h/prompts-dev", "h2h/calib")  # default; --dirs overrides (comma list under /vol)


def gram_hashes(text, n=13):
    w = WORD.findall(text)
    return {hash(" ".join(w[i:i + n])) for i in range(max(0, len(w) - n + 1))}


@app.function(cpu=8.0, memory=65536, timeout=3300, volumes={"/vol": vol, "/hf": hf_shared.read_only()})
def build(tgt: str, n_calib: int, L: int, dirs: list = None):
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from kvx.models import model_path
    tok = AutoTokenizer.from_pretrained(model_path(tgt))
    tok.name_or_path = tgt  # same as fit_eval: the cache key reads it
    # 13-grams of every h2h prompt
    tg, files = set(), {}
    for d in (dirs or H2H_DIRS):
        for f in sorted(os.listdir(f"/vol/{d}")):
            if not f.endswith(".jsonl"):
                continue
            k = 0
            for line in open(f"/vol/{d}/{f}"):
                r = json.loads(line)
                for m in r["messages"]:
                    tg |= gram_hashes(m["content"])
                k += 1
            files[f"{d}/{f}"] = k
    half = n_calib // 2
    eos = tok.eos_token_id
    # same BOS rule as modal_fit_eval (Llama: every sequence starts with BOS, as every served prompt does)
    bos = int(tok.bos_token_id is not None and "qwen" not in tok.name_or_path.lower())
    lead = [tok.bos_token_id] if bos else []
    stats = dict(h2h_files=files, h2h_13grams=len(tg), dropped=0)

    def clean(seq):
        return not (gram_hashes(tok.decode(seq)) & tg)
    # FineWeb half: concatenated docs from row 0
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    fwl, buf, last = [], [], 0
    for i, ex in enumerate(fw):
        last = i
        buf += tok(ex["text"], add_special_tokens=False)["input_ids"] + [eos]
        while len(buf) >= L - len(lead) and len(fwl) < half:
            s = lead + buf[:L - len(lead)]
            buf = buf[L - len(lead):]
            if clean(s):
                fwl.append(s)
            else:
                stats["dropped"] += 1
        if len(fwl) >= half:
            break
    assert len(fwl) == half and last < 20000, (len(fwl), last)
    stats["fineweb_rows_used"] = [0, last]
    # PG19-train half (book chunks, up to 32 per book)
    pg, books = [], 0
    for ex in load_dataset("emozilla/pg19", split="train", streaming=True):
        ids = tok(ex["text"][: L * 32 * 6], add_special_tokens=False)["input_ids"]
        books += 1
        Lb = L - len(lead)
        for c in range(min(32, len(ids) // Lb)):
            s = lead + ids[c * Lb:(c + 1) * Lb]
            if clean(s):
                pg.append(s)
            else:
                stats["dropped"] += 1
            if len(pg) >= half:
                break
        if len(pg) >= half:
            break
    assert len(pg) == half
    stats["pg19_books"] = books
    leak = sum(1 for s in fwl + pg if not clean(s))
    assert leak == 0, leak
    stats["overlap_after"] = leak
    stats["bos"] = bos
    for tag, seqs in ((f"calib_{half}x{L}_bos{bos}", fwl), (f"pg19train_{half}x{L}_bos{bos}", pg)):
        p = _cache_path(tok, tag)
        assert not os.path.exists(p), f"cache exists, refusing to overwrite: {p}"
        torch.save(seqs, p)
        stats[tag] = p
    sd = f"/vol/calib_stats/{tgt.replace('/', '--')}_{n_calib}x{L}"
    os.makedirs(sd, exist_ok=True)
    json.dump(stats, open(f"{sd}/calib_stats.json", "w"), indent=1)
    stats["stats_file"] = f"{sd}/calib_stats.json"
    vol.commit()
    return stats


@app.local_entrypoint()
def main(tgt: str = "Qwen/Qwen3-8B", n_calib: int = 1000, seq_len: int = 8192, dirs: str = ""):
    print(json.dumps(build.remote(tgt, n_calib, seq_len, [d for d in dirs.split(",") if d] or None), indent=1))
