"""CPU Modal job: prompt files for the head-to-head quality gate (Qwen3-1.7B -> Qwen3-8B, OpenAI chat API).

Writes, on KVX_VOLUME under h2h/gate/:
  needles_<L>.jsonl   N retrieval-code needles per length L (total rendered chat-prompt tokens ~ L, Qwen3-8B
                      chat template, thinking off): FineWeb-Edu sample-10BT haystack from row --skip-rows on, the
                      sentence " The retrieval code for project NOUN is WORD-dddd." at depths spread over 5-95%,
                      then the question. Answer = the code (exact substring match in the reply).
  hellaswag100.jsonl  100 HellaSwag rows ([hs_offset:+100]; gate = 0 = exactly the 100 OneTriangle's own evaluator selects (kvt/benchmarks/hellaswag.py:
                      Rowan/hellaswag@218ec52e validation, namespace "qwen3-kv-transfer-hellaswag-v1", sha256
                      selection key sort), in its chat-letter format (spec._chat_options): ctx + "\\nA. ..." lines +
                      "\\n\\nAnswer with the letter of the correct option." as the user turn.
  manifest.json       sha256 of every file, token counts, sources.
Each line: {"id", "kind", "messages": [{"role": "user", "content": ...}], "answer", "max_tokens", ...}.

    cd clone && modal run tools/h2h_gate_prompts.py::main
"""
import hashlib
import json
import os
import random

import modal

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("transformers==5.24.0", "tokenizers", "datasets>=4.0", "huggingface_hub", "jinja2")
         .env({"HF_HOME": "/tmp/hf", **{k: os.environ[k] for k in ("KVX_VOLUME", "KVX_WEIGHTS_VOLUME") if k in os.environ}}))
vol = modal.Volume.from_name(os.environ.get("KVX_VOLUME", "kvx"), create_if_missing=True)
weights = modal.Volume.from_name(os.environ.get("KVX_WEIGHTS_VOLUME", "kvx-hf"))
app = modal.App(os.environ.get("KVX_APP", "kvx-h2h-gate-prompts"), image=image)

QWEN, QWEN_REV = "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"  # h2h default target
HS_ID, HS_REV, HS_SPLIT = "Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76", "validation"
HS_NAMESPACE = "qwen3-kv-transfer-hellaswag-v1"
LETTERS = "ABCDEFGHIJ"
CHAT_INSTRUCTION = "\n\nAnswer with the letter of the correct option."
NOUNS = ("ALBATROSS", "BADGER", "CANOE", "DOLPHIN", "EMERALD", "FALCON", "GLACIER", "HARPOON", "IGLOO", "JAGUAR",
         "KETTLE", "LANTERN", "MAGNOLIA", "NUTMEG", "OCTOPUS", "PELICAN", "QUARTZ", "RACCOON", "SAPPHIRE", "TULIP",
         "UMBRELLA", "VOLCANO", "WALRUS", "ZEPPELIN", "BEACON", "COMET", "FOSSIL", "GONDOLA", "MEADOW", "ORCHID")
WORDS = ("AMBER", "BIRCH", "CEDAR", "DELTA", "EMBER", "FROST", "GROVE", "HAVEN", "IVORY", "JUNIPER", "KESTREL",
         "LUNAR", "MAPLE", "NOVA", "ONYX", "PRISM", "QUILL", "RAVEN", "SOLAR", "TIDAL", "UMBER", "VIOLET", "WILLOW",
         "ZENITH")


def snapshot(name, rev):
    d = f"/hf/hub/models--{name.replace('/', '--')}/snapshots/{rev}"
    assert os.path.exists(f"{d}/config.json"), d
    return d


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def hellaswag_rows(n, offset=0):
    """OneTriangle kvt selection, reimplemented: kvt/data.select_deterministic_samples + spec.build_row (chat)."""
    from datasets import load_dataset
    ds = load_dataset(HS_ID, split=HS_SPLIT, revision=HS_REV)
    cand = []
    for i, r in enumerate(ds):
        raw = r.get("ctx")
        if not isinstance(raw, str) or not raw.strip():
            continue
        text = raw.strip()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        ex_id = str(r["ind"] if r.get("ind") is not None else i)
        key = hashlib.sha256("\0".join([HS_NAMESPACE, HS_ID, HS_SPLIT, str(i), ex_id, digest]).encode("utf-8")).hexdigest()
        cand.append((key, i, ex_id, text, r))
    cand.sort(key=lambda c: c[0])
    out = []
    for key, i, ex_id, text, r in cand[offset:offset + n]:
        choices = [" " + str(e) for e in r["endings"]]
        listing = "".join(f"\n{L}. {c.strip()}" for L, c in zip(LETTERS, choices))
        body = text[:-len("\nAnswer:")] if text.endswith("\nAnswer:") else text.rstrip()
        out.append(dict(id=f"hellaswag/{ex_id}", kind="hellaswag", row_index=i, selection_key=key,
                        messages=[{"role": "user", "content": body + listing + CHAT_INSTRUCTION}],
                        answer=LETTERS[int(r["label"])], max_tokens=8))
    return out


@app.function(cpu=8.0, memory=32768, timeout=3000, volumes={"/vol": vol, "/hf": weights.read_only()})
def build(lengths: list, n: int, skip_rows: int, out_dir: str, seed: int, hs_offset: int, calib: int,
          tgt: str = QWEN, tgt_rev: str = QWEN_REV, max_ctx: int = 32768):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot(tgt, tgt_rev))
    def render(content):  # rendered chat-prompt token count (transformers 5: tokenize=True returns a dict, so render text)
        text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        return len(tok(text, add_special_tokens=False)["input_ids"])
    assert render("hello") > 5, render("hello")
    os.makedirs(f"/vol/{out_dir}", exist_ok=True)
    man = dict(tokenizer=f"{tgt}@{tgt_rev}", chat_template=f"{tgt} pinned (enable_thinking=False passed; ignored by non-Qwen templates)",
               max_ctx=max_ctx, files={})
    # haystack docs: FineWeb-Edu sample-10BT rows >= skip_rows (disjoint from both sides' calibration scans)
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    it = (ex["text"] for k, ex in enumerate(fw) if k >= skip_rows)
    rng = random.Random(seed)
    for L in lengths:
        rows = []
        for j in range(n):
            noun, word = NOUNS[j % len(NOUNS)], WORDS[(j * 7 + L // 1024) % len(WORDS)]
            code = f"{word}-{rng.randrange(1000, 10000)}"
            needle = f" The retrieval code for project {noun} is {code}."
            question = f"\n\nWhat is the retrieval code for project {noun}? Reply with the code only."
            budget = L - render(needle + question) - 8
            ids = []
            while len(ids) < budget + 512:
                ids += tok(next(it) + "\n\n", add_special_tokens=False)["input_ids"]
            depth = 0.05 + 0.90 * j / max(1, n - 1)
            for _ in range(6):  # decode/encode drift grows with length: step the budget toward L
                hay = tok.decode(ids[:budget])
                pos = int(len(hay) * depth)
                cut = hay.rfind(". ", 0, pos)
                pos = cut + 1 if cut > 0 else pos
                content = hay[:pos] + needle + hay[pos:] + question
                ntok = render(content)
                if L - 32 <= ntok <= L:
                    break
                budget += L - 16 - ntok
            assert L - 64 <= ntok <= L + 64 and ntok + 32 <= max_ctx, (L, ntok)
            rows.append(dict(id=f"needle{L}/{j}", kind="needle", length_target=L, prompt_tokens=ntok, input_tokens=ntok, depth=round(depth, 3),
                             messages=[{"role": "user", "content": content}], answer=code, max_tokens=32))
        p = f"/vol/{out_dir}/needles_{L}.jsonl"
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        man["files"][os.path.basename(p)] = dict(sha256=sha(p), n=len(rows),
                                                  prompt_tokens=[min(r["prompt_tokens"] for r in rows), max(r["prompt_tokens"] for r in rows)])
    if calib:  # their acceptance test shapes (ACCEPTANCE_RESULTS.json): 13,837-token AMBER-7421 retrieval + France
        needle, question = " The retrieval code is AMBER-7421.", "\n\nWhat is the retrieval code?"
        ids = []
        while len(ids) < 14000:
            ids += tok(next(it) + "\n\n", add_special_tokens=False)["input_ids"]
        budget = 13837 - render(needle + question)
        for _ in range(50):  # decode/encode boundaries drift by a few tokens: step the budget to the exact count
            hay = tok.decode(ids[:budget])
            pos = hay.rfind(". ", 0, len(hay) // 2) + 1
            content = hay[:pos] + needle + hay[pos:] + question
            ntok = render(content)
            if ntok == 13837:
                break
            budget += 13837 - ntok
        assert ntok == 13837, ntok
        cal = [dict(id="calib/long_retrieval_13837", kind="needle", prompt_tokens=ntok, input_tokens=ntok,
                    messages=[{"role": "user", "content": content}], answer="AMBER-7421", max_tokens=16,
                    published=dict(latency_s=0.5465, reply="The retrieval code is AMBER-7421.")),
               dict(id="calib/short_france", kind="needle", prompt_tokens=render("What is the capital of France?"),
                    input_tokens=render("What is the capital of France?"),
                    messages=[{"role": "user", "content": "What is the capital of France?"}], answer="Paris", max_tokens=16,
                    published=dict(latency_s=0.1877, reply="The capital of France is Paris."))]
        for r in cal:
            p = f"/vol/{out_dir}/{r['id'].split('/')[1]}.jsonl"
            open(p, "w").write(json.dumps(r) + "\n")
            man["files"][os.path.basename(p)] = dict(sha256=sha(p), n=1, prompt_tokens=r["prompt_tokens"],
                                                      note="calibration (their acceptance-test shape; generator not in their package)")
    hs = hellaswag_rows(100, hs_offset)
    p = f"/vol/{out_dir}/hellaswag100.jsonl"
    with open(p, "w") as f:
        for r in hs:
            f.write(json.dumps(r) + "\n")
    man["files"]["hellaswag100.jsonl"] = dict(sha256=sha(p), n=len(hs), source=f"{HS_ID}@{HS_REV} {HS_SPLIT}",
                                              selection=f"OneTriangle kvt namespace {HS_NAMESPACE} (sha256 key sort), rows [{hs_offset}:{hs_offset + 100}]",
                                              format="kvt spec._chat_options user turn", first_ids=[r["id"] for r in hs[:5]])
    man["haystack"] = f"HuggingFaceFW/fineweb-edu sample-10BT streamed from row {skip_rows}"
    man["needle_seed"] = seed
    json.dump(man, open(f"/vol/{out_dir}/manifest.json", "w"), indent=1)
    vol.commit()
    return man


@app.local_entrypoint()
def main(lengths: str = "8192,16384,30720", n: int = 24, skip_rows: int = 50000, out_dir: str = "h2h/prompts",
         seed: int = 20261123, hs_offset: int = 0, calib: int = 1, tgt: str = QWEN, tgt_rev: str = QWEN_REV,
         max_ctx: int = 32768):
    # other targets: --tgt org/name --tgt-rev <sha> --max-ctx <served max-model-len>
    # gate: defaults (their 100 HellaSwag rows). DEV (tail pick, DECISION h2h-tail-frozen): a separate dir, other
    # seed + haystack rows, HellaSwag rows [100:200] of their order:
    #   --out-dir h2h/prompts-dev --seed 7 --skip-rows 150000 --hs-offset 100 --calib 0
    print(json.dumps(build.remote([int(x) for x in lengths.split(",")], n, skip_rows, out_dir, seed, hs_offset, calib,
                                  tgt, tgt_rev, max_ctx), indent=1))
