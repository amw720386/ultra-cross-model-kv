"""Strategy-agnostic quality evaluations for cross-model KV transfer.

Every eval here is written against two abstract callables so that the same code
measures the native target prefill, a mapped (cross-model) cache, or a mapped
cache with the last N prompt tokens recomputed by the target:

* :data:`Scorer` -- loglikelihood of continuations given contexts.
* :data:`Generator` -- greedy generation from a context.

Contents
--------
1. Multiple-choice loglikelihood tasks (lm-eval-harness prompt formats):
   HellaSwag, PIQA, ARC-Challenge, MMLU, WinoGrande.
2. :func:`chance_normalized_retention`.
3. Needle-in-a-haystack / RULER-style retrieval (:func:`make_niah`, :func:`run_niah`).
4. Held-out text loading plus perplexity / agreement helpers.
5. ``python src/xkv/evals.py`` runs an offline, stdlib-only self-check.

Scorer batching contract
------------------------
Each scorer call contains the choices of ``questions_per_call`` questions
(default 1). Pairs whose context token ids are identical (all tasks except
WinoGrande, whose *continuation* is shared instead) can be served from a single
prefilled cache: callers should use :func:`group_by_context` to prefill (or
transfer) once per unique context and then score each continuation on top of a
copy of that cache.

Heavy dependencies (``datasets``) are imported lazily inside loaders only.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "Scorer",
    "Generator",
    "MCItem",
    "group_by_context",
    "encode_pair",
    "run_hellaswag",
    "run_piqa",
    "run_arc_challenge",
    "run_mmlu",
    "run_winogrande",
    "chance_normalized_retention",
    "make_niah",
    "run_niah",
    "load_heldout_text",
    "prefix_continuation_pairs",
    "perplexity",
    "token_agreement",
    "generation_agreement",
]

Pair = tuple[list[int], list[int]]
#: (context_ids, continuation_ids) pairs -> (sum logprob, n continuation tokens, all-greedy).
Scorer = Callable[[list[Pair]], list[tuple[float, int, bool]]]
#: (context_ids, max_new_tokens) -> greedily generated token ids.
Generator = Callable[[list[int], int], list[int]]


# --------------------------------------------------------------------------- #
# Tokenization helpers
# --------------------------------------------------------------------------- #
def _encode(tokenizer: Any, text: str) -> list[int]:
    """Encode without adding BOS/EOS (HF tokenizers and simple fakes)."""
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)
    return list(ids)


def _decode(tokenizer: Any, ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(list(ids), skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(list(ids))


def encode_pair(tokenizer: Any, context: str, continuation: str) -> Pair:
    """lm-eval-harness style encoding of a (context, continuation) pair.

    Trailing whitespace of the context is moved onto the continuation, the
    concatenation is encoded once, and the continuation ids are the suffix
    after the context ids. If the context is empty, a BOS/EOS id is used as the
    one-token context (as lm-eval does).
    """
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    if not context:
        prefix = getattr(tokenizer, "bos_token_id", None)
        if prefix is None:
            prefix = getattr(tokenizer, "eos_token_id", None)
        if prefix is None:
            raise ValueError("empty context and tokenizer has no BOS/EOS id")
        return [int(prefix)], _encode(tokenizer, continuation)
    whole = _encode(tokenizer, context + continuation)
    ctx = _encode(tokenizer, context)
    # Guard against merges across the boundary changing the context ids.
    if whole[: len(ctx)] != ctx:
        return ctx, _encode(tokenizer, continuation)
    return ctx, whole[len(ctx):]


def group_by_context(pairs: Sequence[Pair]) -> dict[tuple[int, ...], list[int]]:
    """Map each unique context (as a tuple) to indices of pairs that share it.

    A scorer can prefill/transfer one cache per key and reuse it for every
    continuation index listed under that key.
    """
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, (ctx, _) in enumerate(pairs):
        groups.setdefault(tuple(ctx), []).append(i)
    return groups


# --------------------------------------------------------------------------- #
# Dataset loading (lazy)
# --------------------------------------------------------------------------- #
_Candidate = tuple[str, "str | None", "str | None"]  # (path, config name, revision)


def _load_hf(candidates: Sequence[_Candidate], split: str) -> Any:
    """Try ``datasets.load_dataset`` on each candidate until one works."""
    import datasets  # lazy: heavy and network-bound

    errors: list[str] = []
    for path, name, revision in candidates:
        try:
            kwargs: dict[str, Any] = {"split": split}
            if revision:
                kwargs["revision"] = revision
            return datasets.load_dataset(path, name, **kwargs)
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            errors.append(f"{path}/{name}@{revision or 'main'}: {type(exc).__name__}: {exc}")
    raise RuntimeError("could not load any dataset candidate:\n  " + "\n  ".join(errors))


_PARQUET = "refs/convert/parquet"
DATASETS: dict[str, tuple[list[_Candidate], str]] = {
    "hellaswag": ([("Rowan/hellaswag", None, None), ("Rowan/hellaswag", None, _PARQUET)], "validation"),
    "piqa": (
        [("ybisk/piqa", None, None), ("baber/piqa", None, None), ("ybisk/piqa", None, _PARQUET)],
        "validation",
    ),
    "arc_challenge": ([("allenai/ai2_arc", "ARC-Challenge", None)], "test"),
    "mmlu": ([("cais/mmlu", "all", None)], "test"),
    "winogrande": (
        [("allenai/winogrande", "winogrande_xl", None), ("allenai/winogrande", "winogrande_xl", _PARQUET)],
        "validation",
    ),
}


def _subsample(n_total: int, limit: int, seed: int) -> list[int]:
    """Deterministic, sorted subsample of ``range(n_total)`` (all if limit<=0)."""
    if limit <= 0 or limit >= n_total:
        return list(range(n_total))
    return sorted(random.Random(seed).sample(range(n_total), limit))


def _get_docs(task: str, docs: Sequence[dict] | None, limit: int, seed: int) -> list[dict]:
    if docs is None:
        candidates, split = DATASETS[task]
        docs = _load_hf(candidates, split)
    return [docs[i] for i in _subsample(len(docs), limit, seed)]


# --------------------------------------------------------------------------- #
# Multiple-choice core
# --------------------------------------------------------------------------- #
@dataclass
class MCItem:
    """One multiple-choice question as per-choice (context, continuation) strings."""

    contexts: list[str]
    continuations: list[str]
    gold: int
    norm_lens: list[int]  # character lengths used for acc_norm


def _shared_ctx_item(context: str, choices: Sequence[str], gold: int, delim: str = " ") -> MCItem:
    return MCItem(
        contexts=[context] * len(choices),
        continuations=[delim + c for c in choices],
        gold=gold,
        norm_lens=[max(1, len(c)) for c in choices],
    )


def _stderr(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / (n - 1)) if n > 1 else float("nan")


def _run_mc(
    task: str,
    tokenizer: Any,
    scorer: Scorer,
    items: Sequence[MCItem],
    chance: float,
    has_norm: bool,
    questions_per_call: int = 1,
) -> dict:
    """Score MC items; returns acc / acc_norm (lm-eval semantics)."""
    correct = correct_norm = 0
    qpc = max(1, questions_per_call)
    for start in range(0, len(items), qpc):
        chunk = items[start:start + qpc]
        pairs: list[Pair] = []
        for it in chunk:
            pairs.extend(encode_pair(tokenizer, c, k) for c, k in zip(it.contexts, it.continuations))
        results = scorer(pairs)
        if len(results) != len(pairs):
            raise ValueError(f"scorer returned {len(results)} results for {len(pairs)} pairs")
        off = 0
        for it in chunk:
            k = len(it.continuations)
            lls = [r[0] for r in results[off:off + k]]
            off += k
            pred = max(range(k), key=lambda j: lls[j])
            pred_norm = max(range(k), key=lambda j: lls[j] / it.norm_lens[j])
            correct += int(pred == it.gold)
            correct_norm += int(pred_norm == it.gold)
    n = len(items)
    acc = correct / n if n else float("nan")
    out: dict[str, Any] = {"task": task, "acc": acc, "acc_stderr": _stderr(acc, n), "n": n, "chance": chance}
    if has_norm:
        acc_norm = correct_norm / n if n else float("nan")
        out.update(acc_norm=acc_norm, acc_norm_stderr=_stderr(acc_norm, n))
    return out


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
def _hellaswag_preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def hellaswag_item(doc: dict) -> MCItem:
    """HellaSwag doc -> MCItem (lm-eval preprocessing)."""
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    query = _hellaswag_preprocess(doc["activity_label"] + ": " + ctx)
    choices = [_hellaswag_preprocess(e) for e in doc["endings"]]
    return _shared_ctx_item(query, choices, int(doc["label"]))


def piqa_item(doc: dict) -> MCItem:
    """PIQA doc -> MCItem ("Question: {goal}\\nAnswer:")."""
    return _shared_ctx_item(f"Question: {doc['goal']}\nAnswer:", [doc["sol1"], doc["sol2"]], int(doc["label"]))


def arc_item(doc: dict) -> MCItem:
    """ARC doc -> MCItem ("Question: {question}\\nAnswer:")."""
    labels = list(doc["choices"]["label"])
    return _shared_ctx_item(
        f"Question: {doc['question']}\nAnswer:", list(doc["choices"]["text"]), labels.index(doc["answerKey"])
    )


def mmlu_item(doc: dict, include_description: bool = True) -> MCItem:
    """MMLU doc -> MCItem (0-shot, lm-eval default template, scoring " A".." D")."""
    letters = ["A", "B", "C", "D"]
    body = doc["question"].strip() + "\n"
    body += "".join(f"{l}. {c}\n" for l, c in zip(letters, doc["choices"])) + "Answer:"
    if include_description:
        subject = str(doc.get("subject", "")).replace("_", " ")
        body = f"The following are multiple choice questions (with answers) about {subject}.\n\n" + body
    return _shared_ctx_item(body, letters, int(doc["answer"]))


def winogrande_item(doc: dict) -> MCItem:
    """WinoGrande doc -> MCItem with lm-eval partial scoring.

    Contexts differ per option (sentence up to the blank with the option filled
    in); the continuation (rest of the sentence) is shared and is what is scored.
    """
    sentence = doc["sentence"]
    idx = sentence.index("_")
    target = sentence[idx + 1:].strip()
    contexts = [sentence[:idx] + opt for opt in (doc["option1"], doc["option2"])]
    return MCItem(contexts, [" " + target] * 2, {"1": 0, "2": 1}[str(doc["answer"])], [1, 1])


def run_hellaswag(tokenizer: Any, scorer: Scorer, limit: int, seed: int = 0,
                  docs: Sequence[dict] | None = None, questions_per_call: int = 1) -> dict:
    """HellaSwag (Rowan/hellaswag validation): acc, acc_norm; chance 0.25."""
    items = [hellaswag_item(d) for d in _get_docs("hellaswag", docs, limit, seed)]
    return _run_mc("hellaswag", tokenizer, scorer, items, 0.25, True, questions_per_call)


def run_piqa(tokenizer: Any, scorer: Scorer, limit: int, seed: int = 0,
             docs: Sequence[dict] | None = None, questions_per_call: int = 1) -> dict:
    """PIQA (ybisk/piqa, fallback baber/piqa, validation): acc, acc_norm; chance 0.5."""
    items = [piqa_item(d) for d in _get_docs("piqa", docs, limit, seed)]
    return _run_mc("piqa", tokenizer, scorer, items, 0.5, True, questions_per_call)


def run_arc_challenge(tokenizer: Any, scorer: Scorer, limit: int, seed: int = 0,
                      docs: Sequence[dict] | None = None, questions_per_call: int = 1) -> dict:
    """ARC-Challenge (allenai/ai2_arc test): acc, acc_norm; chance = mean 1/#choices."""
    items = [arc_item(d) for d in _get_docs("arc_challenge", docs, limit, seed)]
    chance = sum(1 / len(it.continuations) for it in items) / len(items) if items else 0.25
    return _run_mc("arc_challenge", tokenizer, scorer, items, chance, True, questions_per_call)


def run_mmlu(tokenizer: Any, scorer: Scorer, limit: int, seed: int = 0,
             docs: Sequence[dict] | None = None, questions_per_call: int = 1,
             include_description: bool = True) -> dict:
    """MMLU (cais/mmlu "all" test, 0-shot): acc; chance 0.25."""
    items = [mmlu_item(d, include_description) for d in _get_docs("mmlu", docs, limit, seed)]
    return _run_mc("mmlu", tokenizer, scorer, items, 0.25, False, questions_per_call)


def run_winogrande(tokenizer: Any, scorer: Scorer, limit: int, seed: int = 0,
                   docs: Sequence[dict] | None = None, questions_per_call: int = 1) -> dict:
    """WinoGrande (allenai/winogrande winogrande_xl validation, partial scoring): acc; chance 0.5."""
    items = [winogrande_item(d) for d in _get_docs("winogrande", docs, limit, seed)]
    return _run_mc("winogrande", tokenizer, scorer, items, 0.5, False, questions_per_call)


def chance_normalized_retention(mapped_acc: float, oracle_acc: float, chance: float) -> float:
    """OneTriangle retention: (mapped - chance) / (oracle - chance); NaN if oracle <= chance."""
    denom = oracle_acc - chance
    if denom <= 0:
        return float("nan")
    return (mapped_acc - chance) / denom


# --------------------------------------------------------------------------- #
# Needle in a haystack
# --------------------------------------------------------------------------- #
_ADJ = ("quiet ancient bright narrow heavy gentle distant golden hollow silver rapid "
        "patient curious steady crooked frozen humble marble open rustic").split()
_NOUN = ("river valley lantern harbor meadow tower orchard bridge garden mountain "
         "village forest market library window island road castle field shore").split()
_VERB = ("crossed watched followed carried passed reached guarded circled faced "
         "joined left found touched framed shadowed").split()
_PREP = "near beyond beside under above behind toward across".split()
_TIME = ("in the morning|at dusk|during the long winter|after the rain|before the harvest|"
         "late in the evening|on a calm day|through the autumn").split("|")


def _builtin_sentences(n: int, seed: int) -> list[str]:
    """Procedurally generated, digit-free filler sentences (deterministic)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append(
            f"The {rng.choice(_ADJ)} {rng.choice(_NOUN)} {rng.choice(_VERB)} the {rng.choice(_ADJ)} "
            f"{rng.choice(_NOUN)} {rng.choice(_PREP)} the {rng.choice(_NOUN)} {rng.choice(_TIME)}."
        )
    return out


def _wikitext_sentences(max_sentences: int) -> list[str]:
    """Sentences from wikitext-103 train with digits removed (to keep needles unique)."""
    ds = _load_hf([("Salesforce/wikitext", "wikitext-103-raw-v1", None)], "train")
    out: list[str] = []
    for row in ds:
        line = row["text"].strip()
        if not line or line.startswith("="):
            continue
        line = re.sub(r"\d", "", line)
        out.extend(s.strip() + "." for s in line.split(". ") if len(s.split()) >= 5)
        if len(out) >= max_sentences:
            break
    return out


def _chat_prompt(tokenizer: Any, user: str) -> str:
    """Render a single user turn with the chat template (thinking disabled if supported)."""
    messages = [{"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        pass
    except Exception:  # noqa: BLE001 - no template at all
        return f"User: {user}\nAssistant:"
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:  # noqa: BLE001
        return f"User: {user}\nAssistant:"


_NEEDLE = "The special magic number for {key} is {value}."
_QUESTION = "What is the special magic number for {key} mentioned in the provided text?"
_ANSWER_PREFIX = "The special magic number for {key} mentioned in the provided text is"


def _niah_prompt(tokenizer: Any, haystack: str, key: str) -> str:
    user = (
        "Some special magic numbers are hidden within the following text. Make sure to memorize "
        "them. I will quiz you about the numbers afterwards.\n\n"
        f"{haystack}\n\n{_QUESTION.format(key=key)}"
    )
    return _chat_prompt(tokenizer, user) + _ANSWER_PREFIX.format(key=key)


def _compose(filler: list[str], needles: list[tuple[float, str]]) -> str:
    sents = list(filler)
    # Insert deepest first so earlier insertion points are unaffected.
    for depth, needle in sorted(needles, key=lambda x: -x[0]):
        sents.insert(int(round(min(max(depth, 0.0), 1.0) * len(filler))), needle)
    return " ".join(sents)


def make_niah(
    tokenizer: Any,
    context_tokens: int,
    n: int,
    seed: int = 0,
    depth_fracs: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    num_keys: int = 1,
    filler_source: str = "builtin",
) -> list[dict]:
    """Build ``n`` deterministic NIAH prompts of at most ``context_tokens`` tokens.

    Sample ``i`` places the queried needle at ``depth_fracs[i % len(depth_fracs)]``.
    With ``num_keys > 1`` (multi-key variant) extra distractor needles with other
    keys are placed at random depths and only one key is asked for. Each sample is
    ``{"input_ids", "answer", "key", "depth", "num_keys", "n_tokens"}``.
    ``filler_source`` is ``"builtin"`` (offline) or ``"wikitext"`` (falls back to
    builtin on any loading error).
    """
    if num_keys < 1:
        raise ValueError("num_keys must be >= 1")
    rng = random.Random(seed)
    overhead = len(_encode(tokenizer, _niah_prompt(tokenizer, "", "x-y")))
    budget = context_tokens - overhead
    if budget <= 0:
        raise ValueError(f"context_tokens={context_tokens} smaller than prompt overhead {overhead}")

    # Pool of filler sentences with token counts; samples take contiguous runs.
    probe = _builtin_sentences(64, seed)
    avg = max(1.0, sum(len(_encode(tokenizer, " " + s)) for s in probe) / len(probe))
    need = int(budget / avg * 1.3) + 16
    pool: list[str] = []
    if filler_source == "wikitext":
        try:
            pool = _wikitext_sentences(need)
        except Exception:  # noqa: BLE001 - deterministic offline fallback
            pool = []
    if len(pool) < need:
        pool = _builtin_sentences(need, seed)
    lens = [len(_encode(tokenizer, " " + s)) for s in pool]

    samples = []
    for i in range(n):
        depth = float(depth_fracs[i % len(depth_fracs)])
        keys: list[str] = []
        while len(keys) < num_keys:
            k = f"{rng.choice(_ADJ)}-{rng.choice(_NOUN)}"
            if k not in keys:
                keys.append(k)
        values = [str(rng.randint(1_000_000, 9_999_999)) for _ in keys]
        needles = [(depth, _NEEDLE.format(key=keys[0], value=values[0]))]
        needles += [(rng.random(), _NEEDLE.format(key=k, value=v)) for k, v in zip(keys[1:], values[1:])]
        needle_tok = sum(len(_encode(tokenizer, " " + t)) for _, t in needles)

        start = rng.randrange(len(pool))
        filler: list[str] = []
        total = needle_tok
        j = start
        while total < budget and len(filler) < len(pool):
            filler.append(pool[j])
            total += lens[j]
            j = (j + 1) % len(pool)

        ids = _encode(tokenizer, _niah_prompt(tokenizer, _compose(filler, needles), keys[0]))
        for _ in range(20):  # trim until within budget
            excess = len(ids) - context_tokens
            if excess <= 0 or not filler:
                break
            drop = max(1, int(math.ceil(excess / avg)))
            filler = filler[: max(0, len(filler) - drop)]
            ids = _encode(tokenizer, _niah_prompt(tokenizer, _compose(filler, needles), keys[0]))
        samples.append({"input_ids": ids, "answer": values[0], "key": keys[0], "depth": depth,
                        "num_keys": num_keys, "n_tokens": len(ids)})
    return samples


def run_niah(
    tokenizer: Any,
    generator: Generator,
    context_tokens: int = 4096,
    n: int = 20,
    seed: int = 0,
    depth_fracs: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    num_keys: int = 1,
    max_new_tokens: int = 16,
    filler_source: str = "builtin",
    samples: Sequence[dict] | None = None,
) -> dict:
    """Exact-match NIAH accuracy overall and per depth.

    A sample is correct when the first 7-digit number in the decoded output
    equals the needle value (RULER string match, stricter on distractors).
    Pass ``samples`` (from :func:`make_niah`) to reuse identical prompts across
    prefill strategies.
    """
    if samples is None:
        samples = make_niah(tokenizer, context_tokens, n, seed, depth_fracs, num_keys, filler_source)
    per_depth: dict[str, list[int]] = {}
    outputs = []
    for s in samples:
        text = _decode(tokenizer, generator(list(s["input_ids"]), max_new_tokens))
        m = re.search(r"\d{7}", text)
        ok = int(m is not None and m.group(0) == s["answer"])
        per_depth.setdefault(f"{s['depth']:.2f}", []).append(ok)
        outputs.append(text)
    flat = [x for v in per_depth.values() for x in v]
    return {
        "task": f"niah_{num_keys}key" if samples and samples[0]["num_keys"] > 1 else "niah",
        "acc": sum(flat) / len(flat) if flat else float("nan"),
        "acc_by_depth": {d: sum(v) / len(v) for d, v in sorted(per_depth.items())},
        "n": len(flat),
        "chance": 0.0,
        "mean_tokens": sum(s["n_tokens"] for s in samples) / len(samples) if samples else 0,
        "outputs": outputs,
    }


# --------------------------------------------------------------------------- #
# Held-out text, perplexity and agreement
# --------------------------------------------------------------------------- #
def _doc_bucket(key: str) -> str:
    """Stable 50/50 document split: 'calib' or 'eval'."""
    return "calib" if int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 2 == 0 else "eval"


def load_heldout_text(
    tokenizer: Any,
    n_seqs: int,
    seq_len: int,
    seed: int = 0,
    split: str = "eval",
    source: str = "wikitext",
    max_docs: int = 200_000,
) -> list[list[int]]:
    """Return up to ``n_seqs`` token-id sequences of exactly ``seq_len`` tokens.

    ``split="calib"`` and ``split="eval"`` never share documents:

    * ``source="wikitext"`` (Salesforce/wikitext, wikitext-103-raw-v1): calib uses
      the ``train`` split, eval uses ``test`` (disjoint articles). Non-overlapping
      windows are sampled deterministically from the concatenated text.
    * ``source="fineweb"`` (HuggingFaceFW/fineweb-edu, sample-10BT, streaming):
      documents are assigned to calib/eval by a SHA-256 hash of their id; docs are
      packed (EOS-separated) into chunks and leftovers are dropped, so each doc
      contributes to at most one sequence.
    """
    if split not in ("calib", "eval"):
        raise ValueError("split must be 'calib' or 'eval'")
    rng = random.Random(seed)
    if source == "wikitext":
        ds = _load_hf([("Salesforce/wikitext", "wikitext-103-raw-v1", None)],
                      "train" if split == "calib" else "test")
        max_chars = n_seqs * seq_len * 12 if split == "calib" else None
        parts: list[str] = []
        n_chars = 0
        for row in ds:
            parts.append(row["text"])
            n_chars += len(row["text"])
            if max_chars is not None and n_chars >= max_chars:
                break
        text = "".join(parts)
        ids: list[int] = []
        step = 1_000_000  # tokenize in chunks split on newlines
        pos = 0
        while pos < len(text):
            end = min(len(text), pos + step)
            nl = text.rfind("\n", pos, end) if end < len(text) else end
            end = nl + 1 if nl > pos else end
            ids.extend(_encode(tokenizer, text[pos:end]))
            pos = end
        n_windows = len(ids) // seq_len
        chosen = sorted(rng.sample(range(n_windows), min(n_seqs, n_windows)))
        return [ids[w * seq_len:(w + 1) * seq_len] for w in chosen]

    if source == "fineweb":
        import datasets  # lazy

        stream = datasets.load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                                       split="train", streaming=True)
        stream = stream.shuffle(seed=seed, buffer_size=10_000)
        eos = getattr(tokenizer, "eos_token_id", None)
        out: list[list[int]] = []
        buf: list[int] = []
        for k, row in enumerate(stream):
            if k >= max_docs or len(out) >= n_seqs:
                break
            if _doc_bucket(str(row.get("id") or row["text"][:256])) != split:
                continue
            buf.extend(_encode(tokenizer, row["text"]))
            if eos is not None:
                buf.append(int(eos))
            if len(buf) >= seq_len:
                out.append(buf[:seq_len])
                buf = []
        return out
    raise ValueError(f"unknown source {source!r}")


def prefix_continuation_pairs(seqs: Iterable[Sequence[int]], prefix_len: int) -> list[Pair]:
    """Split each sequence into (prefix to prefill/transfer, continuation to score)."""
    out = []
    for s in seqs:
        if not 0 < prefix_len < len(s):
            raise ValueError(f"prefix_len={prefix_len} invalid for length {len(s)}")
        out.append((list(s[:prefix_len]), list(s[prefix_len:])))
    return out


def perplexity(scorer: Scorer, pairs: Sequence[Pair], batch_size: int = 1) -> dict:
    """Token-weighted perplexity of continuations plus sequence-level greedy match rate."""
    total_lp = 0.0
    total_n = 0
    greedy = 0
    for i in range(0, len(pairs), max(1, batch_size)):
        for lp, n, g in scorer(list(pairs[i:i + batch_size])):
            total_lp += lp
            total_n += n
            greedy += int(g)
    nll = -total_lp / total_n if total_n else float("nan")
    return {"ppl": math.exp(nll) if total_n else float("nan"), "nll_per_token": nll,
            "n_tokens": total_n, "n_seqs": len(pairs),
            "seq_greedy_match_rate": greedy / len(pairs) if pairs else float("nan")}


def token_agreement(ref: Sequence[int], hyp: Sequence[int]) -> dict:
    """Agreement between two generations: exact, common-prefix length, positional match rate."""
    prefix = 0
    for a, b in zip(ref, hyp):
        if a != b:
            break
        prefix += 1
    denom = max(len(ref), len(hyp))
    pos = sum(a == b for a, b in zip(ref, hyp)) / denom if denom else 1.0
    return {"exact": list(ref) == list(hyp), "prefix_len": prefix,
            "prefix_frac": prefix / denom if denom else 1.0, "position_match": pos}


def generation_agreement(ref_generator: Generator, test_generator: Generator,
                         contexts: Sequence[Sequence[int]], max_new_tokens: int = 64) -> dict:
    """Mean greedy-generation agreement of ``test_generator`` against ``ref_generator``."""
    rows = [token_agreement(ref_generator(list(c), max_new_tokens), test_generator(list(c), max_new_tokens))
            for c in contexts]
    n = len(rows)
    if not n:
        return {"n": 0}
    return {"n": n,
            "exact_rate": sum(r["exact"] for r in rows) / n,
            "mean_prefix_len": sum(r["prefix_len"] for r in rows) / n,
            "mean_prefix_frac": sum(r["prefix_frac"] for r in rows) / n,
            "mean_position_match": sum(r["position_match"] for r in rows) / n}


# --------------------------------------------------------------------------- #
# Offline self-check
# --------------------------------------------------------------------------- #
class _CharTokenizer:
    """Char-level fake tokenizer with a trivial chat template (no special tokens)."""

    eos_token_id = 0
    bos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(i) for i in ids if i > 0)

    def apply_chat_template(self, messages: list[dict], tokenize: bool = False,
                            add_generation_prompt: bool = True, enable_thinking: bool = True) -> str:
        s = "".join(f"<{m['role']}>\n{m['content']}\n</{m['role']}>\n" for m in messages)
        return s + ("<assistant>\n" if add_generation_prompt else "")


def _self_check() -> None:
    tok = _CharTokenizer()

    def oracle_scorer(pairs: list[Pair]) -> list[tuple[float, int, bool]]:
        # Rewards any (context+continuation) containing the marker "right".
        return [(0.0 if "right" in tok.decode(c + k) else -10.0, len(k), True) for c, k in pairs]

    def hash_scorer(pairs: list[Pair]) -> list[tuple[float, int, bool]]:
        return [(-(zlib.crc32(bytes(str(c + k), "utf-8")) % 1000) / 100.0, len(k), False) for c, k in pairs]

    calls: list[int] = []

    def counting(pairs: list[Pair]) -> list[tuple[float, int, bool]]:
        calls.append(len(group_by_context(pairs)))
        return oracle_scorer(pairs)

    hs = [{"activity_label": "Cooking", "ctx_a": "A man [title] cuts onions.", "ctx_b": "then he",
           "endings": ["fries them right away.", "sings loudly.", "leaves.", "sleeps [step] now."],
           "label": "0"}] * 3
    pq = [{"goal": "Open a jar", "sol1": "twist the lid", "sol2": "twist it right", "label": 1}] * 4
    arc = [{"question": "Q?", "choices": {"text": ["no", "right", "nah"], "label": ["A", "B", "C"]},
            "answerKey": "B"}] * 2
    mm = [{"question": "2+2?", "subject": "high_school_math", "choices": ["1", "4", "3", "5"],
           "answer": 1}] * 5
    wg = [{"sentence": "Tom thanked Bob because _ was kind.", "option1": "Tom", "option2": "right",
           "answer": "2"}] * 4

    res = {
        "hellaswag": run_hellaswag(tok, counting, limit=2, docs=hs),
        "piqa": run_piqa(tok, oracle_scorer, limit=0, docs=pq, questions_per_call=3),
        "arc": run_arc_challenge(tok, oracle_scorer, limit=0, docs=arc),
        "mmlu": run_mmlu(tok, hash_scorer, limit=3, seed=1, docs=mm),
        "winogrande": run_winogrande(tok, oracle_scorer, limit=0, docs=wg),
    }
    assert hellaswag_item(hs[0]).contexts[0] == "Cooking: A man. cuts onions. Then he", hellaswag_item(hs[0]).contexts[0]
    assert calls == [1, 1], calls  # one shared context per HellaSwag question
    for name in ("hellaswag", "piqa", "arc", "winogrande"):
        assert res[name]["acc"] == 1.0, (name, res[name])
    assert res["mmlu"]["n"] == 3 and 0.0 <= res["mmlu"]["acc"] <= 1.0
    assert abs(res["arc"]["chance"] - 1 / 3) < 1e-9
    ctx, cont = encode_pair(tok, "Answer: ", "B")
    assert tok.decode(ctx) == "Answer:" and tok.decode(cont) == " B"
    for name, r in res.items():
        print(f"[mc] {name:10s} " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                              for k, v in r.items() if k != "task"))

    assert chance_normalized_retention(0.7, 0.8, 0.25) == (0.7 - 0.25) / (0.8 - 0.25)
    assert math.isnan(chance_normalized_retention(0.5, 0.25, 0.25))
    print(f"[retention] (0.70 vs 0.80, chance 0.25) = {chance_normalized_retention(0.7, 0.8, 0.25):.4f}")

    def needle_gen(ids: list[int], max_new: int) -> list[int]:
        text = tok.decode(ids)
        key = re.findall(r"special magic number for (\S+) mentioned", text)[-1]
        val = re.search(rf"special magic number for {re.escape(key)} is (\d{{7}})", text).group(1)
        return tok.encode(" " + val + ".")[:max_new]

    def dumb_gen(ids: list[int], max_new: int) -> list[int]:
        return tok.encode(" 1234567.")[:max_new]

    for keys in (1, 4):
        samples = make_niah(tok, context_tokens=3000, n=10, seed=0, num_keys=keys)
        assert all(2500 <= s["n_tokens"] <= 3000 for s in samples), [s["n_tokens"] for s in samples]
        again = make_niah(tok, context_tokens=3000, n=10, seed=0, num_keys=keys)
        assert [s["input_ids"] for s in samples] == [s["input_ids"] for s in again]  # deterministic
        good = run_niah(tok, needle_gen, samples=samples)
        bad = run_niah(tok, dumb_gen, samples=samples)
        assert good["acc"] == 1.0 and bad["acc"] == 0.0, (good["acc"], bad["acc"])
        print(f"[niah] keys={keys} oracle acc={good['acc']:.2f} by_depth={good['acc_by_depth']} "
              f"dumb acc={bad['acc']:.2f} mean_tokens={good['mean_tokens']:.0f}")

    seqs = [tok.encode("hello world, this is held-out text number %d." % i) for i in range(3)]
    pairs = prefix_continuation_pairs(seqs, prefix_len=10)
    ppl = perplexity(lambda ps: [(-0.5 * len(k), len(k), True) for _, k in ps], pairs, batch_size=2)
    assert abs(ppl["ppl"] - math.exp(0.5)) < 1e-9
    agr = generation_agreement(needle_gen, lambda ids, m: needle_gen(ids, m)[:4] + [1, 2],
                               [s["input_ids"] for s in samples[:3]], 16)
    assert agr["exact_rate"] == 0.0 and agr["mean_prefix_len"] == 4
    print(f"[ppl] {ppl}")
    print(f"[agreement] {agr}")
    assert _doc_bucket("doc-a") in ("calib", "eval")
    print("self-check OK (no network, datasets not imported)")


if __name__ == "__main__":
    import sys

    _self_check()
    assert "datasets" not in sys.modules
