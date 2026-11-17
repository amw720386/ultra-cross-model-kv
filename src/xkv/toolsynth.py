"""Multi-turn tool-calling distillation samples: (agent prompt, reference assistant turn) from public data.

Each sample is a Qwen-chat-templated agent prompt -- a long system document, `tools=` schemas rendered by the
template, and a user/assistant/tool conversation -- that stops right before an assistant turn. The reference
continuation is that assistant turn rendered exactly as the template renders it (a tool call in the template's
`<tool_call><function=...>` format, or a natural-language reply), ending with `<|im_end|>`.

Sources (all streamed; nothing is downloaded in full):
  * NousResearch/hermes-function-calling-v1 (Apache-2.0), configs
      - `func_calling`             multi-turn, parallel tool calls, rich JSON-schema tools
      - `glaive_func_calling`      multi-turn conversations re-formatted from glaiveai/glaive-function-calling-v2
                                   (Apache-2.0), with tool calls and tool responses
      - `func_calling_singleturn`  single tool-calling turn over large tool sets
    Hermes rows are ShareGPT style (`human`/`gpt`/`tool`) with `<tool_call>`/`<tool_response>` JSON blocks and a
    `tools` JSON column; they are converted to OpenAI-style messages (assistant `tool_calls`, role "tool").
  * ccdv/govreport-summarization (U.S. GAO / CRS reports, public-domain government works): long policy and
    regulation style prose used as the operating manual in the system prompt to reach long prompt lengths.

No tau-bench / tau2-bench data, no Salesforce APIGen-MT (built from tau-bench environments), and nothing derived
from the airline / retail / telecom tau domains is used.

To reach a prompt length P, earlier complete conversations from other rows are prepended to the history (their
tools are merged into the schema list) and the system document is filled up to roughly 85-95% of P.
"""
from __future__ import annotations

import ast
import json
import random
import re

_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_TOOL_RESP = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.S)

_SYSTEM_HEADERS = [
    "You are an operations assistant for a service desk. Use the available tools to act on the user's behalf. "
    "Follow the reference manual below when answering; do not invent tool arguments the user has not provided.",
    "You are a support agent with access to internal tools. The handbook below is part of your operating "
    "context. Call a tool when it is needed to complete the request; otherwise reply directly to the user.",
    "You are an assistant that helps users by calling the functions provided. Read the policy and background "
    "material below carefully; it governs how requests should be handled.",
]


def _fix_text(s: str) -> str:
    """GovReport replaces curly quotes and dashes with U+FFFD; map them back heuristically."""
    s = re.sub(r"(?<=[A-Za-z])�(?=(s|t|re|ll|ve|d|m)\b)", "'", s)
    s = re.sub(r"(?<=\w)�(?=\w)", " - ", s)
    return s.replace("�", '"')


def _load_json(s):
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except Exception:
        return ast.literal_eval(s)


def _tool_ok(t) -> bool:
    f = t.get("function") if isinstance(t, dict) else None
    return isinstance(f, dict) and isinstance(f.get("name"), str) and f["name"] and \
        isinstance(f.get("parameters", {}), dict)


def convert(row) -> tuple[list[dict], list[dict]] | None:
    """Hermes ShareGPT row -> (messages without system, tools); None if the row is malformed."""
    try:
        tools = _load_json(row["tools"])
        if isinstance(tools, dict):
            tools = [tools]
        tools = [t if "function" in t else {"type": "function", "function": t} for t in tools]
        if not tools or not all(_tool_ok(t) for t in tools):
            return None
        names = {t["function"]["name"] for t in tools}
        msgs: list[dict] = []
        for turn in row["conversations"]:
            who, val = turn["from"], turn["value"]
            if not isinstance(val, str):
                return None
            if who == "system":
                continue
            if who == "human":
                if not val.strip():
                    return None
                msgs.append({"role": "user", "content": val.strip()})
            elif who == "gpt":
                calls = []
                for blob in _TOOL_CALL.findall(val):
                    c = _load_json(blob)
                    args = _load_json(c.get("arguments", {}))
                    if c.get("name") not in names or not isinstance(args, dict):
                        return None
                    calls.append({"type": "function", "function": {"name": c["name"], "arguments": args}})
                text = _TOOL_CALL.sub("", val).strip()
                if "<tool_call>" in text or "</tool_call>" in text or "<tool_response>" in text:
                    return None
                if not text and not calls:
                    return None
                m = {"role": "assistant", "content": text}
                if calls:
                    m["tool_calls"] = calls
                msgs.append(m)
            elif who == "tool":
                resps = _TOOL_RESP.findall(val) or [val.strip()]
                if not msgs or msgs[-1]["role"] not in ("assistant", "tool"):
                    return None
                msgs += [{"role": "tool", "content": r} for r in resps if r]
            else:
                return None
        if not msgs or msgs[0]["role"] != "user" or not any(m["role"] == "assistant" for m in msgs):
            return None
        return msgs, tools
    except Exception:
        return None


class ToolSynth:
    """Agent-style tool-calling distillation samples for the Qwen chat template; see the module docstring."""

    CONFIGS = (("func_calling", 0.45), ("glaive_func_calling", 0.40), ("func_calling_singleturn", 0.15))
    TOOL_TARGET_FRAC = 0.7  # prefer an assistant turn that calls tools when the conversation has one

    def __init__(self, tok, seed=0):
        from datasets import load_dataset

        self.tok, self.rng, self.seed = tok, random.Random(seed), seed
        self.its = {cfg: iter(load_dataset("NousResearch/hermes-function-calling-v1", cfg, split="train",
                                           streaming=True).shuffle(seed=seed, buffer_size=2000))
                    for cfg, _ in self.CONFIGS}
        self.docs = iter(load_dataset("ccdv/govreport-summarization", split="train", streaming=True)
                         .shuffle(seed=seed, buffer_size=200))
        self.end = tok.convert_tokens_to_ids("<|im_end|>")
        self.doc_ids: list[int] = []  # leftover document tokens, consumed front to back

    # ---- helpers ---------------------------------------------------------------------------------------------
    def _enc(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def _render(self, msgs, tools, gen):
        kw = dict(tools=tools, add_generation_prompt=gen, tokenize=False)
        try:
            return self.tok.apply_chat_template(msgs, enable_thinking=False, **kw)
        except TypeError:
            return self.tok.apply_chat_template(msgs, **kw)

    def _conv(self, cfg=None):
        if cfg is None:
            cfg = self.rng.choices([c for c, _ in self.CONFIGS], [w for _, w in self.CONFIGS])[0]
        for _ in range(1000):
            try:
                row = next(self.its[cfg])
            except StopIteration:  # restart the epoch with a new shuffle order
                from datasets import load_dataset
                self.seed += 1
                self.its[cfg] = iter(load_dataset("NousResearch/hermes-function-calling-v1", cfg, split="train",
                                                  streaming=True).shuffle(seed=self.seed, buffer_size=2000))
                continue
            out = convert(row)
            if out is not None:
                return out
        raise RuntimeError(f"no well-formed rows in {cfg}")

    def _doc_tokens(self, n):
        """Next n tokens of system document text (whole reports concatenated as needed)."""
        while len(self.doc_ids) < n:
            try:
                r = next(self.docs)["report"]
            except StopIteration:
                from datasets import load_dataset
                self.seed += 1
                self.docs = iter(load_dataset("ccdv/govreport-summarization", split="train", streaming=True)
                                 .shuffle(seed=self.seed, buffer_size=200))
                continue
            if isinstance(r, str) and len(r) > 2000:
                self.doc_ids += self._enc(("\n\n" if self.doc_ids else "") + _fix_text(r.strip()))
        out, self.doc_ids = self.doc_ids[:n], self.doc_ids[n:]
        return out

    def _system(self, header, doc_ids):
        if not doc_ids:
            return header
        text = self.tok.decode(doc_ids)
        cut = text.rfind("\n")
        if cut > len(text) // 2:
            text = text[:cut]
        return f"{header}\n\n# Reference manual\n\n{text.strip()}"

    @staticmethod
    def _merge_tools(tool_lists):
        seen, out = set(), []
        for tools in tool_lists:
            for t in tools:
                if t["function"]["name"] not in seen:
                    seen.add(t["function"]["name"])
                    out.append(t)
        return out

    # ---- main entry ------------------------------------------------------------------------------------------
    def make(self, P: int, cont: int):
        enc = self._enc
        for _ in range(50):
            msgs, tools = self._conv()
            targets = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
            calls = [i for i in targets if msgs[i].get("tool_calls")]
            k = self.rng.choice(calls if calls and self.rng.random() < self.TOOL_TARGET_FRAC else targets)
            header = self.rng.choice(_SYSTEM_HEADERS)

            def render(history, all_tools, system):
                pre = [{"role": "system", "content": system}] + history + msgs[:k]
                p = self._render(pre, all_tools, True)
                full = self._render(pre + [msgs[k]], all_tools, False)
                return p, full

            try:
                p, full = render([], tools, header)
            except Exception:
                continue
            if not full.startswith(p):
                continue
            if len(enc(p)) > P:
                continue

            # Fill: a share of the slack with earlier conversations, the rest with the system document.
            goal = int(P * self.rng.uniform(0.85, 0.95))
            hist_share = self.rng.choice((0.0, 0.25, 0.5, 0.8))
            history, all_tools = [], list(tools)
            n = len(enc(p))
            tries = 0
            while n < hist_share * goal and tries < 8:
                tries += 1
                h_msgs, h_tools = self._conv()
                if h_msgs[-1]["role"] != "assistant":
                    continue
                cand_tools = self._merge_tools([all_tools, h_tools])
                try:
                    cp, cf = render(history + h_msgs, cand_tools, header)
                except Exception:
                    continue
                cn = len(enc(cp))
                if cn > goal or not cf.startswith(cp):
                    continue
                history, all_tools, p, full, n = history + h_msgs, cand_tools, cp, cf, cn

            budget = goal - n - 32
            system = header
            if budget > 64:
                doc = self._doc_tokens(budget)
                for _shrink in range(6):
                    system = self._system(header, doc)
                    p2, f2 = render(history, all_tools, system)
                    n2 = len(enc(p2))
                    if n2 <= P:
                        p, full, n = p2, f2, n2
                        break
                    doc = doc[: max(0, len(doc) - (n2 - P) - 64)]
                else:
                    system = header
            ids = enc(p)
            if len(ids) > P:
                continue
            c = full[len(p):]
            end = c.find("<|im_end|>")
            if end < 0:
                continue
            c_ids = enc(c[: end]) + [self.end]
            return ids, c_ids[:cont]
        raise RuntimeError("no tool-calling sample fits")


if __name__ == "__main__":
    import statistics
    import time

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    syn = ToolSynth(tok, seed=0)
    for P in (2048, 8192):
        t0, samples = time.time(), [syn.make(P, 256) for _ in range(20)]
        pl = [len(p) for p, _ in samples]
        cl = [len(c) for _, c in samples]
        texts = [(tok.decode(p), tok.decode(c)) for p, c in samples]
        n_call = sum("<tool_call>" in c for _, c in texts)
        n_end = sum(c[-1] == syn.end for _, c in samples)
        print(f"P={P}: {time.time() - t0:.1f}s  prompt len min/median/max = {min(pl)}/{statistics.median(pl)}/"
              f"{max(pl)} (mean frac of P {statistics.mean(pl) / P:.2f})  cont len min/median/max = "
              f"{min(cl)}/{statistics.median(cl)}/{max(cl)}  tool_call conts {n_call}/20  end with im_end "
              f"{n_end}/20  tool msgs in prompt {sum('<tool_response>' in p for p, _ in texts)}/20")
        assert all(n <= P for n in pl)
        assert all("<tools>" in p for p, _ in texts)
        assert n_call >= len(samples) / 2
    p, c = texts[0]
    print("=" * 30, "prompt tail (last 400 chars)", "=" * 30)
    print(p[-400:])
    print("=" * 30, "continuation", "=" * 30)
    print(c)
    print("self-check OK")
