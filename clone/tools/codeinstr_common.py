"""Shared (no GPU) pieces for code map v2: eval-format prompt rendering and the xkv agent-log sample check."""
import json

# src/xkv/genevals.py _EVALPLUS_PREFIX: the bench appends it to the prompt (assistant prefix); the native answer
# continues after it. In the distillation sample it opens the assistant message (continuation side).
EVALPLUS_PREFIX = ("Below is a Python script with a self-contained function that solves the problem and "
                   "passes corresponding tests:\n```python\n")


def eval_prompt(tok, user):
    """The bench's prompt text: chat template (thinking off) + EvalPlus assistant prefix (genevals.chat_prompt)."""
    return tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False) + EVALPLUS_PREFIX


def agent_line(user, answer):
    """One xkv agent_logs JSONL line (task_distill agent_logs): messages, tools, out (the native reply)."""
    return json.dumps(dict(messages=[{"role": "user", "content": user}], tools=[],
                           out={"content": EVALPLUS_PREFIX + answer}))


def check_agent_sample(tok, line, cont=256, max_prefix=2048):
    """Replays xkv task_distill's agent-sample parsing on one line; returns (prompt ids, cont ids) or raises with the
    reason (a sample xkv would silently skip)."""
    from xkv.agentbench import render_prompt, to_template_messages
    r = json.loads(line)
    pre = render_prompt(tok, r["messages"], r["tools"], False)
    out = dict(r["out"], role="assistant")
    full = tok.apply_chat_template(to_template_messages(r["messages"] + [out]), tools=r["tools"] or None,
                                   tokenize=False, add_generation_prompt=False, enable_thinking=False)
    if not full.startswith(pre):
        raise ValueError(f"render mismatch: pre ends {pre[-60:]!r}, full at that point {full[len(pre) - 60:len(pre) + 20]!r}")
    pi = tok.encode(pre, add_special_tokens=False)
    ci = tok.encode(full[len(pre):], add_special_tokens=False)[:cont]
    if len(pi) > max_prefix or len(ci) < 2:
        raise ValueError(f"length: prompt {len(pi)} cont {len(ci)}")
    return pi, ci
