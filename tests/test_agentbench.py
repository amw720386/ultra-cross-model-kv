"""CPU checks for xkv.agentbench: Qwen3.5 template round trip, tool-call parsing, micro-batching, and (optional)
one official tau2 task end to end against a scripted model.

    PYTHONPATH=src python -m pytest tests/test_agentbench.py -q
    # end-to-end too (needs a Python 3.12 venv with tau2 installed from an LF checkout):
    XKV_TAU2_PY=/path/venv/bin/python XKV_TAU2_HOME=/path/tau2-bench PYTHONPATH=src python -m pytest tests -q
"""
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from xkv import agentbench as A

TOOLS = [{"type": "function", "function": {"name": "book", "description": "d", "parameters": {
    "type": "object", "properties": {"user_id": {"type": "string"}, "n": {"type": "integer"},
                                     "items": {"type": "array"}, "ok": {"type": "boolean"}, "note": {"type": "string"}}}}},
         {"type": "function", "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}}}]


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(os.environ.get("XKV_TOKENIZER", "Qwen/Qwen3.5-4B"))


def test_template_roundtrip(tok):
    call = {"user_id": "007", "n": 3, "items": [{"a": 1}, "x"], "ok": False, "note": "line1\nline2"}
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "sure", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "book", "arguments": json.dumps(call)}},
                {"id": "c2", "type": "function", "function": {"name": "ping", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "done"},
            {"role": "tool", "tool_call_id": "c2", "content": "pong"}]
    p = A.render_prompt(tok, msgs, TOOLS, thinking=False)
    assert p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    seg = p.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0]
    m = A.parse_completion(seg, TOOLS)
    assert m["content"] == "sure"
    got = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in m["tool_calls"]]
    assert got == [("book", call), ("ping", {})]
    assert "late" in A.render_prompt(tok, [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                                           {"role": "system", "content": "late"}], None, False)


def test_parse_variants():
    m = A.parse_completion("<think>\nplan\n</think>\n\nhello", TOOLS)
    assert m["content"] == "hello" and m["reasoning_content"] == "plan" and "tool_calls" not in m
    m = A.parse_completion('<tool_call>\n{"name": "ping", "arguments": {}}\n</tool_call>', TOOLS)
    assert m["tool_calls"][0]["function"]["name"] == "ping" and m["content"] is None
    m = A.parse_completion("x\n<tool_call>\n<function=book>\n<parameter=n>\n5\n</parameter>\n", TOOLS)  # truncated
    assert json.loads(m["tool_calls"][0]["function"]["arguments"]) == {"n": 5}
    assert A.strip_json_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_microbatching():
    calls = []

    class B:
        def chat_many(self, reqs):
            calls.append(len(reqs))
            time.sleep(0.2)
            return [{"role": "assistant", "content": f"echo {r.messages[-1]['content']}"} for r in reqs]

    srv = A.ChatServer({"xkv-agent": B(), "xkv-user": A.FnBackend(lambda m, t: {"content": "u"})},
                       {8131: "xkv-agent", 8132: "xkv-user"}, gather_ms=100).start()
    res = {}

    def post(port, model, text):
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": text}]}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req).read())["choices"][0]["message"]["content"]

    try:
        ts = [threading.Thread(target=lambda i=i: res.__setitem__(i, post(8131, "openai/xkv-agent", str(i))))
              for i in range(6)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert res == {i: f"echo {i}" for i in range(6)}
        assert sum(calls) == 6 and len(calls) <= 2
        assert post(8132, "anything", "x") == "u"  # unknown model name -> port default backend
    finally:
        srv.close()


def _scripted(tok, role):
    """Agent looks the user up, then refuses; user opens, then stops; judges answer in the expected schema."""
    end = tok.convert_tokens_to_ids("<|im_end|>")

    def gen(prompts, max_new):
        outs = []
        for ids in prompts:
            p = tok.decode(ids)
            last = p.rsplit("<|im_start|>", 2)[-2]
            if role == "agent":
                if "<tool_response>" in last:
                    t = "I'm sorry, but this reservation cannot be cancelled under our policy."
                else:
                    t = ("Let me look up your profile.\n\n<tool_call>\n<function=get_user_details>\n"
                         "<parameter=user_id>\nemma_kim_9957\n</parameter>\n</function>\n</tool_call>")
            elif role == "user":
                t = "###STOP###" if "cannot" in last else "Hi, please cancel reservation EHGLP3. I'm emma_kim_9957."
            else:
                t = ('```json\n{"results": []}\n```' if "metExpectation" in p
                     else '```json\n{"reasoning": "stub", "met": true}\n```')
            outs.append(tok.encode(t, add_special_tokens=False) + [end])
        return outs

    return gen


@pytest.mark.skipif(not os.environ.get("XKV_TAU2_PY"), reason="set XKV_TAU2_PY / XKV_TAU2_HOME for the tau2 run")
def test_tau2_airline_task0_end_to_end(tok, tmp_path):
    backends = {"xkv-agent": A.QwenBackend(tok, _scripted(tok, "agent"), "agent"),
                "xkv-user": A.QwenBackend(tok, _scripted(tok, "user"), "user"),
                "xkv-judge": A.QwenBackend(tok, _scripted(tok, "judge"), "judge", json_fences=True)}
    srv = A.ChatServer(backends, {8141: "xkv-agent", 8142: "xkv-user"}).start()
    try:
        r = A.run_tau2_domain("airline", srv, 8141, 8142, tmp_path / "airline", task_ids=["0"], max_concurrency=1,
                              python=os.environ["XKV_TAU2_PY"], tau2_home=Path(os.environ["XKV_TAU2_HOME"]),
                              timeout=900)
    finally:
        srv.close()
    assert r["returncode"] == 0, r["stderr_tail"]
    assert r["n_scored"] == 1 and r["pass1"] == 1.0, r  # the scripted agent behaves correctly on task 0
