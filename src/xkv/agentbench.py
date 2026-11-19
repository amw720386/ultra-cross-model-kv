"""Agentic benchmarks (tau2-bench text, Terminal-Bench 2.0) against our own generators.

An in-process OpenAI-compatible chat-completions server sits in front of pluggable chat backends.
The benchmark harnesses (official packages, run as subprocesses in their own Python 3.12 venvs) talk to it
through litellm with a custom api_base, so every LLM call they make lands on our models:

    agent endpoint  (port A, model "xkv-agent")  -> strategy under test (target / transfer / source)
    user endpoint   (port B, model "xkv-user")   -> native target, non-thinking, greedy (user simulator)
    judge           (port B, model "xkv-judge")  -> native target, non-thinking, greedy (tau2 LLM judges)

Concurrent requests are micro-batched: one GPU thread drains the request queue, groups requests by backend and
runs each group through a single batched `gen` call. A backend is any object with
`chat_many(list[ChatRequest]) -> list[dict]`; `FnBackend` wraps a plain `chat_fn(messages, tools) -> message`.

Model-side tasks (run on Modal through modal_jobs/agent_lab.py): `task_tau2`, `task_terminal_bench`.

tau2 protocol: tau2-bench 2.0 (pinned commit, see modal_jobs/agent_common.py), released `text-v1` benchmark,
English cells, frozen 50-task subsets per domain (airline_50 / retail_50 / telecom_50), 1 trial, seed 300, official
prompts, tools, max steps/errors and scoring (pass^1 = mean task reward; overall = equal-weight mean of domains).
Deviations from the official setting:
  - user simulator: native target 27B, non-thinking, greedy (official: openai/gpt-5.5, reasoning_effort low);
  - communicate judge (airline tasks with communicate_info) and NL-assertion judge (retail tasks with
    nl_assertions): native target 27B (official: gpt-5.5 / gpt-5.4-mini). The NL judge model is a hard-coded
    constant in tau2, overridden through a sitecustomize hook; ```json fences are stripped from judge replies;
  - agent: greedy, non-thinking by default (agent_thinking=True enables Qwen thinking), max 2048 new tokens/turn;
  - max_concurrency 8 instead of 3 (scheduling only; batch composition can perturb bf16 greedy decoding).
local_user_and_judges=False restores the official user simulator and judges when an OpenAI key is available.
"""
from __future__ import annotations

import ast
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# --------------------------------------------------------------------------------------------------------------
# OpenAI <-> Qwen3.5 chat template
# --------------------------------------------------------------------------------------------------------------


def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI content parts
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def _args_dict(a):
    if isinstance(a, dict):
        return a
    if not a:
        return {}
    try:
        v = json.loads(a)
        return v if isinstance(v, dict) else {"value": v}
    except (TypeError, ValueError):
        return {"value": a}


def to_template_messages(messages: list[dict]) -> list[dict]:
    """OpenAI chat messages -> messages the Qwen3.5 chat template accepts.

    - assistant tool_call arguments (JSON strings) become dicts (the template renders <parameter=...> blocks);
    - content parts are flattened to text, tool results to strings;
    - a system message that is not the first message becomes a user turn (the template rejects it otherwise);
    - reasoning_content on assistant turns is kept (the template only renders it after the last user query).
    """
    out = []
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "developer":
            role = "system"
        if role == "system":
            if i == 0:
                out.append({"role": "system", "content": _text(m.get("content"))})
            else:
                out.append({"role": "user", "content": _text(m.get("content"))})
        elif role == "user":
            out.append({"role": "user", "content": _text(m.get("content"))})
        elif role == "assistant":
            d = {"role": "assistant", "content": _text(m.get("content"))}
            if m.get("reasoning_content"):
                d["reasoning_content"] = m["reasoning_content"]
            tcs = m.get("tool_calls") or []
            if tcs:
                d["tool_calls"] = [{"type": "function", "id": tc.get("id"),
                                    "function": {"name": (tc.get("function") or tc).get("name"),
                                                 "arguments": _args_dict((tc.get("function") or tc).get("arguments"))}}
                                   for tc in tcs]
            out.append(d)
        elif role in ("tool", "function"):
            out.append({"role": "tool", "content": _text(m.get("content"))})
        else:
            raise ValueError(f"unsupported role {role!r}")
    if not any(m["role"] == "user" and not m["content"].lstrip().startswith("<tool_response>") for m in out):
        # The template needs at least one real user query; add an empty-turn marker rather than fail.
        out.insert(1 if out and out[0]["role"] == "system" else 0, {"role": "user", "content": "(start)"})
    return out


def render_prompt(tok, messages: list[dict], tools: list[dict] | None, thinking: bool) -> str:
    return tok.apply_chat_template(to_template_messages(messages), tools=tools or None, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=thinking)


_TOOL_BLOCK = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
_FUNC = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.S)
_PARAM = re.compile(r"<parameter=([^>\n]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", re.S)


def _param_types(tools) -> dict:
    types = {}
    for t in tools or []:
        f = t.get("function", t)
        props = ((f.get("parameters") or {}).get("properties") or {})
        types[f.get("name")] = {k: (v or {}).get("type") for k, v in props.items()}
    return types


def _convert(raw: str, typ):
    """Parameter text -> JSON value, guided by the schema type (same rules as vLLM's qwen3_coder parser)."""
    if raw.startswith("\n"):
        raw = raw[1:]
    if raw.endswith("\n"):
        raw = raw[:-1]
    if typ == "string" or (isinstance(typ, list) and "string" in typ and len(typ) == 1):
        return raw
    s = raw.strip()
    if s.lower() in ("null", "none") and typ != "string":
        return None
    for f in (json.loads, ast.literal_eval):
        try:
            return f(s)
        except Exception:  # noqa: BLE001
            pass
    if typ in ("boolean",) and s.lower() in ("true", "false"):
        return s.lower() == "true"
    return raw


def parse_completion(text: str, tools: list[dict] | None = None) -> dict:
    """Generated text -> OpenAI assistant message (content, reasoning_content, tool_calls)."""
    reasoning = None
    if "</think>" in text:
        reasoning, text = text.split("</think>", 1)
        reasoning = reasoning.replace("<think>", "").strip()
    text = text.strip()
    types = _param_types(tools)
    calls = []
    start = text.find("<tool_call>")
    if start < 0 and tools and re.search(r"<function=[^>\n]+>", text):  # tool call without the wrapper
        start = text.find("<function=")
        blocks = [text[start:]]
    else:
        blocks = [b for b in _TOOL_BLOCK.findall(text)] if start >= 0 else []
    for b in blocks:
        fm = _FUNC.search(b)
        if fm:
            name = fm.group(1).strip()
            ptypes = types.get(name, {})
            args = {pm.group(1).strip(): _convert(pm.group(2), ptypes.get(pm.group(1).strip()))
                    for pm in _PARAM.finditer(fm.group(2))}
        else:  # Qwen2.5/Qwen3 style JSON payload
            try:
                obj = json.loads(b.strip())
                name, args = obj.get("name"), obj.get("arguments", obj.get("parameters", {}))
                args = _args_dict(args)
            except (ValueError, AttributeError):
                continue
        if not name:
            continue
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    content = text[:start].strip() if start >= 0 else text
    msg = {"role": "assistant", "content": content or (None if calls else "")}
    if reasoning:
        msg["reasoning_content"] = reasoning
    if calls:
        msg["tool_calls"] = calls
    return msg


def strip_json_fences(s: str | None) -> str | None:
    """```json {...}``` -> {...} when the payload parses (judges call json.loads on the raw content)."""
    if not s:
        return s
    m = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", s, re.S)
    if m:
        try:
            json.loads(m.group(1))
            return m.group(1)
        except ValueError:
            pass
    return s


# --------------------------------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------------------------------


class ContextTooLong(Exception):
    pass


@dataclass
class ChatRequest:
    messages: list
    tools: list | None = None
    max_tokens: int | None = None
    stop: list | None = None
    model: str = ""
    future: "queue.Queue" = field(default_factory=lambda: queue.Queue(maxsize=1))
    t0: float = field(default_factory=time.time)


class FnBackend:
    """Wrap a plain chat_fn(messages, tools) -> assistant message dict (requests run one at a time)."""

    def __init__(self, chat_fn, name="fn"):
        self.chat_fn, self.name = chat_fn, name

    def chat_many(self, reqs):
        out = []
        for r in reqs:
            m = dict(self.chat_fn(r.messages, r.tools))
            m.setdefault("role", "assistant")
            out.append(m)
        return out


class QwenBackend:
    """Qwen3.5 chat template + batched token generator `gen(prompts, max_new) -> list[list[int]]`."""

    def __init__(self, tok, gen, name, thinking=False, max_new=2048, max_prompt=131072, json_fences=False,
                 log=None, native_tools=False, native_system=False, native_history=False, req_log=None,
                 drop_mapped=None):
        from .genrun import stop_ids

        self.drop_mapped = drop_mapped  # (k, cap): ablation, see _drop_mapped (policy and older history deleted)

        self.req_log = req_log  # optional JSONL path: every request (messages, tools) and its reply is appended

        self.tok, self.gen, self.name, self.thinking = tok, gen, name, thinking
        self.native_tools = native_tools
        self.native_system = native_system
        self.native_history = native_history
        self.max_new, self.max_prompt, self.json_fences = max_new, max_prompt, json_fences
        self.stops = stop_ids(tok)
        self.log = log or (lambda s: print(s, flush=True))
        self.stats = {"calls": 0, "requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    def encode(self, r: ChatRequest) -> list[int]:
        return self.encode_spans(r)[0]

    def encode_spans(self, r: ChatRequest):
        """Token ids plus the token span of the <tools>...</tools> block (computed natively when native_tools)."""
        msgs = _drop_mapped(self.tok, r.messages, r.tools, *self.drop_mapped) if self.drop_mapped else r.messages
        text = render_prompt(self.tok, msgs, r.tools, self.thinking)
        if self.native_history:
            # session serving: the first request (system + first user turn) is transferred; later turns are appended
            # natively to the kept cache, so everything after the first user turn is computed by the target
            starts = [i for i in range(len(text)) if text.startswith("<|im_start|>", i)]
            if len(starts) >= 3:
                enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
                ids, offs = enc["input_ids"], enc["offset_mapping"]
                a = next(i for i, (x, y) in enumerate(offs) if x >= starts[2])
                return ids, [(a, len(ids))]
            return self.tok.encode(text, add_special_tokens=False), []
        if self.native_system and text.startswith("<|im_start|>system"):
            # the system turn (policy + tool schemas) is identical for every request of a domain: computed
            # natively once and served from a prefix cache in deployment; here it is a native span
            enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            c1 = text.index("<|im_end|>") + len("<|im_end|>")
            b = next((i for i, (x, y) in enumerate(offs) if x >= c1), len(ids))
            return ids, [(0, b)]
        if not self.native_tools or "<tools>" not in text:
            return self.tok.encode(text, add_special_tokens=False), []
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        c0, c1 = text.index("<tools>"), text.index("</tools>") + len("</tools>")
        a = next(i for i, (x, y) in enumerate(offs) if y > c0)
        b = next((i for i, (x, y) in enumerate(offs) if x >= c1), len(ids))
        return ids, [(a, b)]

    def chat_many(self, reqs):
        prompts, keep, out, spans = [], [], [None] * len(reqs), []
        for i, r in enumerate(reqs):
            try:
                ids, sp = self.encode_spans(r)
            except Exception as e:  # noqa: BLE001 - template errors become per-request 400s
                out[i] = e
                continue
            if len(ids) > self.max_prompt:
                out[i] = ContextTooLong(f"This model's maximum context length is {self.max_prompt} tokens. "
                                        f"However, your messages resulted in {len(ids)} tokens.")
                continue
            prompts.append(ids)
            spans.append(sp)
            keep.append(i)
        if prompts:
            new = [min(self.max_new, reqs[i].max_tokens or self.max_new) for i in keep]
            t = time.time()
            toks = (self.gen(prompts, max(new), spans=spans) if (self.native_tools or self.native_system
                                                                 or self.native_history)
                    else self.gen(prompts, max(new)))
            dt = time.time() - t
            for j, i in enumerate(keep):
                ids = list(toks[j])[:new[j]]
                cut = next((k for k, x in enumerate(ids) if x in self.stops), None)
                finish = "stop" if cut is not None else "length"
                ids = ids[:cut] if cut is not None else ids
                text = self.tok.decode(ids, skip_special_tokens=False)
                if self.thinking:
                    text = "<think>\n" + text
                for s in reqs[i].stop or []:
                    if s and s in text:
                        text, finish = text.split(s, 1)[0], "stop"
                msg = parse_completion(text, reqs[i].tools)
                if self.json_fences and not msg.get("tool_calls"):
                    msg["content"] = strip_json_fences(msg["content"])
                msg["_usage"] = {"prompt_tokens": len(prompts[j]), "completion_tokens": len(ids),
                                 "total_tokens": len(prompts[j]) + len(ids)}
                msg["_finish"] = "tool_calls" if msg.get("tool_calls") else finish
                out[i] = msg
                if self.req_log:
                    with open(self.req_log, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"messages": reqs[i].messages, "tools": reqs[i].tools,
                                            "max_tokens": reqs[i].max_tokens, "out": {
                                                k: v for k, v in msg.items() if not k.startswith("_")}},
                                           default=str) + "\n")
                self.stats["prompt_tokens"] += len(prompts[j])
                self.stats["completion_tokens"] += len(ids)
            self.stats["calls"] += 1
            self.stats["requests"] += len(keep)
            self.stats["seconds"] += dt
            self.log(f"[srv] {self.name} batch={len(keep)} prompt={min(map(len, prompts))}-{max(map(len, prompts))} "
                     f"new<={max(len(t) for t in toks)} {dt:.1f}s")
        return out


# --------------------------------------------------------------------------------------------------------------
# Server: one GPU thread, micro-batching, several ports
# --------------------------------------------------------------------------------------------------------------


class ChatServer:
    """OpenAI-compatible /v1/chat/completions on one or more ports, backed by named backends.

    ports: {port: default backend name}. The request's `model` (after an optional "provider/" prefix) selects a
    backend when it names one; otherwise the port's default is used.
    """

    def __init__(self, backends: dict, ports: dict, host="127.0.0.1", max_batch=16, gather_ms=40):
        self.backends, self.ports, self.host = backends, ports, host
        self.max_batch, self.gather = max_batch, gather_ms / 1000
        self.q: queue.Queue = queue.Queue()
        self.pending: list[ChatRequest] = []
        self.httpds, self.threads = [], []
        self.stop_ev = threading.Event()

    # ---- GPU loop -------------------------------------------------------------------------------------------
    def _loop(self):
        while not self.stop_ev.is_set():
            try:
                self.pending.append(self.q.get(timeout=0.2))
            except queue.Empty:
                if not self.pending:
                    continue
            t_end = time.time() + self.gather  # let concurrent requests arrive
            while time.time() < t_end:
                try:
                    self.pending.append(self.q.get(timeout=max(0.0, t_end - time.time())))
                except queue.Empty:
                    break
            if not self.pending:
                continue
            name = min(self.pending, key=lambda r: r.t0).model  # oldest request's backend goes first
            group = [r for r in self.pending if r.model == name][: self.max_batch]
            self.pending = [r for r in self.pending if r not in group]
            try:
                outs = self.backends[name].chat_many(group)
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                outs = [e] * len(group)
            for r, o in zip(group, outs):
                r.future.put(o)

    def submit(self, req: ChatRequest, timeout=3600):
        self.q.put(req)
        return req.future.get(timeout=timeout)

    def resolve(self, model: str, port: int) -> str:
        m = (model or "").split("/")[-1]
        return m if m in self.backends else self.ports[port]

    # ---- HTTP -----------------------------------------------------------------------------------------------
    def _handler(self, port):
        srv = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):  # quiet
                pass

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path.rstrip("/").endswith("/models"):
                    self._send(200, {"object": "list", "data": [{"id": k, "object": "model", "owned_by": "xkv"}
                                                                for k in srv.backends]})
                else:
                    self._send(200, {"ok": True})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except ValueError:
                    return self._send(400, {"error": {"message": "invalid json", "type": "invalid_request_error"}})
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    return self._send(404, {"error": {"message": f"unknown path {self.path}"}})
                if body.get("stream"):
                    return self._send(400, {"error": {"message": "stream=true is not supported",
                                                      "type": "invalid_request_error"}})
                stop = body.get("stop")
                req = ChatRequest(messages=body.get("messages") or [], tools=body.get("tools") or None,
                                  max_tokens=body.get("max_completion_tokens") or body.get("max_tokens"),
                                  stop=[stop] if isinstance(stop, str) else stop,
                                  model=srv.resolve(body.get("model"), port))
                if body.get("tool_choice") == "none":
                    req.tools = None
                out = srv.submit(req)
                if isinstance(out, ContextTooLong):
                    return self._send(400, {"error": {"message": str(out), "type": "invalid_request_error",
                                                      "code": "context_length_exceeded"}})
                if isinstance(out, Exception):
                    return self._send(500, {"error": {"message": repr(out)[:2000], "type": "server_error"}})
                usage = out.pop("_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
                finish = out.pop("_finish", "tool_calls" if out.get("tool_calls") else "stop")
                self._send(200, {"id": f"chatcmpl-{uuid.uuid4().hex[:20]}", "object": "chat.completion",
                                 "created": int(time.time()), "model": body.get("model") or req.model,
                                 "choices": [{"index": 0, "message": out, "finish_reason": finish}],
                                 "usage": usage})

        return H

    def start(self):
        t = threading.Thread(target=self._loop, daemon=True, name="xkv-gpu")
        t.start()
        self.threads.append(t)
        for port in self.ports:
            httpd = ThreadingHTTPServer((self.host, port), self._handler(port))
            httpd.daemon_threads = True
            th = threading.Thread(target=httpd.serve_forever, daemon=True, name=f"xkv-http-{port}")
            th.start()
            self.httpds.append(httpd)
            self.threads.append(th)
        return self

    def url(self, port):
        return f"http://{self.host}:{port}/v1"

    def close(self):
        self.stop_ev.set()
        for h in self.httpds:
            h.shutdown()
            h.server_close()


# --------------------------------------------------------------------------------------------------------------
# tau2-bench (text-v1, English)
# --------------------------------------------------------------------------------------------------------------

TAU2_HOME = Path(os.environ.get("XKV_TAU2_HOME", "/opt/tau2-bench"))  # git checkout (data/ lives here)
TAU2_PY = os.environ.get("XKV_TAU2_PY", "/opt/tau2env/bin/python")  # Python 3.12 venv with tau2 installed

# Run inside the tau2 process: point the hard-coded NL-assertion judge (and the unused env-interface model)
# at our judge endpoint, then hand over to the official CLI. Worker subprocesses re-import this through
# sitecustomize (PYTHONPATH), so the override also holds there.
_TAU2_SITECUSTOMIZE = r'''
import json, os
_o = os.environ.get("XKV_TAU2_OVERRIDES")
if _o:
    import sys
    import tau2.config as _c  # importing tau2 also imports modules that copied the constants already
    _o = json.loads(_o)
    for _k, _v in _o.items():
        setattr(_c, _k, _v)
    for _n, _m in list(sys.modules.items()):
        if _n.startswith("tau2") and _m is not None:
            for _k, _v in _o.items():
                if hasattr(_m, _k):
                    setattr(_m, _k, _v)
'''


def tau2_overrides(judge_base: str, judge_model="openai/xkv-judge") -> dict:
    args = {"temperature": 0.0, "api_base": judge_base, "api_key": "xkv"}
    return {"DEFAULT_LLM_NL_ASSERTIONS": judge_model, "DEFAULT_LLM_NL_ASSERTIONS_ARGS": args,
            "DEFAULT_LLM_ENV_INTERFACE": judge_model, "DEFAULT_LLM_ENV_INTERFACE_ARGS": args,
            "DEFAULT_SEMANTIC_COMMUNICATE_MODEL": judge_model,
            "DEFAULT_SEMANTIC_COMMUNICATE_MODEL_ARGS": {"api_base": judge_base, "api_key": "xkv"}}


def tau2_command(domain, agent_base, user_base, judge_base, save_to, num_tasks=None, task_ids=None,
                 max_concurrency=8, python=None, extra=(), local_user_and_judges=True):
    """Official `tau2 run --benchmark text --language en` with the agent (and by default the user simulator and
    both LLM judges) routed to our endpoints. local_user_and_judges=False keeps the official user simulator and
    judges (OpenAI models; needs OPENAI_API_KEY in the environment)."""
    cmd = [python or TAU2_PY, "-m", "tau2.cli", "run", "--benchmark", "text", "--language", "en",
           "--domain", domain,
           "--agent-llm", "openai/xkv-agent",
           "--agent-llm-args", json.dumps({"temperature": 0.0, "api_base": agent_base, "api_key": "xkv"}),
           "--max-concurrency", str(max_concurrency), "--save-to", str(save_to)]
    if local_user_and_judges:
        cmd += ["--user-llm", "openai/xkv-user",
                "--user-llm-args", json.dumps({"temperature": 0.0, "api_base": user_base, "api_key": "xkv"}),
                "--communicate-judge", json.dumps({"model": "openai/xkv-judge",
                                                   "model_args": {"api_base": judge_base, "api_key": "xkv"}})]
    if task_ids:
        cmd += ["--task-ids", *map(str, task_ids)]
    elif num_tasks:
        cmd += ["--num-tasks", str(num_tasks)]
    return cmd + list(extra)


def tau2_env(judge_base, tau2_home=None, workdir=None, local_user_and_judges=True):
    workdir = Path(workdir or "/tmp/xkv_tau2_site")
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "sitecustomize.py").write_text(_TAU2_SITECUSTOMIZE)
    env = dict(os.environ)
    env.update({"TAU2_DATA_DIR": str(Path(tau2_home or TAU2_HOME) / "data"),
                "PYTHONPATH": str(workdir), "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
                "LITELLM_LOG": "ERROR"})
    if local_user_and_judges:
        env.update({"XKV_TAU2_OVERRIDES": json.dumps(tau2_overrides(judge_base)), "OPENAI_API_KEY": "xkv"})
    elif not env.get("OPENAI_API_KEY"):
        raise RuntimeError("local_user_and_judges=False needs OPENAI_API_KEY (attach it as a Modal secret)")
    return env


def _main_tau2_cell(save_to: Path, domain: str) -> Path | None:
    for p in (save_to / f"en-{domain}" / "results.json", save_to / "results.json"):
        if p.exists():
            return p
    hits = sorted(save_to.rglob("results.json"))
    return hits[0] if hits else None


def read_tau2_results(save_to: Path, domain: str) -> dict:
    """pass^1 over scored simulations; infrastructure errors are excluded like the official scorer does."""
    p = _main_tau2_cell(Path(save_to), domain)
    if p is None:
        return {"error": f"no results.json under {save_to}"}
    d = json.loads(p.read_text(encoding="utf-8"))
    sims = d.get("simulations") or []
    rewards, errors, per_task, breakdown = [], 0, {}, {}
    for s in sims:
        ri = s.get("reward_info") or {}
        r = ri.get("reward")
        if r is None or s.get("termination_reason") == "infrastructure_error":
            errors += 1
            continue
        rewards.append(float(r))
        per_task.setdefault(str(s.get("task_id")), []).append(float(r))
        for k, v in (ri.get("reward_breakdown") or {}).items():
            breakdown.setdefault(k, []).append(float(v))
    term = {}
    for s in sims:
        term[str(s.get("termination_reason"))] = term.get(str(s.get("termination_reason")), 0) + 1
    task_means = [sum(v) / len(v) for v in per_task.values()]
    return {"pass1": sum(task_means) / len(task_means) if task_means else None, "n_scored": len(rewards),
            "n_tasks": len(per_task), "n_errored": errors, "termination": term,
            "breakdown": {k: sum(v) / len(v) for k, v in breakdown.items()},
            "per_task": {k: sum(v) / len(v) for k, v in per_task.items()}, "results_json": str(p)}


def run_tau2_domain(domain, server: ChatServer, agent_port, user_port, save_to, num_tasks=None, task_ids=None,
                    max_concurrency=8, python=None, tau2_home=None, timeout=6 * 3600, log=print,
                    local_user_and_judges=True):
    judge = server.url(user_port)
    cmd = tau2_command(domain, server.url(agent_port), server.url(user_port), judge, save_to, num_tasks=num_tasks,
                       task_ids=task_ids, max_concurrency=max_concurrency, python=python,
                       local_user_and_judges=local_user_and_judges)
    env = tau2_env(judge, tau2_home, Path(save_to).parent / "_site", local_user_and_judges)
    log(f"[tau2] {domain}: {' '.join(c if len(c) < 120 else c[:117] + '...' for c in cmd)}")
    t = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=timeout, cwd=str(tau2_home or TAU2_HOME))
    out = read_tau2_results(Path(save_to), domain)
    out.update({"returncode": p.returncode, "seconds": time.time() - t,
                "stdout_tail": p.stdout[-3000:], "stderr_tail": "\n".join(
                    l for l in p.stderr.splitlines() if "DEBUG" not in l)[-3000:]})
    return out


def overall(res: dict, domains) -> float | None:
    v = [res[d]["pass1"] for d in domains if isinstance(res.get(d), dict) and res[d].get("pass1") is not None]
    return sum(v) / len(v) if len(v) == len(domains) else None


def task_tau2(src="4b", tgt="27b", mapper_name="4b-27b-w8192-distill-ret", strategies=("target", "transfer"),
              domains=("airline", "retail", "telecom"), n_per_domain=None, task_ids=None, tail=128, window=8192,
              max_concurrency=8, agent_thinking=False, max_new=2048, max_prompt=131072, run_name=None,
              agent_port=8101, user_port=8102, local_user_and_judges=True, full_until=None,
              turn_tail_max=None, native_tools=False, native_system=False, native_history=False, turn_tail_msgs=1,
              log_requests=False):
    """tau2-bench text-v1 (English) pass^1 per domain + overall (equal domain weights) for each agent strategy.

    By default the user simulator and the LLM judges run on the native target (non-thinking, greedy);
    local_user_and_judges=False uses the official OpenAI user simulator/judges instead (needs OPENAI_API_KEY).
    n_per_domain=None runs the full frozen 50-task subset per domain; otherwise the first n tasks of each subset.
    """
    from .experiments import ART, get, load_mapper
    from .genrun import make_batch_gen

    sm, _ = get(src) if any(s.startswith(("transfer", "source")) for s in strategies) else (None, None)
    tm, tok = get(tgt)
    mapper = load_mapper(mapper_name, sm, tm) if any(s.startswith("transfer") for s in strategies) else None
    run_name = run_name or time.strftime("%Y%m%d-%H%M%S")
    root = ART / "tau2" / run_name
    gen_kw = dict(tail=tail, window=window, batch=max_concurrency, full_until=full_until)  # one call per task
    agent_kw = dict(gen_kw, turn_tail_max=turn_tail_max, turn_tail_msgs=turn_tail_msgs)
    native = make_batch_gen("target", sm, tm, mapper, tok, **gen_kw)
    res = {"config": {"strategies": list(strategies), "domains": list(domains), "n_per_domain": n_per_domain,
                      "mapper": mapper_name, "tail": tail, "window": window, "agent_thinking": agent_thinking,
                      "max_concurrency": max_concurrency, "max_new": max_new, "turn_tail_max": turn_tail_max, "turn_tail_msgs": turn_tail_msgs, "native_tools": native_tools, "native_system": native_system, "native_history": native_history, "root": str(root),
                      "benchmark": "tau2 text-v1 en", "user_sim": f"{tgt} native, non-thinking, greedy" if local_user_and_judges
                      else "official (OpenAI)"}}
    for strat in strategies:
        # "target-drop": ablation, native target on only the part the agent setting computes natively
        # (tools + last turn_tail_msgs messages up to turn_tail_max tokens); the policy and older history are deleted
        nat = strat in ("target", "target-drop")
        agent_gen = native if nat else make_batch_gen(strat, sm, tm, mapper, tok, **agent_kw)
        backends = {
            "xkv-agent": QwenBackend(tok, agent_gen, f"agent:{strat}", thinking=agent_thinking, max_new=max_new,
                                     native_tools=native_tools and not nat,
                                     native_system=native_system and not nat,
                                     native_history=native_history and not nat,
                                     max_prompt=max_prompt,
                                     drop_mapped=(turn_tail_msgs, turn_tail_max or 8192) if strat == "target-drop"
                                     else None),
            "xkv-user": QwenBackend(tok, native, "user", thinking=False, max_new=1024, max_prompt=max_prompt),
            "xkv-judge": QwenBackend(tok, native, "judge", thinking=False, max_new=2048, max_prompt=max_prompt,
                                     json_fences=True),
        }
        server = ChatServer(backends, {agent_port: "xkv-agent", user_port: "xkv-user"},
                            max_batch=max_concurrency).start()
        res[strat] = {}
        try:
            for d in domains:
                if log_requests:  # agent requests -> <root>/<strat>/<domain>_requests.jsonl (for task_tau2_replay)
                    (root / strat).mkdir(parents=True, exist_ok=True)
                    backends["xkv-agent"].req_log = str(root / strat / f"{d}_requests.jsonl")
                r = run_tau2_domain(d, server, agent_port, user_port, root / strat / d, num_tasks=n_per_domain,
                                    task_ids=task_ids, max_concurrency=max_concurrency,
                                    local_user_and_judges=local_user_and_judges)
                res[strat][d] = r
                short = {k: r.get(k) for k in ("pass1", "n_scored", "n_errored", "returncode", "seconds")}
                print(f"[tau2] {strat} {d} {json.dumps(short)}", flush=True)
                (ART / "results").mkdir(parents=True, exist_ok=True)
                (ART / "results" / f"tau2_partial_{run_name}.json").write_text(json.dumps(res, default=str))
        finally:
            server.close()
        res[strat]["overall"] = overall(res[strat], domains)
        res[strat]["server_stats"] = {k: b.stats for k, b in backends.items()}
        print(f"[tau2] {strat} overall={res[strat]['overall']}", flush=True)
    return res


# --------------------------------------------------------------------------------------------------------------
# Terminal-Bench 2.0 through Harbor (Terminus-2 agent on our endpoint, task sandboxes on Modal)
# --------------------------------------------------------------------------------------------------------------

HARBOR_BIN = os.environ.get("XKV_HARBOR_BIN", "/opt/harborenv/bin/harbor")


def harbor_command(agent_base, jobs_dir, job_name, n_tasks=None, task_names=(), concurrency=4, max_prompt=131072,
                   max_new=8192, temperature=None, environment="modal", attempts=1, harbor=None, extra=(),
                   sandbox_app="xkv-harbor-tb2"):
    model_info = {"max_input_tokens": max_prompt, "max_output_tokens": max_new, "input_cost_per_token": 0.0,
                  "output_cost_per_token": 0.0}
    cmd = [harbor or HARBOR_BIN, "run", "-d", "terminal-bench@2.0", "-a", "terminus-2", "-m", "openai/xkv-agent",
           "--base-url", agent_base, "--ak", f"model_info={json.dumps(model_info)}",
           "-e", environment, "-n", str(concurrency), "-k", str(attempts), "-o", str(jobs_dir),
           "--job-name", job_name, "--ek", f"app_name={sandbox_app}", "-y"]
    if temperature is not None:
        cmd += ["--ak", f"temperature={temperature}"]
    if n_tasks:
        cmd += ["-l", str(n_tasks)]
    for t in task_names:
        cmd += ["-i", t]
    return cmd + list(extra)


def read_harbor_result(jobs_dir, job_name) -> dict:
    p = Path(jobs_dir) / job_name / "result.json"
    if not p.exists():
        return {"error": f"missing {p}"}
    d = json.loads(p.read_text(encoding="utf-8"))
    stats = d.get("stats") or {}
    evals = stats.get("evals") or {}
    out = {"result_json": str(p), "n_errored_trials": stats.get("n_errored_trials"),
           "n_trials": stats.get("n_trials")}
    for k, v in evals.items():
        ms = v.get("metrics") or []
        out["pass_rate"] = ms[0].get("mean") if ms else None
        out["eval_key"] = k
        out["eval"] = v
    return out


def task_terminal_bench(src="4b", tgt="27b", mapper_name="4b-27b-w8192-distill-ret",
                        strategies=("transfer",), n_tasks=10, task_names=(), concurrency=4, tail=128, window=8192,
                        agent_thinking=False, max_new=8192, max_prompt=131072, temperature=None, run_name=None,
                        agent_port=8201, attempts=1, turn_tail_max=None, turn_tail_msgs=1, full_until=None,
                        agent_timeout_multiplier=None):
    """Terminal-Bench 2.0 (Harbor, Terminus-2 agent, sandboxes on Modal) pass rate per strategy.

    Needs Modal credentials inside the container (MODAL_TOKEN_ID / MODAL_TOKEN_SECRET via a Modal Secret) so
    Harbor can create the task sandboxes. The model endpoint stays on localhost: Terminus-2 calls the LLM from
    the Harbor host process and only drives tmux inside the sandbox.
    """
    from .experiments import ART, get, load_mapper
    from .genrun import make_batch_gen

    sm, _ = get(src) if any(s.startswith(("transfer", "source")) for s in strategies) else (None, None)
    tm, tok = get(tgt)
    mapper = load_mapper(mapper_name, sm, tm) if any(s.startswith("transfer") for s in strategies) else None
    run_name = run_name or time.strftime("%Y%m%d-%H%M%S")
    jobs = ART / "terminal_bench" / run_name
    res = {"config": {"strategies": list(strategies), "n_tasks": n_tasks, "concurrency": concurrency,
                      "agent_thinking": agent_thinking, "max_prompt": max_prompt, "max_new": max_new,
                      "temperature": temperature, "jobs_dir": str(jobs), "mapper": mapper_name,
                      "turn_tail_max": turn_tail_max, "turn_tail_msgs": turn_tail_msgs, "full_until": full_until,
                      "agent_timeout_multiplier": agent_timeout_multiplier}}
    extra = ["--agent-timeout-multiplier", str(agent_timeout_multiplier)] if agent_timeout_multiplier else []
    for strat in strategies:
        gen = make_batch_gen(strat, sm, tm, mapper, tok, tail=tail, window=window, batch=max(8, concurrency),
                             full_until=full_until, turn_tail_max=turn_tail_max, turn_tail_msgs=turn_tail_msgs)
        backends = {"xkv-agent": QwenBackend(tok, gen, f"agent:{strat}", thinking=agent_thinking, max_new=max_new,
                                             max_prompt=max_prompt)}
        server = ChatServer(backends, {agent_port: "xkv-agent"}, max_batch=max(8, concurrency)).start()
        cmd = harbor_command(server.url(agent_port), jobs, strat.replace("@", "_"), n_tasks=n_tasks,
                             task_names=task_names, concurrency=concurrency, max_prompt=max_prompt, max_new=max_new,
                             temperature=temperature, attempts=attempts, extra=extra)
        env = dict(os.environ, OPENAI_API_KEY="xkv", PYTHONIOENCODING="utf-8")
        print("[tb2] " + " ".join(cmd), flush=True)
        t = time.time()
        try:
            p = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=12 * 3600)
        finally:
            server.close()
        r = read_harbor_result(jobs, strat.replace("@", "_"))
        r.update({"returncode": p.returncode, "seconds": time.time() - t, "stdout_tail": p.stdout[-3000:],
                  "stderr_tail": p.stderr[-3000:], "server_stats": backends["xkv-agent"].stats})
        res[strat] = r
        print(f"[tb2] {strat} pass={r.get('pass_rate')} errored={r.get('n_errored_trials')}", flush=True)
    return res


def _action(msg) -> tuple:
    """(kind, tool name, canonical args) of an assistant reply; kind is 'tool' or 'text'."""
    tc = (msg or {}).get("tool_calls") or []
    if not tc:
        return ("text", None, None)
    f = tc[0].get("function", tc[0])
    a = f.get("arguments")
    try:
        a = json.dumps(json.loads(a) if isinstance(a, str) else a, sort_keys=True)
    except (TypeError, ValueError):
        a = str(a)
    return ("tool", f.get("name"), a)


def _drop_mapped(tok, messages, tools, k=10, cap=8192):
    """Ablation: only what the agent setting computes natively, with the mapped part deleted instead of mapped.
    The system turn (policy) is dropped (the tool schemas still render, since tools are passed separately), and
    only the last k non-system messages are kept, oldest dropped first until they fit in cap tokens."""
    msgs = [m for m in messages if m.get("role") != "system"][-k:]
    try:
        base = len(tok.encode(render_prompt(tok, [], tools, False), add_special_tokens=False)) if tools else 0
    except Exception:  # noqa: BLE001 - template may refuse an empty conversation
        base = 0
    while msgs and (msgs[0].get("role") == "tool" or (len(msgs) > 1 and len(tok.encode(
            render_prompt(tok, msgs, tools, False), add_special_tokens=False)) - base > cap)):
        msgs = msgs[1:]
    return msgs


def _native_share(be, reqs, turn_tail_max, turn_tail_msgs, repair):
    """Token accounting of a transfer variant: share of prompt tokens the target computes natively (tools span +
    turn tail), and an upper bound including the repair budget (repair re-encodes up to that many tokens)."""
    from .mapper import adaptive_tail

    im = be.tok.convert_tokens_to_ids("<|im_start|>")
    tot = nat = nat_rep = 0
    per, per_rep = [], []
    for r in reqs:
        ids, sp = be.encode_spans(r)
        n = len(ids)
        base = adaptive_tail(n)
        starts = [i for i, t in enumerate(ids) if t == im]
        tail = base
        if turn_tail_max and len(starts) >= 2:
            k = min(turn_tail_msgs + 1, len(starts) - 1)
            tail = max(base, min(turn_tail_max, n - starts[-k]))
        tail = min(tail, n)
        tools = sum(max(0, min(b, n - tail) - a) for a, b in sp)
        x = tail + tools
        y = min(n, x + repair)
        tot, nat, nat_rep = tot + n, nat + x, nat_rep + y
        per.append(x / n)
        per_rep.append(y / n)
    m = max(1, len(per))
    return {"n": len(per), "mean_prompt_tokens": round(tot / m), "native_token_share": round(nat / max(1, tot), 4),
            "native_share_mean": round(sum(per) / m, 4),
            "native_token_share_with_repair_upper": round(nat_rep / max(1, tot), 4),
            "native_share_with_repair_upper_mean": round(sum(per_rep) / m, 4)}


def task_tau2_replay(src="4b", tgt="27b", mapper_name="4b-27b-fid+4b-27b-fid-attn", run_name="replay",
                     domains=("telecom", "retail"), record=True, variants=None, max_prompts=200, seed=0,
                     max_new=160, full_until=32768, batch=8, record_kw=None, ttft_n=0):
    """Teacher-forced agent replay: every agent request of a native tau2 run (recorded with log_requests) is
    answered again by each variant; the score is how often the variant's next action matches the native one
    (same tool, and same tool + arguments). A cheap proxy for tau2 that needs no user simulator.

    variants: [{"name", "strategy", "native_tools", "native_system", "turn_tail_max", "turn_tail_msgs", "env",
    "drop_mapped"}]. drop_mapped: native target on only the part the agent setting computes natively (see
    _drop_mapped; uses turn_tail_msgs / turn_tail_max). ttft_n > 0: also time n requests one at a time with one
    output token (batch-1 time to first token incl. source prefill, mapping and repair).
    record=True first runs the native agent (local 27B user simulator) with request logging if no log exists.
    """
    import random

    import torch

    from .experiments import ART, get, load_mapper
    from .genrun import make_batch_gen

    root = ART / "tau2" / run_name
    logs = {d: root / "target" / f"{d}_requests.jsonl" for d in domains}
    res = {"config": {"mapper": mapper_name, "run_name": run_name, "max_prompts": max_prompts, "seed": seed,
                      "max_new": max_new, "full_until": full_until}}
    missing = [d for d, p in logs.items() if not p.exists()]
    if missing and record:
        kw = dict(dict(max_concurrency=8, full_until=full_until), **(record_kw or {}))
        r = task_tau2(src, tgt, mapper_name, strategies=("target",), domains=tuple(missing), run_name=run_name,
                      log_requests=True, **kw)
        res["record"] = {d: {k: v for k, v in (r["target"].get(d) or {}).items()
                             if k in ("pass1", "n_scored", "termination")} for d in missing}
    sm, _ = get(src)
    tm, tok = get(tgt)
    mappers = {mapper_name: load_mapper(mapper_name, sm, tm)}
    variants = variants or [{"name": "native", "strategy": "target"}]
    for d in domains:
        rows = [json.loads(l) for l in logs[d].read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:
            prior = [m for m in r["messages"] if m.get("role") == "assistant" and m.get("tool_calls")]
            r["first_tool"] = not prior and _action(r["out"])[0] == "tool"
        firsts = [r for r in rows if r["first_tool"]]
        rest = [r for r in rows if not r["first_tool"]]
        random.Random(seed).shuffle(rest)
        pick = firsts + rest[:max(0, max_prompts - len(firsts))]
        res[d] = {"n_requests": len(rows), "n_picked": len(pick), "n_first_tool": len(firsts)}
        ref = [_action(r["out"]) for r in pick]
        for v in variants:
            old = {k: os.environ.get(k) for k in (v.get("env") or {})}
            os.environ.update({k: str(x) for k, x in (v.get("env") or {}).items()})
            try:
                strat = v.get("strategy", "target")
                mn = v.get("mapper", mapper_name)  # per-variant map (e.g. old vs retrained)
                if strat != "target" and mn not in mappers:
                    mappers[mn] = load_mapper(mn, sm, tm)
                mapper = mappers.get(mn)
                g = make_batch_gen(strat, sm, tm, mapper if strat != "target" else None, tok, batch=batch,
                                   full_until=full_until, turn_tail_max=v.get("turn_tail_max"),
                                   turn_tail_msgs=v.get("turn_tail_msgs", 1))
                be = QwenBackend(tok, g, v["name"], max_new=max_new,
                                 native_tools=bool(v.get("native_tools")) and strat != "target",
                                 native_system=bool(v.get("native_system")) and strat != "target")
                if v.get("drop_mapped"):
                    reqs = [ChatRequest(messages=_drop_mapped(tok, r["messages"], r["tools"],
                                                              v.get("turn_tail_msgs", 10), v.get("turn_tail_max") or 8192),
                                        tools=r["tools"]) for r in pick]
                else:
                    reqs = [ChatRequest(messages=r["messages"], tools=r["tools"]) for r in pick]
                t0 = time.time()
                outs = []
                for k in range(0, len(reqs), batch):
                    outs += be.chat_many(reqs[k:k + batch])
                acts = [_action(o if isinstance(o, dict) else {}) for o in outs]
                extra = {}
                if strat.startswith("transfer"):
                    rb = int(strat.split("+rep")[1].rstrip("zm")) if "+rep" in strat else 0
                    extra["native_share"] = _native_share(be, reqs, v.get("turn_tail_max"),
                                                          v.get("turn_tail_msgs", 1), rb)
                if ttft_n:
                    g1 = make_batch_gen(strat, sm, tm, mapper if strat != "target" else None, tok, batch=1,
                                        full_until=full_until, turn_tail_max=v.get("turn_tail_max"),
                                        turn_tail_msgs=v.get("turn_tail_msgs", 1))
                    be1 = QwenBackend(tok, g1, v["name"] + ":ttft", max_new=1,
                                      native_tools=bool(v.get("native_tools")) and strat != "target",
                                      native_system=bool(v.get("native_system")) and strat != "target")
                    be1.chat_many([reqs[0]])  # warm-up
                    lat, plen = [], []
                    for r in reqs[1:ttft_n + 1]:
                        torch.cuda.synchronize()
                        t1 = time.time()
                        be1.chat_many([r])
                        torch.cuda.synchronize()
                        lat.append(time.time() - t1)
                        plen.append(len(be1.encode(r)))
                    lat_s = sorted(lat)
                    extra["ttft"] = {"n": len(lat), "median_s": round(lat_s[len(lat_s) // 2], 4),
                                     "mean_s": round(sum(lat) / max(1, len(lat)), 4),
                                     "mean_prompt_tokens": round(sum(plen) / max(1, len(plen)))}
            finally:
                for k, x in old.items():
                    if x is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = x

            def rate(sel, full):
                idx = [i for i in range(len(pick)) if sel(i)]
                if not idx:
                    return None
                f = (lambda a, b: a == b) if full else (lambda a, b: a[:2] == b[:2])
                return round(sum(f(acts[i], ref[i]) for i in idx) / len(idx), 4)

            def first(i):
                return pick[i]["first_tool"]

            def tool(i):
                return ref[i][0] == "tool"

            out = {"tool_match": rate(lambda i: True, False), "full_match": rate(lambda i: True, True),
                   "first_tool_match": rate(first, False), "first_full_match": rate(first, True),
                   "tool_turn_match": rate(tool, False), "tool_turn_full_match": rate(tool, True),
                   "seconds": round(time.time() - t0, 1), "spec": v, "first_tool_counts": {}, "samples": [],
                   **extra}
            names = [{t.get("function", t).get("name") for t in (r["tools"] or [])} for r in pick]
            out["invalid_tool_rate"] = round(sum(a[0] == "tool" and a[1] not in names[i]
                                                 for i, a in enumerate(acts)) / max(1, len(acts)), 4)
            out["action_counts"] = {}
            for a in acts:
                out["action_counts"][a[1] or "text"] = out["action_counts"].get(a[1] or "text", 0) + 1
            for i in range(len(pick)):
                if first(i):
                    n = acts[i][1] or "text"
                    out["first_tool_counts"][n] = out["first_tool_counts"].get(n, 0) + 1
                if acts[i] != ref[i] and len(out["samples"]) < 12:
                    out["samples"].append({"ref": ref[i], "got": acts[i]})
            res[d][v["name"]] = out
            print(f"[replay] {d} {v['name']} " + json.dumps({k: out[k] for k in (
                "tool_match", "full_match", "first_tool_match", "tool_turn_match", "invalid_tool_rate", "seconds")}), flush=True)
            (ART / "results").mkdir(parents=True, exist_ok=True)
            (ART / "results" / f"replay_partial_{run_name}.json").write_text(json.dumps(res, default=str))
    return res


TASKS = {"tau2": task_tau2, "terminal_bench": task_terminal_bench, "tau2_replay": task_tau2_replay}


if __name__ == "__main__":  # tiny manual check: python -m xkv.agentbench '<text>'
    print(json.dumps(parse_completion(sys.argv[1] if len(sys.argv) > 1 else ""), indent=1))
