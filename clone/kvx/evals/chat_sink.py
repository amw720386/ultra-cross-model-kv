"""Exact attention-sink template for chat prompts: every chat-formatted prompt starts with the same tokens
("<|im_start|>user\\n" for Qwen), and attention is causal, so the target's KV at those positions is identical for
every request. One native forward of that prefix gives an EXACT template (vs the calibration-mean sink template),
used by kvx.vllm_connector's sink_template option for positions < A. Zero per-request cost.

python -m kvx.evals.chat_sink --model Qwen/Qwen3-32B --out /vol/runs/<id>/chat_sink.pt
Output: {"K": [Lt, H, A, D] pre-RoPE, "V": [Lt, H, A, D], "ids": prefix ids, "text": prefix} (connector format).
"""
import argparse

import torch


def template_prefix(tok):
    """Longest prefix shared by the chat template of two different user messages (token level)."""
    a = tok.apply_chat_template([{"role": "user", "content": "alpha"}], tokenize=False, add_generation_prompt=True)
    b = tok.apply_chat_template([{"role": "user", "content": "Zebra 42"}], tokenize=False, add_generation_prompt=True)
    ia, ib = tok(a, add_special_tokens=False)["input_ids"], tok(b, add_special_tokens=False)["input_ids"]
    n = 0
    while n < min(len(ia), len(ib)) and ia[n] == ib[n]:
        n += 1
    return ia[:n]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from kvx import core
    from kvx.models import load_lm
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = template_prefix(tok)
    if not ids:
        raise SystemExit("no constant chat prefix")
    m = load_lm(a.model).cuda().eval()
    x = torch.tensor([ids], device="cuda")
    cos, sin = core.rope_cos_sin(m, len(ids), "cuda")
    K, V, _ = core.capture_kv(m, x, cos, sin)  # [L, 1, H, A, D], K pre-RoPE
    torch.save(dict(K=K[:, 0].float().cpu(), V=V[:, 0].float().cpu(), ids=ids, text=tok.decode(ids)), a.out)
    print(dict(A=len(ids), text=tok.decode(ids), K=tuple(K.shape)))


if __name__ == "__main__":
    main()
