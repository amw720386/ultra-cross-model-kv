"""Cross-tokenizer alignment for cross-family KV transfer (notes 015 / 017 / 018).

Contract (agreed with kvx-speed for the vLLM connector): the source runs on its own tokenization of the same text;
target token j takes the source token whose character span contains the LAST character of target token j. Its pre-RoPE
K/V (all selected source layers) is mapped to target position j and roped at j. Same rule as HeteroFold's end-boundary
alignment (arXiv 2609.32259) up to tie-breaking.
"""
import bisect

import torch


def align(src_offsets, tgt_offsets):
    """src_offsets / tgt_offsets: lists of (start, end) character spans (end exclusive), in order, over the same text.
    Returns LongTensor[n_tgt]: for each target token, the index of the source token containing its last character
    (end - 1). Tokens with empty spans (special tokens) or characters not covered by any source token (whitespace
    folded by a tokenizer) take the last source token that ends at or before that character (0 if none)."""
    s_start = [s for s, _ in src_offsets]
    out = []
    for (ts, te) in tgt_offsets:
        c = te - 1 if te > ts else ts  # last character of the target token
        i = bisect.bisect_right(s_start, c) - 1  # last source token starting at or before c
        while i > 0 and src_offsets[i][1] <= src_offsets[i][0]:  # skip empty-span source tokens
            i -= 1
        out.append(max(i, 0))  # a gap after source token i keeps i (the latest earlier boundary)
    return torch.tensor(out, dtype=torch.long)


def encode_pair(tok_s, tok_t, text, n_tgt, bos_s=None, bos_t=None):
    """Tokenize `text` with both tokenizers, cut to the first n_tgt target tokens (and the matching text prefix).
    bos_*: token id prepended (attention sink) on that side; the alignment index accounts for it, and the target BOS
    (if any) is aligned to the source BOS (or source token 0).
    Returns (src_ids list, tgt_ids list, idx LongTensor[len(tgt_ids)] into src_ids)."""
    et = tok_t(text, add_special_tokens=False, return_offsets_mapping=True)
    t_ids, t_off = et["input_ids"][:n_tgt], et["offset_mapping"][:n_tgt]
    cut = t_off[-1][1] if t_off else 0
    es = tok_s(text[:cut], add_special_tokens=False, return_offsets_mapping=True)
    s_ids, s_off = es["input_ids"], es["offset_mapping"]
    idx = align(s_off, t_off)
    if bos_s is not None:
        s_ids = [bos_s] + s_ids
        idx = idx + 1
    if bos_t is not None:
        t_ids = [bos_t] + t_ids
        idx = torch.cat([torch.zeros(1, dtype=torch.long), idx])
    return s_ids, t_ids, idx
