# CPU test of the cross-tokenizer alignment on real tokenizers (Llama-3.1 / TinyLlama / Mistral vs Qwen3).
import sys
sys.path.insert(0, '.')
from transformers import AutoTokenizer
from research.xalign import align, encode_pair
text = "The quick brown fox jumps over the lazy dog. Prices rose 12.5% in 2024 -- naïve café, 東京, emoji 🙂 end."
tq = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
for name in ("unsloth/Llama-3.1-8B-Instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "mistralai/Ministral-3-3B-Instruct-2512-BF16"):
    try:
        ts = AutoTokenizer.from_pretrained(name)
    except Exception as e:
        print(name, "SKIP", repr(e)[:100]); continue
    s_ids, t_ids, idx = encode_pair(ts, tq, text, 10 ** 6)
    es = ts(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    et = tq(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    bad = 0
    for j, i in enumerate(idx.tolist()):
        c = et[j][1] - 1
        s0, s1 = es[i]
        ok = s0 <= c < s1 or (s1 <= c)  # contains, or latest earlier boundary
        bad += not ok
        assert i == 0 or es[i][0] <= c, (j, i)
    assert idx.tolist() == sorted(idx.tolist()), "alignment must be monotone"
    print(name, "src", len(s_ids), "tgt", len(t_ids), "violations", bad)
    for j in (0, 5, len(t_ids) - 1):
        i = idx[j].item()
        print("   tgt", repr(tq.decode([t_ids[j]])), "-> src", repr(ts.decode([s_ids[i]])))
