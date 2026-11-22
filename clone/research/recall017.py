"""Note 017 request set: synthetic box-color recall at ~512 / 2048 / 7500 context tokens (their llama-to-qwen-72b.json:
4 independent fact sets x 3 lengths = 12 cases, 3 repetitions per mode = 72 measured requests, 2 warmups excluded;
complete sequential HTTP request latency, warm models; answers graded exact after normalization).

Their fact format is unpublished; ours (labelled): N boxes "Box <id> is <color>." interleaved with filler sentences,
one question "What color is box <id>? Answer with one word." on a box placed in the middle third. Lengths are counted in
TARGET (Qwen2.5) tokens.  python research/recall017.py out.jsonl
"""
import json, random, sys

COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "brown", "pink", "gray", "teal"]
FILLER = ["The warehouse lights hummed quietly through the night.", "Inventory checks happen every Tuesday morning.",
          "Forklifts are parked along the north wall after each shift.", "Labels are printed on recycled paper stock.",
          "The loading dock faces the river and gets the morning sun.", "Visitors must sign in at the front desk.",
          "Pallets are stacked no higher than four levels.", "The break room coffee machine was replaced last spring."]


def make_case(seed, target_tokens, tok):
    rng = random.Random(seed)
    sents, facts = [], {}
    i = 0
    while True:
        bid = f"{rng.randrange(100, 999)}-{chr(65 + rng.randrange(26))}"
        if bid in facts:
            continue
        facts[bid] = rng.choice(COLORS)
        sents.append(f"Box {bid} is {facts[bid]}.")
        sents += rng.sample(FILLER, 2)
        text = " ".join(sents)
        if len(tok(text, add_special_tokens=False)["input_ids"]) >= target_tokens - 40:
            break
        i += 1
    keys = list(facts)
    q = keys[len(keys) // 3 + rng.randrange(max(1, len(keys) // 3))]
    return dict(seed=seed, nominal_tokens=target_tokens, context=text, question=f"What color is box {q}? Answer with one word.",
                answer=facts[q])


def grade(reply, answer):
    return int(reply.strip().lower().strip(".!\"' ").split()[:1] == [answer])


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct")
    with open(sys.argv[1], "w") as f:
        for fs in range(4):
            for n in (512, 2048, 7500):
                f.write(json.dumps(make_case(1000 * fs + n, n, tok)) + "\n")
