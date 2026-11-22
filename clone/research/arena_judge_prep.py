"""Build the pairwise-judging packet for note 020 (their 36 annotated Arena prompts, plus all 100).

For each prompt: reference = THEIR native Qwen3-8B reply (from their CSV); candidates = OUR mapped reply (arena run) and
THEIR KV-Lingo reply. One LLM judge labels each candidate vs the reference better / same / worse, in a
randomized A/B order, so ours and theirs are judged by the same rubric (their own labels were hand annotations).
python research/arena_judge_prep.py ARENA_ITEMS_JSON THEIR_CSV THEIR_INTERESTING_CSV OUT_DIR
"""
import csv, json, os, random, sys

RUBRIC = ("You compare two assistant replies to the same user prompt. Judge helpfulness, correctness, faithfulness to "
          "the prompt's exact details (names, numbers, strings, requested format and language) and completeness. Ignore "
          "length unless it hurts. Answer for each pair with one line of JSON: {\"pair\": <id>, \"winner\": \"A\"|\"B\"|\"tie\", "
          "\"why\": \"<10 words>\"}. Use tie only when the replies are equally good.")


def main(items_path, csv_path, csv36_path, out_dir, arm="mapped"):
    csv.field_size_limit(10 ** 9)
    ours = json.load(open(items_path))
    by_id = dict(zip(ours["ids"], ours[arm]["texts"]))
    rows = {r["id"]: r for r in csv.DictReader(open(csv_path))}
    ids36 = [r["id"] for r in csv.DictReader(open(csv36_path))]
    rng = random.Random(0)
    os.makedirs(out_dir, exist_ok=True)
    pairs, key = [], []
    for subset, ids in (("annot36", ids36), ("all100", [i for i in rows if i not in set(ids36)])):
        for pid in ids:
            r = rows[pid]
            for cand_name, cand in (("ours_" + arm, by_id.get(pid)), ("theirs_KVLingo", r["KVLingo"])):
                if cand is None:
                    continue
                flip = rng.random() < 0.5
                a, b = (cand, r["8B"]) if flip else (r["8B"], cand)
                n = len(pairs)
                pairs.append(dict(pair=n, prompt=r["prompt"], A=a, B=b))
                key.append(dict(pair=n, id=pid, subset=subset, cand=cand_name, cand_is="A" if flip else "B"))
    with open(os.path.join(out_dir, "pairs.jsonl"), "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    json.dump(key, open(os.path.join(out_dir, "key.json"), "w"))
    open(os.path.join(out_dir, "RUBRIC.txt"), "w").write(RUBRIC)
    print(len(pairs), "pairs ->", out_dir)


def score(key_path, verdict_lines, out_path):
    """verdict_lines: JSON lines {pair, winner}. Writes better/same/worse counts per (subset, cand) vs native 8B."""
    key = {k["pair"]: k for k in json.load(open(key_path))}
    tally = {}
    for line in verdict_lines:
        v = json.loads(line)
        k = key[int(v["pair"])]
        w = v["winner"]
        res = "same" if w == "tie" else ("better" if w == k["cand_is"] else "worse")
        t = tally.setdefault(f"{k['subset']}|{k['cand']}", dict(better=0, same=0, worse=0))
        t[res] += 1
    json.dump(tally, open(out_path, "w"), indent=1)
    return tally


if __name__ == "__main__":
    main(*sys.argv[1:5], **({"arm": sys.argv[5]} if len(sys.argv) > 5 else {}))
