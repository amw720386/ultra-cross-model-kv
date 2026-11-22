"""Calibrate kvx/evals/judge.py's QMSum judge against OneTriangle's native Qwen3.8-27B QMSum score (.4933, n=98).
Same judge (judge.judge_one: model, reasoning_effort, integer 0-10 / 10); only the rubric text varies. Variants are
scored on the SAME native-27B generations; the variant closest to .4933 is proposed as the frozen judge version, then
applied unchanged to native and mapped arms of the same item set.
python research/qmsum_calib.py GENS.jsonl DATA_ROOT OUT_DIR [variants]
"""
import json, math, os, sys
from concurrent.futures import ThreadPoolExecutor

from kvx.evals import judge, data

HEAD = ("You grade answers to queries about a meeting transcript. You see the query, a reference answer written by\n"
        "an annotator who read the transcript, and a candidate answer.\n")
TAIL = "\n\nQuery: {query}\n\nReference answer: {ref}\n\nCandidate answer: {cand}\n\nReply with the integer score only."
VARIANTS = {
    "v0": judge.PROMPT,  # kvx-quality's current rubric
    "v1strict": HEAD + (
        "Score the candidate from 0 to 10 by strict anchors:\n"
        "10 = states every key point of the reference, nothing contradicts it, no unsupported claims;\n"
        "8 = all main points, minor omissions;\n6 = most key points but at least one important omission or one unsupported claim;\n"
        "4 = about half of the key points;\n2 = few key points or several errors;\n0 = wrong, empty, or unrelated.\n"
        "Unsupported or invented details lower the score. Do not reward length or style.") + TAIL,
    "v2frac": HEAD + (
        "Silently list the distinct key points of the reference answer. For each, decide whether the candidate states it\n"
        "(1), states it partly (0.5) or not (0). Score = round(10 x covered / number of key points), then subtract 2 for\n"
        "each statement that contradicts the reference (minimum 0). Do not reward length or style.") + TAIL,
}


def run(gens, root, out_dir, names):
    queries = {it["id"]: it["question"].split("\n\n")[0].removeprefix("Query: ") for it in data.qmsum(root, 10 ** 6)}
    rows = [json.loads(l) for l in open(gens)]
    rows = [r for r in rows if r["bench"] == "qmsum"]
    os.makedirs(out_dir, exist_ok=True)
    summ = {}
    for name in names:
        judge.PROMPT = VARIANTS[name]
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(lambda r: judge.judge_one(r, queries[r["id"]], "gpt-5.5"), rows))
        with open(f"{out_dir}/{name}.jsonl", "w") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")
        ok = [r["judge"] for r in res if r["judge"] is not None]
        m = sum(ok) / len(ok)
        sd = math.sqrt(sum((x - m) ** 2 for x in ok) / max(1, len(ok) - 1))
        summ[name] = dict(n=len(res), scored=len(ok), mean=m, ci_half=1.96 * sd / math.sqrt(len(ok)),
                          cost_usd=sum(r["cost"] for r in res))
        print(name, json.dumps(summ[name]), flush=True)
    json.dump(dict(target=0.4933, judge_model="gpt-5.5", reasoning_effort="low", variants=summ, prompts=VARIANTS),
              open(f"{out_dir}/summary.json", "w"), indent=1)
    return summ


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], (sys.argv[4] if len(sys.argv) > 4 else "v1strict,v2frac").split(","))
