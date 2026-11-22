"""OUR LM judge for QMSum (query-focused meeting summarization, arXiv 2104.05938). OneTriangle scores QMSum with an
unnamed "canonical judge"; this is not theirs. Reference-based: the judge sees the query, the reference answer and
the candidate, and returns an integer 0-10 (coverage of the reference's key points + faithfulness, no length bonus).
Score = integer / 10. Same judge, prompt and settings for every arm, so arms compare; the absolute value does not
compare with OneTriangle's.

python -m kvx.evals.judge gens.jsonl judged.jsonl [--model gpt-5.5]   (needs OPENAI_API_KEY; litellm)
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

PROMPT = """You grade answers to queries about a meeting transcript. You see the query, a reference answer written by
an annotator who read the transcript, and a candidate answer. Score the candidate from 0 to 10:
10 = covers all key points of the reference and states nothing that contradicts it;
5 = covers about half of the key points, or mixes correct points with clear errors;
0 = wrong, empty, or unrelated.
Do not reward length or style. Judge only content against the reference.

Query: {query}

Reference answer: {ref}

Candidate answer: {cand}

Reply with the integer score only."""


def _strip_think(t):
    return re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()


def judge_one(row, query, model):
    import litellm
    cand = _strip_think(row["text"])[:6000]
    msg = PROMPT.format(query=query, ref=row["answer"], cand=cand or "(empty)")
    for _ in range(4):
        try:
            # gpt-5 family: no temperature control; reasoning tokens count against the completion budget
            r = litellm.completion(model=model, messages=[{"role": "user", "content": msg}],
                                   max_completion_tokens=2000, reasoning_effort="low")
            txt = r.choices[0].message.content or ""
            m = re.search(r"\d+", txt)
            if m:
                s = min(10, int(m.group()))
                return dict(id=row["id"], arm=row.get("arm"), judge=s / 10, raw=txt.strip(),
                            cost=litellm.completion_cost(r))
        except Exception as e:  # rate limits / transient errors: retry
            err = repr(e)
    return dict(id=row["id"], arm=row.get("arm"), judge=None, raw=None, cost=0.0, error=locals().get("err"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gens")
    ap.add_argument("out")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    from kvx.evals import data
    queries = {it["id"]: it["question"].split("\n\n")[0].removeprefix("Query: ") for it in data.qmsum(a.data_root, 10**6)}
    rows = [json.loads(l) for l in open(a.gens)]
    rows = [r for r in rows if r["bench"] == "qmsum"]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda r: judge_one(r, queries[r["id"]], a.model), rows))
    with open(a.out, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    ok = [r["judge"] for r in res if r["judge"] is not None]
    import math
    m = sum(ok) / len(ok) if ok else float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in ok) / max(1, len(ok) - 1)) if ok else float("nan")
    print(json.dumps(dict(n=len(res), scored=len(ok), mean=m, ci_half=1.96 * sd / math.sqrt(max(1, len(ok))),
                          cost_usd=sum(r["cost"] for r in res), judge=a.model)))


if __name__ == "__main__":
    main()
