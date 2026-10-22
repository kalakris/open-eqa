"""J1r: J1 with unparseable rows re-asked at a larger completion budget.

With max_tokens 32 (J0's setting), Qwen sometimes starts a rationale and is cut off before
giving a mark, which leaves score null. Those rows skew toward long, hedging predictions, so
dropping them biases agreement. This re-asks only those rows with max_tokens 512 and takes the mark
from the reply ("Your mark: N", a bare digit, or the last "mark/score N" phrase).

Writes scores/<set>__J1r.jsonl: every J1 row, with re-asked rows replaced and flagged retried=true.
The original J1 file is untouched.

  python -m calib.judge_retry --sets C D A ...
"""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

from calib.common import DATASET, ROOT, chat, make_client, now, read_jsonl
from openeqa.evaluation.llm_match import parse_score
from openeqa.utils.prompt_utils import load_prompt

MARK = re.compile(r"(?:mark|score|rating)\W{0,3}(?:is|of)?\W{0,3}([1-5])\b", re.IGNORECASE)


FIRST_LINE = re.compile(r"^[\s*#_`]*(?:your mark\W*)?([1-5])(?:\s*/\s*5)?[\s*_`.]*$", re.IGNORECASE)


def parse_lenient(raw):
    raw = (raw or "").strip()
    try:
        return parse_score(raw)
    except ValueError:
        pass
    # a mark alone on the first line, possibly bold: "**4**", "4/5", "Your mark: 4"
    first = FIRST_LINE.match(raw.splitlines()[0]) if raw else None
    if first:
        return int(first.group(1))
    hits = MARK.findall(raw)
    if hits:
        return int(hits[-1])
    # a mark alone on the last line; a stray number inside prose ("2 speakers") is not a mark
    last = FIRST_LINE.match(raw.splitlines()[-1]) if raw else None
    return int(last.group(1)) if last else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--src-tag", default="J1")
    ap.add_argument("--tag", default="J1r")
    ap.add_argument("--model", default="qwen3.6-27b-fast")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("-j", "--workers", type=int, default=4)
    args = ap.parse_args()

    items = {q["question_id"]: q for q in json.load(open(DATASET))}
    prompts = {"mmbench": load_prompt("mmbench"), "mmbench-extra": load_prompt("mmbench-extra")}
    client = make_client("qwen")

    def retry(row):
        q = items[row["question_id"]]
        extra = q.get("extra_answers")
        content = prompts["mmbench" if extra is None else "mmbench-extra"].format(
            question=q["question"], answer=q["answer"], prediction=row["prediction"], extra_answers=extra)
        r = chat(client, model=args.model, messages=[{"role": "user", "content": content}],
                 seed=1234, temperature=0, max_tokens=args.max_tokens)
        raw = r.choices[0].message.content or ""
        score = parse_lenient(raw)
        return {**row, "judge": args.tag, "raw": raw, "score": score,
                "parse_error": None if score is not None else "no mark found after retry",
                "settings": {**row["settings"], "max_tokens": args.max_tokens}, "retried": True,
                "first_raw": row["raw"], "prompt_tokens": r.usage.prompt_tokens,
                "completion_tokens": r.usage.completion_tokens, "cost_usd": 0.0, "timestamp": now()}

    for s in args.sets:
        rows = read_jsonl(ROOT / "scores" / f"{s}__{args.src_tag}.jsonl")
        bad = [r for r in rows if r["score"] is None and r.get("raw") is not None]
        with ThreadPoolExecutor(args.workers) as pool:
            fixed = {r["question_id"]: r for r in pool.map(retry, bad)}
        out = [fixed.get(r["question_id"], {**r, "judge": args.tag}) for r in rows]
        with (ROOT / "scores" / f"{s}__{args.tag}.jsonl").open("w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        still = sum(1 for r in fixed.values() if r["score"] is None)
        print(f"{s}: {len(bad)} re-asked, {len(bad) - still} recovered, {still} still unscored")


if __name__ == "__main__":
    main()
