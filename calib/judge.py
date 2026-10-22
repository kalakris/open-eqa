"""LLM-Match judging with raw outputs saved, concurrent and resumable.

Same prompt, last-period stripping and settings as evaluate-predictions.py /
llm_match.py, but it records every raw judge reply. Writes scores/<set>__<tag>.jsonl.

  python calib/judge.py --set A --tag J0p --model gpt-4-turbo-2024-04-09
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from calib.common import (
    BUDGET_USD, DATASET, ROOT, JsonlWriter, chat, cost, make_client, now, read_jsonl, spent_so_far,
)
from openeqa.evaluation.llm_match import parse_score
from openeqa.utils.prompt_utils import load_prompt


def strip_after_last_period(answer):
    # evaluate-predictions.py: keep up to and including the last ".", only for truthy answers
    if answer:
        end_idx = answer.rfind(".")
        if end_idx >= 0 and end_idx + 1 < len(answer):
            answer = answer[: end_idx + 1]
    return answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--tag", required=True, help="judge label, e.g. J0p")
    ap.add_argument("--endpoint", choices=["openai", "qwen"], default="openai")
    ap.add_argument("--model", required=True)
    ap.add_argument("--predictions", help="default predictions/<set>.json")
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("-j", "--workers", type=int, default=12)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    items = {q["question_id"]: q for q in json.load(open(args.dataset))}
    preds = json.load(open(args.predictions or ROOT / "predictions" / f"{args.set}.json"))
    if args.limit:
        preds = preds[: args.limit]
    out_path = ROOT / "scores" / f"{args.set}__{args.tag}.jsonl"
    done = {r["question_id"] for r in read_jsonl(out_path)}
    todo = [p for p in preds if p["question_id"] not in done]
    print(f"{args.set} x {args.tag}: {len(preds)} predictions, {len(done)} done, {len(todo)} to go")

    prompts = {"mmbench": load_prompt("mmbench"), "mmbench-extra": load_prompt("mmbench-extra")}
    client = make_client(args.endpoint)
    out = JsonlWriter(out_path)
    settings = {"seed": args.seed, "temperature": args.temperature, "max_tokens": args.max_tokens}
    spent = spent_so_far()

    def run(pred):
        item = items[pred["question_id"]]
        extra = item.get("extra_answers")
        prediction = strip_after_last_period(pred["answer"])
        row = {
            "question_id": pred["question_id"],
            "prediction": prediction,
            "judge": args.tag,
            "model": args.model,
            "settings": settings,
            "timestamp": now(),
        }
        if prediction is None:
            # llm_match.get_llm_match_score returns 0 without calling the model
            return {**row, "raw": None, "score": 0, "parse_error": None, "cost_usd": 0.0}
        name = "mmbench" if extra is None else "mmbench-extra"
        content = prompts[name].format(
            question=item["question"], answer=item["answer"],
            prediction=prediction, extra_answers=extra,
        )
        r = chat(client, model=args.model, messages=[{"role": "user", "content": content}], **settings)
        raw = r.choices[0].message.content or ""
        try:
            score, parse_error = parse_score(raw), None
        except ValueError as e:
            score, parse_error = None, str(e)
        return {
            **row,
            "prompt": name,
            "raw": raw,
            "score": score,
            "parse_error": parse_error,
            "response_model": r.model,
            "system_fingerprint": r.system_fingerprint,
            "prompt_tokens": r.usage.prompt_tokens,
            "completion_tokens": r.usage.completion_tokens,
            "cost_usd": cost(args.model, r.usage),
        }

    failures = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(run, p) for p in todo]
        bar = tqdm(as_completed(futs), total=len(futs))
        for f in bar:
            try:
                row = f.result()
            except Exception as e:
                failures += 1
                tqdm.write(f"FAILED: {e!r}")
                continue
            out.write(row)
            spent += row["cost_usd"]
            bar.set_postfix(spent=f"${spent:.2f}")
            if spent > BUDGET_USD and args.endpoint != "qwen":  # Qwen on omen is free
                tqdm.write(f"budget ${BUDGET_USD} reached, stopping")
                for g in futs:
                    g.cancel()
                break
    out.close()

    rows = read_jsonl(out_path)
    scores = np.array([r["score"] for r in rows if r["score"] is not None], dtype=float)
    n_err = sum(1 for r in rows if r["parse_error"])
    if len(scores):
        agg = np.mean(100.0 * (np.clip(scores, 1, 5) - 1) / 4)
        hist = {k: int((scores == k).sum()) for k in range(1, 6)}
        print(f"{len(rows)}/{len(preds)} judged; LLM-Match {agg:.1f}; marks {hist}; "
              f"{n_err} parse errors; {failures} failed calls; total spend ${spent_so_far():.2f}")


if __name__ == "__main__":
    main()
