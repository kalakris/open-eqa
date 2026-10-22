"""Blind-LLM prediction sets (A: gpt-4-0613, B: Qwen), concurrent and resumable.

Same prompt, parsing and sampling settings as openeqa/baselines/gpt4.py, plus raw
text and usage per call. Writes predictions/raw/<set>.jsonl (every call) and
predictions/<set>.json ([{question_id, answer}], evaluate-predictions.py format).

  python calib/gen_blind.py --set A --endpoint openai --model gpt-4-0613
  python calib/gen_blind.py --set B --endpoint qwen --model qwen3.6-27b-fast --temperature 0 -j 4
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from calib.common import (
    BUDGET_USD, DATASET, ROOT, JsonlWriter, chat, cost, make_client, now, read_jsonl, spent_so_far,
)
from openeqa.baselines.gpt4 import parse_output
from openeqa.utils.prompt_utils import load_prompt


def first_line(raw: str):
    lines = [s.strip() for s in raw.splitlines() if s.strip()]
    return lines[0] if lines else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--endpoint", choices=["openai", "qwen"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("-j", "--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    args = ap.parse_args()

    dataset = json.load(open(args.dataset))
    if args.limit:
        dataset = dataset[: args.limit]
    raw_path = ROOT / "predictions" / "raw" / f"{args.set}.jsonl"
    done = {r["question_id"] for r in read_jsonl(raw_path)}
    todo = [q for q in dataset if q["question_id"] not in done]
    print(f"set {args.set}: {len(dataset)} questions, {len(done)} done, {len(todo)} to go")

    prompt = load_prompt("blind-llm")
    client = make_client(args.endpoint)
    out = JsonlWriter(raw_path)
    settings = {"seed": args.seed, "temperature": args.temperature, "max_tokens": args.max_tokens}
    spent = spent_so_far()

    def run(item):
        r = chat(
            client,
            model=args.model,
            messages=[{"role": "user", "content": prompt.format(question=item["question"])}],
            **settings,
        )
        raw = r.choices[0].message.content or ""
        try:
            answer, parse_error = parse_output(raw), None
        except ValueError as e:
            # gpt4.py with --force would store None (scored 0 unjudged). Qwen sometimes drops
            # the "A:" prefix, so fall back to the first line; parse_error flags these rows.
            answer, parse_error = first_line(raw), str(e)
        return {
            "question_id": item["question_id"],
            "raw": raw,
            "answer": answer,
            "parse_error": parse_error,
            "model": args.model,
            "response_model": r.model,
            "system_fingerprint": r.system_fingerprint,
            "settings": settings,
            "prompt_tokens": r.usage.prompt_tokens,
            "completion_tokens": r.usage.completion_tokens,
            "cost_usd": cost(args.model, r.usage),
            "timestamp": now(),
        }

    failures = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(run, q) for q in todo]
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

    rows = {r["question_id"]: r for r in read_jsonl(raw_path)}
    for r in rows.values():
        if r["parse_error"] and r["answer"] is None:  # rows written before the fallback existed
            r["answer"] = first_line(r["raw"])
    preds = [{"question_id": q["question_id"], "answer": rows[q["question_id"]]["answer"]}
             for q in dataset if q["question_id"] in rows]
    pred_path = ROOT / "predictions" / f"{args.set}.json"
    json.dump(preds, pred_path.open("w"), indent=2)
    n_err = sum(1 for r in rows.values() if r["parse_error"])
    print(f"{len(preds)}/{len(dataset)} predictions -> {pred_path} "
          f"({n_err} parse errors, {failures} failed calls, total spend ${spent_so_far():.2f})")


if __name__ == "__main__":
    main()
