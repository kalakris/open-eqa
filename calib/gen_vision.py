"""Vision prediction sets (C: gpt-4o, D: Qwen), concurrent and resumable.

Same prompt split, frame sampling, resizing and settings as
openeqa/baselines/gpt4v.py. The answer is the raw model output, unparsed, as there.
Writes predictions/raw/<set>.jsonl and predictions/<set>.json.

  python calib/gen_vision.py --set C --endpoint openai --model gpt-4o-2024-08-06 --limit 5
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from calib.common import (
    BUDGET_USD, ROOT, JsonlWriter, chat, cost, make_client, now, read_jsonl, spent_so_far,
)
from openeqa.utils.openai_utils import prepare_openai_vision_messages
from openeqa.utils.prompt_utils import load_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--endpoint", choices=["openai", "qwen"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default=str(ROOT / "data" / "open-eqa-hm3d-sub.json"))
    ap.add_argument("--frames-directory", default=str(ROOT / "data" / "frames"))
    ap.add_argument("--num-frames", type=int, default=15)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("-j", "--workers", type=int, default=6)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    dataset = json.load(open(args.dataset))
    if args.limit:
        dataset = dataset[: args.limit]
    raw_path = ROOT / "predictions" / "raw" / f"{args.set}.jsonl"
    done = {r["question_id"] for r in read_jsonl(raw_path)}
    todo = [q for q in dataset if q["question_id"] not in done]
    print(f"set {args.set}: {len(dataset)} questions, {len(done)} done, {len(todo)} to go")

    prefix, suffix = load_prompt("gpt4v").split("User Query:")
    client = make_client(args.endpoint)
    out = JsonlWriter(raw_path)
    settings = {"seed": args.seed, "temperature": args.temperature, "max_tokens": args.max_tokens,
                "num_frames": args.num_frames, "image_size": args.image_size}
    spent = spent_so_far()

    def run(item):
        frames = sorted((Path(args.frames_directory) / item["episode_history"]).glob("*-rgb.png"))
        indices = np.round(np.linspace(0, len(frames) - 1, args.num_frames)).astype(int)
        paths = [str(frames[i]) for i in indices]
        messages = prepare_openai_vision_messages(
            prefix=prefix,
            suffix="User Query:" + suffix.format(question=item["question"]),
            image_paths=paths,
            image_size=args.image_size,
        )
        r = chat(client, model=args.model, messages=messages, seed=args.seed,
                 temperature=args.temperature, max_tokens=args.max_tokens)
        raw = r.choices[0].message.content
        return {
            "question_id": item["question_id"],
            "raw": raw,
            "answer": raw,
            "frames": [Path(p).name for p in paths],
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
            if spent > BUDGET_USD:
                tqdm.write(f"budget ${BUDGET_USD} reached, stopping")
                for g in futs:
                    g.cancel()
                break
    out.close()

    rows = {r["question_id"]: r for r in read_jsonl(raw_path)}
    preds = [{"question_id": q["question_id"], "answer": rows[q["question_id"]]["answer"]}
             for q in dataset if q["question_id"] in rows]
    json.dump(preds, (ROOT / "predictions" / f"{args.set}.json").open("w"), indent=2)
    per_q = [r["cost_usd"] for r in rows.values()]
    print(f"{len(preds)}/{len(dataset)} predictions; {failures} failed calls; "
          f"mean ${np.mean(per_q):.4f}/question, mean prompt tokens "
          f"{np.mean([r['prompt_tokens'] for r in rows.values()]):.0f}; total spend ${spent_so_far():.2f}")


if __name__ == "__main__":
    main()
