"""Vision predictions for GPT-5.6+ models via the Responses API, with prompt caching per episode.

Prompt, frame sampling and resizing match openeqa/baselines/gpt4v.py. An explicit cache
breakpoint sits on the last frame, so an episode's questions share the prefix up to and
including the frames. Questions within an episode run sequentially (the first one writes
the cache), and episodes run in parallel. temperature/seed aren't accepted by these models.
See calib/probe_cache.py for the measurements behind this.

  python -m calib.gen_vision_cached --set F --model gpt-6-astra --dataset data/open-eqa-hm3d.json
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from calib.common import BUDGET_USD, ROOT, JsonlWriter, make_client, now, read_jsonl, spent_so_far
from calib.common import _retryable
from openeqa.utils.openai_utils import prepare_openai_vision_messages
from openeqa.utils.prompt_utils import load_prompt

# USD per 1M tokens, standard tier
PRICES = {"gpt-6-astra": {"input": 10.0, "cached": 1.0, "write_30m": 12.5, "output": 50.0}}


def build_input(prefix, suffix, question, paths, image_size):
    msgs = prepare_openai_vision_messages(
        prefix=prefix, suffix="User Query:" + suffix.format(question=question),
        image_paths=paths, image_size=image_size)
    content = []
    for c in msgs[0]["content"]:
        if c["type"] == "text":
            content.append({"type": "input_text", "text": c["text"]})
        else:
            content.append({"type": "input_image", "image_url": c["image_url"]["url"]})
    last_image = max(i for i, c in enumerate(content) if c["type"] == "input_image")
    content[last_image]["prompt_cache_breakpoint"] = {"mode": "explicit"}
    return [{"role": "user", "content": content}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--frames-directory", default=str(ROOT / "data" / "frames"))
    ap.add_argument("--num-frames", type=int, default=15)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--max-output-tokens", type=int, default=4000)
    ap.add_argument("--effort", help="reasoning.effort; omit for the model default (medium on gpt-6-astra)")
    ap.add_argument("--mode", help="reasoning.mode: standard (default) or pro")
    ap.add_argument("-j", "--workers", type=int, default=8, help="episodes in parallel")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    dataset = json.load(open(args.dataset))
    if args.limit:
        dataset = dataset[: args.limit]
    raw_path = ROOT / "predictions" / "raw" / f"{args.set}.jsonl"
    done = {r["question_id"] for r in read_jsonl(raw_path)}
    by_ep = defaultdict(list)
    for q in dataset:
        if q["question_id"] not in done:
            by_ep[q["episode_history"]].append(q)
    print(f"set {args.set}: {len(dataset)} questions, {len(done)} done, "
          f"{sum(map(len, by_ep.values()))} to go over {len(by_ep)} episodes")

    prefix, suffix = load_prompt("gpt4v").split("User Query:")
    client = make_client("openai")
    out = JsonlWriter(raw_path)
    p = PRICES[args.model]
    settings = {"max_output_tokens": args.max_output_tokens, "num_frames": args.num_frames,
                "image_size": args.image_size, "cache_breakpoint": "after last frame, ttl 30m"}
    reasoning = {k: v for k, v in (("effort", args.effort), ("mode", args.mode)) if v}
    extra = {"reasoning": reasoning} if reasoning else {}
    settings["reasoning"] = reasoning or "model default"
    state = {"spent": spent_so_far(), "stop": False}

    def ask(item, paths):
        for attempt in range(8):
            try:
                return client.responses.create(
                    model=args.model,
                    input=build_input(prefix, suffix, item["question"], paths, args.image_size),
                    max_output_tokens=args.max_output_tokens, **extra)
            except Exception as e:
                if not _retryable(e) or attempt == 7:
                    raise
                import random, time
                time.sleep(min(60, 2**attempt) * (0.5 + random.random()))

    def run_episode(ep, items):
        frames = sorted((Path(args.frames_directory) / ep).glob("*-rgb.png"))
        idx = np.round(np.linspace(0, len(frames) - 1, args.num_frames)).astype(int)
        paths = [str(frames[i]) for i in idx]
        rows = []
        for item in items:
            if state["stop"]:
                break
            r = ask(item, paths)
            d = r.usage.input_tokens_details
            cached = d.cached_tokens or 0
            write = getattr(d, "cache_write_tokens", 0) or 0
            uncached = r.usage.input_tokens - cached - write
            cost = (uncached * p["input"] + cached * p["cached"] + write * p["write_30m"]
                    + r.usage.output_tokens * p["output"]) / 1e6
            row = {
                "question_id": item["question_id"],
                "raw": r.output_text,
                "answer": r.output_text,
                "frames": [Path(x).name for x in paths],
                "model": args.model,
                "response_model": r.model,
                "api": "responses",
                "settings": settings,
                "prompt_tokens": r.usage.input_tokens,
                "cached_tokens": cached,
                "cache_write_tokens": write,
                "completion_tokens": r.usage.output_tokens,
                "reasoning_tokens": r.usage.output_tokens_details.reasoning_tokens,
                "cost_usd": cost,
                "timestamp": now(),
            }
            out.write(row)
            rows.append(row)
        return rows

    failures = 0
    bar = tqdm(total=sum(map(len, by_ep.values())))
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(run_episode, ep, items): ep for ep, items in by_ep.items()}
        for f in as_completed(futs):
            try:
                rows = f.result()
            except Exception as e:
                failures += 1
                tqdm.write(f"FAILED episode {futs[f]}: {e!r}")
                continue
            state["spent"] += sum(r["cost_usd"] for r in rows)
            bar.update(len(rows))
            bar.set_postfix(spent=f"${state['spent']:.2f}")
            if state["spent"] > BUDGET_USD:
                tqdm.write(f"budget ${BUDGET_USD} reached, stopping")
                state["stop"] = True
    out.close()

    rows = {r["question_id"]: r for r in read_jsonl(raw_path)}
    preds = [{"question_id": q["question_id"], "answer": rows[q["question_id"]]["answer"]}
             for q in dataset if q["question_id"] in rows]
    json.dump(preds, (ROOT / "predictions" / f"{args.set}.json").open("w"), indent=2)
    tot_in = sum(r["prompt_tokens"] for r in rows.values())
    tot_cached = sum(r["cached_tokens"] for r in rows.values())
    print(f"{len(preds)}/{len(dataset)} predictions; {failures} failed episodes; "
          f"set cost ${sum(r['cost_usd'] for r in rows.values()):.2f}; "
          f"cache hit {tot_cached / max(tot_in, 1):.0%} of input tokens; total spend ${spent_so_far():.2f}")


if __name__ == "__main__":
    main()
