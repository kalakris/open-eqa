"""Vision predictions from Claude models, cached per episode.

Prompt, frame sampling and resizing match openeqa/baselines/gpt4v.py: text prefix, N frames,
"User Query: ..." suffix, in one user message. A cache_control breakpoint on the last frame
lets an episode's questions share the cached frames. Questions within an episode run
sequentially (the first writes the cache), and episodes run in parallel.

Thinking can't be disabled on these models, so it runs adaptive at a fixed effort. Temperature
is rejected by the API. Refusals are recorded as-is, with no fallback model, so every
answer comes from the named model. The key is read from CLAUDE_API_KEY in .env.

  python -m calib.gen_vision_claude --set G_opus --model claude-opus-5-5 --effort low \
      --dataset data/open-eqa-hm3d-sub.json --limit 5
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import numpy as np
from tqdm import tqdm

from calib.common import (
    ANTHROPIC_BUDGET_USD, ROOT, JsonlWriter, load_env, now, read_jsonl, spent_so_far,
)
from openeqa.utils.openai_utils import prepare_openai_vision_messages
from openeqa.utils.prompt_utils import load_prompt

# USD per 1M tokens: input, 5-minute cache write, cache read, output
PRICES = {
    "claude-opus-5-5": (4.0, 5.0, 0.20, 20.0),
    "claude-fable-5-1": (10.0, 12.5, 0.25, 50.0),
}


def build_content_files(prefix, suffix, question, keys, manifest):
    """Same layout as build_content, but frames are Files API references (see upload_frames_claude)."""
    content = [{"type": "text", "text": prefix}]
    content += [{"type": "image", "source": {"type": "file", "file_id": manifest[k]}} for k in keys]
    content[-1]["cache_control"] = {"type": "ephemeral"}
    content.append({"type": "text", "text": "User Query:" + suffix.format(question=question)})
    return content


def build_content(prefix, suffix, question, paths, image_size):
    msgs = prepare_openai_vision_messages(
        prefix=prefix, suffix="User Query:" + suffix.format(question=question),
        image_paths=paths, image_size=image_size)
    content = []
    for c in msgs[0]["content"]:
        if c["type"] == "text":
            content.append({"type": "text", "text": c["text"]})
        else:
            data = c["image_url"]["url"].split(",", 1)[1]
            content.append({"type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": data}})
    last_image = max(i for i, c in enumerate(content) if c["type"] == "image")
    content[last_image]["cache_control"] = {"type": "ephemeral"}
    return content


def frame_paths(frames_dir, episode, num_frames):
    frames = sorted((Path(frames_dir) / episode).glob("*-rgb.png"))
    if num_frames <= 0 or num_frames >= len(frames):
        return [str(f) for f in frames]
    idx = np.round(np.linspace(0, len(frames) - 1, num_frames)).astype(int)
    return [str(frames[i]) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--model", required=True, choices=sorted(PRICES))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--frames-directory", default=str(ROOT / "data" / "frames"))
    ap.add_argument("--num-frames", type=int, default=15, help="0 = every frame")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-tokens", type=int, default=8000, help="covers thinking + answer")
    ap.add_argument("-j", "--workers", type=int, default=6, help="episodes in parallel")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--files-manifest", help="use uploaded frames (calib/upload_frames_claude.py)")
    args = ap.parse_args()
    manifest = json.load(open(args.files_manifest)) if args.files_manifest else None

    load_env()
    client = anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"], timeout=600, max_retries=0)
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
    out = JsonlWriter(raw_path)
    pin, pwrite, pread, pout = PRICES[args.model]
    settings = {"effort": args.effort, "thinking": "adaptive (always on)", "max_tokens": args.max_tokens,
                "num_frames": args.num_frames, "image_size": args.image_size,
                "cache": "ephemeral 5m after last frame", "fallbacks": "none",
                "frames_via": "files api" if manifest else "base64"}
    state = {"spent": spent_so_far("anthropic"), "stop": False}

    def ask(content):
        for attempt in range(8):
            try:
                return client.messages.create(
                    model=args.model, max_tokens=args.max_tokens,
                    output_config={"effort": args.effort},
                    messages=[{"role": "user", "content": content}])
            except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
                if attempt == 7:
                    raise
                time.sleep(min(60, 2**attempt) * (0.5 + random.random()))
            except anthropic.APIStatusError as e:
                if e.status_code < 500 or attempt == 7:
                    raise
                time.sleep(min(60, 2**attempt) * (0.5 + random.random()))

    def run_episode(ep, items):
        paths = frame_paths(args.frames_directory, ep, args.num_frames)
        rows = []
        for item in items:
            if state["stop"]:
                break
            t = time.time()
            if manifest:
                keys = [f"{ep}/{Path(p).name}" for p in paths]
                content = build_content_files(prefix, suffix, item["question"], keys, manifest)
            else:
                content = build_content(prefix, suffix, item["question"], paths, args.image_size)
            r = ask(content)
            text = "".join(b.text for b in r.content if b.type == "text").strip()
            u = r.usage
            write = u.cache_creation_input_tokens or 0
            read = u.cache_read_input_tokens or 0
            cost = (u.input_tokens * pin + write * pwrite + read * pread + u.output_tokens * pout) / 1e6
            row = {
                "question_id": item["question_id"],
                "raw": text,
                "answer": text if r.stop_reason != "refusal" else None,
                "stop_reason": r.stop_reason,
                "stop_details": r.stop_details.model_dump() if getattr(r, "stop_details", None) else None,
                "frames": [Path(p).name for p in paths],
                "model": args.model,
                "response_model": r.model,
                "settings": settings,
                "input_tokens": u.input_tokens,
                "cache_write_tokens": write,
                "cache_read_tokens": read,
                "prompt_tokens": u.input_tokens + write + read,
                "completion_tokens": u.output_tokens,  # includes thinking
                "cost_usd": cost,
                "seconds": round(time.time() - t, 1),
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
            if state["spent"] > ANTHROPIC_BUDGET_USD:
                tqdm.write(f"anthropic budget ${ANTHROPIC_BUDGET_USD} reached, stopping")
                state["stop"] = True
    out.close()

    rows = {r["question_id"]: r for r in read_jsonl(raw_path)}
    preds = [{"question_id": q["question_id"], "answer": rows[q["question_id"]]["answer"]}
             for q in dataset if q["question_id"] in rows]
    json.dump(preds, (ROOT / "predictions" / f"{args.set}.json").open("w"), indent=2)
    vals = list(rows.values())
    print(f"{len(preds)}/{len(dataset)} predictions; {failures} failed episodes; "
          f"{sum(1 for r in vals if r['stop_reason'] == 'refusal')} refusals; "
          f"set cost ${sum(r['cost_usd'] for r in vals):.2f}; "
          f"mean output tokens {np.mean([r['completion_tokens'] for r in vals]):.0f}; "
          f"cache read share {sum(r['cache_read_tokens'] for r in vals) / max(sum(r['prompt_tokens'] for r in vals), 1):.0%}; "
          f"anthropic spend ${spent_so_far('anthropic'):.2f}")


if __name__ == "__main__":
    main()
