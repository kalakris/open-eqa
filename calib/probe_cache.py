"""Check prompt caching for vision runs: 3 questions from one episode, sent back to back.

GPT-5.6+ models default to an implicit cache breakpoint at the end of the last message,
which in gpt4v.py's single-message prompt falls after the per-question text, so nothing
would be reused. This puts an explicit breakpoint on the last frame instead.

Tries Chat Completions first, then the Responses API, and logs the full usage object.

  python -m calib.probe_cache --model gpt-6-astra --mode chat
"""

import argparse
import json
import time

import numpy as np

from calib.common import ROOT, JsonlWriter, make_client, now
from openeqa.utils.openai_utils import prepare_openai_vision_messages
from openeqa.utils.prompt_utils import load_prompt

EPISODE = "hm3d-v0/014-hm3d-rsggHU7g7dh"  # 14 questions, the most of any HM3D episode
PRICES = {"gpt-6-astra": {"input": 10.0, "cached": 1.0, "write_30m": 12.5, "output": 50.0}}
BREAKPOINT = {"mode": "explicit"}


def chat_messages(question, paths):
    prefix, suffix = load_prompt("gpt4v").split("User Query:")
    msgs = prepare_openai_vision_messages(
        prefix=prefix, suffix="User Query:" + suffix.format(question=question),
        image_paths=paths, image_size=512)
    content = msgs[0]["content"]
    last_image = max(i for i, c in enumerate(content) if c["type"] == "image_url")
    content[last_image]["prompt_cache_breakpoint"] = BREAKPOINT
    return msgs


def responses_input(question, paths):
    # same content, Responses API block types
    content = []
    for c in chat_messages(question, paths)[0]["content"]:
        if c["type"] == "text":
            content.append({"type": "input_text", "text": c["text"]})
        else:
            block = {"type": "input_image", "image_url": c["image_url"]["url"]}
            if "prompt_cache_breakpoint" in c:
                block["prompt_cache_breakpoint"] = c["prompt_cache_breakpoint"]
            content.append(block)
    return [{"role": "user", "content": content}]


def cost(model, u):
    p = PRICES[model]
    return (u["uncached"] * p["input"] + u["cached"] * p["cached"]
            + u["write"] * p["write_30m"] + u["output"] * p["output"]) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--mode", choices=["chat", "responses"], default="chat")
    ap.add_argument("-n", type=int, default=3)
    args = ap.parse_args()

    qs = [q for q in json.load(open(ROOT / "data" / "open-eqa-v0.json"))
          if q["episode_history"] == EPISODE][: args.n]
    frames = sorted((ROOT / "data" / "frames" / EPISODE).glob("*-rgb.png"))
    paths = [str(frames[i]) for i in np.round(np.linspace(0, len(frames) - 1, 15)).astype(int)]
    client = make_client("openai")
    out = JsonlWriter(ROOT / "predictions" / "raw" / f"probe_cache_{args.model}_{args.mode}.jsonl")

    for q in qs:
        t = time.time()
        if args.mode == "chat":
            r = client.chat.completions.create(
                model=args.model, messages=chat_messages(q["question"], paths),
                seed=1234, max_completion_tokens=4000)
            raw = r.choices[0].message.content
            usage = r.usage.model_dump()
            d = usage.get("prompt_tokens_details") or {}
            total_in, total_out = usage["prompt_tokens"], usage["completion_tokens"]
        else:
            r = client.responses.create(
                model=args.model, input=responses_input(q["question"], paths),
                max_output_tokens=4000)
            raw = r.output_text
            usage = r.usage.model_dump()
            d = usage.get("input_tokens_details") or {}
            total_in, total_out = usage["input_tokens"], usage["output_tokens"]
        cached = d.get("cached_tokens") or 0
        write = d.get("cache_write_tokens") or 0
        u = {"uncached": total_in - cached - write, "cached": cached, "write": write, "output": total_out}
        row = {"question_id": q["question_id"], "model": args.model, "api": args.mode, "raw": raw,
               "usage": usage, "split": u, "cost_usd": cost(args.model, u),
               "seconds": round(time.time() - t, 1), "timestamp": now()}
        out.write(row)
        print(json.dumps({"q": q["question"][:50], "split": u, "cost": round(row["cost_usd"], 5),
                          "raw": (raw or "")[:80]}))
        print("   usage:", json.dumps(usage)[:400])
    out.close()


if __name__ == "__main__":
    main()
