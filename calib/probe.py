"""Small cost/behaviour probes for reasoning models before committing to full runs.

  vision: first N questions of the HM3D subset, gpt4v.py prompt and frames
  judge:  first N C-set predictions, mmbench / mmbench-extra prompts

Logs every call (usage incl. reasoning and cached tokens) to predictions/raw/probe_<name>.jsonl,
so the spend is counted with everything else. Reasoning models may reject
temperature/seed/max_tokens; the parameters actually accepted are recorded per row.

  python -m calib.probe vision --model gpt-6-astra -n 5
  python -m calib.probe judge --model gpt-5.6-sol -n 5
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import openai

from calib.common import ROOT, JsonlWriter, chat, make_client, now
from calib.judge import strip_after_last_period
from openeqa.evaluation.llm_match import parse_score
from openeqa.utils.openai_utils import prepare_openai_vision_messages
from openeqa.utils.prompt_utils import load_prompt

# USD per 1M tokens: input, cached input, output
PRICES = {"gpt-6-astra": (10.0, 1.0, 50.0), "gpt-5.6-sol": (4.0, 0.4, 20.0)}


def call(client, model, messages, max_out):
    """Try the repo's settings first, then drop what a reasoning model refuses."""
    attempts = [
        {"seed": 1234, "temperature": 0.2, "max_tokens": max_out},
        {"seed": 1234, "temperature": 0.2, "max_completion_tokens": max_out},
        {"seed": 1234, "max_completion_tokens": max_out},
        {"max_completion_tokens": max_out},
    ]
    for params in attempts:
        try:
            return chat(client, model=model, messages=messages, **params), params
        except openai.BadRequestError as e:
            last = e
    raise last


def usage_row(model, r):
    u = r.usage
    cached = getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
    reasoning = getattr(u.completion_tokens_details, "reasoning_tokens", 0) or 0
    pin, pcached, pout = PRICES[model]
    cost = ((u.prompt_tokens - cached) * pin + cached * pcached + u.completion_tokens * pout) / 1e6
    return {"prompt_tokens": u.prompt_tokens, "cached_tokens": cached,
            "completion_tokens": u.completion_tokens, "reasoning_tokens": reasoning, "cost_usd": cost}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["vision", "judge"])
    ap.add_argument("--model", required=True)
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--max-out", type=int, default=4000,
                    help="completion budget incl. reasoning; generous so reasoning isn't cut off")
    args = ap.parse_args()

    client = make_client("openai")
    name = f"probe_{args.kind}_{args.model}"
    out = JsonlWriter(ROOT / "predictions" / "raw" / f"{name}.jsonl")
    items = {q["question_id"]: q for q in json.load(open(ROOT / "data" / "open-eqa-v0.json"))}

    if args.kind == "vision":
        qs = json.load(open(ROOT / "data" / "open-eqa-hm3d-sub.json"))[: args.n]
        prefix, suffix = load_prompt("gpt4v").split("User Query:")
        jobs = []
        for q in qs:
            frames = sorted((ROOT / "data" / "frames" / q["episode_history"]).glob("*-rgb.png"))
            idx = np.round(np.linspace(0, len(frames) - 1, 15)).astype(int)
            msgs = prepare_openai_vision_messages(
                prefix=prefix, suffix="User Query:" + suffix.format(question=q["question"]),
                image_paths=[str(frames[i]) for i in idx], image_size=512)
            jobs.append((q["question_id"], msgs))
    else:
        preds = json.load(open(ROOT / "predictions" / "C.json"))[: args.n]
        prompts = {"mmbench": load_prompt("mmbench"), "mmbench-extra": load_prompt("mmbench-extra")}
        jobs = []
        for p in preds:
            q = items[p["question_id"]]
            extra = q.get("extra_answers")
            content = prompts["mmbench" if extra is None else "mmbench-extra"].format(
                question=q["question"], answer=q["answer"],
                prediction=strip_after_last_period(p["answer"]), extra_answers=extra)
            jobs.append((p["question_id"], [{"role": "user", "content": content}]))

    for qid, msgs in jobs:
        t = time.time()
        r, params = call(client, args.model, msgs, args.max_out)
        raw = r.choices[0].message.content
        row = {"question_id": qid, "model": args.model, "params": params, "raw": raw,
               "finish_reason": r.choices[0].finish_reason, "seconds": round(time.time() - t, 1),
               "timestamp": now(), **usage_row(args.model, r)}
        if args.kind == "judge":
            try:
                row["score"] = parse_score(raw or "")
            except ValueError:
                row["score"] = None
        out.write(row)
        print(json.dumps({k: row[k] for k in row if k not in ("question_id", "timestamp")})[:400])
    out.close()


if __name__ == "__main__":
    main()
