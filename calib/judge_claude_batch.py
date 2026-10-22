"""LLM-Match judging with a Claude model through the Message Batches API (J3 variants).

Same prompts and last-period stripping as calib/judge.py; the prediction text is taken from
the J0p rows, so every judge saw identical inputs. One batch per variant covers every set
that has J0p scores. Temperature is rejected by the API, so sampling is the model default.

Variants (tag: thinking setting):
  J3off  thinking disabled, effort low  (closest to the original judge's direct answer)
  J3low  adaptive thinking, effort low
  J3med  adaptive thinking, effort medium

  python -m calib.judge_claude_batch submit --variant J3low
  python -m calib.judge_claude_batch collect
"""

import argparse
import json
import os
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from calib.common import DATASET, ROOT, load_env, now, read_jsonl
from calib.judge_retry import parse_lenient
from openeqa.utils.prompt_utils import load_prompt

MODEL = "claude-haiku-5-5"
# Batch tier: half of standard ($0.10 / $0.50 per 1M tokens)
PRICE_IN, PRICE_OUT = 0.05, 0.25
VARIANTS = {
    "J3off": {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}, "max_tokens": 16},
    "J3low": {"output_config": {"effort": "low"}, "max_tokens": 4000},
    "J3med": {"output_config": {"effort": "medium"}, "max_tokens": 4000},
}
STATE = ROOT / "scores" / "batches_J3.json"


def client():
    load_env()
    return anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"])


def all_sets():
    return sorted(p.name.removesuffix("__J0p.jsonl") for p in (ROOT / "scores").glob("*__J0p.jsonl")
                  if not p.name.startswith("probe_"))


def submit(variant):
    state = json.load(open(STATE)) if STATE.exists() else {}
    if variant in state:
        print(f"{variant}: already submitted ({state[variant]['batch_id']})")
        return
    items = {q["question_id"]: q for q in json.load(open(DATASET))}
    prompts = {"mmbench": load_prompt("mmbench"), "mmbench-extra": load_prompt("mmbench-extra")}
    cfg = VARIANTS[variant]
    requests = []
    for s in all_sets():
        for r in read_jsonl(ROOT / "scores" / f"{s}__J0p.jsonl"):
            if r["prediction"] is None:
                continue  # llm_match scores these 0 without a call
            q = items[r["question_id"]]
            extra = q.get("extra_answers")
            content = prompts["mmbench" if extra is None else "mmbench-extra"].format(
                question=q["question"], answer=q["answer"], prediction=r["prediction"], extra_answers=extra)
            params = {"model": MODEL, "max_tokens": cfg["max_tokens"],
                      "messages": [{"role": "user", "content": content}]}
            for k in ("thinking", "output_config"):
                if k in cfg:
                    params[k] = cfg[k]
            requests.append(Request(custom_id=f"{s}_{r['question_id']}",
                                    params=MessageCreateParamsNonStreaming(**params)))
    b = client().messages.batches.create(requests=requests)
    state[variant] = {"batch_id": b.id, "requests": len(requests), "model": MODEL,
                      "config": {k: v for k, v in cfg.items()}, "submitted": now()}
    json.dump(state, open(STATE, "w"), indent=2)
    print(f"{variant}: submitted {len(requests)} requests as {b.id}")


def collect():
    state = json.load(open(STATE))
    c = client()
    for variant, info in state.items():
        if info.get("collected"):
            continue
        b = c.messages.batches.retrieve(info["batch_id"])
        n = b.request_counts
        print(f"{variant}: {b.processing_status} (succeeded {n.succeeded}, errored {n.errored}, "
              f"processing {n.processing}, expired {n.expired})")
        if b.processing_status != "ended":
            continue
        by_set = {}
        for res in c.messages.batches.results(info["batch_id"]):
            s, qid = res.custom_id[:-37], res.custom_id[-36:]
            row = {"question_id": qid, "judge": variant, "model": MODEL, "settings": info["config"],
                   "batch_id": info["batch_id"], "timestamp": now()}
            if res.result.type != "succeeded":
                row.update(raw=None, score=None, parse_error=f"batch {res.result.type}", cost_usd=0.0)
            else:
                m = res.result.message
                text = "".join(bl.text for bl in m.content if bl.type == "text").strip()
                score = parse_lenient(text)
                row.update(raw=text, score=score, stop_reason=m.stop_reason,
                           parse_error=None if score is not None else "no mark found",
                           prompt_tokens=m.usage.input_tokens, completion_tokens=m.usage.output_tokens,
                           cost_usd=(m.usage.input_tokens * PRICE_IN + m.usage.output_tokens * PRICE_OUT) / 1e6)
            by_set.setdefault(s, []).append(row)
        total = 0.0
        for s, rows in by_set.items():
            # predictions that were None never went out; record them as llm_match would
            sent = {r["question_id"] for r in rows}
            for r0 in read_jsonl(ROOT / "scores" / f"{s}__J0p.jsonl"):
                if r0["prediction"] is None and r0["question_id"] not in sent:
                    rows.append({"question_id": r0["question_id"], "judge": variant, "model": MODEL,
                                 "raw": None, "score": 0, "parse_error": None, "cost_usd": 0.0})
            with (ROOT / "scores" / f"{s}__{variant}.jsonl").open("w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            total += sum(r["cost_usd"] for r in rows)
        info.update(collected=now(), cost_usd=total, sets=len(by_set))
        json.dump(state, open(STATE, "w"), indent=2)
        print(f"   -> {len(by_set)} sets, ${total:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["submit", "collect"])
    ap.add_argument("--variant", choices=sorted(VARIANTS))
    args = ap.parse_args()
    submit(args.variant) if args.action == "submit" else collect()


if __name__ == "__main__":
    main()
