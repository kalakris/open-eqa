"""LLM-Match judging through the OpenAI Batch API (J2), one batch per prediction set.

Same prompts and last-period stripping as calib/judge.py. GPT-5.6+ models reject temperature,
so requests carry seed 1234 and a generous max_completion_tokens (it includes reasoning tokens).
Batch IDs are tracked in scores/batches_<tag>.json, so submit/collect can be re-run safely.

  python -m calib.judge_batch submit --tag J2 --model gpt-5.6-sol --sets A B C ...
  python -m calib.judge_batch collect --tag J2
"""

import argparse
import io
import json

from calib.common import ROOT, DATASET, make_client, now, read_jsonl
from calib.judge import strip_after_last_period
from openeqa.evaluation.llm_match import parse_score
from openeqa.utils.prompt_utils import load_prompt

# Batch-tier USD per 1M tokens: input, cached input, output
BATCH_PRICES = {"gpt-5.6-sol": (2.0, 0.2, 10.0)}


def state_path(tag):
    return ROOT / "scores" / f"batches_{tag}.json"


def load_state(tag):
    p = state_path(tag)
    return json.load(open(p)) if p.exists() else {}


def submit(args, client):
    items = {q["question_id"]: q for q in json.load(open(DATASET))}
    prompts = {"mmbench": load_prompt("mmbench"), "mmbench-extra": load_prompt("mmbench-extra")}
    state = load_state(args.tag)
    for s in args.sets:
        if s in state:
            print(f"{s}: already submitted ({state[s]['batch_id']})")
            continue
        done = {r["question_id"] for r in read_jsonl(ROOT / "scores" / f"{s}__{args.tag}.jsonl")}
        lines = []
        for p in json.load(open(ROOT / "predictions" / f"{s}.json")):
            if p["question_id"] in done:
                continue
            prediction = strip_after_last_period(p["answer"])
            if prediction is None:
                continue  # scored 0 without a call at collect time, as llm_match does
            q = items[p["question_id"]]
            extra = q.get("extra_answers")
            content = prompts["mmbench" if extra is None else "mmbench-extra"].format(
                question=q["question"], answer=q["answer"], prediction=prediction, extra_answers=extra)
            lines.append(json.dumps({
                "custom_id": p["question_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": args.model, "seed": 1234, "max_completion_tokens": args.max_tokens,
                         "messages": [{"role": "user", "content": content}]},
            }))
        if not lines:
            print(f"{s}: nothing to judge")
            continue
        f = client.files.create(file=(f"{s}__{args.tag}.jsonl", io.BytesIO("\n".join(lines).encode())),
                                purpose="batch")
        b = client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions",
                                  completion_window="24h",
                                  metadata={"set": s, "tag": args.tag})
        state[s] = {"batch_id": b.id, "file_id": f.id, "requests": len(lines), "model": args.model,
                    "max_completion_tokens": args.max_tokens, "submitted": now()}
        json.dump(state, open(state_path(args.tag), "w"), indent=2)
        print(f"{s}: submitted {len(lines)} requests as {b.id}")


def collect(args, client):
    state = load_state(args.tag)
    for s, info in state.items():
        if info.get("collected"):
            continue
        b = client.batches.retrieve(info["batch_id"])
        counts = b.request_counts
        print(f"{s}: {b.status} ({counts.completed}/{counts.total} done, {counts.failed} failed)")
        if b.status != "completed":
            continue
        model = info["model"]
        pin, pcached, pout = BATCH_PRICES[model]
        settings = {"seed": 1234, "max_completion_tokens": info["max_completion_tokens"],
                    "temperature": "default (not accepted)", "api": "batch"}
        rows = []
        for line in client.files.content(b.output_file_id).text.splitlines():
            r = json.loads(line)
            body = r["response"]["body"] if r.get("response") else None
            if not body or r["response"]["status_code"] != 200:
                continue  # failed requests stay unjudged and get picked up by a re-submit
            raw = body["choices"][0]["message"]["content"] or ""
            try:
                score, parse_error = parse_score(raw.strip()), None
            except ValueError as e:
                score, parse_error = None, str(e)
            u = body["usage"]
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            reasoning = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            rows.append({
                "question_id": r["custom_id"], "judge": args.tag, "model": model, "settings": settings,
                "raw": raw, "score": score, "parse_error": parse_error,
                "response_model": body.get("model"), "system_fingerprint": body.get("system_fingerprint"),
                "prompt_tokens": u["prompt_tokens"], "cached_tokens": cached,
                "completion_tokens": u["completion_tokens"], "reasoning_tokens": reasoning,
                "cost_usd": ((u["prompt_tokens"] - cached) * pin + cached * pcached
                             + u["completion_tokens"] * pout) / 1e6,
                "batch_id": info["batch_id"], "timestamp": now(),
            })
        # predictions that were None never went out; record them as llm_match would
        sent = {r["question_id"] for r in rows}
        for p in json.load(open(ROOT / "predictions" / f"{s}.json")):
            if p["answer"] is None and p["question_id"] not in sent:
                rows.append({"question_id": p["question_id"], "judge": args.tag, "model": model,
                             "settings": settings, "raw": None, "score": 0, "parse_error": None,
                             "cost_usd": 0.0, "timestamp": now()})
        out = ROOT / "scores" / f"{s}__{args.tag}.jsonl"
        with out.open("a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        info.update(collected=now(), rows=len(rows), cost_usd=sum(r["cost_usd"] for r in rows),
                    parse_errors=sum(1 for r in rows if r["parse_error"]))
        json.dump(state, open(state_path(args.tag), "w"), indent=2)
        print(f"   -> {len(rows)} rows, ${info['cost_usd']:.2f}, {info['parse_errors']} parse errors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["submit", "collect"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--sets", nargs="*", default=[])
    ap.add_argument("--max-tokens", type=int, default=2000)
    args = ap.parse_args()
    client = make_client("openai")
    (submit if args.action == "submit" else collect)(args, client)


if __name__ == "__main__":
    main()
