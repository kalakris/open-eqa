"""Shared helpers for the judge-calibration runs: clients, retries, resumable JSONL, cost."""

import json
import os
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import openai

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "open-eqa-v0.json"

# USD per 1M tokens (input, output). Qwen on omen is free.
PRICES = {
    "gpt-4-0613": (30.0, 60.0),
    "gpt-4-turbo-2024-04-09": (10.0, 30.0),
    "gpt-4o-2024-08-06": (2.5, 10.0),
}

# Hard stop across all runs. Raised with approval on 2026-10-22: $70 -> $95 when the account
# held $100, then -> $140 after a $50 top-up.
BUDGET_USD = 140.0


def load_env():
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def make_client(endpoint: str) -> openai.OpenAI:
    """endpoint: 'openai' or 'qwen'. Timeouts are generous: Qwen cold-starts in ~150 s."""
    load_env()
    if endpoint == "qwen":
        return openai.OpenAI(
            base_url=os.environ["QWEN_BASE_URL"],
            api_key=os.environ["QWEN_API_KEY"],
            timeout=400,
            max_retries=0,
        )
    # Ignore any OPENAI_BASE_URL in the shell so OpenAI runs can't leak to the proxy.
    return openai.OpenAI(
        base_url="https://api.openai.com/v1",
        api_key=os.environ["OPENAI_API_KEY"],
        timeout=90,
        max_retries=0,
    )


def _retryable(e: Exception) -> bool:
    if isinstance(e, (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError)):
        return True
    if isinstance(e, openai.APIStatusError):
        return e.status_code >= 500
    return False


def chat(client, attempts: int = 8, **kwargs):
    """chat.completions.create with backoff on 5xx/429/timeouts; other 4xx raise immediately."""
    for i in range(attempts):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            if not _retryable(e) or i == attempts - 1:
                raise
            time.sleep(min(60, 2**i) * (0.5 + random.random()))


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cost(model: str, usage) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return (usage.prompt_tokens * pin + usage.completion_tokens * pout) / 1e6


# Anthropic credits are a separate $200 pool (2026-10-22), so Claude spend has its own stop.
ANTHROPIC_BUDGET_USD = 190.0


def spent_so_far(provider: str = "openai") -> float:
    """Sum of cost_usd over every raw log for one provider, so budgets hold across runs.

    Rows for claude-* models count as "anthropic"; everything else (OpenAI, and free Qwen) as "openai".
    """
    total = 0.0
    for d in ("predictions/raw", "scores"):
        for p in (ROOT / d).glob("*.jsonl"):
            for row in read_jsonl(p):
                is_claude = str(row.get("model", "")).startswith("claude-")
                if is_claude == (provider == "anthropic"):
                    total += row.get("cost_usd", 0.0)
    return total


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


class JsonlWriter:
    """Thread-safe append-and-flush, so a crash loses at most the in-flight calls."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("a")
        self.lock = threading.Lock()

    def write(self, row: dict):
        with self.lock:
            self.f.write(json.dumps(row) + "\n")
            self.f.flush()

    def close(self):
        self.f.close()
