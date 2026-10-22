#!/bin/sh
# Run a calib module from the repo root with the project venv; never inherit OPENAI_BASE_URL.
cd "$(dirname "$0")/.." && exec env -u OPENAI_BASE_URL .venv/bin/python -m "$@"
