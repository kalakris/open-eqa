"""Upload episode frames to the Anthropic Files API, resized as gpt4v.py does (long side 512).

Every-frame runs exceed the 32 MB request limit as base64 (~0.27 MB per 512 px PNG), so they
reference uploaded files by file_id instead. The manifest maps "<episode>/<frame>.png" to a
file_id and is written as uploads finish, so the upload can be re-run safely.

  python -m calib.upload_frames_claude --episodes-from data/open-eqa-hm3d.json
"""

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import cv2
from tqdm import tqdm

from calib.common import ROOT, load_env

MANIFEST = ROOT / "data" / "frames" / "claude-files-512.json"


def encode(path, image_size):
    # same resize as openeqa.utils.openai_utils.prepare_openai_vision_messages
    frame = cv2.imread(str(path))
    factor = image_size / max(frame.shape[:2])
    frame = cv2.resize(frame, dsize=None, fx=factor, fy=factor)
    ok, buf = cv2.imencode(".png", frame)
    assert ok
    return buf.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-from", required=True, help="dataset JSON whose episodes to upload")
    ap.add_argument("--frames-directory", default=str(ROOT / "data" / "frames"))
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("-j", "--workers", type=int, default=8)
    args = ap.parse_args()

    load_env()
    client = anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"], max_retries=5)
    manifest = json.load(open(MANIFEST)) if MANIFEST.exists() else {}
    episodes = sorted({q["episode_history"] for q in json.load(open(args.episodes_from))})
    todo = []
    for ep in episodes:
        for f in sorted((Path(args.frames_directory) / ep).glob("*-rgb.png")):
            key = f"{ep}/{f.name}"
            if key not in manifest:
                todo.append((key, f))
    print(f"{len(episodes)} episodes, {len(manifest)} already uploaded, {len(todo)} to go")

    lock = threading.Lock()

    def upload(key, path):
        data = encode(path, args.image_size)
        name = key.replace("/", "__")
        return key, client.files.upload(file=(name, data, "image/png")).id

    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(upload, k, p) for k, p in todo]
        for i, f in enumerate(tqdm(as_completed(futs), total=len(futs))):
            key, file_id = f.result()
            with lock:
                manifest[key] = file_id
                if i % 200 == 0:
                    json.dump(manifest, open(MANIFEST, "w"))
    json.dump(manifest, open(MANIFEST, "w"), indent=0)
    print(f"manifest: {len(manifest)} files -> {MANIFEST}")


if __name__ == "__main__":
    main()
