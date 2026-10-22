"""Import published third-party prediction runs (data/external/, see each SOURCES.md) as
extra calibration sets, restricted to one shared stratified sample of questions.

Every set uses the same sample, so sets are comparable item by item. ScanNet-only runs
get the ScanNet part of it. Writes predictions/X_<name>.json and
predictions/external-sample-question-ids.txt.
"""

import json
import random
from collections import defaultdict

from calib.common import DATASET, ROOT

SEED = 20261022
SAMPLE = 300
EXT = ROOT / "data" / "external"

SETS = {
    "graphpad_full": "graph-pad/results/full.json",
    "graphpad_roomfix": "graph-pad/results/full_roomfix.json",
    "graphpad_baseline": "graph-pad/results/full_baseline.json",
    "scw_llava_next": "samuelcahyawijaya-open-eqa/predictions/llava-next-predictions.json",
    "scw_videollama2": "samuelcahyawijaya-open-eqa/predictions/videollama2-predictions.json",
    "scw_rag_vlm": "samuelcahyawijaya-open-eqa/predictions/rag-vlm-predictions.json",
    "scw_rag_vlm_0725": "samuelcahyawijaya-open-eqa/predictions/rag-vlm-0725-predictions.json",
    "scw_rag_vlm_0726": "samuelcahyawijaya-open-eqa/predictions/rag-vlm-0726-predictions.json",
    "scw_blind_gpt4o": "samuelcahyawijaya-open-eqa/predictions/blind-gpt4-predictions.json",
    "agentic_v3_tools": "agentic-3d-task/v3_tools_full1079_predictions_20260612.json",
    "agentic_v4_noframes": "agentic-3d-task/v4_noframes_full1079_predictions_20260612.json",
}


def sample_ids(data):
    # proportional to category size, so the sample mirrors the benchmark mix
    rng = random.Random(SEED)
    by_cat = defaultdict(list)
    for q in sorted(data, key=lambda q: q["question_id"]):
        by_cat[q["category"]].append(q["question_id"])
    ids = []
    for cat in sorted(by_cat):
        ids += rng.sample(by_cat[cat], round(SAMPLE * len(by_cat[cat]) / len(data)))
    return set(ids)


def main():
    data = json.load(open(DATASET))
    ids = sample_ids(data)
    (ROOT / "predictions" / "external-sample-question-ids.txt").write_text(
        "\n".join(sorted(ids)) + "\n"
    )
    for name, rel in SETS.items():
        preds = {p["question_id"]: p["answer"] for p in json.load(open(EXT / rel))}
        out = [{"question_id": q, "answer": preds[q]} for q in sorted(ids) if q in preds]
        json.dump(out, (ROOT / "predictions" / f"X_{name}.json").open("w"), indent=2)
        empty = sum(1 for o in out if not o["answer"])
        print(f"X_{name}: {len(out)} of {len(ids)} sampled ({empty} empty)")


if __name__ == "__main__":
    main()
