"""Control prediction sets.

E1: each question's own reference answer (expect mark 5).
E2: the reference answer of a different question in the same category, with a
    different answer string (plausible but wrong; expect mostly 1).
"""

import json
import random
from collections import defaultdict

from calib.common import DATASET, ROOT

SEED = 20261022


def main():
    data = json.load(open(DATASET))
    rng = random.Random(SEED)

    e1 = [{"question_id": q["question_id"], "answer": q["answer"]} for q in data]

    by_cat = defaultdict(list)
    for q in data:
        by_cat[q["category"]].append(q)
    e2, e2_src = [], []
    for q in data:
        pool = [o for o in by_cat[q["category"]]
                if o["question_id"] != q["question_id"]
                and o["answer"].strip().lower() != q["answer"].strip().lower()]
        donor = rng.choice(pool)
        e2.append({"question_id": q["question_id"], "answer": donor["answer"]})
        e2_src.append({"question_id": q["question_id"], "donor_question_id": donor["question_id"]})

    out = ROOT / "predictions"
    out.mkdir(exist_ok=True)
    json.dump(e1, (out / "E1.json").open("w"), indent=2)
    json.dump(e2, (out / "E2.json").open("w"), indent=2)
    json.dump({"seed": SEED, "pairs": e2_src}, (out / "E2_donors.json").open("w"), indent=2)
    print(f"E1: {len(e1)}  E2: {len(e2)} (seed {SEED})")


if __name__ == "__main__":
    main()
