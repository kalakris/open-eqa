"""data/open-eqa-hm3d-sub.json: ~200 HM3D questions for the vision sets (C, D).

Stratified by category (equal quota per category, capped by availability, with the
remainder going to the larger categories). Within each category, round-robin over
episodes in shuffled order so questions spread over as many episodes as possible.
"""

import json
import random
from collections import Counter, defaultdict

from calib.common import DATASET, ROOT

SEED = 20261022
TARGET = 200


def main():
    data = [q for q in json.load(open(DATASET)) if q["episode_history"].startswith("hm3d-v0/")]
    rng = random.Random(SEED)
    by_cat = defaultdict(list)
    for q in data:
        by_cat[q["category"]].append(q)
    cats = sorted(by_cat)

    # equal quotas, then hand leftover slots to categories that still have questions
    quota = {c: min(len(by_cat[c]), TARGET // len(cats)) for c in cats}
    while sum(quota.values()) < TARGET and any(quota[c] < len(by_cat[c]) for c in cats):
        for c in sorted(cats, key=lambda c: len(by_cat[c]) - quota[c], reverse=True):
            if sum(quota.values()) < TARGET and quota[c] < len(by_cat[c]):
                quota[c] += 1

    episode_load = Counter()
    picked = []
    for c in cats:
        by_ep = defaultdict(list)
        for q in by_cat[c]:
            by_ep[q["episode_history"]].append(q)
        for qs in by_ep.values():
            rng.shuffle(qs)
        eps = sorted(by_ep)
        rng.shuffle(eps)
        for _ in range(quota[c]):
            # least-used episode overall that still has a question in this category
            ep = min((e for e in eps if by_ep[e]), key=lambda e: episode_load[e])
            picked.append(by_ep[ep].pop())
            episode_load[ep] += 1

    picked.sort(key=lambda q: q["question_id"])
    json.dump(picked, (ROOT / "data" / "open-eqa-hm3d-sub.json").open("w"), indent=2)
    (ROOT / "predictions" / "hm3d-sub-question-ids.txt").write_text(
        "\n".join(q["question_id"] for q in picked) + "\n"
    )
    print(f"{len(picked)} questions over {len(episode_load)} of "
          f"{len({q['episode_history'] for q in data})} episodes (seed {SEED})")
    print(dict(Counter(q["category"] for q in picked)))
    print("extra_answers:", sum(1 for q in picked if "extra_answers" in q))


if __name__ == "__main__":
    main()
