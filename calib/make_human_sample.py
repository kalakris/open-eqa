"""Sample predictions for human adjudication of judge disagreements.

Draws ~200 (set, question) pairs, stratified by:
  tier       frontier models vs the rest
  agreement  strong (J0p/J1r/J2 marks span >= 2), mild (span 1), agree (all equal)
  category   round-robin over the 7 categories inside each stratum
Each question appears once in the sample. The unlicensed external sets (graphpad,
agentic) are left out because the page shows prediction text.

Writes calib/human_sample.json (item ids with set, judge marks, prediction),
which the analysis joins against the labels. The labeling page embeds only
the blind fields.
"""

import json
import random
from collections import defaultdict

from calib.common import DATASET, ROOT, read_jsonl

SEED = 20261022
FRONTIER = ["H_opus55_f50", "I_fable51_f50", "H_opus55", "I_fable51", "F"]
MID = ["C", "D", "X_scw_rag_vlm_0726", "X_scw_llava_next", "X_scw_videollama2", "A", "B"]
TARGETS = {("frontier", "strong"): 70, ("frontier", "mild"): 30, ("frontier", "agree"): 20,
           ("mid", "strong"): 45, ("mid", "mild"): 20, ("mid", "agree"): 15}
JUDGES = ["J0p", "J1r", "J2"]


def main():
    items = {q["question_id"]: q for q in json.load(open(DATASET))}
    pools = defaultdict(lambda: defaultdict(list))  # (tier, agreement) -> category -> rows
    for tier, sets in [("frontier", FRONTIER), ("mid", MID)]:
        for s in sets:
            marks = {j: {r["question_id"]: r for r in read_jsonl(ROOT / "scores" / f"{s}__{j}.jsonl")}
                     for j in JUDGES}
            for qid, r0 in marks["J0p"].items():
                m = {j: marks[j].get(qid, {}).get("score") for j in JUDGES}
                if any(v is None for v in m.values()) or not r0.get("prediction"):
                    continue
                span = max(m.values()) - min(m.values())
                agreement = "strong" if span >= 2 else "mild" if span == 1 else "agree"
                pools[(tier, agreement)][items[qid]["category"]].append(
                    {"set": s, "question_id": qid, "marks": m, "prediction": r0["prediction"]})

    rng = random.Random(SEED)
    used, picked = set(), []
    for (tier, agreement), n in TARGETS.items():
        by_cat = pools[(tier, agreement)]
        for rows in by_cat.values():
            rng.shuffle(rows)
        cats = sorted(by_cat)
        got = 0
        while got < n and any(by_cat[c] for c in cats):
            for c in cats:
                while by_cat[c] and by_cat[c][-1]["question_id"] in used:
                    by_cat[c].pop()
                if by_cat[c] and got < n:
                    row = by_cat[c].pop()
                    used.add(row["question_id"])
                    picked.append({**row, "tier": tier, "agreement": agreement})
                    got += 1

    rng.shuffle(picked)
    for i, row in enumerate(picked, 1):
        row["item"] = f"L{i:03d}"
    json.dump(picked, (ROOT / "calib" / "human_sample.json").open("w"), indent=1)

    blind = [{"id": r["item"], "category": items[r["question_id"]]["category"],
              "question": items[r["question_id"]]["question"],
              "answer": items[r["question_id"]]["answer"],
              "extra": items[r["question_id"]].get("extra_answers") or [],
              "prediction": r["prediction"]} for r in picked]
    json.dump(blind, (ROOT / "calib" / "human_sample_blind.json").open("w"), separators=(",", ":"))
    counts = defaultdict(int)
    for r in picked:
        counts[(r["tier"], r["agreement"])] += 1
    print(len(picked), dict(counts))


if __name__ == "__main__":
    main()
