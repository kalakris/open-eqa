# Judge-calibration prediction sets

Generated for the OpenEQA judge calibration (see `OPENEQA_JUDGE_CALIBRATION_BRIEF.md`).
Each `<set>.json` is `[{question_id, answer}]`, the format `evaluate-predictions.py`
reads. `raw/<set>.jsonl` keeps every generation call: raw text, parsed answer, model,
settings, usage, cost, and timestamp. Judge outputs are in `../scores/<set>__<judge>.jsonl`.

Code is in `calib/`, run via `calib/run.sh calib.<module> ...`.

## Judges

| Tag | Model | Settings | Notes |
|---|---|---|---|
| J0 | `gpt-4-1106-preview` | seed 1234, T 0.2, max_tokens 32 | The paper's LLM-Match judge. **Not runnable**: `model_not_found` on 2026-10-21 (OpenAI page lists a 2026-10-23 shutdown, but the base model appears to have been shut off with the 2026-03-26 preview batch). |
| J0p | `gpt-4-turbo-2024-04-09` | seed 1234, T 0.2, max_tokens 32 | Closest available stand-in for J0 (same GPT-4 Turbo family, GA snapshot). Shuts down 2026-10-23. `mmbench` / `mmbench-extra` prompts and last-period stripping as in `evaluate-predictions.py`. |
| J1 | Qwen3.8-27B-NVFP4 on omen (alias `qwen3.6-27b-fast`, thinking off) | seed 1234, T 0, max_tokens 32 | Candidate permanent replacement for J0. |
| J2 | `gpt-5.6-sol` | seed 1234, max_completion_tokens 2000, default temperature (T not accepted); Batch API | Commercial reference; OpenAI's listed replacement for the retired GPT-4 snapshots. Batch IDs in `../scores/batches_J2.json`. |
| J0g | `gpt-4-0613` | seed 1234, T 0.2, max_tokens 32 | Second stand-in from the original GPT-4 generation, run on C, D, E1, E2 for a cross-check against J0p. Shuts down 2026-10-23. |

## Sets

| Set | Predictions | Questions | Model / source | Prompt | Generated |
|---|---|---|---|---|---|
| A | blind | 1,636 | `gpt-4-0613`, seed 1234, T 0.2, max_tokens 128 (the paper's blind GPT-4 setup) | `blind-llm.txt` | 2026-10-22 |
| B | blind | 1,636 | Qwen3.8-27B-NVFP4 on omen (served as alias `qwen3.6-27b-fast`, thinking off), seed 1234, T 0, max_tokens 128. 10 replies lacked the `A:` prefix; their first line is used as the answer (`parse_error` set in `raw/B.jsonl`) instead of `gpt4.py`'s unjudged 0. | `blind-llm.txt` | 2026-10-22 |
| C | vision | 200 (HM3D subset) | `gpt-4o-2024-08-06`, 15 frames evenly spaced, 512 px, seed 1234, T 0.2, max_tokens 128; raw output is the answer, as in `gpt4v.py` | `gpt4v.txt` | 2026-10-22 |
| D | vision | 200 (HM3D subset) | Qwen3.8-27B-NVFP4 on omen (alias `qwen3.6-27b-fast`, thinking off), same frames and settings as C except T 0; images enabled on omen 2026-10-22 | `gpt4v.txt` | 2026-10-22 |
| F | vision | 557 (all HM3D) | `gpt-6-astra` via Responses API, 15 frames evenly spaced, 512 px, max_output_tokens 4000 (temperature/seed not accepted); explicit prompt-cache breakpoint after the last frame, so an episode's questions share the cached frames (89% of input tokens cached, $4.37) | `gpt4v.txt` | 2026-10-22 |
| E1 | control | 1,636 | each question's own reference `answer` | n/a | 2026-10-22 |
| E2 | control | 1,636 | a reference answer from another question in the same category with a different answer string (`E2_donors.json`, seed 20261022) | n/a | 2026-10-22 |

The vision subset `data/open-eqa-hm3d-sub.json` has 200 HM3D questions: 28–29 per category,
all 63 episodes, seed 20261022. Its IDs are in `hm3d-sub-question-ids.txt`.
Built by `calib/make_hm3d_subset.py`.

## HM3D frames

The `data/README.md` Dropbox link had expired by 2026-10-21. The frames came from the Hugging Face
mirror `AIGeeksGroup/OpenEQA` (63 per-episode tars, each checked against HF's sha256). Every
frame (10,267 PNGs, 1920x1080) was cross-checked by CRC32 and size against a second,
independent mirror, `the-Lin/openeqa` (uploaded 2025-01-22), and all matched. The frames are not committed.

## Sanity check: J0p vs. the paper's J0

Blind GPT-4 on EM-EQA, LLM-Match (bootstrap SE):

| | ScanNet | HM3D | ALL |
|---|---|---|---|
| Paper Table 2 (J0) | 32.5±1.2 | 35.5±1.7 | 33.5±1.0 |
| Set A, J0p | 32.9±1.2 | 36.8±1.7 | 34.3±1.0 |

Every column agrees within 1 SE. This is aggregate agreement only, not per-item agreement.

## External sets (X_*)

These are published third-party prediction runs. The files and provenance are in `data/external/<source>/SOURCES.md`,
imported by `calib/import_external.py`. All were judged in full with J0p on 2026-10-22: 1,636 questions per run, or 1,079 for the
ScanNet-only agentic runs. The 299-question stratified sample judged first (seed 20261022) is listed in
`external-sample-question-ids.txt`.

| Set | Source | Model | J0p |
|---|---|---|---|
| X_graphpad_full | saeejithnair/graph-robotics | Graph-Pad (Gemini 2.0 Flash scene-graph agent) | 56.7 |
| X_graphpad_roomfix | same | Graph-Pad, room fix | 56.4 |
| X_graphpad_baseline | same | Gemini 2.0 Flash with frames | 50.8 |
| X_scw_llava_next | SamuelCahyawijaya/open-eqa fork | LLaVA-NeXT-Video | 46.4 |
| X_scw_videollama2 | same | VideoLLaMA2 | 46.5 |
| X_scw_rag_vlm | same | retrieval + InternVL | 51.6 |
| X_scw_rag_vlm_0725 | same | retrieval + InternVL (07-25) | 53.1 |
| X_scw_rag_vlm_0726 | same | retrieval + InternVL (07-26) | 58.5 |
| X_scw_blind_gpt4o | same | blind GPT-4o | 34.0 |
| X_agentic_v3_tools | yruns agentic-3d-task | gpt-5.4 agent, frames + tools (ScanNet) | 71.5 |
| X_agentic_v4_noframes | same | gpt-5.4 agent, tools only (ScanNet) | 61.2 |

graph-pad and agentic-3d-task state no license, so their prediction and score files stay out of
this public repo (see `.gitignore`). Only the MIT-licensed `scw` sets are committed.
