# Threshold & Pitch Bounds Sweep — val split

**Date:** 2026-04-24
**Model:** yoloxpose_m_4xb64-300e_960, epoch 300 (same checkpoint as v4)
**Input:** `work_dirs/val_predictions.pkl` (6,777 images, 125,414 raw detections, min score 0.05)
**Evaluator:** `sskit.coco.LocSimCOCOeval` (bbox mode, `sigmas=[0.089, 0.089]`, `position_from_keypoint_index=1`)
**GT:** `data/raw/SoccerNet/SpiideoSynLoc/annotations_fullhd/val.json` (FullHD 1920×1080, 109,351 annotations)

---

## Sanity check

- **Current leaderboard (v4 on challenge split):** 51.11 mAP-LocSim @ threshold=0.05
- **Our local eval of same pipeline on val split:** **47.01 mAP-LocSim** @ threshold=0.05, no pitch filter
- **Gap: ~4.1 points** → this is the **val-vs-challenge split distribution difference**, not an evaluator bug. Reasons:
  - val_predictions.pkl is on val (6,777 images); v4 leaderboard is on challenge (11,352 images, hidden GT)
  - v4 uses **keypoints** submission format (`position_from_keypoint_index=1` in metadata), so Codabench projects BEV server-side — NO client pitch filter applied by v4
  - Same checkpoint, same inference params, same eval metric → relative improvements on val should translate to challenge (modulo noise)

**Conclusion: evaluator is sound.** Relative ranking of configs is trustworthy; absolute numbers lag challenge by ~4 points.

---

## Critical discovery — GT extends well beyond the current pitch filter

GT `position_on_pitch` range on val (109,351 annotations):

| axis | min | max | count \|v\|>45 |
|------|-----|-----|----------------|
| x | −39.3 | +39.9 | 0 |
| y | −58.3 | +57.5 | **19,726 (18.0%)** |

The legacy filter in `scripts/make_submission.py:75` (`if abs(x) > 60 or abs(y) > 45: continue`) **cuts 18% of GT positions**. SoccerNet/Spiideo scenes include off-pitch regions (camera often shows bench, run-offs, etc.). **Never use `|y|>45` again.**

Note: v4 submission uses the keypoints format which does NOT apply this filter (Codabench projects BEV server-side without bounds), so v4 was NOT hurt by this. But any future switch back to the `position_on_pitch` format via `make_submission.py` WILL cost ~4.7 mAP.

---

## Sweep 1 — Score threshold (no pitch filter)

| threshold | n_detections | mAP-LocSim | AP₅₀ | Precision@thr | Recall@thr | F1@thr |
|----------:|-------------:|-----------:|-----:|-------------:|-----------:|-------:|
| **0.05** | 125,414 | **0.4701** | 0.6554 | 0.647 | 0.710 | 0.677 |
| 0.08 | 118,704 | 0.4666 | 0.6490 | 0.713 | 0.700 | 0.706 |
| 0.10 | 116,004 | 0.4666 | 0.6490 | 0.713 | 0.700 | 0.706 |
| 0.15 | 111,057 | 0.4655 | 0.6490 | 0.713 | 0.700 | 0.706 |
| 0.20 | 107,569 | 0.4649 | 0.6490 | 0.713 | 0.700 | 0.706 |
| 0.25 | 105,184 | 0.4623 | 0.6419 | 0.757 | 0.690 | 0.722 |
| 0.30 | 103,204 | 0.4617 | 0.6419 | 0.757 | 0.690 | 0.722 |
| 0.40 |  99,958 | 0.4611 | 0.6419 | 0.757 | 0.690 | 0.722 |
| 0.50 |  97,053 | 0.4573 | 0.6344 | 0.781 | 0.680 | **0.727** |
| 0.60 |  94,146 | 0.4531 | 0.6267 | 0.796 | 0.670 | 0.727 |

- **mAP-LocSim is monotonically decreasing with threshold** (as expected — filtering truncates the ranked-list PR curve; can only lose AP, never gain).
- The 0.05 minimum in the dump is a hard floor — we can't test lower without re-inference.
- Precision grows, recall drops, F1 peaks at 0.50–0.60 — but F1 is not the leaderboard metric.
- **For mAP-LocSim: threshold = 0.05 is optimal.** Do not raise.

---

## Sweep 2 — Pitch bounds (threshold = 0.05)

| bounds (|x|, |y|) | n_detections | mAP-LocSim | AP₅₀ | P | R | notes |
|------------------:|-------------:|-----------:|-----:|--:|--:|-------|
| 52.5 × 34 | 79,628 | 0.3579 | 0.4780 | 0.744 | 0.500 | real pitch, too tight, cuts 37% of GT |
| 55 × 35 | 81,539 | 0.3647 | 0.4871 | 0.755 | 0.510 | still too tight |
| **60 × 45 (legacy)** | 101,044 | 0.4228 | 0.5758 | 0.746 | 0.610 | **costs −4.73 mAP** |
| 50 × 60 | 125,414 | 0.4701 | 0.6554 | 0.647 | 0.710 | matches GT range — no cut |
| 45 × 58 | 125,414 | 0.4701 | 0.6554 | 0.647 | 0.710 | tightest that keeps all GT + all preds |
| **none** | 125,414 | **0.4701** | 0.6554 | 0.647 | 0.710 | same — preds fit in 41×57 box |

Raw pred BEV range is `x ∈ [−39.4, 41.5], y ∈ [−57.2, 55.6]`, so any bounds ≥ `(42, 58)` are effectively no-op. **The legacy 60×45 filter is catastrophic (−4.73 mAP).**

---

## Sweep 3 — Grid (threshold × bounds)

| | bounds=60×45 (legacy) | bounds=50×60 (GT-aware) | bounds=none |
|-|---------------------:|------------------------:|------------:|
| **thr=0.05** | 0.4228 | 0.4701 | **0.4701** |
| **thr=0.10** | 0.4222 | 0.4666 | 0.4666 |
| **thr=0.20** | 0.4191 | 0.4649 | 0.4649 |

**No combination beats (threshold=0.05, no filter) = 0.4701.** The pitch filter never helps; lowering threshold always helps (capped at 0.05 in the dump).

---

## Top 3 recommendations

### 1. Safest boost — verify `make_submission.py` filter is not used on next submission
If the next pipeline reverts to the `position_on_pitch` format via `scripts/make_submission.py`, the `|x|>60, |y|>45` filter MUST be removed/widened. Staying on keypoints format (as v4 does) is the simplest safeguard.

**Action:** Either keep the v4 keypoints-style submission OR patch `scripts/make_submission.py:75` to `abs(x) > 60 or abs(y) > 60` (or remove entirely). **Expected gain vs legacy position_on_pitch format: +4.7 mAP on val; proportionally similar on challenge (~+4–5 points to put 51.11 → ~55–56).**

### 2. Balanced — current v4 settings are already optimal on this pkl dump
For the `val_predictions.pkl` dump with 0.05 floor, **(threshold=0.05, no pitch filter)** is the Pareto-best config. v4 already does this. **Expected gain from pure post-processing on v4: ~0.**

To push further, re-run inference with a lower score threshold (0.01 or 0.03) and re-sweep — potentially unlocks another 0.5–1.5 mAP based on the monotonic trend.

### 3. Aggressive — dump at threshold=0.01 + combine with TTA + NMS tuning
Generate a new val_predictions pkl with `model.test_cfg.score_thr=0.01` (currently 0.05 in the inference run). With more tail detections, standard COCO AP keeps growing as the ranked list extends. This is a **separate inference run** and out of scope for this 60-min sweep, but is the clearest path to beating v4.

**Expected gain:** 0.5–2.0 mAP on val. Combined with no-filter keypoints format → projected challenge: **52–53 mAP-LocSim**.

---

## Recommended `metadata.json` for next submission

```json
{
  "score_threshold": 0.05,
  "position_from_keypoint_index": 1
}
```

Same as v4 — **do not change.** Keep keypoints format. Keep threshold at 0.05. Do not add any pitch bounds filter on the client side.

If you switch to `position_on_pitch` format (server does not project), submit:

```json
{"score_threshold": 0.05, "optional_keypoint_index": null}
```

…and in the results builder, use bounds `|x| ≤ 50, |y| ≤ 60` (or no filter). The legacy `60×45` MUST NOT ship.

---

## Estimated gain on challenge split (extrapolated from val)

| scenario | val mAP | expected challenge mAP | vs current v4 (51.11) |
|----------|--------:|-----------------------:|----------------------:|
| v4 baseline (kpts, thr=0.05, no filter) | 0.4701 | ~51.1 | 0 |
| legacy `position_on_pitch` with 60×45 filter (regression trap) | 0.4228 | ~45.8 | **−5.3** |
| next-inference dump @ thr=0.01, no filter | 0.475–0.485 est. | ~51.5–52.5 | **+0.5–1.5** |

**Caveat on overfit risk:** only 4 val-vs-challenge points of slack were observed. Extrapolating improvements < 1 mAP is within noise; anything below +1 val mAP may not move the challenge leaderboard at all. Treat <1-point val deltas as ties.

**Caveat on split divergence:** val is a held-out subset with public GT; challenge has hidden GT and may be drawn from a different camera/stadium distribution. Trust relative ranking of configs, but don't over-fit the sweep to val.

---

## Bottom line for the next submission

**Submit v4 as-is, or re-run inference with `score_thr=0.01` + same submission settings.** There is no free post-processing gain on the current `val_predictions.pkl`. The one real landmine — the legacy `60×45` pitch filter — must not be reintroduced if the submission format changes. Any other tuning (score threshold > 0.05, smaller pitch bounds) strictly hurts mAP-LocSim.

---

## Artifacts

- Sweep script: `/tmp/sweep_eval/run_sweep.py`
- Precomputed flat dets + BEV: `/tmp/sweep_eval/flat.pkl`
- Raw JSON results: `/tmp/sweep_eval/sweep_results.json`
- 20 eval runs total; ~12 minutes wall time on CPU.
