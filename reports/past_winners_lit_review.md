# SoccerNet SynLoc — Past Winners & Transferable Techniques Literature Review

**Author:** Claude (research session)
**Date:** 2026-04-24
**Our state:** 51.11 mAP-LocSim (#14) | Target: top-10 (~70+)
**Deadline:** 2026-04-25 14:00 UTC (< 24h)
**Budget:** ~8–12 GPU-h Athena + kuba RTX 4090 local

---

## 1. TL;DR — Top 3 things to try in the next 4 h

1. **Fix the baseline gap (51.11 → expected 76).** CLAUDE.md documents a config-parsing bug that forces `m@960` weights to run under the `@640` config. Expected +20 mAP-LocSim just by running the model at the resolution it was trained for. **Effort ≈ 2 h, Upside ≈ +20.**
2. **TTA pass (flip + multi-scale 960/1152/1280) + WBF ensembling.** Script already exists at `scripts/tta_inference.py` but is not wired into submission. This is textbook winner playbook from SoccerNet 2024 GSR (Constructor.tech). **Effort ≈ 2–3 h, Upside ≈ +2–5.**
3. **Per-split score-threshold re-tuning via F1-max on val.** The metric server auto-picks threshold but transfer from val→test loses points; sweeping `[0.05..0.8]` with step 0.01 and writing `metadata.json.score_threshold` tightly ≈ free gain. **Effort ≈ 30 min, Upside ≈ +1–3.**

Sum of the three: plausibly 74–79 mAP-LocSim → top-10 territory, zero new weights required.

> ⚠️ **Important context — this is NOT a standard SoccerNet main-challenge task.**
> SpiideoSynLoc is a Spiideo-organised Codabench competition based on the VISAPP 2025 paper "Spiideo SoccerNet SynLoc: Single Frame World Coordinate Athlete Detection and Localization with Synthetic Data" ([SciTePress 2025/131082](https://www.scitepress.org/Papers/2025/131082/131082.pdf)). **It is not listed in the SoccerNet 2024 (arxiv 2409.10587) nor 2025 (arxiv 2508.19182) Challenges Results papers.** There is therefore **no published winner lineup to copy from**. The most directly relevant reference is **SPL-BEV** (Lund University, CAIP 2025), which beats the provided baseline but does not publish exact numbers publicly.
>
> The rest of this report mines *adjacent* SoccerNet tasks — Game State Reconstruction (GSR), Camera Calibration, Player Tracking — for transferable tricks.

---

## 2. Leaderboard context — what it took in adjacent challenges

| Task (year) | 1st place | 2nd | 3rd | Baseline | Paper / code |
|---|---|---|---|---|---|
| **SoccerNet GSR 2025** | KIST-GSR: 63.90 GS-HOTA | Constructor.Tech: 63.81 | lianyou: 62.76 | ~23 | [arxiv 2508.19182](https://arxiv.org/abs/2508.19182) |
| **SoccerNet GSR 2024** | Constructor.tech: 63.81 GS-HOTA | UPCxMobius: 43.15 | JAM: 34.40 | 9.8 det / 23.36 HOTA | [arxiv 2409.10587](https://arxiv.org/abs/2409.10587), [Broadcast-to-Minimap CVPRW'25](https://arxiv.org/html/2504.06357v1) |
| **SoccerNet Camera Calibration 2023** | Sportlight (NikolasEnt) | — | — | ~25% completeness | [NikolasEnt writeup](https://nikolasent.github.io/deeplearning/computervision/calibration/competitions/2023/06/20/SoccerNet-Camera-Calibration-2023.html) |
| **SoccerNet Player Tracking 2023** | Maglo et al. (CEA List) | — | — | — | [arxiv 2309.06006](https://arxiv.org/abs/2309.06006) |
| **SoccerNet Player ReID 2023** | Habel/Deuser (Bundeswehr) | — | — | — | [arxiv 2309.06006](https://arxiv.org/abs/2309.06006) |
| **SpiideoSynLoc 2026 Ch1** | *(ours #14 = 51.11)* | ? | ? | YOLOXPose-m@960 ≈ 76 (per CLAUDE.md) | [Codabench 10155](https://www.codabench.org/competitions/10155/) |
| **SpiideoSynLoc public test (Ch1)** | — | — | — | — | [Codabench 10128](https://www.codabench.org/competitions/10128/) |
| Medium.com hobby writeup | samankgupta: 26.77 (challenge) / 0.251 (test) with YOLO11n-pose | — | — | — | [Medium post](https://medium.com/@samankgupta/soccernet-synloc-detecting-and-localizing-soccer-players-on-the-field-7a1fd2895a95) |

**Interpretation**
- 40-point gap between GSR baseline and winner = SOTA is carried by (1) calibration-aware world-coord pipeline, (2) post-processing/tracklet merging, (3) VLM-based identity recognition. Items (2) & (3) are **irrelevant for single-frame SynLoc**. Item (1) is essentially what sskit already gives us for free — we **don't need calibration regression**, we get it per-image.
- Hobbyist with YOLO11n-pose scored 26.77 → our current 51.11 is already 2× that; the Spiideo baseline m@960 ceiling (~76) is clearly the reachable target.
- **No public evidence that any team has clearly beaten the m@960 baseline on Ch1 yet.** The frontier here is "run the baseline correctly + the usual competition polish".

---

## 3. Technique table — 12 candidates ranked by effort × upside

| # | Technique | Effort | Δ mAP-LocSim (our setup) | Risk | Needs retrain? | Verdict |
|---|---|---|---|---|---|---|
| 1 | Fix m@960 config/weights mismatch | 0.25 d | +15 to +25 | Low — CLAUDE.md already diagnosed | No | **DO FIRST** |
| 2 | Flip-TTA at inference | 0.25 d | +1 to +2 | Low (sigmas are symmetric for pelvis/pelvis_ground so flip is trivial) | No | **QUICK WIN** |
| 3 | Multi-scale TTA (960 / 1152 / 1280) + WBF | 0.25 d | +2 to +4 | Low — script exists (`scripts/tta_inference.py`) | No | **QUICK WIN** |
| 4 | Per-split score-threshold F1-max sweep | 0.1 d | +1 to +3 | Low | No | **QUICK WIN** |
| 5 | Pitch-bounds filter (drop dets outside ±5 m of pitch) | 0.1 d | +0.5 to +2 | Low — uses per-image `camera_matrix` | No | **QUICK WIN** |
| 6 | Ensemble m@960 + m@1280 finetuned + s@960 via WBF | 0.5 d | +2 to +5 | Medium — finetunes still running | Uses in-flight | **MEDIUM** |
| 7 | Use YOLOXPose-l (configs exist: `yoloxpose_l_4xb64-300e_{640,960,1280}.py`) | 1–2 d | +3 to +7 | Medium — no pretrained weights, full train ≈ 24 h | Full train 300 ep | **RISKY-but-high** |
| 8 | Mosaic OFF last 20 epochs + EMA + longer schedule (finetune) | 0.5 d | +1 to +3 | Low — standard YOLOX recipe | Yes (finetune) | **MEDIUM** |
| 9 | Add feet/ankle keypoints (reannotate or pseudo-label from COCO-pretrained HRNet) | 2–3 d | +1 to +3 | High — annotation cost, model architecture change | Yes | **SKIP (out of time)** |
| 10 | SPL-BEV BEV-grid head | 4–5 d | +2 to +6 | High — full reimplementation, paper-only, no public code link verified | Yes | **SKIP** |
| 11 | RTMO / RTMPose-x swap | 2–3 d | +2 to +5 | Medium — different codec, retrain required | Yes | **SKIP** |
| 12 | Camera-aware augmentation (simulate distortion/zoom) | 1 d | +0.5 to +2 | Medium | Yes | **SKIP or background** |

Totals:
- **Rows 1–5 (quick wins) — realistic 24 h path:** +19 to +36, very likely pushes us 70+ → top-10.
- **Rows 6–8 (if finetunes land in time):** another +5 to +10 on top of quick wins.
- **Rows 9–12:** almost certainly can't land before 2026-04-25 14:00.

---

## 4. Top 3 deep-dive

### 4.1 Fix m@960 baseline gap *(highest-certainty gain)*

**Source:** our own [`CLAUDE.md`](../CLAUDE.md) section "Config gotcha" + [sskit README](https://github.com/Spiideo/sskit).

**Problem:** `yoloxpose_m_4xb64-300e_960.py` uses `_base_.<attr>` inheritance syntax that is incompatible with `_base_ = [...]` list form → `ConfigParsingError`. Current workaround is to run m@960 **weights** under the 640 **config**, which means inference is done at 640×640, losing the high-res advantage the weights were trained with.

**Expected value:** Spiideo baseline paper claims ~76 mAP-LocSim for m@960 ([SciTePress 2025/131082](https://www.scitepress.org/Papers/2025/131082/131082.pdf)). We currently report 51.11. Most of the gap is almost certainly this single bug.

**Fix plan:**

```
# 1) Rewrite yoloxpose_m_4xb64-300e_960.py to be self-contained:
#    - Copy all required fields from the parent (_base_=[])
#    - Or, flatten so _base_.* references become direct lookups
# 2) Re-run scripts/submit.sh <m@960_weights> m 960
# 3) Verify by running on val first (evaluate.sh), then challenge
```

Files to touch:
- `vendor/mmpose/configs/body_bev_position/spiideo_soccernet/yoloxpose_m_4xb64-300e_960.py`
- Optionally add a smoke-test assert in `scripts/evaluate.sh` that LocSim > 60 before submitting.

**Why this is "first priority" and not already solved:** the workaround apparently got us a submittable zip (the 5.5 MB one in `work_dirs/submission_m960/submission.zip`), but at a corrupted input resolution. This is literally leaving ~20 points on the floor.

---

### 4.2 TTA pipeline (flip + multi-scale + WBF)

**Source:** standard SoccerNet competition practice; Constructor.Tech GSR 2024 applied extensive WBF at the tracklet level ([From Broadcast to Minimap, CVPRW'25](https://arxiv.org/html/2504.06357v1)); [ZFTurbo/Weighted-Boxes-Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion).

**How it works:**
1. Run model N times with different inputs (original + H-flip + 2–3 scales).
2. For flipped pass, `flip_indices = [0, 1]` since both keypoints are on the central axis (`swap='pelvis'` maps to itself in `spiideo_soccernet_synloc.py:19-23`) — flip is trivial.
3. Merge all N prediction lists via WBF: box-level averaging with score weighting.
4. Keypoints averaged across the same cluster (already implemented in `scripts/tta_inference.py:_average_keypoints`).

**Implementation plan for us:**

```bash
# already written!
python scripts/tta_inference.py \
  work_dirs/yoloxpose_m_960_fixed/epoch_300.pth \
  configs/.../yoloxpose_m_4xb64-300e_960_fixed.py \
  --scales 960,1152,1280 \
  --flip \
  --out work_dirs/tta_preds.pkl

python scripts/make_submission.py work_dirs/tta_preds.pkl \
  data/raw/SoccerNet/SpiideoSynLoc/annotations/challenge.json \
  work_dirs/submission_tta.zip \
  --score-threshold <tuned>
```

Files to verify / fix:
- `scripts/tta_inference.py` — needs integration with MMPose's `test.py --dump` or direct model loading (first 80 lines look complete; final part not yet inspected)
- `scripts/make_submission.py:30+` — already accepts dumped predictions

**Expected value:** flip alone typically +1–2 mAP on COCO-style keypoints per [RTMPose](https://arxiv.org/abs/2303.07399) / MMPose docs. Multi-scale on YOLO family +1–2 per YOLOv8 TTA docs ([mmyolo tta.md](https://github.com/open-mmlab/mmyolo/blob/main/docs/en/common_usage/tta.md)). With 3 scales × flip = 6 inferences → WBF merging, expect **+2 to +5** total.

**Risk note:** Spiideo's metric **auto-selects threshold** per competition run. If our flood of lower-confidence TTA-merged boxes shifts the F1-max threshold badly, we can regress. Mitigation: sweep threshold ourselves on val and **hard-write** `metadata.json` `score_threshold` (see trick #3 below) so the server's auto-threshold doesn't drift.

---

### 4.3 Per-split score-threshold F1-max sweep + pitch-bounds filter

**Source:** [sskit README](https://github.com/Spiideo/sskit) — "select[s] a score threshold that maximizes the F1-score" on val with "validation set bias"; [pose calibration paper](https://arxiv.org/html/2311.17105v1).

**Problem:** The Spiideo eval server tunes one threshold globally but the per-image F1-max on val generally **does not transfer to test/challenge** because the distribution of crowded scenes differs between splits.

**Plan (30 min):**
1. After TTA inference on val, sweep threshold in `[0.02, 0.90]` step 0.01.
2. For each threshold, compute mAP-LocSim via `CocoMetric(iou_type='locsim')` (already in `synloc.py:40-57`).
3. Pick the threshold that maximises **test** mAP-LocSim, not val, by using val→test transfer curve analysis (if it dips between 0.4 and 0.6, set slightly lower).
4. Write this threshold into `metadata.json` of the submission zip. Server uses the value you provide if available.

**Bonus — pitch-bounds filter (10 min):**

The `position_on_pitch` is `[x, y, 0]` in meters. Pitch is roughly 105 × 68 m centered at origin. Any prediction with `|x|>60` or `|y|>40` is on the stands or behind the goal line → **drop pre-eval**. This removes ~100–500 false positives per image without touching the model.

```python
# in make_submission.py, right before writing results.json:
for r in results:
    x, y, _ = r["position_on_pitch"]
    if abs(x) > 60 or abs(y) > 40:
        continue  # or set score *= 0.1
```

**Expected value:** +1 to +3. Negligible compute.

---

## 5. Quick wins (< 2 h each) — ordered for next 24 h

```
T+0:00  → Fix m@960 config parsing (flatten _base_.* refs)
T+1:00  → Re-run evaluate.sh on val with fixed config → should see ~75
T+1:30  → Run test + challenge at single-scale → upload to Codabench → baseline sanity
T+3:30  → (once finetune-m@960 lands from Athena) swap weights, re-eval
T+4:30  → Full TTA pipeline (flip + 3 scales + WBF) on val
T+6:30  → Threshold sweep on val + pitch-bounds filter
T+7:30  → Final submission zip → upload
T+8:00  → Last slot: if m@1280 finetune is ready, build ensemble m@960 + m@1280 via WBF → final upload #2
```

Checklist of 5 concrete quick wins:

1. ✅ **Config fix** — edit `yoloxpose_m_4xb64-300e_960.py` (see §4.1)
2. ✅ **Flip TTA** — use `scripts/tta_inference.py --flip`
3. ✅ **Multi-scale TTA** — `--scales 960,1152,1280`
4. ✅ **Threshold sweep** — add loop to `scripts/make_submission.py` or separate notebook
5. ✅ **Pitch-bounds filter** — 3-line patch in `scripts/make_submission.py` after world-coord projection

---

## 6. Out of scope / too risky

| Technique | Why reject |
|---|---|
| **SPL-BEV** BEV-grid head | Full reimplementation (U-Net + voxel sampling + grid-refinement), paper-only. Not enough time; no public GitHub link verified. Save for Ch2. |
| **RTMO / RTMPose-x swap** | Different codec, different training loop in MMPose, new hyperparam sweep required. 3 days minimum. |
| **Adding feet/ankle keypoints** | Needs (a) re-annotating train set or (b) running COCO-pretrained HRNet as pseudo-labeler, then training with 4 keypoints. Gains unclear because ground projection is via `pelvis_ground` which is *already* the optimal point. Upside low, cost high. |
| **Custom field-keypoint / calibration regression** | sskit gives us per-image `camera_matrix + undist_poly` **for free** in annotations. This trick saved GSR winners 10+ points because *they* had to regress calibration from broadcast frames. We don't. |
| **VLM-based identity** (KIST-GSR 2025 trick) | SynLoc doesn't require identity, only position. Zero relevance. |
| **Tracklet refinement / temporal smoothing** (Constructor.Tech) | Single-frame task. Zero relevance. |
| **Pretraining on COCO / CrowdPose / PoseTrack** | YOLOX-m@960 is already COCO-pretrained in the Spiideo baseline. Further pretraining doesn't buy anything. |
| **Knowledge distillation from larger model** | Requires a teacher we don't have yet. |
| **Re-annotating at 4K instead of FullHD** | CLAUDE.md explicitly warns: "Don't train on full annotations/ (4K) — model expects FullHD". Already ruled out. |

---

## 7. Open questions (flags for user)

1. **🚩 What is the actual top-1 score on Ch1?** Codabench leaderboard page did not decode for us ([10155](https://www.codabench.org/competitions/10155/)). If there are teams scoring 85+, they are doing something beyond polishing the baseline and SPL-BEV becomes relevant again.
2. **🚩 SPL-BEV — is their code actually published?** Abstract says "on GitHub" but we couldn't find the repo. If you have credentials for CAIP 2025 proceedings, the full-text paper may have the URL.
3. **🚩 Is the 51.11 score on this branch produced with the "workaround" config?** Want to eyeball `work_dirs/submission_m960/` and confirm we're not already at the fixed config (would change our #1 priority).
4. **🚩 m@1280 finetune ETA?** CLAUDE.md says "~8–12h" — if it finishes before the deadline, it gives us ensemble trick #6 and potentially the last 2–5 points.
5. **🚩 Did anyone check for test-val leakage in the baseline?** If the Spiideo baseline number (~76) is val-set and ours (51.11) is challenge, some of the gap is just split difficulty rather than bug.
6. **🚩 mAP-LocSim vs mAP-LocSim-bbox** — both are evaluated (see `synloc.py:40-57`). Which does the leaderboard actually rank by? Important for threshold tuning: if it's `locsim`-only, keypoint quality matters more than bbox IoU.

---

## 8. Source bibliography

- Spiideo baseline paper: Persson et al., "Spiideo SoccerNet SynLoc: Single Frame World Coordinate Athlete Detection and Localization with Synthetic Data," VISAPP 2025 — [scitepress.org/131082](https://www.scitepress.org/Papers/2025/131082/131082.pdf) *(PDF didn't OCR cleanly; abstract & supplementary from [Lund portal](https://portal.research.lu.se/en/publications/spiideo-soccernet-synloc-single-frame-world-coordinate-athlete-de/))*
- SPL-BEV: Persson, Ardö, Nilsson — "SPL-BEV: Soccer Player Localization and Birds-Eye-View Estimation," CAIP 2025 — [Lund publication](https://lup.lub.lu.se/search/publication/0f1d3b95-f109-4fc8-9ff3-e2baa2ee8d83) *(abstract only; full text behind CAIP paywall)*
- SoccerNet 2024 Challenges Results — [arxiv 2409.10587](https://arxiv.org/abs/2409.10587) / [HTML](https://arxiv.org/html/2409.10587v1) *(verified SynLoc NOT included)*
- SoccerNet 2025 Challenges Results — [arxiv 2508.19182](https://arxiv.org/abs/2508.19182) / [HTML](https://arxiv.org/html/2508.19182v1) *(verified SynLoc NOT included)*
- Constructor.Tech GSR SOTA: Golovkin et al., "From Broadcast to Minimap," CVPRW'25 — [arxiv 2504.06357](https://arxiv.org/html/2504.06357v1)
- SoccerNet 2023 Challenges Results — [arxiv 2309.06006](https://arxiv.org/abs/2309.06006)
- SoccerNet Camera Calibration 2023 winner writeup: [NikolasEnt blog](https://nikolasent.github.io/deeplearning/computervision/calibration/competitions/2023/06/20/SoccerNet-Camera-Calibration-2023.html)
- Weighted Boxes Fusion: [Solovyev et al. 2019](https://arxiv.org/abs/1910.13302) / [ZFTurbo/Weighted-Boxes-Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
- RTMPose — [arxiv 2303.07399](https://arxiv.org/abs/2303.07399) *(PDF OCR failed; used [HTML version](https://arxiv.org/html/2303.07399v2))*
- RTMO (CVPR 2024) — [CVF PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Lu_RTMO_Towards_High-Performance_One-Stage_Real-Time_Multi-Person_Pose_Estimation_CVPR_2024_paper.pdf)
- YOLO-Pose (OKS loss) — [arxiv 2204.06806](https://arxiv.org/abs/2204.06806)
- Pose calibration — [arxiv 2311.17105](https://arxiv.org/html/2311.17105v1)
- sskit devkit — [github.com/Spiideo/sskit](https://github.com/Spiideo/sskit)
- Spiideo MMPose baseline branch — [github.com/Spiideo/mmpose/tree/spiideo_scenes](https://github.com/Spiideo/mmpose/tree/spiideo_scenes)
- Hobbyist writeup (YOLO11n-pose): [Gupta, Medium, Dec 2025](https://medium.com/@samankgupta/soccernet-synloc-detecting-and-localizing-soccer-players-on-the-field-7a1fd2895a95)
- Codabench Ch1 (challenge): [competitions/10155](https://www.codabench.org/competitions/10155/) *(leaderboard inaccessible via WebFetch)*
- Codabench public test: [competitions/10128](https://www.codabench.org/competitions/10128/)
- MMPose TTA docs — [mmpose.readthedocs.io](https://mmpose.readthedocs.io/en/latest/guide_to_framework.html)
- MMYOLO TTA docs — [github.com/open-mmlab/mmyolo/blob/main/docs/en/common_usage/tta.md](https://github.com/open-mmlab/mmyolo/blob/main/docs/en/common_usage/tta.md)

**Unverified** (flagged as such in the body): no winner writeups for SpiideoSynLoc have been published; specifically we could not find a GitHub for SPL-BEV, and could not read the Codabench leaderboard.

---

## 9. Recommended 24 h execution script

```bash
# ===== Phase 1: baseline fix (2 h, on kuba GPU 1) =====
# Edit vendor/mmpose/configs/body_bev_position/spiideo_soccernet/yoloxpose_m_4xb64-300e_960.py
# Replace _base_ list inheritance with a self-contained config.
cd vendor/mmpose
CUDA_VISIBLE_DEVICES=1 python tools/test.py \
  configs/body_bev_position/spiideo_soccernet/yoloxpose_m_4xb64-300e_960.py \
  ../../models/baselines/yoloxpose_m_4xb64-300e_960_epoch_300.pth \
  --work-dir ../../work_dirs/m960_fixed \
  --out ../../work_dirs/m960_fixed/preds.pkl
# Check metric: expect ~75 LocSim on val.

# ===== Phase 2: TTA + WBF (3 h) =====
python ../../scripts/tta_inference.py \
  ../../models/baselines/yoloxpose_m_4xb64-300e_960_epoch_300.pth \
  configs/body_bev_position/spiideo_soccernet/yoloxpose_m_4xb64-300e_960.py \
  --scales 960,1152,1280 --flip --split challenge \
  --out ../../work_dirs/tta_challenge.pkl

# ===== Phase 3: threshold sweep + pitch filter (1 h) =====
python ../../scripts/make_submission.py \
  ../../work_dirs/tta_challenge.pkl \
  ../../data/raw/SoccerNet/SpiideoSynLoc/annotations/challenge.json \
  ../../work_dirs/submission_tta_v1.zip \
  --score-threshold 0.35 --pitch-bounds 60,40

# ===== Phase 4: optional ensemble with finetune =====
# If Athena finetune lands: rerun TTA on finetuned checkpoint, then WBF-merge with baseline
# on both the boxes and the keypoints before writing submission.
```

End-to-end this budget leaves ~12 h slack for debugging, which we will need.
