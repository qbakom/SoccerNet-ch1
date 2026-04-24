# SoccerNet SynLoc — 24h Submission Plan

**Status (2026-04-24, deadline 2026-04-25 14:00).** Current leaderboard: **#14 @ 51.11 mAP-LocSim**. Top-3 ≈ 91+. Gap to top-10 ≈ ~10–15 pts, gap to top-3 ≈ ~40 pts. Best working submission = m@960 keypoints flat format (`work_dirs/submission_v4_challenge/submission.zip`, 216k dets).

**Budget.** 2 challenge submissions/day (4 total before deadline). Test split 500/day (unlimited for us in practice). 1 GPU on kuba (RTX 4090) + Athena A100 with **4423 GPUh remaining** (577/5000 used, 11.5%). Assumed ambition: **top-10 (~70 mAP)** — awaiting user confirmation.

---

## 0. CRITICAL ATHENA DECISIONS (read first)

Athena recon revealed both running jobs are **from-scratch training**, not finetunes. Both misidentified in prior CLAUDE.md notes.

### Job 2544643 — `yoloxpose_m_960` from-scratch
- **State**: epoch 32/300, elapsed 13h33m, **ETA ~7.5h** (finishes tonight ~20:00–22:00 local).
- **Loss**: 4.6–4.8 stable — converging, but 300 epochs of a from-scratch m@960 run is unlikely to beat the **Spiideo-provided m@960 baseline** (which was trained on a superset of data with tuned hyperparams). Unknown until we evaluate.
- **Action**: let it finish. When done → scp checkpoint to kuba → single inference on test split → compare vs 51.11 / 86.5 baseline. If worse, discard. If better, use as seed for ensembling (E10) and as base for E9b finetune below.

### Job 2544644 — `yoloxpose_m_1280` from-scratch ⚠️
- **State**: epoch 18/300, elapsed 13h16m, **ETA ~2.5 days** → **WILL NOT FINISH before deadline**.
- **Prior m@1280 test**: near-zero mAP — either misconfigured checkpoint or very-early-training artifact. Either way, not usable as-is.
- **Recommendation A (STRONGLY PREFERRED) — kill + restart as finetune**:
  ```
  scancel 2544644
  # then launch: finetune m@1280 for 30–50 epochs starting FROM m@960 Spiideo baseline weights
  #   wall time ≈ 8–12h on A100 → finishes well before 2026-04-25 14:00
  ```
  **Why**: Spiideo's m@960 is 300-epoch converged. The 960→1280 resolution bump changes input stride + FPN feature scales; the head needs re-alignment, but backbone features transfer almost 1:1. 30–50 epochs of finetune typically recovers 90%+ of full-training gain on resolution bumps. This is the **only path to getting a real m@1280 checkpoint before the deadline**.
  **Risk**: config compat — `yoloxpose_m_4xb64-300e_1280.py` (if exists) vs using m@960 config with `input_size=1280` override. Validate config parses before submitting (takes 30 sec).
- **Recommendation B (NOT recommended)**: keep running, grab epoch 18 as a mid-training ablation. Too early — loss ≈ 4.2–4.4 vs converged target ~3.0. Would produce near-useless checkpoint.

**Action required from user**: authorize Recommendation A (kill 2544644 + submit finetune). This is the single highest-leverage decision in the next 24h.

### Timeout (2544075) + completed short runs
Housekeeping only — no action needed.

---

## 1. Local recon summary (kuba, read-only)

| Artefact | Path | Size | Notes |
|---|---|---|---|
| Challenge submission (51.11) | `work_dirs/submission_v4_challenge/submission.zip` | 17.6 MB | keypoints flat format, 216709 dets, score≥?, threshold in metadata=0.05 |
| Test submission (86.5) | `work_dirs/submission_m960_test/submission_v3.zip` | — | last known v3 matches 86.5 on test |
| Test keypoints v4 | `work_dirs/submission_v4_keypoints/submission.zip` | 14.4 MB | 177070 dets / 9309 imgs |
| **Val pkl dump** | `work_dirs/val_predictions.pkl` | 13 MB | **2026-04-24 11:47** — fresh, usable for CPU-only sweeps |
| Baseline weights | `models/baselines/yoloxpose_m_4xb64-300e_960_epoch_300.pth` | — | m@960 baseline |
| TTA stub | `scripts/tta_inference.py` | — | WBF merging coded, but **MMPose integration is TODO** (see `scripts/tta_inference.py:109`) |
| Submission builder | `scripts/make_submission.py` | — | position_on_pitch format with analytical `image_to_ground`; accepts `--score-threshold` |

**Format divergence**: `make_submission.py` builds `position_on_pitch` format (older, gave 51.11? or lower — need to confirm). The 51.11 submission on Codabench appears to be the **keypoints flat format** per CLAUDE.md + v4 log. Confirm which file was last uploaded before experimenting.

**SSH from kuba → Athena**: key auth fails (`Permission denied (publickey)`). I cannot probe Athena directly — you must run recon commands yourself (section 3).

---

## 2. Gap analysis — what I need from you

Pick answers only where the information is non-obvious. Grouped by priority.

### A. Team strategy (highest priority — decides whether to parallelize)
1. **Do the other 3 teammates have independent Athena grants / separate GPUs?** If yes, we can split model×resolution combos; if no, serialize on the 4090.
2. **Is anyone already running TTA, multi-scale, or ensembling experiments?** Avoid duplication.
3. **Ambition level**: _just finish ok (≥51)_ vs _top-10 (~70)_ vs _top-3 (~91)_? This determines whether we take risk on m@1280 finetune or play safe with post-proc sweeps.

### B. Athena recon (I cannot SSH — please paste outputs)
Run on your laptop (or from any terminal with your key):
```bash
ssh plgjkomosa@athena.cyfronet.pl '
  squeue -u plgjkomosa --format="%.10i %.9P %.30j %.8T %.10M %.10l %R"
  echo "===ckpts==="
  ls -lht $SCRATCH/synloc/work_dirs/ 2>/dev/null | head -25
  echo "===last-slurm==="
  ls -1t $SCRATCH/synloc/slurm-*.out 2>/dev/null | head -3
  echo "===history==="
  hpc-jobs-history -s $(date -d "7 days ago" +%Y-%m-%d) 2>/dev/null | tail -20
'
```
Plus `tail -150 <latest slurm-*.out>` so I can see the m@1280 finetune status (loss curve, current epoch).

### C. What's already been tried
4. **TTA attempts**: anything beyond the stub in `scripts/tta_inference.py`? Any flip/multi-scale run logs?
5. **Score threshold sweep**: was 0.05 chosen empirically or just "low enough to not drop anything"?
6. **NMS / per-class tuning**: default only?
7. **Ensembling**: tried combining tiny/s/m predictions with WBF?
8. **Higher keypoint count / headtop detector**: anything beyond pelvis_ground@idx1?

### D. Compute budget & constraints
9. **m@1280 finetune job**: still running? ETA? Can we kill it if something better comes up, or is it mandatory finish?
10. **GPUh remaining**: is ~4980 actually ours, or shared with others draining it?

---

## 3. 24h plan — proposed schedule

Three windows + 4 submission slots (2 today, 2 tomorrow before 14:00).

### Window 1 (now → +4h) — P0 free wins (CPU-only, no GPU needed)
Everything here runs on `work_dirs/val_predictions.pkl` and the existing `submission_v4_*.zip` files.

| ID | Experiment | Effort | Expected | Risk | Prio |
|----|-----------|--------|----------|------|------|
| E1 | **Score threshold sweep on val pkl** — rebuild submission at thresholds {0.01, 0.03, 0.05, 0.1, 0.2, 0.3}, evaluate locally on val split if GT available | 30 min CPU | ±1–3 mAP | low — purely offline | **P0** |
| E2 | **Out-of-bounds filter tuning** — `make_submission.py:75` rejects \|x\|>60 or \|y\|>45; tighten to pitch bounds (52.5×34) | 15 min CPU | +0.3–1 mAP (remove spurious far detections) | low | **P0** |
| E3 | **Per-camera score threshold** — if some stadia have systematically lower/higher confidence, calibrate per-stadium | 45 min CPU | +0.5–1 mAP | medium — overfit to val | P1 |
| E4 | **Format verification** — diff current challenge zip vs `04_submission_sanity_check.ipynb` checks; confirm which of keypoints vs position_on_pitch Codabench actually accepted for 51.11 | 15 min | unlocks upload confidence | zero — just check | **P0** |

**→ Submission #2 (today, challenge)**: best of E1+E2+E4 combined. Expected: **51 → 53–55**.

### Window 2 (+4h → +12h, kuba 4090 overnight) — P1 model-side wins

| ID | Experiment | Effort | Expected | Risk | Prio |
|----|-----------|--------|----------|------|------|
| E5 | **TTA single model (m@960 + hflip)** — minimal TTA, skip multi-scale | 2–3h @ 4090 | +1–2 mAP | medium — TTA script has TODO (`scripts/tta_inference.py:109`), needs implementation | P1 |
| E6 | **Multi-scale inference m@960 at 640/960/1280 + hflip, WBF merge** | 5–7h @ 4090 | +2–3 mAP | medium — same TODO + WBF keypoint averaging is naïve (`scripts/tta_inference.py:81`, takes best-conf model instead of weighted mean) | P1 |
| E7 | **Ensemble m@640 + m@960 outputs** (WBF bboxes, conf-weighted keypoints) | 3h @ 4090 (need m@640 inference first, ~1.5h) | +1–2 mAP | low-medium | P1 |
| E8 | **Higher-res single inference m@960 weights @1280 input** (if m@960 config supports it) | 1.5h @ 4090 | +1–2 mAP (resolution bumps small-player recall — 43.6%→77.2% per `reports/top1_strategy.md:60`) | medium — may drop mAP if weights don't generalize to unseen scale | **P0** (quickest high-upside) |

**→ Submission #3 (today, challenge)**: E8 if it beats E1+E2 on test; otherwise the combined E5/E6.

### Window 3 (+12h → +22h, Athena + kuba) — P1/P2 high-upside

| ID | Experiment | Effort | Expected | Risk | Prio |
|----|-----------|--------|----------|------|------|
| E9 | **Eval Athena m@960-from-scratch (epoch 300)** vs Spiideo baseline — scp checkpoint, run inference on test, compare to 86.5 | 1h scp + 1h inference + 15min eval | ±0 to +2 mAP (likely worse than Spiideo baseline — let data decide) | low — pure comparison | P1 |
| E9b | **m@1280 finetune 30–50ep from m@960 Spiideo baseline** (requires scancel 2544644 first) — see section 0 | 8–12h @ A100 + 1.5h inference on kuba | **+3–6 mAP** (resolution bump is the biggest single lever, per `reports/top1_strategy.md:56`) | medium-high — config validation + convergence risk | **P0** (highest-upside single action) |
| E10 | **Ensemble {Spiideo m@960, Athena m@960-scratch (if ≥51), E9b m@1280} with WBF + conf-weighted keypoints** | 1–2h CPU post-proc after E9b | +1–3 on top of E9b | medium — needs correct WBF keypoint averaging (fix `scripts/tta_inference.py:81`) | P1 |
| E11 | **Per-class / per-stadium NMS + IoU tuning** on best-so-far pkl | 1h CPU | +0.3–1 mAP | low | P2 |
| E12 | **Headtop keypoint fallback** — for detections with low pelvis confidence, use alternate kpt (verify which is pelvis_ground vs headtop in `annotations_fullhd/` schema) | 2h coding + 1h sweep | +0.5–2 mAP | medium | P2 |
| E13 | **Progressive score-floor per image** — guarantee N≥2 detections per image even below global threshold | 45 min CPU | +0.3–1 mAP (many images drop below threshold entirely) | low | P2 |

**→ Submission #4 (tomorrow AM, challenge)**: if E9b finished and eval on test shows uplift → submit E9b+E10. Otherwise fall back to E8+E11+E13 combo. Save one final slot for emergency resubmit.

### What NOT to do in 24h (per `reports/top1_strategy.md`)
- ❌ Retrain from scratch (300 epochs = 2+ days)
- ❌ New architectures (RT-DETR, deformable attn)
- ❌ Custom position_on_pitch regression loss
- ❌ Heavy augmentation retraining

---

## 4. Expected scoreline with this plan

Updated with corrected understanding: Athena has no usable m@1280 checkpoint today; m@960-from-scratch is a question mark, not an upside.

| After | Submission combo | Expected mAP-LocSim | Confidence |
|---|---|---|---|
| Now | current | 51.11 | ✅ confirmed |
| +4h | E1 + E2 + E4 (CPU post-proc) | 52–55 | **high** |
| +12h | + E8 (m@960 weights @1280 input) | 55–60 | medium |
| +22h (if Rec. A authorized now) | + E9b m@1280 finetune + E10 ensemble | **63–72** | medium — hinges on finetune convergence |
| +22h (if Rec. A NOT authorized) | + E9 eval only + E11+E13 | 56–62 | medium |
| Stretch | + E12 (headtop fallback) | +1–2 on best | low |

**Top-10 (~70) is realistic ONLY if Recommendation A is authorized within the next 1–2 hours** — so the m@1280 finetune has a full 10–12h window before challenge submission slot #4 tomorrow morning. Delay past Window 2 → E9b won't finish → cap at ~60.

**Top-3 (91+)** is not reachable from 51.11 baseline in 24h. Would need: different backbone (L-size or RT-DETR) + 1280 training from scratch + custom loss — multi-day work.

---

## 5. Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Wrong submission format → 0 score | -51 mAP | E4 verification first; keep v4_challenge.zip as safety fallback |
| Burn both daily challenge slots on bad submission | 24h dead time | Evaluate on test split (500/day) BEFORE challenge submission |
| E9b m@1280 finetune doesn't converge / config fails | cap at ~60 mAP | Pre-validate config (30 sec dry run with 1 iter); if fails, fall back to E8 single-inference |
| Kill 2544644 loses 13h of compute | 13 GPUh sunk | Accepted — 13h of unusable from-scratch training is already sunk cost; finetune path is strictly better |
| Athena m@960-from-scratch is worse than Spiideo baseline | none (just discard) | E9 is a comparison experiment with low cost — discard if worse |
| TTA script TODO blocks E5/E6 | -2–3 mAP upside | Run MMPose `test.py` 3× with `--cfg-options codec.input_size=(N,N)` + merge pkls manually, skip rewriting `scripts/tta_inference.py` |
| Score threshold over-tuned to test | Drop on challenge | Keep threshold conservative (0.03–0.05) on challenge; tune aggressively only on test |
| m@1280 config doesn't exist at 1280 res | E9b blocked | Use m@960 config + `codec.input_size=(1280,1280)` + retrain head — same pattern as CLAUDE.md gotcha for inference |

---

## 6. Decision points (need user auth before executing)

Presented as OPTIONS in the session reply. Approve which experiments I may spawn as child sessions (or run directly) once you authorize.
