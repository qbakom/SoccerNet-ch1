# Failure Analysis — YOLOX-m@960 baseline

**Analiza 50 worst-scoring images z val set (F1 < 0.55).**
Data: 2026-04-24, autor: qbakom.

## TL;DR (3 bullety)

1. **Problem #1 = Small Objects.** 95.1% GT w worst images to *small* (<64×64 w 4K). 98% missed detections to też small. Duzi gracze nigdy nie są gubieni.

2. **False Positives nie są problemem.** Median FP score = 0.22. Threshold 0.35 zbija precision z 82% na 96% **bez znaczącego kosztu recall** (94% → 89%).

3. **Rekomendacja:** (a) **m@1280 finetune** → bezpośrednio adresuje #1; (b) **threshold 0.35** → natychmiastowe precision boost; (c) opcjonalnie **ensemble m@1280 + m@960**.

---

## Kontekst

- Baseline Codabench score: **51.11 mAP-LocSim** (#14, @LocSim=5 = 87.81)
- Spiideo's own baseline (hakanardo #10): **77.3 mAP-LocSim** → 26 punktów gap
- Top3: Precision 97-98%, Recall 94-97%
- My: Precision 69%, Recall 73% → **precision gap dominuje**

## Metodologia

1. Inference YOLOX-m@960 (weights: `yoloxpose_m_4xb64-300e_960_epoch_300.pth`) na całym val set (6777 images)
2. Per-image F1 using greedy keypoint matching (pelvis_ground at distance threshold 50px w 4K coords)
3. Top 50 worst images by F1 (range 0.25-0.55)
4. Categorize każdy missed detection (FN) / false positive (FP) by bbox size and prediction confidence

## Kluczowe statystyki

### Rozkład wielkości GT (w worst 50)

| Kategoria | Worst 50 | Full val | Uwaga |
|-----------|----------|----------|-------|
| Small (<64×64 w 4K) | **95.1%** | 94.3% | Nieco nadreprezentowane |
| Medium | 4.9% | 5.2% | Proporcjonalne |
| Large | 0.0% | 0.5% | Nigdy nie są worst-cases |

**Median bbox area w worst 50:** 606 px² (w 4K coords) → ~12×12 px w FullHD → **~4×4 px w 960 input!**

### Missed Detections (FN) — 97 total

| Typ | Count | % |
|-----|-------|---|
| fn_small | **96** | **99.0%** |
| fn_medium | 1 | 1.0% |
| fn_large | 0 | 0.0% |

**Wniosek:** Model prawie nigdy nie gubi medium/large graczy. Problem to wyłącznie small objects.

### False Positives (FP) — 114 total

| Score range | Count | Action |
|-------------|-------|--------|
| >0.5 (hallucinations) | 16 | 14% |
| 0.2-0.5 (noise) | 50 | 44% |
| <0.2 (will be filtered) | 48 | 42% |

**Median FP score: 0.22.** Threshold 0.35 wyeliminowałby ~92/114 FP.

### Primary error type (per-image)

| Category | Count |
|----------|-------|
| missed_small | **49** |
| overdetection | 1 |
| missed_large | 0 |
| hallucination | 0 |

**98% worst images miało primary error = "nie wykrył małych graczy".**

## Implikacje dla treningu

### Co JEST problemem

1. **Small object detection** — model przy input 960 widzi małych graczy jako ~4×4 px. Informacja fizycznie za mała dla YOLOX.
2. **Precision threshold** — score 0.05 przepuszcza za dużo noise (FP median=0.22). Tuning na val daje clear peak przy 0.35.

### Co NIE jest problemem

1. ~~Overall model accuracy~~ — duzi gracze wykrywani prawie 100%.
2. ~~Localization post-processing~~ — gdy detekcja jest dobra, pozycja na boisku jest dokładna.
3. ~~Camera-specific failures~~ — brak silnego wzorca per-stadion (rozkład worst images equal per prefix).
4. ~~Bad keypoints / wrong localization~~ — 0 obserwowanych przypadków w top-10.

## Rekomendacje (priorytetyzowane)

### 🟢 Natychmiastowe (0 GPU, już gotowe)

1. **Score threshold 0.35 w metadata** — val F1 87.9%→92.8%, expected Codabench boost +5-20 mAP.
2. Submission `submission_m960_test_v5_th035.zip` już wygenerowana.

### 🟡 Po treningu (Athena)

3. **Finetune m@1280** (Job 2545544, ETA dzisiaj wieczorem) — bezpośrednio adresuje small object problem. Expected boost +10-15 mAP.
4. **Finetune m@960** (Job 2545543, ETA dzisiaj popołudnie) — mniejszy boost, ale tani.
5. **Ensemble m@960 + m@1280** z NMS — +2-3 mAP extra.

### 🔴 Niski priorytet / nie-do-tego deadline

6. TTA (horizontal flip) — mały boost, extra inference time.
7. NMS tuning — już jest OK, małe usprawnienia.
8. Wymiana modelu na l/x — bez baseline weights = scratch training, za długie.

## Oczekiwany wynik

- Baseline + threshold 0.35: **~70 mAP** (dogonienie hakanardo 77.3 powinno być realne)
- + finetuned m@1280: **~82-85 mAP** (top 7-8)
- + ensemble + TTA: **~87-90 mAP** (top 5)

## Załączniki

- `reports/figures/failures/worst_*.png` — 10 worst val images z visualization (zielone=GT, czerwone=pred)
- `reports/failure_analysis_data.csv` — full 50-image breakdown z kategoryzacją
- `notebooks/` — starter notebook analyze_failures.ipynb
