# Val Failure Analysis Bundle

Cel: zidentyfikować **gdzie** baseline YOLOX-m@960 najbardziej gubi na val set SynLoc,
żeby wiedzieć **co** poprawiać w treningu.

Baseline na Codabench Challenge phase: **mAP-LocSim 51.11** (pozycja 14). Top 3 ma ~91.
Chcemy wiedzieć czemu 40-point gap.

## Contents

- `val_predictions.pkl` — predictions z modelu YOLOX-m@960 na val set (6777 images). FullHD coords.
- `val_annotations.json` — ground truth (4K coords, trzeba skalować ×0.5 do matchowania z predictions).
- `top50_worst_images/` — 50 obrazów z najniższym per-image F1. Zaczynaj analizę od nich.
- `analyze_failures.ipynb` — starter notebook: load data, helpers, przykładowa wizualizacja.
- `per_image_scores.csv` — per-image TP/FP/FN/F1 dla wszystkich 6777 val images (nie tylko top-50).

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install numpy pandas matplotlib pillow jupyter
jupyter notebook analyze_failures.ipynb
```

**Nie potrzebujesz** torch/mmcv/GPU — predictions są już wygenerowane.

## Zadanie (TL;DR)

1. Uruchom notebook, zobacz jak loaderzy działają.
2. Obejrzyj 50 worst images — dla każdego pytaj: "co model zrobił źle?"
3. Kategoryzuj błędy:
   - `missed_small` — model nie wykrył małego zawodnika (daleko od kamery, <32×32px)
   - `missed_occluded` — gracz zasłonięty / w tłumie
   - `missed_edge` — gracz blisko krawędzi pola
   - `false_positive_ref` — detektor zaznaczył sędziego / trenera / background
   - `false_positive_crowd` — detektor halucynował w tłumie
   - `bad_keypoint` — detekcja OK, ale pelvis keypoint w złym miejscu
   - `wrong_localization` — keypoint OK ale pozycja na pitchu zła (problem z camera matrix?)
4. Per-camera / per-stadium analiza: czy pewne kamery konsekwentnie gorsze?
5. Napisz `failure_analysis_m960.md` (1 strona) + 10 annotated PNG.

## Kluczowe pytania do odpowiedzi

| Jeśli dominują... | To rekomendacja |
|---|---|
| `missed_small` | Finetune m@1280 (wyższa rozdzielczość = no-brainer) |
| `false_positive_*` | Podnieść score_threshold 0.05 → 0.3+ |
| `missed_occluded` | NMS tuning, lub większy model (l/x) |
| Jedna kamera/stadion dominuje | Camera-specific augmentation |
| `wrong_localization` | Post-processing camera matrix / undist polynomial |
| `bad_keypoint` | Model underfits — więcej epok / większy model |

## Format predictions (val_predictions.pkl)

```python
import pickle
with open('val_predictions.pkl', 'rb') as f:
    preds = pickle.load(f)

# preds = {image_id: [detection_dict, ...]}
# detection_dict = {
#   'keypoints_fullhd': [[pelvis_x, pelvis_y, conf], [pelvis_ground_x, pelvis_ground_y, conf]],
#   'bbox_fullhd': [x1, y1, x2, y2],   # FullHD coords
#   'score': 0.95
# }
```

**Uwaga na coords:**
- Predictions są w FullHD (1920×1080)
- GT w val_annotations.json są w 4K (3840×2160) → skaluj ×0.5 albo rób annotations_fullhd scaling

## Deliverable

- `failure_analysis_m960.md` — markdown report (1 strona)
- `failures/*.png` — 10 annotated screenshots
- Slack TL;DR — 3 bullety

Deadline: **dziś wieczorem (22:00)**, żebyśmy mogli rano 25.04 wdrożyć wnioski.

Ping qbakom jak:
- Coś nie działa w setupie
- Masz wstępne findings (nawet z 3 images, lepiej iterować)
- Pytania dot. kategoryzacji błędów
