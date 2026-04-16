# SoccerNet SynLoc Dataset Analysis

## Dataset Summary

| Split | Images | Annotations | Players/img (avg) | Disk |
|-------|--------|-------------|-------------------|------|
| train | 42,504 | 668,259 | 15.7 | 13 GB |
| val | 6,777 | 109,351 | 16.1 | 2.0 GB |
| test | 9,309 | 148,164 | 15.9 | 2.8 GB |
| challenge | 11,352 | 0 (hidden) | — | 3.5 GB |
| mini | 7 | 127 | 18.1 | — |
| **Total** | **69,949** | **925,901** | | **~21 GB** |

## Key Observations

### 1. Resolution Mismatch
- **Annotations** reference 4K resolution (3840×2160)
- **Downloaded images** are FullHD (1920×1080)
- **Action needed:** Scale bboxes and keypoints by 0.5x when using FullHD images

### 2. Coordinate System
- Position on pitch uses **centered coordinates**: X ∈ [-41, 41], Y ∈ [-59, 59]
- NOT the [0, 105] × [0, 68] standard pitch coordinates
- Origin appears to be at pitch center

### 3. Player Detection Stats
- Average ~16 players per image (22 players + refs - occluded)
- Range: 1-28 per image
- Some images have very few players (close-up shots, replays)

### 4. Bounding Box Sizes (4K reference)
- Median area: ~750 px² (very small — distant players)
- Mean area: 1,423 px² 
- Range: 1 to 35,413 px²
- In FullHD: median ~188 px² (roughly 14×14 pixels) — tiny targets

### 5. Category
- Single category: `person` (id=1)
- No distinction between players, referees, goalkeepers

### 6. Challenge Set
- 11,352 images with NO annotations — submit-only evaluation via Codabench

### 7. Camera Parameters
- Each image has `camera_matrix` (3x4 projection), `dist_poly`, `undist_poly`
- Enables analytical world coordinate localization (bypass keypoint regression)

## Smoke Test Results (YOLOX-tiny@640, 5 epochs)

| Metric | Value |
|--------|-------|
| locsim/AP (mAP-LocSim) | 28.8 |
| locsim/AP .5 | 45.3 |
| locsim/precision | 64.9% |
| locsim/recall | 45.0% |
| locsim/f1 | 53.2% |
| bbox/AP | 26.2 |
| bbox/AP .5 | 71.1 |

Baseline target (YOLOX-m@960, 300 epochs): 76.17 mAP-LocSim

## Files
- Images: `data/raw/SoccerNet/SpiideoSynLoc/{train,val,test,challenge}/`
- Annotations (4K): `annotations/{train,val,test,challenge_public,mini}.json`
- Annotations (FullHD): `annotations_fullhd/` (scaled 0.5x — use these for training)
- Format: COCO with `position_on_pitch: [x, y, z]` extension
