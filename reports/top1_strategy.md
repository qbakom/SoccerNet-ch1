# Strategy for Top-1 in SoccerNet SynLoc 2026

## Core Insight

mAP-LocSim measures world-coordinate localization accuracy. The camera matrix
provides near-perfect pixel→world mapping (0.05m error). Therefore:

**The entire challenge reduces to: detect pelvis_ground keypoint as accurately as possible in pixels.**

## Priority Stack (ordered by expected impact)

### P0: Keypoint accuracy is everything
- Larger model = better keypoints: tiny → small → medium → large
- Higher resolution = better small players: 640 → 960 → 1280
- More epochs = better convergence: 300 epochs baseline

### P1: Multi-scale training and inference
- Train at multiple resolutions (640+960+1280)
- TTA: predict at 3 scales + horizontal flip, average keypoints
- Expected gain: +2-3 mAP

### P2: Ensemble diverse models
- YOLOX-m@960 + YOLOX-l@1280 + potentially RT-DETR
- Weighted Box Fusion (WBF) for bbox merging
- Keypoint averaging weighted by confidence
- Expected gain: +2-3 mAP

### P3: Post-processing with camera model
- Use analytical projection (sskit) instead of learned position_on_pitch
- Filter physically impossible positions (outside pitch boundaries)
- Expected gain: +1-2 mAP (eliminates localization regression error)

### P4: Training tricks
- Stronger augmentations: mosaic, mixup, heavy color jitter
- Focal loss for hard examples (distant/occluded players)
- Progressive resizing: start 640, increase to 1280 during training
- Expected gain: +1-2 mAP

### P5: Architecture improvements (high effort)
- RT-DETR backbone (transformer, better global context)
- Deformable attention for small object detection
- Expected gain: +2-4 mAP but requires significant engineering

## Estimated scoreline

| Run | Model | Resolution | Epochs | Expected mAP |
|-----|-------|-----------|--------|-------------|
| Baseline | YOLOX-m | 960 | 300 | ~76 |
| Larger | YOLOX-l | 1280 | 300 | ~79 |
| +TTA | YOLOX-l | multi-scale | - | ~81 |
| +Ensemble | m+l | multi | - | ~83 |
| +Camera post-proc | - | - | - | ~84 |
| +Training tricks | - | - | 300+ | ~85+ |

## Critical: Small Object Problem

94.3% of players are "small" by COCO standards (area < 1024 px² in FullHD).
51.7% have a side < 16 pixels. Resolution is the #1 factor:

| Input resolution | Players well-detected (>8×8 px) |
|-----------------|--------------------------------|
| 640px | 15.0% |
| 960px | 43.6% |
| 1280px | 77.2% |

**1280px input is non-negotiable for competitive results.**

## What NOT to spend time on
- Custom loss functions for position_on_pitch regression (camera model is better)
- Complex NMS strategies (default works fine for non-overlapping players)
- Data augmentation that distorts player shape (players are already tiny)
- Training on 4K images (same effective object size as FullHD with matched resolution)
