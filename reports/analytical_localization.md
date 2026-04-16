# Analytical Localization via Camera Matrix

## Key Finding

The dataset provides per-image `camera_matrix` (3x4), `dist_poly`, and `undist_poly`.
Using sskit's `image_to_ground(camera_matrix, undist_poly, normalize(keypoint, shape))`,
we can convert pixel keypoints → world coordinates with near-perfect accuracy.

## Accuracy (GT keypoints → world coords, 200 images, 3142 annotations)

| Metric | Value |
|--------|-------|
| Mean error | 0.060 m |
| Median error | 0.050 m |
| P95 error | 0.148 m |
| <0.1m | 83.3% |
| <0.5m | 100.0% |

## Implication

The localization bottleneck is **keypoint detection accuracy in pixels**, not the
world-coordinate mapping. A model that detects pelvis_ground keypoint within 5 pixels
(FullHD) = 10 pixels (4K) will achieve <0.5m world error.

## Pipeline

```
Image → Detector (YOLOX/RT-DETR) → pelvis_ground keypoint [u, v] in FullHD
                    ↓
            Scale ×2 to 4K coords
                    ↓
     normalize(keypoint, (3, 2160, 3840))
                    ↓
     image_to_ground(camera_matrix, undist_poly, normalized_pt)
                    ↓
     position_on_pitch [x, y, 0] in meters
```

## Current state (YOLOX-tiny, 5 epochs)

With the smoke test model: median localization error = 10.35 m (bad keypoints after 5 epochs).
Expected with YOLOX-m 300 epochs: <1m median error.

## IMPORTANT: Coordinate scaling

- Predictions from FullHD model are in 1920×1080 pixel space
- Camera model expects 4K (3840×2160) coordinates
- Must multiply predicted keypoints by 2.0 before using sskit camera functions
