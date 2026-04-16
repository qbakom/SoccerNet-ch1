"""Convert YOLOX predictions to world coordinates using camera model.

Usage:
    python scripts/analytical_localize.py predictions.json annotations_4k.json output.json

The script:
1. Takes YOLOX predictions (keypoints in FullHD pixel space)
2. Scales keypoints to 4K (×2)
3. Uses sskit camera model (camera_matrix + undist_poly) to project pelvis_ground → world
4. Outputs predictions with position_on_pitch field
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sskit.camera import image_to_ground, normalize


def convert_predictions(
    predictions: list[dict],
    annotations_4k: dict,
    scale: float = 2.0,
    score_threshold: float = 0.01,
) -> list[dict]:
    img_info = {img["id"]: img for img in annotations_4k["images"]}

    results = []
    for pred in predictions:
        if pred["score"] < score_threshold:
            continue

        img_id = pred["image_id"]
        if img_id not in img_info:
            continue

        info = img_info[img_id]
        cam = np.array(info["camera_matrix"])
        undist = np.array(info["undist_poly"])
        shape_4k = (3, info["height"], info["width"])

        kpts = pred["keypoints"]
        # pelvis_ground is keypoint index 1 (flat format: x0,y0,v0,x1,y1,v1)
        u_4k = kpts[3] * scale
        v_4k = kpts[4] * scale

        norm_pt = normalize(
            torch.tensor([[u_4k, v_4k]], dtype=torch.float64), shape_4k
        )
        ground = image_to_ground(cam, undist, norm_pt)

        x, y = ground[0][0].item(), ground[0][1].item()

        # Filter physically impossible positions (outside pitch + margin)
        if abs(x) > 60 or abs(y) > 45:
            continue

        result = {
            "image_id": img_id,
            "category_id": pred["category_id"],
            "position_on_pitch": [x, y, 0.0],
            "score": pred["score"],
            "area": 0,
        }
        results.append(result)

    return results


def main():
    if len(sys.argv) < 4:
        print("Usage: python analytical_localize.py predictions.json annotations_4k.json output.json")
        sys.exit(1)

    pred_path, ann_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(pred_path) as f:
        preds = json.load(f)
    with open(ann_path) as f:
        anns = json.load(f)

    results = convert_predictions(preds, anns)

    with open(out_path, "w") as f:
        json.dump(results, f)

    print(f"Converted {len(results)} predictions → {out_path}")


if __name__ == "__main__":
    main()
