"""Create challenge submission from YOLOX predictions.

Usage:
    python scripts/make_submission.py predictions.pkl annotations_4k.json output.zip [--score-threshold 0.3]

    predictions.pkl: --dump output from MMPose test.py
    annotations_4k.json: original 4K annotations (for camera_matrix)
    output.zip: submission zip for Codabench
"""

import argparse
import json
import os
import pickle
import zipfile

import numpy as np
import torch
from sskit.camera import image_to_ground, normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", help="Predictions .pkl from MMPose test.py --dump")
    parser.add_argument("annotations", help="4K annotations JSON (for camera params)")
    parser.add_argument("output", help="Output .zip path")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--fullhd-scale", type=float, default=2.0,
                        help="Scale factor from FullHD predictions to 4K camera coords")
    args = parser.parse_args()

    with open(args.predictions, "rb") as f:
        preds = pickle.load(f)

    with open(args.annotations) as f:
        anns = json.load(f)

    img_info = {img["id"]: img for img in anns["images"]}

    results = []
    for pred_item in preds:
        img_id = pred_item["img_id"]
        if img_id not in img_info:
            continue

        info = img_info[img_id]
        cam = np.array(info["camera_matrix"])
        undist = np.array(info["undist_poly"])
        shape_4k = (3, info["height"], info["width"])

        pred_instances = pred_item.get("pred_instances", {})
        bboxes = pred_instances.get("bboxes", [])
        scores = pred_instances.get("bbox_scores", [])
        keypoints = pred_instances.get("keypoints", [])

        for i in range(len(scores)):
            score = float(scores[i])
            if score < args.score_threshold:
                continue

            kpts = keypoints[i]
            # pelvis_ground is keypoint index 1
            u_fullhd = float(kpts[1][0])
            v_fullhd = float(kpts[1][1])

            u_4k = u_fullhd * args.fullhd_scale
            v_4k = v_fullhd * args.fullhd_scale

            pt_norm = normalize(
                torch.tensor([[u_4k, v_4k]], dtype=torch.float64), shape_4k
            )
            ground = image_to_ground(cam, undist, pt_norm)
            x, y = ground[0][0].item(), ground[0][1].item()

            if abs(x) > 60 or abs(y) > 45:
                continue

            results.append({
                "image_id": img_id,
                "category_id": 1,
                "position_on_pitch": [x, y, 0.0],
                "score": score,
                "area": 0,
            })

    metadata = {
        "score_threshold": args.score_threshold,
        "optional_keypoint_index": None,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results.json", json.dumps(results))
        zf.writestr("metadata.json", json.dumps(metadata))

    print(f"Submission created: {args.output}")
    print(f"  {len(results)} predictions (score >= {args.score_threshold})")
    print(f"  From {len(set(r['image_id'] for r in results))} images")


if __name__ == "__main__":
    main()
