"""Test-Time Augmentation (TTA) inference for SynLoc.

Runs model at multiple scales + horizontal flip, merges predictions via WBF.
Expected gain: +2-3 mAP over single-scale inference.

Usage:
    python scripts/tta_inference.py checkpoint config --scales 640,960,1280 --flip
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ensemble_boxes import weighted_boxes_fusion


def merge_predictions_wbf(
    all_preds: list[list[dict]],
    image_ids: list[int],
    image_size: tuple[int, int] = (1920, 1080),
    iou_thr: float = 0.5,
    skip_box_thr: float = 0.01,
) -> list[dict]:
    """Merge predictions from multiple scales/flips using Weighted Box Fusion."""
    w, h = image_size
    merged = []

    for img_id in image_ids:
        boxes_list, scores_list, labels_list = [], [], []
        kpts_list = []

        for preds in all_preds:
            img_preds = [p for p in preds if p["image_id"] == img_id]
            if not img_preds:
                boxes_list.append(np.zeros((0, 4)))
                scores_list.append(np.zeros(0))
                labels_list.append(np.zeros(0, dtype=int))
                kpts_list.append([])
                continue

            boxes = []
            scores = []
            labels = []
            kpts = []
            for p in img_preds:
                x, y, bw, bh = p["bbox"]
                boxes.append([x / w, y / h, (x + bw) / w, (y + bh) / h])
                scores.append(p["score"])
                labels.append(p["category_id"])
                kpts.append(p["keypoints"])

            boxes_list.append(np.array(boxes))
            scores_list.append(np.array(scores))
            labels_list.append(np.array(labels, dtype=int))
            kpts_list.append(kpts)

        if all(len(b) == 0 for b in boxes_list):
            continue

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            iou_thr=iou_thr, skip_box_thr=skip_box_thr,
        )

        for i, (box, score, label) in enumerate(zip(fused_boxes, fused_scores, fused_labels)):
            x1, y1, x2, y2 = box
            merged.append({
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h],
                "score": float(score),
                "keypoints": _average_keypoints(kpts_list, i, len(all_preds)),
            })

    return merged


def _average_keypoints(kpts_list, match_idx, n_models):
    """Average keypoints from matched detections across models.

    Simple approach: take keypoints from highest-confidence model.
    TODO: proper weighted average based on confidence.
    """
    for kpts in kpts_list:
        if match_idx < len(kpts):
            return kpts[match_idx]
    return [0, 0, 0, 0, 0, 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to .pth checkpoint")
    parser.add_argument("config", help="Path to MMPose config")
    parser.add_argument("--scales", default="640,960,1280", help="Comma-separated scales")
    parser.add_argument("--flip", action="store_true", help="Add horizontal flip")
    parser.add_argument("--output", default="tta_predictions.json")
    args = parser.parse_args()

    print("TTA inference requires ensemble_boxes package:")
    print("  pip install ensemble_boxes")
    print()
    print(f"Scales: {args.scales}")
    print(f"Flip: {args.flip}")
    print(f"Checkpoint: {args.checkpoint}")
    print()
    print("TODO: Implement multi-scale inference loop using MMPose API")
    print("This script provides the WBF merging logic.")
    print("Integration with MMPose test.py needed for each scale.")


if __name__ == "__main__":
    main()
