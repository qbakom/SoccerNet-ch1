"""N-way WBF ensemble of Codabench submissions.

Usage:
  python ensemble_wbf_n.py OUTPUT.zip --iou 0.55 --threshold 0.05 \
      --input zip1 WEIGHT1 --input zip2 WEIGHT2 [--input zip3 WEIGHT3 ...]

All inputs must be in 4K-coord format with keypoints [x,y,v, x,y,v].
Output is 4K Codabench-ready zip.
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from ensemble_boxes import weighted_boxes_fusion

IMG_W, IMG_H = 3840.0, 2160.0


def load_zip(path):
    with zipfile.ZipFile(path) as z:
        with z.open("results.json") as f:
            return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output", help="Output zip path")
    ap.add_argument("--input", nargs=2, action="append", metavar=("ZIP", "WEIGHT"),
                    required=True, help="Repeat: --input zip1 1.0 --input zip2 1.0")
    ap.add_argument("--iou", type=float, default=0.55)
    ap.add_argument("--threshold", type=float, default=0.05)
    args = ap.parse_args()

    zips = [inp[0] for inp in args.input]
    weights = [float(inp[1]) for inp in args.input]
    results_all = [load_zip(z) for z in zips]

    # Group by image_id
    groups = []
    for r in results_all:
        g = {}
        for item in r:
            g.setdefault(item["image_id"], []).append(item)
        groups.append(g)

    all_ids = sorted(set().union(*(g.keys() for g in groups)))

    merged = []
    det_id = 1

    for img_id in all_ids:
        per_model_data = []
        for g in groups:
            lst = g.get(img_id, [])
            if not lst:
                per_model_data.append((np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int), []))
                continue
            boxes, scores, labels, kpts_list = [], [], [], []
            for it in lst:
                x, y, w, h = it["bbox"]
                boxes.append([x / IMG_W, y / IMG_H, (x + w) / IMG_W, (y + h) / IMG_H])
                scores.append(it["score"])
                labels.append(it.get("category_id", 1))
                kpts_list.append(it["keypoints"])
            per_model_data.append((np.array(boxes), np.array(scores),
                                    np.array(labels, dtype=int), kpts_list))

        if all(len(pm[0]) == 0 for pm in per_model_data):
            continue

        boxes_list = [pm[0] for pm in per_model_data]
        scores_list = [pm[1] for pm in per_model_data]
        labels_list = [pm[2] for pm in per_model_data]

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=args.iou, skip_box_thr=0.01,
        )

        for fb, fs, fl in zip(fused_boxes, fused_scores, fused_labels):
            fx1, fy1, fx2, fy2 = fb
            bx = fx1 * IMG_W
            by = fy1 * IMG_H
            bw = (fx2 - fx1) * IMG_W
            bh = (fy2 - fy1) * IMG_H
            cx_f, cy_f = (fx1 + fx2) / 2 * IMG_W, (fy1 + fy2) / 2 * IMG_H

            # pick keypoints from nearest matching detection across all models
            best_kpts = None
            best_dist = float("inf")
            for (boxes, scores, labels, kpts_list) in per_model_data:
                for bi in range(len(boxes)):
                    bb = boxes[bi]
                    cx = (bb[0] + bb[2]) / 2 * IMG_W
                    cy = (bb[1] + bb[3]) / 2 * IMG_H
                    d = (cx - cx_f) ** 2 + (cy - cy_f) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_kpts = kpts_list[bi]

            if best_kpts is None:
                continue

            merged.append({
                "id": det_id,
                "image_id": img_id,
                "category_id": int(fl),
                "keypoints": list(best_kpts),
                "score": float(fs),
                "bbox": [bx, by, bw, bh],
                "area": bw * bh,
            })
            det_id += 1

    meta = {"score_threshold": args.threshold, "position_from_keypoint_index": 1}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results.json", json.dumps(merged))
        zf.writestr("metadata.json", json.dumps(meta))

    print(f"Wrote {args.output}")
    for i, (zp, w) in enumerate(zip(zips, weights)):
        cnt = len(results_all[i])
        print(f"  [{i}] {zp}  dets={cnt}  weight={w}")
    print(f"  merged: {len(merged)} dets from {len(set(m['image_id'] for m in merged))} imgs")
    print(f"  iou={args.iou}  threshold={args.threshold}")


if __name__ == "__main__":
    main()
