"""Ensemble two Codabench submissions via Weighted Box Fusion (WBF).

Usage: python ensemble_wbf.py zip1 zip2 OUTPUT.zip [threshold] [weight1] [weight2] [iou_thr]

Both inputs must already be in 4K-coord format with keypoints [x,y,v, x,y,v].
Output is 4K-coord submission ready for Codabench.
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

try:
    from ensemble_boxes import weighted_boxes_fusion
except ImportError:
    print("pip install ensemble_boxes")
    raise

IMG_W = 3840.0
IMG_H = 2160.0


def load_zip(path):
    with zipfile.ZipFile(path) as z:
        with z.open("results.json") as f:
            return json.load(f)


def main():
    zip1 = sys.argv[1]
    zip2 = sys.argv[2]
    out_zip = sys.argv[3]
    threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
    w1 = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    w2 = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
    iou_thr = float(sys.argv[7]) if len(sys.argv) > 7 else 0.55

    r1 = load_zip(zip1)
    r2 = load_zip(zip2)

    # Group by image_id
    def group(results):
        g = {}
        for r in results:
            g.setdefault(r["image_id"], []).append(r)
        return g

    g1 = group(r1)
    g2 = group(r2)
    all_ids = sorted(set(g1) | set(g2))

    merged = []
    det_id = 1

    for img_id in all_ids:
        list1 = g1.get(img_id, [])
        list2 = g2.get(img_id, [])

        # Build lists for WBF — normalize boxes to [0,1]
        def to_norm(lst):
            boxes, scores, labels, kpts_list = [], [], [], []
            for r in lst:
                x, y, w, h = r["bbox"]
                boxes.append([x / IMG_W, y / IMG_H, (x + w) / IMG_W, (y + h) / IMG_H])
                scores.append(r["score"])
                labels.append(r["category_id"])
                kpts_list.append(r["keypoints"])
            return (np.array(boxes) if boxes else np.zeros((0, 4)),
                    np.array(scores) if scores else np.zeros(0),
                    np.array(labels, dtype=int) if labels else np.zeros(0, dtype=int),
                    kpts_list)

        b1, s1, l1, k1 = to_norm(list1)
        b2, s2, l2, k2 = to_norm(list2)

        if len(b1) == 0 and len(b2) == 0:
            continue

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            [b1, b2], [s1, s2], [l1, l2],
            weights=[w1, w2], iou_thr=iou_thr, skip_box_thr=0.01,
        )

        # For each fused box, pick keypoints from the closest detection (highest IoU) from any model
        for fi, (fb, fs, fl) in enumerate(zip(fused_boxes, fused_scores, fused_labels)):
            fx1, fy1, fx2, fy2 = fb
            # Convert fused box back to 4K
            bx = fx1 * IMG_W
            by = fy1 * IMG_H
            bw = (fx2 - fx1) * IMG_W
            bh = (fy2 - fy1) * IMG_H

            # Find best matching keypoints (by center distance) from either model
            best_kpts = None
            best_dist = float('inf')
            cx_f, cy_f = (fx1 + fx2) / 2 * IMG_W, (fy1 + fy2) / 2 * IMG_H
            for b, k in [(b1, k1), (b2, k2)]:
                if len(b) == 0:
                    continue
                for bi in range(len(b)):
                    bb = b[bi]
                    cx = (bb[0] + bb[2]) / 2 * IMG_W
                    cy = (bb[1] + bb[3]) / 2 * IMG_H
                    d = (cx - cx_f) ** 2 + (cy - cy_f) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_kpts = k[bi]

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

    metadata = {"score_threshold": threshold, "position_from_keypoint_index": 1}
    Path(out_zip).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results.json", json.dumps(merged))
        zf.writestr("metadata.json", json.dumps(metadata))

    print(f"Wrote {out_zip}")
    print(f"  input1: {len(r1)} dets from {len(g1)} imgs (weight {w1})")
    print(f"  input2: {len(r2)} dets from {len(g2)} imgs (weight {w2})")
    print(f"  merged: {len(merged)} dets from {len(set(m['image_id'] for m in merged))} imgs")
    print(f"  iou_thr={iou_thr}  threshold={threshold}")


if __name__ == "__main__":
    main()
