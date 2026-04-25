"""Build Codabench submission zip from mmpose CocoMetric .keypoints.json output.

Input JSON format (each item):
    {bbox: [x,y,w,h], category_id, image_id, keypoints: [x,y,score, x,y,score], score}
    — all in FullHD coords, visibility = keypoint score (float).

Output: Codabench zip with 4K coords, visibility=2 (int), id, area.

Usage: python build_from_keypoints_json.py INPUT.keypoints.json OUTPUT.zip THRESHOLD
"""
import json
import sys
import zipfile
from pathlib import Path

SCALE = 2.0


def main():
    src = sys.argv[1]
    dst = sys.argv[2]
    threshold = float(sys.argv[3])

    with open(src) as f:
        preds = json.load(f)

    new_results = []
    for det_id, r in enumerate(preds, start=1):
        kpts = r["keypoints"]
        # mmpose writes [x,y,score, x,y,score, ...] — replace score with int 2 and scale
        flat = []
        for i in range(0, len(kpts), 3):
            flat.extend([kpts[i] * SCALE, kpts[i + 1] * SCALE, 2])

        bx, by, bw, bh = [v * SCALE for v in r["bbox"]]

        new_results.append({
            "id": det_id,
            "image_id": r["image_id"],
            "category_id": r.get("category_id", 1),
            "keypoints": flat,
            "score": r["score"],
            "bbox": [bx, by, bw, bh],
            "area": bw * bh,
        })

    metadata = {
        "score_threshold": threshold,
        "position_from_keypoint_index": 1,
    }

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results.json", json.dumps(new_results))
        zf.writestr("metadata.json", json.dumps(metadata))

    kxs = [r["keypoints"][0] for r in new_results[:1000]]
    print(f"Wrote {dst}")
    print(f"  dets={len(new_results)}  imgs={len(set(r['image_id'] for r in new_results))}")
    print(f"  metadata={metadata}")
    print(f"  kpt x range: {min(kxs):.1f} .. {max(kxs):.1f}")


if __name__ == "__main__":
    main()
