"""Rescale existing FullHD submission zip to 4K + fix format (add id, area, vis=2).

Usage: python rescale_submission.py INPUT.zip OUTPUT.zip [THRESHOLD]
"""
import json
import sys
import zipfile
from pathlib import Path

SCALE = 2.0


def main():
    src_zip = sys.argv[1]
    dst_zip = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else None

    with zipfile.ZipFile(src_zip) as z:
        with z.open("results.json") as f:
            results = json.load(f)
        with z.open("metadata.json") as f:
            metadata = json.load(f)

    if threshold is not None:
        metadata["score_threshold"] = threshold
    metadata["position_from_keypoint_index"] = 1

    new_results = []
    for det_id, r in enumerate(results, start=1):
        kpts_raw = r["keypoints"]
        # Flatten if nested or keep flat; input is flat [x,y,v, x,y,v, ...]
        flat_out = []
        # every 3 values is one keypoint (x,y,v). Scale x,y ×2; replace v with 2 (int).
        for i in range(0, len(kpts_raw), 3):
            x, y = kpts_raw[i], kpts_raw[i + 1]
            flat_out.extend([x * SCALE, y * SCALE, 2])

        bbox = r["bbox"]  # [x, y, w, h]
        bx, by, bw, bh = [v * SCALE for v in bbox]

        new_results.append({
            "id": det_id,
            "image_id": r["image_id"],
            "category_id": r.get("category_id", 1),
            "keypoints": flat_out,
            "score": r["score"],
            "bbox": [bx, by, bw, bh],
            "area": bw * bh,
        })

    Path(dst_zip).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dst_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results.json", json.dumps(new_results))
        zf.writestr("metadata.json", json.dumps(metadata))

    kxs = [r["keypoints"][0] for r in new_results[:1000]]
    print(f"Wrote {dst_zip}")
    print(f"  dets={len(new_results)}  imgs={len(set(r['image_id'] for r in new_results))}")
    print(f"  metadata={metadata}")
    print(f"  kpt x range (first 1k): {min(kxs):.1f} .. {max(kxs):.1f}")
    print(f"  sample[0]: {new_results[0]}")


if __name__ == "__main__":
    main()
