"""Rescale FullHD-coords zip (from --challenge flag pipeline, vis=score float, no id/area)
to 4K Codabench format (vis=2 int, id sequential, area = w*h).

Usage: rescale_keypoints_zip.py INPUT.zip OUTPUT.zip [THRESHOLD]
"""
import json, sys, zipfile
from pathlib import Path

SCALE = 2.0


def main():
    src = sys.argv[1]
    dst = sys.argv[2]
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05

    with zipfile.ZipFile(src) as z:
        with z.open("results.json") as f:
            results = json.load(f)
        try:
            with z.open("metadata.json") as f:
                meta_in = json.load(f)
            if "score_threshold" in meta_in and len(sys.argv) <= 3:
                thr = meta_in["score_threshold"]  # use F1-opt from val if available
        except KeyError:
            pass

    out = []
    for det_id, r in enumerate(results, start=1):
        kpts = r["keypoints"]
        flat = []
        # input format: [x,y,score, x,y,score] (vis=score float)
        for i in range(0, len(kpts), 3):
            flat.extend([kpts[i] * SCALE, kpts[i + 1] * SCALE, 2])
        bx, by, bw, bh = [v * SCALE for v in r["bbox"]]
        out.append({
            "id": det_id,
            "image_id": r["image_id"],
            "category_id": r.get("category_id", 1),
            "keypoints": flat,
            "score": r["score"],
            "bbox": [bx, by, bw, bh],
            "area": bw * bh,
        })

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results.json", json.dumps(out))
        zf.writestr("metadata.json", json.dumps({"score_threshold": thr, "position_from_keypoint_index": 1}))

    kxs = [r["keypoints"][0] for r in out[:1000]]
    print(f"{src} -> {dst}")
    print(f"  dets={len(out)}  imgs={len(set(r['image_id'] for r in out))}")
    print(f"  thr={thr}  kpt_x={min(kxs):.1f}..{max(kxs):.1f}")


if __name__ == "__main__":
    main()
