#!/usr/bin/env python3
"""Per-camera BEV-LocSim analysis.

Projects predicted and GT keypoint[1] to ground via sskit.image_to_ground,
greedy-matches in BEV, computes mean distance + recall@{0.5,1,2,3,5}m
per camera (camera identity = hash of camera_matrix).
"""
import json, sys, zipfile, hashlib
from collections import defaultdict
import numpy as np
from sskit import image_to_ground

PRED_ZIP = sys.argv[1] if len(sys.argv) > 1 else '/srv/soccernet-synloc/submissions/v11_ensemble_test.zip'
GT_JSON  = sys.argv[2] if len(sys.argv) > 2 else '/srv/soccernet-synloc/data/raw/SoccerNet/SpiideoSynLoc/annotations_fullhd/test.json'
THRS = [0.5, 1.0, 2.0, 3.0, 5.0]


def cam_fp(cm):
    flat = tuple(round(x, 3) for row in cm for x in row)
    return hashlib.md5(str(flat).encode()).hexdigest()[:8]


def to_ground(kp_xy, cam, undist, W, H):
    """kp_xy in pixel coords (any space) -> ground (X, Y) in field meters.
    sskit convention: centered+W-normalized: ((x-W/2)/W, (y-H/2)/W)."""
    cam = np.asarray(cam, dtype=np.float64)
    undist = np.asarray(undist, dtype=np.float64)
    pts = np.array([[(kp_xy[0] - W/2) / W, (kp_xy[1] - H/2) / W]], dtype=np.float64)
    g = image_to_ground(cam, undist, pts)
    g = g.numpy() if hasattr(g, 'numpy') else g
    return float(g[0, 0]), float(g[0, 1])


def main():
    print(f"Loading GT: {GT_JSON}")
    gt = json.load(open(GT_JSON))
    img_meta = {im['id']: im for im in gt['images']}
    img_to_cam = {im['id']: cam_fp(im['camera_matrix']) for im in gt['images']}

    # GT player ground positions per image — use position_on_pitch directly (no projection)
    gt_per_img = defaultdict(list)
    for a in gt['annotations']:
        pos = a.get('position_on_pitch')
        if pos is None: continue
        gt_per_img[a['image_id']].append((float(pos[0]), float(pos[1])))

    print(f"Loading preds: {PRED_ZIP}")
    with zipfile.ZipFile(PRED_ZIP) as z:
        with z.open('results.json') as f:
            preds = json.load(f)
        with z.open('metadata.json') as f:
            meta = json.load(f)
    score_thr = meta.get('score_threshold', 0.05)
    print(f"Score threshold from meta: {score_thr:.3f}")

    pred_per_img = defaultdict(list)
    for p in preds:
        if p['score'] < score_thr:
            continue
        kp = p['keypoints']
        # Predictions are in 4K coords. camera_matrix is for FullHD → use FullHD W/H
        # but project from 4K coords scaled to FullHD pixel space.
        x, y = kp[3] * 0.5, kp[4] * 0.5
        im = img_meta.get(p['image_id'])
        if im is None: continue
        gx, gy = to_ground((x, y), im['camera_matrix'], im['undist_poly'],
                           im['width'], im['height'])
        pred_per_img[p['image_id']].append((gx, gy, p['score']))

    # Per-camera matching
    per_cam = defaultdict(lambda: {
        'imgs': 0, 'tp': defaultdict(int), 'fn': defaultdict(int),
        'n_gt': 0, 'n_pred': 0, 'distances': []
    })
    for img_id, gts in gt_per_img.items():
        cam = img_to_cam[img_id]
        ps = pred_per_img.get(img_id, [])
        per_cam[cam]['imgs'] += 1
        per_cam[cam]['n_gt'] += len(gts)
        per_cam[cam]['n_pred'] += len(ps)

        if not gts or not ps:
            for t in THRS: per_cam[cam]['fn'][t] += len(gts)
            continue

        # Distance matrix gt x pred
        dm = np.zeros((len(gts), len(ps)))
        for i, g in enumerate(gts):
            for j, p in enumerate(ps):
                dm[i, j] = np.hypot(g[0] - p[0], g[1] - p[1])

        # Greedy: at each threshold compute matching
        for t in THRS:
            mat = dm.copy()
            matched_gt = set(); matched_p = set()
            while True:
                if mat.size == 0: break
                idx = np.unravel_index(np.argmin(mat), mat.shape)
                if mat[idx] > t: break
                if idx[0] in matched_gt or idx[1] in matched_p:
                    mat[idx] = np.inf
                    continue
                matched_gt.add(idx[0])
                matched_p.add(idx[1])
                if t == 5.0:  # collect once
                    per_cam[cam]['distances'].append(mat[idx])
                mat[idx] = np.inf
            per_cam[cam]['tp'][t] += len(matched_gt)
            per_cam[cam]['fn'][t] += len(gts) - len(matched_gt)

    # Report
    rows = []
    for cam, d in per_cam.items():
        recalls = {t: d['tp'][t] / max(1, d['n_gt']) for t in THRS}
        precs = {t: d['tp'][t] / max(1, d['n_pred']) for t in THRS}
        # Integrated mAP-LocSim approximation: mean recall*precision F-ish
        # Real metric integrates AP, here we use a proxy: mean of recall over thresholds
        proxy_score = np.mean(list(recalls.values())) * 100
        med_dist = np.median(d['distances']) if d['distances'] else float('nan')
        rows.append((cam, d['imgs'], d['n_gt'] / d['imgs'], d['n_pred'] / d['imgs'],
                     med_dist, recalls[0.5]*100, recalls[1.0]*100, recalls[5.0]*100,
                     proxy_score))

    rows.sort(key=lambda r: -r[8])  # by proxy score desc
    print(f"\n{'cam':>10} {'imgs':>6} {'gt/img':>7} {'pr/img':>7} {'med_d_m':>8} {'r@0.5':>7} {'r@1':>7} {'r@5':>7} {'proxy_mAP':>10}")
    for r in rows:
        print(f"{r[0]:>10} {r[1]:>6d} {r[2]:>7.1f} {r[3]:>7.1f} {r[4]:>8.2f} {r[5]:>7.1f} {r[6]:>7.1f} {r[7]:>7.1f} {r[8]:>10.2f}")

    overall_proxy = np.mean([r[8] for r in rows])
    print(f"\nOverall proxy mean recall: {overall_proxy:.2f}")


if __name__ == '__main__':
    main()
