#!/usr/bin/env python3
"""Per-camera FP-rate chart — supports thesis that gap to top is FP filtering."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# (cam, n_imgs, gt_per_img, pred_per_img, r@0.5, r@1, r@5)
data = [
    ("2aa01f64", 217,  16.3, 21.9, 97.0, 99.0, 99.5),
    ("1b0300c6", 212,  15.8, 18.8, 96.7, 98.3, 98.8),
    ("5a2118f7", 239,  14.5, 18.0, 93.8, 97.8, 98.9),
    ("7cba6618", 225,  14.2, 17.4, 93.5, 97.6, 98.9),
    ("838e9636", 236,  14.8, 17.3, 88.9, 95.7, 99.0),
    ("bd854658", 243,  14.0, 16.7, 89.0, 96.2, 98.3),
    ("464f7d03", 1765, 15.9, 23.6, 87.8, 95.0, 98.3),
    ("9e2a2aab", 1712, 16.5, 24.3, 87.5, 95.0, 98.3),
    ("b6606402", 1588, 15.9, 20.2, 86.4, 94.3, 98.2),
    ("8e9979b7", 1596, 15.5, 18.4, 83.8, 93.0, 97.6),
    ("b67c180d", 329,  16.2, 21.9, 81.8, 92.1, 98.0),
    ("3d4aaa0e", 305,  16.8, 21.8, 79.6, 91.7, 97.9),
    ("f44447a9", 300,  17.9, 20.8, 79.2, 91.5, 97.8),
    ("8beaa435", 342,  16.2, 20.1, 78.9, 90.7, 97.7),
]

# Compute per-camera TP and FP rate at r@0.5m threshold
# TP@0.5 = r@0.5 × gt_per_img
# FP@0.5 = pred_per_img - TP@0.5  (predictions not matched within 0.5m)
records = []
for cam, n, gt, pr, r05, r1, r5 in data:
    tp = gt * r05 / 100.0
    fp = pr - tp
    precision = tp / pr if pr > 0 else 0
    over_pred = (pr - gt) / gt * 100  # % over-prediction
    records.append({
        'cam': cam, 'n': n, 'gt': gt, 'pred': pr, 'tp': tp, 'fp': fp,
        'precision': precision * 100, 'over_pred': over_pred,
        'r05': r05, 'big': n > 1000
    })

# Sort by FP per image (descending — worst over-predictors at top)
records.sort(key=lambda r: -r['fp'])

labels = [f"C{i+1:<2}  n={r['n']:>4}" for i, r in enumerate(records)]
tps = [r['tp'] for r in records]
fps = [r['fp'] for r in records]
gts = [r['gt'] for r in records]

fig, ax = plt.subplots(figsize=(8.0, 4.4))
y = np.arange(len(records))[::-1]

# Stacked bars: TP (green) + FP (red)
ax.barh(y, tps, height=0.7, color="#5DA972", edgecolor="black",
        linewidth=0.5, label="True positives  ($r@0.5\,$m matches)")
ax.barh(y, fps, height=0.7, left=tps, color="#C44E52", edgecolor="black",
        linewidth=0.5, label="False positives  (unmatched at $r@0.5\,$m)")

# GT count line marker per row
for yi, g in zip(y, gts):
    ax.plot([g, g], [yi - 0.35, yi + 0.35], color='black',
            linewidth=1.8, zorder=5)

# Annotate FP count at end of bar
for yi, r in zip(y, records):
    total = r['tp'] + r['fp']
    ax.text(total + 0.4, yi, f"{r['fp']:.1f} FP", va='center',
            fontsize=8, color='#900', fontweight='bold')

# Highlight high-volume cameras with red border on label
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9, fontfamily='monospace')
for tick, r in zip(ax.get_yticklabels(), records):
    if r['big']:
        tick.set_color('#900')
        tick.set_fontweight('bold')

ax.set_xlabel("Detections per image", fontsize=10)
ax.set_xlim(0, max(r['tp'] + r['fp'] for r in records) + 4)
ax.tick_params(axis='x', labelsize=9)
ax.grid(axis='x', linewidth=0.3, alpha=0.4)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)

# Marker explanation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor="#5DA972", edgecolor="black",
          label="True positives ($r@0.5\,$m)"),
    Patch(facecolor="#C44E52", edgecolor="black",
          label="False positives"),
    Line2D([0], [0], color='black', linewidth=1.8,
           label="GT player count per image"),
    Patch(facecolor="white", edgecolor="white",
          label=r"$\bf{C7-C10}$" + " = high-volume cams"),
]
ax.legend(handles=legend_handles, loc='center left',
          bbox_to_anchor=(1.02, 0.5), fontsize=8.5,
          framealpha=0.95, borderaxespad=0)

ax.set_title("Per-camera detection breakdown — FP over-prediction (test split, v11)",
             fontsize=10, pad=6)

plt.tight_layout()
plt.subplots_adjust(right=0.72)

out_pdf = '/srv/soccernet-synloc/reports/techreport_cvpr/fig_per_camera_fp.pdf'
out_png = '/srv/soccernet-synloc/reports/techreport_cvpr/fig_per_camera_fp.png'
plt.savefig(out_pdf, dpi=200, bbox_inches='tight')
plt.savefig(out_png, dpi=200, bbox_inches='tight')
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")

# Also print summary
print("\nSummary:")
for r in records:
    print(f"  {r['cam']}  n={r['n']:>4}  GT={r['gt']:.1f}  Pred={r['pred']:.1f}  "
          f"TP={r['tp']:.1f}  FP={r['fp']:.1f}  prec={r['precision']:.1f}%  "
          f"overpr={r['over_pred']:+.0f}%")
