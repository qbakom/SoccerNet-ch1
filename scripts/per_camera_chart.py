#!/usr/bin/env python3
"""Per-camera chart — honest version: full 0-100 axis, single metric focus."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

data = [
    ("2aa01f64", 217, 97.0, 99.0, 99.5),
    ("1b0300c6", 212, 96.7, 98.3, 98.8),
    ("5a2118f7", 239, 93.8, 97.8, 98.9),
    ("7cba6618", 225, 93.5, 97.6, 98.9),
    ("838e9636", 236, 88.9, 95.7, 99.0),
    ("bd854658", 243, 89.0, 96.2, 98.3),
    ("464f7d03", 1765, 87.8, 95.0, 98.3),
    ("9e2a2aab", 1712, 87.5, 95.0, 98.3),
    ("b6606402", 1588, 86.4, 94.3, 98.2),
    ("8e9979b7", 1596, 83.8, 93.0, 97.6),
    ("b67c180d", 329, 81.8, 92.1, 98.0),
    ("3d4aaa0e", 305, 79.6, 91.7, 97.9),
    ("f44447a9", 300, 79.2, 91.5, 97.8),
    ("8beaa435", 342, 78.9, 90.7, 97.7),
]

r05 = [d[2] for d in data]
r1  = [d[3] for d in data]
r5  = [d[4] for d in data]
imgs = [d[1] for d in data]
big = [n > 1000 for n in imgs]

labels = [f"C{i+1:<2}  n={n:>4}" for i, n in enumerate(imgs)]

fig, ax = plt.subplots(figsize=(8.0, 4.4))
y = np.arange(len(labels))[::-1]

# Layered bars: largest threshold bottom, tightest on top
ax.barh(y, r5,  height=0.74, color="#E0E8F0", edgecolor="#AAB8C8",
        linewidth=0.4, zorder=1)
ax.barh(y, r1,  height=0.74, color="#93B5D6", edgecolor="#5C7AA0",
        linewidth=0.4, zorder=2)
colors = ["#C44E52" if b else "#2E6DA4" for b in big]
ax.barh(y, r05, height=0.74, color=colors, edgecolor="black",
        linewidth=0.5, zorder=3)

# Value annotation on r@0.5 bar end
for yi, v in zip(y, r05):
    ax.text(v + 0.8, yi, f"{v:.1f}", va='center', fontsize=8.5,
            fontweight='bold', color='black')

# Honest full-range x-axis
ax.set_xlim(0, 105)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9, fontfamily='monospace')
ax.set_xlabel("Player recall on test split [%]", fontsize=10)
ax.tick_params(axis='x', labelsize=9)
ax.grid(axis='x', linewidth=0.3, alpha=0.4)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)

# Mean reference line at r@0.5m
mean_r05 = np.mean(r05)
ax.axvline(mean_r05, ymin=0, ymax=1, color='#444', linestyle='--',
           linewidth=0.7, alpha=0.6, zorder=0.5)
ax.text(mean_r05, len(labels) - 0.3, f"mean r@0.5: {mean_r05:.1f}%",
        fontsize=7.5, color='#444', ha='center')

# Legend on RIGHT, outside plot area
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor="#2E6DA4", edgecolor="black",
          label=r"$r@0.5\,$m  small camera"),
    Patch(facecolor="#C44E52", edgecolor="black",
          label=r"$r@0.5\,$m  high-volume (n$>$1k)"),
    Patch(facecolor="#93B5D6", edgecolor="#5C7AA0",
          label=r"$r@1\,$m"),
    Patch(facecolor="#E0E8F0", edgecolor="#AAB8C8",
          label=r"$r@5\,$m"),
]
ax.legend(handles=legend_handles, loc='center left',
          bbox_to_anchor=(1.02, 0.5), fontsize=8.5,
          framealpha=0.95, borderaxespad=0)

ax.set_title("Per-camera localization recall (test split, v11 ensemble)",
             fontsize=10, pad=6)

plt.tight_layout()
plt.subplots_adjust(right=0.74)

out_pdf = '/srv/soccernet-synloc/reports/techreport_cvpr/fig_per_camera.pdf'
out_png = '/srv/soccernet-synloc/reports/techreport_cvpr/fig_per_camera.png'
plt.savefig(out_pdf, dpi=200, bbox_inches='tight')
plt.savefig(out_png, dpi=200, bbox_inches='tight')
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
