"""Visualization utilities for SynLoc predictions and data exploration."""

import json
import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from synloc.config import FIGURES_DIR, SOCCERNET_DIR

logger = logging.getLogger(__name__)

# Standard pitch dimensions (meters)
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def draw_pitch(ax: plt.Axes | None = None, color: str = "white", bg: str = "#2d8a4e") -> plt.Axes:
    """Draw a standard soccer pitch on matplotlib axes."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.set_facecolor(bg)
    ax.set_xlim(-5, PITCH_LENGTH + 5)
    ax.set_ylim(-5, PITCH_WIDTH + 5)
    ax.set_aspect("equal")

    # Pitch outline
    ax.plot([0, PITCH_LENGTH, PITCH_LENGTH, 0, 0],
            [0, 0, PITCH_WIDTH, PITCH_WIDTH, 0], color=color, lw=2)

    # Center line and circle
    ax.plot([PITCH_LENGTH / 2, PITCH_LENGTH / 2], [0, PITCH_WIDTH], color=color, lw=1.5)
    circle = plt.Circle((PITCH_LENGTH / 2, PITCH_WIDTH / 2), 9.15, fill=False, color=color, lw=1.5)
    ax.add_patch(circle)

    # Penalty areas
    for x_start in [0, PITCH_LENGTH - 16.5]:
        ax.plot([x_start, x_start + 16.5, x_start + 16.5, x_start],
                [PITCH_WIDTH / 2 - 20.15, PITCH_WIDTH / 2 - 20.15,
                 PITCH_WIDTH / 2 + 20.15, PITCH_WIDTH / 2 + 20.15],
                color=color, lw=1.5)

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    return ax


def plot_positions(
    annotations_file: Path | None = None,
    split: str = "train",
    max_images: int = 5,
    save: bool = True,
) -> None:
    """Plot ground-truth player positions on a pitch diagram.

    Args:
        annotations_file: Path to COCO annotation JSON. Defaults to train split.
        split: Dataset split name.
        max_images: Max number of images to plot.
        save: Whether to save figures.
    """
    annotations_file = annotations_file or (SOCCERNET_DIR / f"{split}.json")

    with open(annotations_file) as f:
        data = json.load(f)

    # Group annotations by image
    img_anns: dict[int, list] = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        img_anns.setdefault(img_id, []).append(ann)

    image_ids = list(img_anns.keys())[:max_images]

    for img_id in image_ids:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        draw_pitch(ax)

        anns = img_anns[img_id]
        for ann in anns:
            pos = ann.get("position_on_pitch")
            if pos:
                ax.scatter(pos[0], pos[1], c="yellow", s=80, edgecolors="black", zorder=5)

        ax.set_title(f"Image {img_id} — {len(anns)} players")

        if save:
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            out = FIGURES_DIR / f"positions_{split}_img{img_id}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            logger.info(f"Saved: {out}")
        plt.close(fig)


def plot_predictions_on_image(
    image_path: Path,
    predictions: list[dict],
    ground_truth: list[dict] | None = None,
    save_path: Path | None = None,
) -> None:
    """Overlay predicted and GT positions on a broadcast image.

    Args:
        image_path: Path to the input image.
        predictions: List of prediction dicts with 'position_on_pitch' and 'score'.
        ground_truth: Optional list of GT annotation dicts.
        save_path: Where to save the figure.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"Could not read image: {image_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, (ax_img, ax_pitch) = plt.subplots(1, 2, figsize=(20, 8))

    # Image view
    ax_img.imshow(img)
    ax_img.set_title("Broadcast View")
    ax_img.axis("off")

    # Pitch view
    draw_pitch(ax_pitch)

    if ground_truth:
        for ann in ground_truth:
            pos = ann.get("position_on_pitch")
            if pos:
                ax_pitch.scatter(pos[0], pos[1], c="lime", s=100,
                                 edgecolors="black", zorder=5, label="GT")

    for pred in predictions:
        pos = pred.get("position_on_pitch")
        score = pred.get("score", 0)
        if pos and score > 0.3:
            ax_pitch.scatter(pos[0], pos[1], c="red", s=80, marker="x",
                             zorder=6, label="Pred")

    ax_pitch.set_title("Pitch Positions")

    # Deduplicate legend
    handles, labels = ax_pitch.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax_pitch.legend(unique.values(), unique.keys(), loc="upper right")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.close(fig)
