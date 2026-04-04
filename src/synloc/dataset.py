"""Data download and validation for SoccerNet SynLoc dataset."""

import json
import logging
from pathlib import Path

from synloc.config import DATA_RAW, SOCCERNET_DIR

logger = logging.getLogger(__name__)

SPLITS = ("train", "valid", "test", "challenge")


def download(version: str = "fullhd", splits: tuple[str, ...] = SPLITS) -> Path:
    """Download SoccerNet SynLoc dataset.

    Credentials can be provided via SPIIDEO_USER and SPIIDEO_PASSWORD env vars
    (or in a .env file at project root). Otherwise prompts interactively.

    Args:
        version: Image resolution version ("fullhd" or default 4K).
        splits: Which splits to download.

    Returns:
        Path to downloaded dataset root.
    """
    import os
    from dotenv import load_dotenv
    from SoccerNet.Downloader import SoccerNetDownloader

    load_dotenv()

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    downloader = SoccerNetDownloader(LocalDirectory=str(DATA_RAW / "SoccerNet"))

    user = os.environ.get("SPIIDEO_USER")
    password = os.environ.get("SPIIDEO_PASSWORD")
    if user and password:
        downloader.spiideoUser = user
        downloader.spiideoPassword = password

    downloader.downloadDataTask(
        task="SpiideoSynLoc",
        split=list(splits),
        version=version,
    )
    logger.info(f"Dataset downloaded to {SOCCERNET_DIR}")
    return SOCCERNET_DIR


def create_dummy_dataset(num_images: int = 20, num_players_per_image: int = 10) -> Path:
    """Create a small dummy dataset for pipeline testing.

    Generates fake COCO-format annotations and blank images.

    Returns:
        Path to dummy dataset root.
    """
    import numpy as np

    dummy_dir = SOCCERNET_DIR
    dummy_dir.mkdir(parents=True, exist_ok=True)

    categories = [{"id": 1, "name": "player", "supercategory": "person"}]
    ann_id = 0

    for split in SPLITS:
        n = num_images if split == "train" else max(5, num_images // 4)
        images = []
        annotations = []

        for img_idx in range(n):
            img_name = f"{split}_{img_idx:04d}.jpg"
            img_h, img_w = 1080, 1920
            images.append({
                "id": img_idx,
                "file_name": img_name,
                "width": img_w,
                "height": img_h,
            })

            # Create a blank image
            img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            img[:] = (34, 139, 34)  # green pitch
            import cv2
            cv2.imwrite(str(dummy_dir / img_name), img)

            for _ in range(num_players_per_image):
                # Random position on pitch (105m x 68m)
                px = float(np.random.uniform(0, 105))
                py = float(np.random.uniform(0, 68))
                # Random bbox in image
                cx = int(np.random.uniform(100, img_w - 100))
                cy = int(np.random.uniform(100, img_h - 100))
                bw, bh = int(np.random.uniform(20, 60)), int(np.random.uniform(40, 120))

                annotations.append({
                    "id": ann_id,
                    "image_id": img_idx,
                    "category_id": 1,
                    "bbox": [cx - bw // 2, cy - bh // 2, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                    "position_on_pitch": [px, py, 0.0],
                    "keypoints": [cx, cy, 2, cx, cy + bh // 2, 2],
                    "num_keypoints": 2,
                })
                ann_id += 1

        coco_data = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        with open(dummy_dir / f"{split}.json", "w") as f:
            json.dump(coco_data, f)

    logger.info(f"Dummy dataset created at {dummy_dir}")
    return dummy_dir


def validate(data_root: Path | None = None) -> dict:
    """Validate dataset integrity: check annotation files and count images per split.

    Returns:
        Dict with split names as keys and stats dict as values.
    """
    data_root = data_root or SOCCERNET_DIR
    stats = {}

    for split in SPLITS:
        ann_file = data_root / f"{split}.json"
        if not ann_file.exists():
            logger.warning(f"Missing annotation file: {ann_file}")
            stats[split] = {"exists": False}
            continue

        with open(ann_file) as f:
            data = json.load(f)

        n_images = len(data.get("images", []))
        n_annotations = len(data.get("annotations", []))
        n_categories = len(data.get("categories", []))

        stats[split] = {
            "exists": True,
            "images": n_images,
            "annotations": n_annotations,
            "categories": n_categories,
        }
        logger.info(f"{split}: {n_images} images, {n_annotations} annotations")

    return stats


def get_class_distribution(data_root: Path | None = None, split: str = "train") -> dict[int, int]:
    """Get annotation count per category for a given split."""
    data_root = data_root or SOCCERNET_DIR
    ann_file = data_root / f"{split}.json"

    with open(ann_file) as f:
        data = json.load(f)

    counts: dict[int, int] = {}
    for ann in data.get("annotations", []):
        cat_id = ann["category_id"]
        counts[cat_id] = counts.get(cat_id, 0) + 1

    return counts
