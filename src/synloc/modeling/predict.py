"""Evaluation and submission wrapper for MMPose."""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from synloc.config import (
    MMPOSE_DIR,
    REPORTS_DIR,
    SOCCERNET_DIR,
    WORK_DIRS,
    get_config_path,
)

logger = logging.getLogger(__name__)


def evaluate(
    checkpoint: str,
    model_size: str = "m",
    resolution: int = 960,
    split: str = "valid",
    work_dir: Path | None = None,
) -> Path:
    """Run evaluation on a given split and save metrics.

    Args:
        checkpoint: Path to model checkpoint (.pth).
        model_size: YOLOX variant.
        resolution: Input resolution.
        split: Dataset split to evaluate on (valid, test).
        work_dir: Override work directory.

    Returns:
        Path to metrics JSON file.
    """
    config_path = get_config_path(model_size, resolution)

    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = REPORTS_DIR / f"metrics_{split}_{model_size}_{resolution}_{timestamp}.json"

    # Override the eval split
    cfg_options = [
        f"val_dataloader.dataset.data_root={SOCCERNET_DIR}",
        f"val_dataloader.dataset.ann_file={split}.json",
    ]

    cmd = [
        sys.executable,
        str(MMPOSE_DIR / "tools" / "test.py"),
        str(config_path),
        str(checkpoint),
        "--out", str(out_file),
        "--cfg-options", *cfg_options,
    ]

    if work_dir:
        cmd.extend(["--work-dir", str(work_dir)])

    logger.info(f"Evaluating on {split} split: {model_size} @ {resolution}px")
    logger.info(f"Checkpoint: {checkpoint}")
    logger.info(f"Command: {' '.join(cmd)}")

    subprocess.run(cmd, check=True, cwd=str(MMPOSE_DIR))

    if out_file.exists():
        logger.info(f"Metrics saved to {out_file}")
        _print_metrics(out_file)
    else:
        logger.warning(f"Expected metrics file not found: {out_file}")

    return out_file


def submit(
    checkpoint: str,
    model_size: str = "m",
    resolution: int = 960,
    work_dir: Path | None = None,
) -> Path:
    """Generate challenge submission zip.

    Args:
        checkpoint: Path to model checkpoint.
        model_size: YOLOX variant.
        resolution: Input resolution.
        work_dir: Override work directory.

    Returns:
        Path to submission zip file.
    """
    config_path = get_config_path(model_size, resolution)

    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    work_dir = work_dir or WORK_DIRS / f"submission_{model_size}_{resolution}"
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg_options = [
        f"val_dataloader.dataset.data_root={SOCCERNET_DIR}",
    ]

    cmd = [
        sys.executable,
        str(MMPOSE_DIR / "tools" / "test.py"),
        str(config_path),
        str(checkpoint),
        "--challenge",
        "--work-dir", str(work_dir),
        "--cfg-options", *cfg_options,
    ]

    logger.info(f"Generating submission: {model_size} @ {resolution}px")
    subprocess.run(cmd, check=True, cwd=str(MMPOSE_DIR))

    # Find generated zip
    zips = list(work_dir.glob("*.zip"))
    if zips:
        logger.info(f"Submission zip: {zips[0]}")
        return zips[0]

    logger.warning("No submission zip found — check work_dir for results.json")
    return work_dir


def _print_metrics(metrics_file: Path) -> None:
    """Pretty-print key metrics from evaluation output."""
    try:
        with open(metrics_file) as f:
            data = json.load(f)

        # MMPose stores metrics in various formats; try common keys
        for key in ("coco/AP", "coco/mAP", "mAP-LocSim", "locsim/AP"):
            if key in data:
                logger.info(f"  {key}: {data[key]:.4f}")

        # Print all metric keys for debugging
        logger.info(f"  All metric keys: {list(data.keys())}")
    except Exception as e:
        logger.warning(f"Could not parse metrics: {e}")
