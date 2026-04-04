"""Training wrapper for MMPose YOLOX-Pose baseline."""

import logging
import subprocess
import sys
from pathlib import Path

from synloc.config import (
    MMPOSE_DIR,
    SOCCERNET_DIR,
    WORK_DIRS,
    get_config_path,
)

logger = logging.getLogger(__name__)


def train(
    model_size: str = "m",
    resolution: int = 960,
    num_gpus: int = 1,
    epochs: int | None = None,
    resume_from: str | None = None,
    amp: bool = True,
    work_dir: Path | None = None,
    extra_args: list[str] | None = None,
) -> Path:
    """Launch MMPose training.

    Args:
        model_size: YOLOX variant (tiny, s, m, l).
        resolution: Input resolution (640, 960, 1280).
        num_gpus: Number of GPUs for distributed training.
        epochs: Override default 300 epochs.
        resume_from: Path to checkpoint to resume from.
        amp: Use automatic mixed precision.
        work_dir: Override default work directory.
        extra_args: Additional args to pass to MMPose train.py.

    Returns:
        Path to work directory with checkpoints.
    """
    config_path = get_config_path(model_size, resolution)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            f"Did you initialize the mmpose submodule? Run: git submodule update --init"
        )

    work_dir = work_dir or WORK_DIRS / f"yoloxpose_{model_size}_{resolution}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Build config overrides
    cfg_options = [
        f"train_dataloader.dataset.data_root={SOCCERNET_DIR}",
        f"val_dataloader.dataset.data_root={SOCCERNET_DIR}",
    ]
    if epochs is not None:
        cfg_options.append(f"train_cfg.max_epochs={epochs}")

    if num_gpus > 1:
        cmd = [
            "bash", str(MMPOSE_DIR / "tools" / "dist_train.sh"),
            str(config_path),
            str(num_gpus),
        ]
    else:
        cmd = [
            sys.executable,
            str(MMPOSE_DIR / "tools" / "train.py"),
            str(config_path),
        ]

    cmd.extend(["--work-dir", str(work_dir)])
    cmd.extend(["--cfg-options"] + cfg_options)

    if amp:
        cmd.append("--amp")
    if resume_from:
        cmd.extend(["--resume", resume_from])
    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"Starting training: {model_size} @ {resolution}px on {num_gpus} GPU(s)")
    logger.info(f"Config: {config_path}")
    logger.info(f"Work dir: {work_dir}")
    logger.info(f"Command: {' '.join(cmd)}")

    subprocess.run(cmd, check=True, cwd=str(MMPOSE_DIR))

    return work_dir
