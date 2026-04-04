"""Project paths and configuration."""

from pathlib import Path

# Project root (two levels up from this file: src/synloc/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
SOCCERNET_DIR = DATA_RAW / "SoccerNet" / "SpiideoSynLoc"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"

# Reports
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Vendor (submodules)
VENDOR_DIR = PROJECT_ROOT / "vendor"
MMPOSE_DIR = VENDOR_DIR / "mmpose"
MMPOSE_CONFIGS = MMPOSE_DIR / "configs" / "body_bev_position" / "spiideo_soccernet"

# MMPose work dirs (training output)
WORK_DIRS = PROJECT_ROOT / "work_dirs"

# Available model configs
YOLOX_CONFIGS = {
    (size, res): f"yoloxpose_{size}_4xb64-300e_{res}.py"
    for size in ("tiny", "s", "m", "l")
    for res in (640, 960, 1280)
}

# Default training config
DEFAULT_MODEL_SIZE = "m"
DEFAULT_RESOLUTION = 960


def get_config_path(size: str = DEFAULT_MODEL_SIZE, resolution: int = DEFAULT_RESOLUTION) -> Path:
    """Get the MMPose config file path for a given model size and resolution."""
    key = (size, resolution)
    if key not in YOLOX_CONFIGS:
        available = [f"{s}-{r}" for s, r in YOLOX_CONFIGS]
        raise ValueError(f"Invalid config {size}-{resolution}. Available: {available}")
    return MMPOSE_CONFIGS / YOLOX_CONFIGS[key]
