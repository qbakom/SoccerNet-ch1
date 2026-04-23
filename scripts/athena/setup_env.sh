#!/bin/bash
# Idempotent conda env setup on Athena.
# Usage: bash scripts/athena/setup_env.sh
# Prereq: repo cloned to $SCRATCH/synloc with submodules initialized.

set -euo pipefail
export SCRATCH="/net/tscratch/people/$USER"

PROJECT_DIR="$SCRATCH/synloc"
ENV_PATH="$SCRATCH/conda_envs/synloc"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: $PROJECT_DIR does not exist."
    echo "Clone first:  git clone --recurse-submodules <repo> $PROJECT_DIR"
    exit 1
fi

if [ ! -d "$PROJECT_DIR/vendor/mmpose" ] || [ -z "$(ls -A "$PROJECT_DIR/vendor/mmpose" 2>/dev/null)" ]; then
    echo "ERROR: vendor/mmpose submodule missing or empty."
    echo "Run:  git -C $PROJECT_DIR submodule update --init --recursive"
    exit 1
fi

echo "=== SynLoc env setup on Athena ==="
module load Miniconda3/23.3.1-0
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "$ENV_PATH" ]; then
    echo "Creating conda env at $ENV_PATH"
    conda create -p "$ENV_PATH" python=3.11 -y
fi

# Use env binaries directly — do NOT rely on `conda activate` / PATH resolution.
# Also block user-site fallback so nothing ever leaks to ~/.local.
PY="$ENV_PATH/bin/python"
PIP="$ENV_PATH/bin/pip"
export PYTHONNOUSERSITE=1
export PIP_USER=0
export PIP_NO_USER=1

# Short-circuit if env is already fully provisioned.
if "$PY" -c "import mmpose, mmdet, mmcv, torch, SoccerNet, sskit" 2>/dev/null; then
    echo "Env already provisioned — nothing to do."
    exit 0
fi

cd "$PROJECT_DIR"

echo "[1/5] PyTorch 2.1.0 + CUDA 12.1"
"$PIP" install torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

echo "[2/5] MMEngine + MMCV 2.1.0"
"$PIP" install mmengine "mmcv==2.1.0" \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

echo "[3/5] Core deps (pinned: numpy<2, setuptools<70)"
"$PIP" install mmdet "numpy<2" "setuptools<70" cython

echo "[4/5] Build deps (no-build-isolation — need numpy<2 present)"
"$PIP" install --no-build-isolation xtcocotools chumpy

echo "[5/5] Runtime + MMPose (editable from vendor/)"
"$PIP" install SoccerNet sskit json_tricks munkres scipy \
    opencv-python pillow matplotlib boto3
"$PIP" install --no-build-isolation -e vendor/mmpose

echo ""
echo "=== Env ready at $ENV_PATH ==="
echo "Activate:  conda activate $ENV_PATH"
echo "Next:      bash scripts/athena/setup_data.sh"
