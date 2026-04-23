#!/bin/bash
# Full environment setup on Athena
# Usage: bash scripts/athena/setup_env.sh

set -e
export SCRATCH=/net/tscratch/people/$USER

echo "=== Setting up SynLoc conda env on Athena ==="

# Load conda
module load Miniconda3/23.3.1-0
eval "$(conda shell.bash hook)"

# Create env if missing
if [ ! -d "$SCRATCH/conda_envs/synloc" ]; then
    echo "Creating conda env at $SCRATCH/conda_envs/synloc..."
    conda create -p $SCRATCH/conda_envs/synloc python=3.11 -y
fi

conda activate $SCRATCH/conda_envs/synloc

cd $SCRATCH/synloc

# Install torch+cuda
echo "Installing PyTorch 2.1.0 + CUDA 12.1..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Install MMEngine/MMCV
echo "Installing MMEngine + MMCV..."
pip install mmengine "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# Core deps (pinned versions critical)
echo "Installing core deps..."
pip install mmdet "numpy<2" "setuptools<70" cython

# Build deps (no build isolation — need numpy<2 already installed)
echo "Installing build deps..."
pip install --no-build-isolation xtcocotools chumpy

# Runtime deps
echo "Installing SoccerNet + sskit..."
pip install SoccerNet sskit json_tricks munkres scipy opencv-python pillow matplotlib boto3

# MMPose editable from vendor
echo "Installing MMPose (editable)..."
cd vendor/mmpose && pip install --no-build-isolation -e . && cd ../..

echo ""
echo "=== Env setup complete ==="
echo "Next step: bash scripts/athena/setup_data.sh"
