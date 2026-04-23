#!/bin/bash -l
#SBATCH --job-name=synloc-train
#SBATCH --time=24:00:00
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --output=slurm-%j.out

# --- Environment ---
export SCRATCH=/net/tscratch/people/$USER
export HF_HOME=$SCRATCH/huggingface
export PYTHONPYCACHEPREFIX=$SCRATCH/pycache

# Load conda
module load Miniconda3/23.3.1-0
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate or create env
ENV_PATH=$SCRATCH/conda_envs/synloc
if [ ! -d "$ENV_PATH" ]; then
    echo "Creating conda env at $ENV_PATH ..."
    conda create -p $ENV_PATH python=3.11 -y
fi
conda activate $ENV_PATH

# Install deps (only first time)
if ! python -c "import mmpose" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
    pip install mmengine "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
    pip install mmdet "numpy<2" "setuptools<70" cython
    pip install --no-build-isolation xtcocotools chumpy
    pip install SoccerNet sskit json_tricks munkres scipy opencv-python pillow matplotlib
    cd $SCRATCH/synloc/vendor/mmpose && pip install --no-build-isolation -e .
    cd $SCRATCH/synloc
fi

# --- Parse arguments ---
MODEL=${1:-m}
RESOLUTION=${2:-960}
EPOCHS=${3:-300}

echo "=========================================="
echo "Training YOLOX-${MODEL} @ ${RESOLUTION}px"
echo "Epochs: ${EPOCHS}"
echo "Node: $(hostname), GPU: $(nvidia-smi -L)"
echo "=========================================="

# --- Training ---
cd $SCRATCH/synloc/vendor/mmpose

CONFIG="configs/body_bev_position/spiideo_soccernet/yoloxpose_${MODEL}_4xb64-300e_${RESOLUTION}.py"
WORK_DIR="$SCRATCH/synloc/work_dirs/yoloxpose_${MODEL}_${RESOLUTION}"

# Batch size per model/resolution
case "${MODEL}_${RESOLUTION}" in
    tiny_640) BATCH=64 ;;
    tiny_960) BATCH=32 ;;
    s_640)    BATCH=48 ;;
    s_960)    BATCH=24 ;;
    m_640)    BATCH=32 ;;
    m_960)    BATCH=16 ;;
    m_1280)   BATCH=8  ;;
    l_640)    BATCH=16 ;;
    l_960)    BATCH=8  ;;
    l_1280)   BATCH=4  ;;
    *)        BATCH=8  ;;
esac

srun python -m torch.distributed.run --standalone --nproc_per_node=1 \
    tools/train.py $CONFIG \
    --launcher pytorch \
    --cfg-options \
        train_cfg.max_epochs=$EPOCHS \
        train_dataloader.batch_size=$BATCH \
        train_dataloader.num_workers=8 \
        train_dataloader.dataset.ann_file=annotations_fullhd/train.json \
        val_dataloader.dataset.ann_file=annotations_fullhd/val.json \
        "val_evaluator.0.ann_file=data/SoccerNet/SpiideoSynLoc/annotations_fullhd/val.json" \
        "val_evaluator.1.ann_file=data/SoccerNet/SpiideoSynLoc/annotations_fullhd/val.json" \
        "val_evaluator.2.ann_file=data/SoccerNet/SpiideoSynLoc/annotations_fullhd/val.json" \
    --amp \
    --work-dir $WORK_DIR

echo "Training complete. Checkpoints in: $WORK_DIR"
