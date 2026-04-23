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
export WANDB_DIR=$SCRATCH/wandb
export WANDB_CACHE_DIR=$SCRATCH/.cache/wandb
export WANDB_CONFIG_DIR=$SCRATCH/.config/wandb
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

# Ensure env exists (idempotent — no-op if already provisioned)
bash $SCRATCH/synloc/scripts/athena/setup_env.sh

module load Miniconda3/23.3.1-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $SCRATCH/conda_envs/synloc

# Use env python explicitly — srun does not inherit conda-activated PATH reliably
PY="$SCRATCH/conda_envs/synloc/bin/python"

# --- Parse arguments ---
MODEL=${1:-m}
RESOLUTION=${2:-960}
EPOCHS=${3:-300}
LOAD_FROM=${4:-}     # optional: path to pretrained .pth
WANDB_PROJECT=${5:-} # optional: wandb project name, enables wandb logging

RUN_NAME="yoloxpose_${MODEL}_${RESOLUTION}_$(date +%m%d_%H%M)"

echo "=========================================="
echo "Training YOLOX-${MODEL} @ ${RESOLUTION}px"
echo "Epochs: ${EPOCHS}"
echo "Load from: ${LOAD_FROM:-<none, from scratch>}"
echo "Wandb:    ${WANDB_PROJECT:-<disabled>}"
echo "Run:      $RUN_NAME"
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

CFG_OPTS=(
    "train_cfg.max_epochs=$EPOCHS"
    "train_dataloader.batch_size=$BATCH"
    "train_dataloader.num_workers=8"
    "train_dataloader.dataset.ann_file=annotations_fullhd/train.json"
    "val_dataloader.dataset.ann_file=annotations_fullhd/val.json"
    "val_evaluator.0.ann_file=data/SoccerNet/SpiideoSynLoc/annotations_fullhd/val.json"
    "val_evaluator.1.ann_file=data/SoccerNet/SpiideoSynLoc/annotations_fullhd/val.json"
    "val_evaluator.2.ann_file=data/SoccerNet/SpiideoSynLoc/annotations_fullhd/val.json"
)
if [ -n "$LOAD_FROM" ]; then
    CFG_OPTS+=("load_from=$LOAD_FROM")
fi

if [ -n "$WANDB_PROJECT" ]; then
    CFG_OPTS+=(
        "visualizer.vis_backends=[dict(type='LocalVisBackend'),dict(type='WandbVisBackend',init_kwargs=dict(project='$WANDB_PROJECT',name='$RUN_NAME'))]"
    )
fi

srun "$PY" -m torch.distributed.run --standalone --nproc_per_node=1 \
    tools/train.py $CONFIG \
    --launcher pytorch \
    --cfg-options "${CFG_OPTS[@]}" \
    --amp \
    --work-dir $WORK_DIR

echo "Training complete. Checkpoints in: $WORK_DIR"
