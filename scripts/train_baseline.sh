#!/bin/bash
# SoccerNet SynLoc — Full Baseline Training
# Usage: ./scripts/train_baseline.sh [NUM_GPUS] [MODEL] [RESOLUTION]
# Example: ./scripts/train_baseline.sh 4 m 960

set -e

NUM_GPUS=${1:-1}
MODEL=${2:-m}
RESOLUTION=${3:-960}
EPOCHS=${4:-300}

CONFIG="configs/body_bev_position/spiideo_soccernet/yoloxpose_${MODEL}_4xb64-300e_${RESOLUTION}.py"
WORK_DIR="../../work_dirs/yoloxpose_${MODEL}_${RESOLUTION}_${NUM_GPUS}gpu"

cd "$(dirname "$0")/../vendor/mmpose"

# Activate venv if exists
if [ -f "../../.venv/bin/activate" ]; then
    source ../../.venv/bin/activate
fi

echo "=========================================="
echo "SynLoc Training: YOLOX-${MODEL} @ ${RESOLUTION}px"
echo "GPUs: ${NUM_GPUS}, Epochs: ${EPOCHS}"
echo "Config: ${CONFIG}"
echo "Work dir: ${WORK_DIR}"
echo "=========================================="

# Calculate batch size per GPU
# RTX 4090 (24GB): tiny=64, s=32, m=16, l=8 at 640px
# A100 (80GB): tiny=64, s=64, m=64, l=32 at 960px
if [ "$RESOLUTION" -ge 960 ]; then
    case $MODEL in
        tiny) BATCH=32 ;;
        s)    BATCH=24 ;;
        m)    BATCH=16 ;;
        l)    BATCH=8  ;;
    esac
else
    case $MODEL in
        tiny) BATCH=64 ;;
        s)    BATCH=48 ;;
        m)    BATCH=32 ;;
        l)    BATCH=16 ;;
    esac
fi

echo "Batch size per GPU: ${BATCH}"

CFG_OPTIONS="train_cfg.max_epochs=${EPOCHS} train_dataloader.batch_size=${BATCH} train_dataloader.num_workers=8"

if [ "$NUM_GPUS" -gt 1 ]; then
    PORT=${MASTER_PORT:-29500}
    python -m torch.distributed.run \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$PORT \
        tools/train.py $CONFIG \
        --launcher pytorch \
        --amp \
        --work-dir $WORK_DIR \
        --cfg-options $CFG_OPTIONS
else
    python -m torch.distributed.run \
        --nproc_per_node=1 \
        --master_port=29500 \
        tools/train.py $CONFIG \
        --launcher pytorch \
        --amp \
        --work-dir $WORK_DIR \
        --cfg-options $CFG_OPTIONS
fi

echo "Training complete. Checkpoints in: ${WORK_DIR}"
