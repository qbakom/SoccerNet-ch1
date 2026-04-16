#!/bin/bash
# SoccerNet SynLoc — Evaluate checkpoint on val/test
# Usage: ./scripts/evaluate.sh CHECKPOINT [MODEL] [RESOLUTION] [SPLIT]
# Example: ./scripts/evaluate.sh work_dirs/yoloxpose_m_960/epoch_300.pth m 960 val

set -e

CHECKPOINT=$1
MODEL=${2:-m}
RESOLUTION=${3:-960}
SPLIT=${4:-val}

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: $0 CHECKPOINT [MODEL] [RESOLUTION] [SPLIT]"
    echo "  CHECKPOINT: path to .pth file"
    echo "  MODEL: tiny, s, m, l (default: m)"
    echo "  RESOLUTION: 640, 960, 1280 (default: 960)"
    echo "  SPLIT: val, test (default: val)"
    exit 1
fi

CONFIG="configs/body_bev_position/spiideo_soccernet/yoloxpose_${MODEL}_4xb64-300e_${RESOLUTION}.py"

cd "$(dirname "$0")/../vendor/mmpose"

if [ -f "../../.venv/bin/activate" ]; then
    source ../../.venv/bin/activate
fi

echo "=========================================="
echo "Evaluating: YOLOX-${MODEL} @ ${RESOLUTION}px on ${SPLIT}"
echo "Checkpoint: ${CHECKPOINT}"
echo "=========================================="

# Override val dataloader to use requested split
CFG_OPTIONS="val_dataloader.dataset.ann_file=annotations/${SPLIT}.json val_dataloader.dataset.data_prefix.img=${SPLIT}"

python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port=29501 \
    tools/test.py $CONFIG $CHECKPOINT \
    --launcher pytorch \
    --cfg-options $CFG_OPTIONS \
    --out ../../reports/metrics_${MODEL}_${RESOLUTION}_${SPLIT}.json

echo "Metrics saved to reports/metrics_${MODEL}_${RESOLUTION}_${SPLIT}.json"
