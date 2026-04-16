#!/bin/bash
# SoccerNet SynLoc — Generate challenge submission
# Usage: ./scripts/submit.sh CHECKPOINT [MODEL] [RESOLUTION]

set -e

CHECKPOINT=$1
MODEL=${2:-m}
RESOLUTION=${3:-960}

CONFIG="configs/body_bev_position/spiideo_soccernet/yoloxpose_${MODEL}_4xb64-300e_${RESOLUTION}.py"
WORK_DIR="../../work_dirs/submission_${MODEL}_${RESOLUTION}"

cd "$(dirname "$0")/../vendor/mmpose"

if [ -f "../../.venv/bin/activate" ]; then
    source ../../.venv/bin/activate
fi

echo "Generating challenge submission..."

python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port=29502 \
    tools/test.py $CONFIG $CHECKPOINT \
    --launcher pytorch \
    --work-dir $WORK_DIR \
    --cfg-options "val_dataloader.dataset.ann_file=annotations/challenge_public.json val_dataloader.dataset.data_prefix.img=challenge" \
    --challenge

echo "Submission zip saved in: ${WORK_DIR}"
