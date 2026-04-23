#!/bin/bash -l
#SBATCH --job-name=synloc-test-submit
#SBATCH --time=01:30:00
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-test-%j.out

# Generates predictions on Codabench "Test" phase (test split).
# Usage: sbatch run_test.sbatch <checkpoint.pth> [model] [resolution]

export SCRATCH=/net/tscratch/people/$USER
export HF_HOME=$SCRATCH/huggingface

module load Miniconda3/23.3.1-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $SCRATCH/conda_envs/synloc

CHECKPOINT=$1
MODEL=${2:-m}
RESOLUTION=${3:-960}

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: sbatch run_test.sbatch <checkpoint.pth> [model] [resolution]"
    exit 1
fi

echo "Generating TEST submission: YOLOX-${MODEL} @ ${RESOLUTION}px"
echo "Checkpoint: $CHECKPOINT"

cd $SCRATCH/synloc/vendor/mmpose

CONFIG="configs/body_bev_position/spiideo_soccernet/yoloxpose_${MODEL}_4xb64-300e_${RESOLUTION}.py"
WORK_DIR="$SCRATCH/synloc/work_dirs/test_submission_${MODEL}_${RESOLUTION}"

srun python -m torch.distributed.run --standalone --nproc_per_node=1 \
    tools/test.py $CONFIG $CHECKPOINT \
    --launcher pytorch \
    --cfg-options \
        "test_dataloader.dataset.ann_file=annotations_fullhd/test.json" \
        "test_evaluator.0.ann_file=data/SoccerNet/SpiideoSynLoc/annotations_fullhd/test.json" \
        "test_evaluator.1.ann_file=data/SoccerNet/SpiideoSynLoc/annotations_fullhd/test.json" \
        "test_evaluator.2.ann_file=data/SoccerNet/SpiideoSynLoc/annotations_fullhd/test.json" \
    --work-dir $WORK_DIR

echo "Test submission in: $WORK_DIR"
ls -la $WORK_DIR/test_submission.zip 2>/dev/null
echo "Upload test_submission.zip to Codabench Test phase."
