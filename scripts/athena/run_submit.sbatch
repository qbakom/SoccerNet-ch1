#!/bin/bash -l
#SBATCH --job-name=synloc-submit
#SBATCH --time=01:00:00
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-submit-%j.out

export SCRATCH=/net/tscratch/people/$USER
export HF_HOME=$SCRATCH/huggingface

module load Miniconda3/23.3.1-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $SCRATCH/conda_envs/synloc

PY="$SCRATCH/conda_envs/synloc/bin/python"

CHECKPOINT=$1
MODEL=${2:-m}
RESOLUTION=${3:-960}

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: sbatch run_submit.sbatch <checkpoint.pth> [model] [resolution]"
    exit 1
fi

echo "Generating submission: YOLOX-${MODEL} @ ${RESOLUTION}px"
echo "Checkpoint: $CHECKPOINT"

cd $SCRATCH/synloc/vendor/mmpose

CONFIG="configs/body_bev_position/spiideo_soccernet/yoloxpose_${MODEL}_4xb64-300e_${RESOLUTION}.py"

# Generate challenge predictions
srun "$PY" -m torch.distributed.run --standalone --nproc_per_node=1 \
    tools/test.py $CONFIG $CHECKPOINT \
    --launcher pytorch \
    --cfg-options \
        "challenge_dataset.ann_file=annotations_fullhd/challenge_public.json" \
        "bev_challenge_evaluator.0.ann_file=data/SoccerNet/SpiideoSynLoc/annotations_fullhd/challenge_public.json" \
    --work-dir $SCRATCH/synloc/work_dirs/submission_${MODEL}_${RESOLUTION} \
    --challenge

echo "Submission generated in: $SCRATCH/synloc/work_dirs/submission_${MODEL}_${RESOLUTION}"
echo "Look for *.zip file to upload to Codabench"
ls -la $SCRATCH/synloc/work_dirs/submission_${MODEL}_${RESOLUTION}/*.zip 2>/dev/null
