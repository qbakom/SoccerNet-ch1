#!/bin/bash
# SoccerNet SynLoc — Generate submission (test or challenge phase)
# Usage: ./scripts/submit.sh CHECKPOINT [MODEL] [RESOLUTION] [PHASE]
#   PHASE: test (default) | challenge
#
# Fixed 2026-04-24 v3:
# - val_dataloader na annotations_fullhd/val.json → runner.val() liczy poprawnie F1-optimal
#   score_threshold do metadata.json (zamiast fallback 0.05)
# - Zawsze używamy --challenge flag (wywołuje bev_challenge_evaluator z format_only=True
#   który produkuje poprawny submission format: id + area + visibility=2, w 4K coords)
# - Dla TEST phase podmieniamy challenge_dataset na test split (4K annotations)

set -e

CHECKPOINT=$1
MODEL=${2:-m}
RESOLUTION=${3:-960}
PHASE=${4:-test}

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: $0 CHECKPOINT [MODEL] [RESOLUTION] [PHASE]"
    echo "  PHASE: test (default) | challenge"
    exit 1
fi

CONFIG="configs/body_bev_position/spiideo_soccernet/yoloxpose_${MODEL}_4xb64-300e_${RESOLUTION}.py"
WORK_DIR="../../work_dirs/submission_${MODEL}_${RESOLUTION}_${PHASE}"

cd "$(dirname "$0")/../vendor/mmpose"

if [ -f "../../.venv/bin/activate" ]; then
    source ../../.venv/bin/activate
fi

ANN_ROOT="/srv/soccernet-synloc/data/raw/SoccerNet/SpiideoSynLoc"

# val_dataloader + val_evaluator → annotations_fullhd/val.json
# Dlaczego fullhd? Model był trenowany na fullhd, więc F1 threshold liczony na fullhd
# jest semantycznie poprawny. (Dla samej rozsądnej wartości threshold skala nie ma
# znaczenia — to jest punkt odcięcia score'u detekcji, nie coord.)
COMMON_OVERRIDES=(
    "val_dataloader.dataset.ann_file=annotations_fullhd/val.json"
    "val_dataloader.dataset.data_prefix.img=val"
    "val_evaluator.0.ann_file=${ANN_ROOT}/annotations_fullhd/val.json"
    "val_evaluator.1.ann_file=${ANN_ROOT}/annotations_fullhd/val.json"
    "val_evaluator.2.ann_file=${ANN_ROOT}/annotations_fullhd/val.json"
)

# KLUCZOWE: Dla phase=test podmieniamy challenge_dataset na test split (4K).
# Dla phase=challenge — zostawiamy domyślne challenge_dataset (annotations/challenge_public.json).
# --challenge flag w test.py robi:
#   cfg.test_evaluator = cfg.bev_challenge_evaluator  (format_only=True → id+area+vis=2)
#   cfg.test_dataloader['dataset'].update(cfg.challenge_dataset)
# Dzięki temu output jest w 4K coords (annotations/ nie annotations_fullhd/) i w poprawnym
# submission format.
if [ "$PHASE" = "test" ]; then
    PHASE_OVERRIDES=(
        "challenge_dataset.ann_file=annotations/test.json"
        "challenge_dataset.data_prefix.img=test"
        "test_evaluator.0.ann_file=${ANN_ROOT}/annotations/test.json"
    )
else
    PHASE_OVERRIDES=(
        "challenge_dataset.ann_file=annotations/challenge_public.json"
        "challenge_dataset.data_prefix.img=challenge"
        "test_evaluator.0.ann_file=${ANN_ROOT}/annotations/challenge_public.json"
    )
fi

echo "=========================================="
echo "Generating $PHASE submission"
echo "Checkpoint: $CHECKPOINT"
echo "Config:     $CONFIG"
echo "Work dir:   $WORK_DIR"
echo "Coords:     4K (via annotations/$PHASE.json)"
echo "=========================================="

python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port=29502 \
    tools/test.py $CONFIG $CHECKPOINT \
    --launcher pytorch \
    --work-dir $WORK_DIR \
    --cfg-options "${COMMON_OVERRIDES[@]}" "${PHASE_OVERRIDES[@]}" \
    --challenge

# test.py zawsze pisze challenge_submission.zip gdy --challenge, w CWD (vendor/mmpose/).
ZIP_SRC="challenge_submission.zip"
if [ -f "$ZIP_SRC" ]; then
    mkdir -p "$WORK_DIR"
    mv "$ZIP_SRC" "$WORK_DIR/submission.zip"
    [ -f results.json ] && cp results.json "$WORK_DIR/"
    [ -f metadata.json ] && cp metadata.json "$WORK_DIR/"
    echo ""
    echo "Submission:  $WORK_DIR/submission.zip"
    echo "Metadata:    $(cat "$WORK_DIR/metadata.json" 2>/dev/null || echo N/A)"
    echo "Coords scale: 4K (per annotations/$PHASE.json width=3840)"
else
    echo "WARN: $ZIP_SRC not found in $(pwd)"
fi
