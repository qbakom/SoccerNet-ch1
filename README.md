# SoccerNet SynLoc 2026

Single-frame world-coordinate athlete detection and localization from synthetic broadcast data.

[Challenge](https://www.soccer-net.org/challenges/2026) | [Codabench](https://www.codabench.org/competitions/10155/) | [Baseline paper (VISAPP 2025)](https://research.spiideo.com/)

## Task

Detect all players in a single broadcast frame and predict their **position on the pitch in meters** (world coordinates). Evaluated via mAP-LocSim.

## Approach

YOLOX-Pose detects players and predicts a `pelvis_ground` keypoint. That keypoint is projected from image coordinates to world coordinates using the per-image camera matrix and undistortion polynomial via [sskit](https://pypi.org/project/sskit/).

## Setup

```bash
git clone --recurse-submodules git@github.com:qbakom/SoccerNet-ch1.git
cd SoccerNet-ch1
python3 -m venv .venv && source .venv/bin/activate

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install mmengine "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install mmdet "numpy<2" "setuptools<70" cython
pip install --no-build-isolation xtcocotools chumpy
pip install SoccerNet sskit json_tricks munkres scipy opencv-python-headless pillow matplotlib boto3
cd vendor/mmpose && pip install --no-build-isolation -e . && cd ../..
```

For Athena (Cyfronet): `bash scripts/athena/setup_env.sh`

## Dataset

```bash
python -c "
from SoccerNet.Downloader import SoccerNetDownloader
d = SoccerNetDownloader(LocalDirectory='data/raw/SoccerNet')
d.downloadDataTask(task='SpiideoSynLoc', split=['train','valid','test','challenge'], version='fullhd')
"
```

Images are FullHD (1920x1080). Annotations ship in 4K and **must be scaled to FullHD** before training — see `scripts/athena/setup_data.sh` for the scaling step.

## Training

All training runs from `vendor/mmpose/`:

```bash
# Local (single GPU)
bash scripts/train_baseline.sh 1 m 960 300

# Athena (SLURM)
sbatch scripts/athena/run_train.sbatch m 960 300

# Fine-tune from pretrained checkpoint
sbatch scripts/athena/run_train.sbatch m 1280 100 /path/to/pretrained.pth
```

## Evaluation

```bash
bash scripts/evaluate.sh work_dirs/yoloxpose_m_960/best.pth m 960 val
```

## Submission

Generate submission zip for Codabench:

```bash
# Test phase (test split)
sbatch scripts/athena/run_test.sbatch /path/to/checkpoint.pth m 960

# Challenge phase (challenge split)
sbatch scripts/athena/run_submit.sbatch /path/to/checkpoint.pth m 960
```

Or locally: `bash scripts/submit.sh /path/to/checkpoint.pth m 960`

## Project structure

```
├── vendor/mmpose/          <- Spiideo MMPose fork (submodule)
├── scripts/
│   ├── train_baseline.sh   <- Local training wrapper
│   ├── evaluate.sh         <- Evaluation on val/test
│   ├── submit.sh           <- Local submission generation
│   ├── make_submission.py  <- Predictions .pkl → Codabench zip
│   ├── analytical_localize.py
│   ├── tta_inference.py    <- Test-time augmentation
│   └── athena/             <- SLURM scripts for Cyfronet A100
├── reports/                <- Analysis and figures
├── data/                   <- Dataset (not in git)
├── models/                 <- Checkpoints (not in git)
└── work_dirs/              <- Training outputs (not in git)
```

## Baseline results

| Model | Resolution | mAP-LocSim |
|-------|-----------|------------|
| YOLOX-tiny | 640 | ~60.6 |
| YOLOX-s | 640 | ~70.0 |
| YOLOX-m | 960 | **76.17** |

## Acknowledgements

This research was supported in part by PL-Grid Infrastructure grant nr PLG/2025/018167.
