# SoccerNet SynLoc 2026

Single-Frame World-Coordinate Athlete Detection & Localization with Synthetic Data.

[Challenge page](https://www.soccer-net.org/challenges/2026) | [Leaderboard](https://www.codabench.org/competitions/10128/) | [Paper (VISAPP 2025)](https://research.spiideo.com/)

## Task

Detect all players in a single broadcast frame and predict their position on the pitch in meters (world coordinates). The baseline uses YOLOX-Pose with a pelvis keypoint projected to ground-plane coordinates, evaluated via mAP-LocSim.

## Quickstart

```bash
# 1. Clone with submodules
git clone --recursive git@github.com:qbakom/SoccerNet.git
cd SoccerNet

# 2. Set up environment
bash setup_env.sh
source venv/bin/activate

# 3. Download dataset
synloc data download

# 4. Validate
synloc data validate

# 5. Train (smoke test)
synloc train --model tiny --resolution 640 --epochs 5

# 6. Evaluate
synloc eval run models/checkpoint.pth --split valid

# 7. Submit
synloc eval submit models/checkpoint.pth
```

## Project Structure

```
├── data/raw/SoccerNet/SpiideoSynLoc/  <- Dataset (downloaded, not in git)
├── models/                             <- Trained checkpoints
├── reports/                            <- Evaluation metrics
│   └── figures/                        <- Visualizations
├── notebooks/                          <- Exploration notebooks
├── src/synloc/                         <- Source code
│   ├── cli.py                          <- CLI entry point
│   ├── config.py                       <- Paths and configuration
│   ├── dataset.py                      <- Data download & validation
│   ├── visualization.py                <- Pitch plotting utilities
│   └── modeling/
│       ├── train.py                    <- Training wrapper
│       └── predict.py                  <- Evaluation & submission
├── vendor/mmpose/                      <- Spiideo MMPose fork (submodule)
├── setup_env.sh                        <- One-shot environment setup
├── Makefile                            <- Make targets
└── requirements.txt                    <- Python dependencies
```

## Available Models

| Model | Resolution | Baseline mAP-LocSim | GFLOPs |
|-------|-----------|---------------------|--------|
| YOLOX-tiny | 640 | ~60.6 | 10.3 |
| YOLOX-small | 640 | ~70.0 | — |
| YOLOX-m | 960 | **76.17** | 108.0 |
| YOLOX-l | 960 | ~79.3 | — |

## CLI Reference

```
synloc data download       # Download dataset
synloc data validate       # Check dataset integrity
synloc train               # Train (default: m @ 960px)
synloc eval run <ckpt>     # Evaluate on valid/test
synloc eval submit <ckpt>  # Generate submission zip
synloc visualize           # Plot pitch positions
```
