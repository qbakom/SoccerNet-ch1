# SoccerNet SynLoc — Training Plan

## Hardware
- **Local (kuba):** 1× RTX 4090 (24GB) + 1× Titan V (12GB)
- **Supercomputer:** 1-2× A100 (80GB)
- **Alternative:** 8× RTX 4090 (na innym serwerze)

## Training Strategy (5 runs, priority order)

### Run 1: Baseline YOLOX-m@960 (PRIORITY)
```bash
# On A100 (1 GPU, 80GB → batch=64)
./scripts/train_baseline.sh 1 m 960 300
# Time estimate: ~12-16h on A100
# Expected: ~76 mAP-LocSim (Spiideo baseline)
```

### Run 2: Large model YOLOX-l@960
```bash
./scripts/train_baseline.sh 1 l 960 300
# Time estimate: ~24-30h on A100
# Expected: ~78-79 mAP-LocSim
```

### Run 3: High-res YOLOX-m@1280
```bash
./scripts/train_baseline.sh 1 m 1280 300
# Time estimate: ~24h on A100
# Expected: ~77-78 mAP-LocSim
```

### Run 4: Large + High-res YOLOX-l@1280
```bash
./scripts/train_baseline.sh 1 l 1280 300
# Time estimate: ~48h on A100
# Expected: ~79-80 mAP-LocSim
```

### Run 5: Small fast model for ensembling YOLOX-s@960
```bash
./scripts/train_baseline.sh 1 s 960 300
# Time estimate: ~8-10h on A100
# Expected: ~74 mAP-LocSim
```

## After Training

### Evaluate all models
```bash
for ckpt in work_dirs/*/epoch_300.pth; do
    ./scripts/evaluate.sh $ckpt
done
```

### Ensemble best models (post-processing)
TODO: implement Weighted Box Fusion (WBF) from multiple models

### TTA (Test-Time Augmentation)
TODO: implement multi-scale + flip inference

### Generate submission
```bash
./scripts/submit.sh work_dirs/best_model/epoch_300.pth m 960
# Upload zip to https://www.codabench.org/competitions/10155/
```

## Multi-GPU (if 2+ A100 available)
```bash
# 2× A100
./scripts/train_baseline.sh 2 l 1280 300
# ~2× faster (24h instead of 48h)
```

## Data Transfer to Supercomputer
```bash
# Pack dataset (no zip files, just images + annotations)
tar czf synloc_data.tar.gz -C data/raw/SoccerNet/SpiideoSynLoc \
    train/ val/ test/ challenge/ annotations/
# Size: ~21 GB compressed

# Pack repo (without data)
tar czf synloc_code.tar.gz \
    --exclude='data/' --exclude='.venv/' --exclude='work_dirs/' \
    --exclude='*.zip' --exclude='__pycache__' \
    -C /srv/soccernet-synloc .
```
