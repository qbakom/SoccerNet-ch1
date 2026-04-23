#!/bin/bash
# Run this ONCE on Athena to download dataset
# Usage: bash scripts/athena/setup_data.sh

set -e
export SCRATCH=/net/tscratch/people/$USER

echo "=== Setting up SynLoc data on Athena ==="
cd $SCRATCH/synloc

# Create data symlink for MMPose
mkdir -p vendor/mmpose/data/SoccerNet
ln -sfn $SCRATCH/synloc/data/raw/SoccerNet/SpiideoSynLoc vendor/mmpose/data/SoccerNet/SpiideoSynLoc

# Cache Spiideo credentials
# Provide via env vars SPIIDEO_EMAIL / SPIIDEO_PASSWORD, or place the JSON manually at
#   ~/.cache/spiideo_research/credentials.json   (format: ["email", "password"])
mkdir -p ~/.cache/spiideo_research
CRED_FILE=~/.cache/spiideo_research/credentials.json
if [ ! -f "$CRED_FILE" ]; then
    if [ -n "${SPIIDEO_EMAIL:-}" ] && [ -n "${SPIIDEO_PASSWORD:-}" ]; then
        python -c "import json,sys; json.dump([sys.argv[1],sys.argv[2]], open(sys.argv[3],'w'))" \
            "$SPIIDEO_EMAIL" "$SPIIDEO_PASSWORD" "$CRED_FILE"
        chmod 600 "$CRED_FILE"
    else
        echo "ERROR: $CRED_FILE missing and SPIIDEO_EMAIL/SPIIDEO_PASSWORD not set."
        echo "Either export both vars or create the file manually."
        exit 1
    fi
fi

# Download dataset
echo "Downloading SynLoc dataset..."
module load Miniconda3/23.3.1-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $SCRATCH/conda_envs/synloc || {
    echo "Create conda env first: conda create -p $SCRATCH/conda_envs/synloc python=3.11 -y && conda activate $SCRATCH/conda_envs/synloc && pip install SoccerNet boto3"
    exit 1
}

python -c "
from SoccerNet.Downloader import SoccerNetDownloader
d = SoccerNetDownloader(LocalDirectory='data/raw/SoccerNet')
d.downloadDataTask(task='SpiideoSynLoc', split=['train','valid','test','challenge'], version='fullhd')
print('Download complete!')
"

# Unzip
echo "Unzipping..."
cd data/raw/SoccerNet/SpiideoSynLoc
for z in train.zip val.zip test.zip challenge.zip annotations.zip; do
    if [ -f "$z" ]; then
        echo "  Unzipping $z..."
        unzip -o -q "$z" -d .
    fi
done

# Create FullHD annotations (scale 4K → FullHD)
echo "Scaling annotations 4K → FullHD..."
cd $SCRATCH/synloc
python -c "
import json, os
SCALE = 0.5
ann_dir = 'data/raw/SoccerNet/SpiideoSynLoc/annotations'
out_dir = 'data/raw/SoccerNet/SpiideoSynLoc/annotations_fullhd'
os.makedirs(out_dir, exist_ok=True)
for fname in ['train.json', 'val.json', 'test.json', 'challenge_public.json', 'mini.json']:
    src = f'{ann_dir}/{fname}'
    if not os.path.exists(src): continue
    with open(src) as f: data = json.load(f)
    for img in data['images']:
        img['width'] = int(img['width'] * SCALE)
        img['height'] = int(img['height'] * SCALE)
    for ann in data.get('annotations', []):
        ann['bbox'] = [v * SCALE for v in ann['bbox']]
        ann['area'] = ann['area'] * SCALE * SCALE
        kpts = ann.get('keypoints', [])
        if kpts and isinstance(kpts[0], list):
            ann['keypoints'] = [[k[0]*SCALE, k[1]*SCALE, k[2]] for k in kpts]
        elif kpts:
            scaled = []
            for i in range(0, len(kpts), 3):
                scaled.extend([kpts[i]*SCALE, kpts[i+1]*SCALE, kpts[i+2]])
            ann['keypoints'] = scaled
    with open(f'{out_dir}/{fname}', 'w') as f: json.dump(data, f)
    print(f'  {fname}: {len(data[\"images\"])} images')
"

echo "=== Data setup complete ==="
echo "Images: $(ls data/raw/SoccerNet/SpiideoSynLoc/train/ | wc -l) train"
echo "Now run: sbatch scripts/athena/run_train.sbatch m 960 300"
