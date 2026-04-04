#!/bin/bash
set -e

echo "=== SoccerNet SynLoc Environment Setup ==="

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Upgrade pip
pip install -U pip setuptools wheel

# Install requirements
pip install -r requirements.txt

# Initialize submodules
echo "Initializing git submodules..."
git submodule update --init --recursive

# Install MMPose in development mode
if [ -d "vendor/mmpose" ]; then
    echo "Installing MMPose..."
    pip install -e vendor/mmpose
fi

echo ""
echo "=== Setup complete ==="
echo "Activate environment: source venv/bin/activate"
echo "Download data:        synloc data download"
echo "Validate data:        synloc data validate"
echo "Start training:       synloc train --model tiny --resolution 640 --epochs 5"
