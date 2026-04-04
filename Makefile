.PHONY: setup data validate train eval submit visualize clean help

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON := python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up the environment (venv, deps, submodules)
setup:
	bash setup_env.sh

## Download SoccerNet SynLoc dataset (fullhd)
data:
	$(PYTHON) -m synloc.cli data download

## Validate dataset integrity
validate:
	$(PYTHON) -m synloc.cli data validate

## Train YOLOX-Pose model (default: medium @ 960px)
train:
	$(PYTHON) -m synloc.cli train

## Train tiny model for quick smoke test (5 epochs)
train-smoke:
	$(PYTHON) -m synloc.cli train --model tiny --resolution 640 --epochs 5

## Evaluate on validation set
eval:
	$(PYTHON) -m synloc.cli eval run $(CHECKPOINT) --split valid

## Generate challenge submission
submit:
	$(PYTHON) -m synloc.cli eval submit $(CHECKPOINT)

## Generate visualizations
visualize:
	$(PYTHON) -m synloc.cli visualize

## Delete compiled Python files and caches
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

help:
	@echo "Available commands:"
	@echo ""
	@echo "  make setup          Set up environment (venv, deps, submodules)"
	@echo "  make data           Download SoccerNet SynLoc dataset"
	@echo "  make validate       Validate dataset integrity"
	@echo "  make train          Train YOLOX-Pose (default: medium @ 960px)"
	@echo "  make train-smoke    Quick smoke test (tiny @ 640px, 5 epochs)"
	@echo "  make eval           Evaluate checkpoint (set CHECKPOINT=path/to/model.pth)"
	@echo "  make submit         Generate submission zip (set CHECKPOINT=path/to/model.pth)"
	@echo "  make visualize      Generate pitch visualizations"
	@echo "  make clean          Remove Python cache files"
