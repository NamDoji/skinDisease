# CLAUDE.md — skinDisease

## Project Overview

Multi-class skin disease classifier (23 categories) built with EfficientNet-B0 fine-tuned on the DermNet dataset (~19,500 images). Includes Grad-CAM explainability for visualising model attention. Secondary backbone support for ResNet-50 and ConvNeXt-Tiny via `timm`.

## Repository

- **GitHub**: https://github.com/NamDoji/skinDisease
- **Commit as**: NamDoji
- **Email**: use the GitHub account email for NamDoji

## Tech Stack

- Python 3.10+
- PyTorch 2.x + torchvision
- `timm` for backbone models
- Albumentations for augmentation
- `grad-cam` library for explainability
- OmegaConf / YAML for config
- pytest + ruff for testing/linting

## Project Structure

```
skinDisease/
├── configs/default.yaml      # all hyperparameters (single source of truth)
├── data/
│   ├── processed/            # generated CSVs — do not commit
│   └── samples/              # demo images
├── src/dermnet/              # core library package
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── transforms.py
│   ├── trainer.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── utils.py
├── scripts/                  # entry-point scripts (train, predict, prepare)
├── notebooks/                # EDA, training demo, results analysis
├── tests/                    # pytest unit tests
└── Makefile                  # convenience targets
```

## Key Design Decisions

- **Backbone**: EfficientNet-B0 (default) — best accuracy/param ratio for medical imaging
- **Fine-tuning**: 2-phase (freeze head → unfreeze all) to protect ImageNet weights
- **Class imbalance**: `WeightedRandomSampler` + `CrossEntropyLoss(weight=...)`
- **LR schedule**: Linear warmup + cosine annealing
- **Mixed precision**: AMP `float16` with `GradScaler`
- **Explainability**: Grad-CAM via `grad-cam` library

## Common Commands

```bash
# Install
pip install -e ".[dev,notebook]"

# Download dataset (requires ~/.kaggle/kaggle.json)
make download

# Prepare splits
make prepare

# Train (EfficientNet-B0 default)
make train

# Predict with Grad-CAM
make predict

# Tests
make test
```

## Config

All hyperparameters live in `configs/default.yaml`. Override at CLI with OmegaConf syntax:
```bash
python scripts/train.py backbone=resnet50 training.epochs=30
```

## Dataset

DermNet on Kaggle — ~19,500 images, 23 classes, CC BY-NC-ND 4.0 license.
Do **not** commit raw or processed data to the repository.

## Git Conventions

- Commit and push as **NamDoji** (`nam@itsol.vn`) — never as `duongphamminhdung`
- Push target: `https://github.com/NamDoji/skinDisease`
- **Never** add Claude as a co-author or contributor in any commit message
- Write short, casual commit messages (e.g. `add gradcam`, `fix trainer bug`)
- Do not commit `data/`, `outputs/`, or `*.pt` checkpoint files
- Run `make test` and `ruff check src scripts` before committing
