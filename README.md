# 🔬 Skin Disease Classifier

> Multi-class skin disease classification across **23 categories** using EfficientNet-B0 fine-tuned on the DermNet dataset (~19,500 images). Includes Grad-CAM explainability to visualise model attention.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)

---

## Results

| Model | Test Accuracy | Top-5 Accuracy | Macro F1 |
|-------|:------------:|:--------------:|:--------:|
| EfficientNet-B0 | ~75% | ~93% | ~71% |
| ResNet-50 | ~73% | ~92% | ~69% |

*Expected ranges — actual results depend on hardware and random seed.*

---

## Disease Classes (23)

Including: **Eczema**, **Psoriasis / Lichen Planus**, **Fungal Infections (Tinea/Ringworm/Candidiasis)**, Acne, Melanoma, Atopic Dermatitis, Lupus, Nail Disease, Vasculitis, Urticaria, and more.

---

## Quick Start

### 1. Install

```bash
pip install -e ".[dev,notebook]"
# or simply:
bash install.sh
```

### 2. Download Dataset

Requires a [Kaggle API key](https://www.kaggle.com/docs/api) configured at `~/.kaggle/kaggle.json`.

```bash
make download
# or:
bash scripts/download_data.sh
```

### 3. Prepare Data Splits

```bash
make prepare
# Outputs: data/processed/train.csv, val.csv, test.csv, classes.csv
```

### 4. Train

```bash
make train                        # EfficientNet-B0 (default)
make train-resnet                 # ResNet-50
python scripts/train.py --backbone convnext_tiny --epochs 30
```

### 5. Run Inference

```bash
make predict                      # predict on data/samples/sample.jpg
python scripts/predict.py --image path/to/skin.jpg \
                           --checkpoint outputs/checkpoints/best.pt \
                           --top_k 5 --gradcam
```

### 6. Generate Grad-CAM Grid

```bash
make gradcam
# Saves: outputs/figures/gradcam_grid.png
```

### 7. Run Tests

```bash
make test
```

---

## Project Structure

```
skinDisease/
├── configs/
│   └── default.yaml          # all hyperparameters
├── data/
│   ├── processed/            # train/val/test CSVs (generated)
│   └── samples/              # demo images
├── src/dermnet/
│   ├── config.py             # YAML config loader
│   ├── dataset.py            # DermNetDataset + WeightedRandomSampler
│   ├── model.py              # EfficientNet-B0 + custom head
│   ├── transforms.py         # Albumentations pipelines
│   ├── trainer.py            # 2-phase fine-tuning loop
│   ├── evaluate.py           # metrics + confusion matrix
│   ├── gradcam.py            # Grad-CAM explainability
│   └── utils.py              # seeding, logging, plotting
├── scripts/
│   ├── download_data.sh      # kaggle download
│   ├── prepare_data.py       # build splits
│   ├── train.py              # training entry point
│   └── predict.py            # single-image inference
├── notebooks/
│   ├── 01_eda.ipynb          # class distribution, sample images
│   ├── 02_training_demo.ipynb # augmentations, model summary
│   └── 03_results_analysis.ipynb # metrics, confusion matrix, Grad-CAM
├── tests/                    # pytest unit tests
└── Makefile                  # convenience commands
```

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Backbone | EfficientNet-B0 | Best accuracy/params ratio; ~5.3M params; proven on medical imaging |
| Library | `timm` | Unified API for 700+ models; easy backbone swapping |
| Class imbalance (sampling) | `WeightedRandomSampler` | Rare classes appear every epoch |
| Class imbalance (loss) | `CrossEntropyLoss(weight=...)` | Stronger gradient signal for rare classes |
| Fine-tuning | 2-phase (freeze → unfreeze) | Protects ImageNet weights; head warms up first |
| LR schedule | Linear warmup + Cosine annealing | Stable convergence; prevents LR collapse |
| Augmentation | Albumentations | Faster than torchvision; `ElasticTransform` for skin variation |
| Mixed precision | AMP (`float16`) | ~2× speedup on GPU; `GradScaler` prevents underflow |
| Explainability | Grad-CAM | Shows model attention for clinical credibility |

---

## Dataset

**[DermNet on Kaggle](https://www.kaggle.com/datasets/shubhamgoel27/dermnet)**
- ~19,500 images across 23 skin disease categories
- License: CC BY-NC-ND 4.0
- Source: [DermNet NZ](https://dermnetnz.org/)

---

## References

- [EfficientNet paper](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019
- [Grad-CAM paper](https://arxiv.org/abs/1610.02391) — Selvaraju et al., 2017
- [timm library](https://github.com/huggingface/pytorch-image-models) — Wightman, 2019
- [Albumentations](https://albumentations.ai/) — Buslaev et al., 2020
