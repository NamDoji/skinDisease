PYTHON  := python
CONFIG  := configs/default.yaml
CKPT    := outputs/checkpoints/best.pt

.PHONY: install prepare train test gradcam predict lint clean

## Install package in development mode
install:
	pip install -e ".[dev,notebook]"

## Download dataset from Kaggle (requires ~/.kaggle/kaggle.json)
download:
	bash scripts/download_data.sh

## Build stratified train/val/test split CSVs
prepare:
	$(PYTHON) scripts/prepare_data.py --config $(CONFIG)

## Train the model (EfficientNet-B0 by default)
train:
	$(PYTHON) scripts/train.py --config $(CONFIG)

## Train with ResNet-50 backbone
train-resnet:
	$(PYTHON) scripts/train.py --config $(CONFIG) --backbone resnet50

## Run unit tests with coverage
test:
	pytest tests/ -v --tb=short --cov=src/dermnet --cov-report=term-missing

## Run single-image inference with Grad-CAM
predict:
	$(PYTHON) scripts/predict.py \
		--image data/samples/sample.jpg \
		--checkpoint $(CKPT) \
		--gradcam

## Generate Grad-CAM grid for 8 random test images
gradcam:
	$(PYTHON) -c "\
import torch, pandas as pd, sys; sys.path.insert(0,'src'); \
from dermnet.config import load_config; \
from dermnet.model import build_model; \
from dermnet.dataset import DermNetDataset; \
from dermnet.transforms import get_val_transforms; \
from dermnet.gradcam import visualize_gradcam_grid, get_target_layer; \
cfg = load_config('$(CONFIG)'); \
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); \
model = build_model(cfg.model.backbone, cfg.data.num_classes, False, cfg.model.dropout); \
ckpt = torch.load('$(CKPT)', map_location=device, weights_only=True); \
model.load_state_dict(ckpt['model_state']); model.to(device); \
df = pd.read_csv('data/processed/test.csv'); \
class_names = pd.read_csv('data/processed/classes.csv', index_col=0)['class_name'].tolist(); \
ds = DermNetDataset(df, get_val_transforms(cfg.data.image_size)); \
visualize_gradcam_grid(model, get_target_layer(model), ds, class_names, device, \
    save_path='outputs/figures/gradcam_grid.png')"

## Lint and type-check
lint:
	ruff check src/ scripts/ tests/

## Remove generated outputs (keep checkpoints)
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

## Remove ALL outputs including checkpoints
clean-all: clean
	rm -rf outputs/figures/ outputs/logs/
	@echo "NOTE: checkpoints preserved. Delete outputs/checkpoints/ manually if needed."
