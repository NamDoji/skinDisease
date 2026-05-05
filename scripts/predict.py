"""
Single-image inference script.

Usage:
    python scripts/predict.py --image data/samples/sample.jpg \\
                               --checkpoint outputs/checkpoints/best.pt

    python scripts/predict.py --image path/to/skin.jpg \\
                               --checkpoint outputs/checkpoints/best.pt \\
                               --top_k 5 --gradcam

Output example:
    ┌───────────────────────────────────────────────────────┐
    │  Prediction : Psoriasis                               │
    │  Confidence : 87.3%                                   │
    │                                                       │
    │  Top-5 predictions:                                   │
    │    1. Psoriasis                        87.3%          │
    │    2. Eczema                            6.1%          │
    │    3. Pityriasis rosea                  2.8%          │
    │    4. Seborrheic Dermatitis             2.0%          │
    │    5. Tinea Ringworm                    0.8%          │
    └───────────────────────────────────────────────────────┘
"""
import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dermnet.config import load_config
from dermnet.gradcam import GradCAM, get_target_layer, overlay_heatmap
from dermnet.model import build_model
from dermnet.transforms import get_inference_transforms


def load_class_names(processed_dir: str) -> list[str]:
    path = Path(processed_dir) / "classes.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"classes.csv not found at {path}. Run prepare_data.py first."
        )
    return pd.read_csv(path, index_col=0)["class_name"].tolist()


def load_model_from_checkpoint(checkpoint_path: str, cfg) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(
        backbone=ckpt.get("backbone", cfg.model.backbone),
        num_classes=ckpt.get("num_classes", cfg.data.num_classes),
        pretrained=False,
        dropout=cfg.model.dropout,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict(
    image_path: str,
    model: torch.nn.Module,
    class_names: list[str],
    cfg,
    device: torch.device,
    top_k: int = 5,
    save_gradcam: bool = False,
) -> dict:
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    transform = get_inference_transforms(cfg.data.image_size)
    tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()

    top_probs, top_indices = probs.topk(min(top_k, len(class_names)))

    results = {
        "prediction": class_names[top_indices[0].item()],
        "confidence": top_probs[0].item(),
        "top_k": [
            {"class": class_names[idx.item()], "prob": prob.item()}
            for idx, prob in zip(top_indices, top_probs)
        ],
    }

    if save_gradcam:
        _save_gradcam_overlay(image_rgb, tensor, model, top_indices[0].item(),
                               device, image_path)

    return results


def _save_gradcam_overlay(
    image_rgb, tensor, model, target_class, device, source_path
) -> None:
    """Run Grad-CAM and save overlay image next to the source."""
    target_layer = get_target_layer(model)
    # Grad-CAM needs gradients — pass fresh tensor
    tensor_grad = tensor.clone().detach().requires_grad_(True)
    gc = GradCAM(model, target_layer)
    heatmap = gc(tensor_grad, target_class)
    gc.remove_hooks()

    img_resized = cv2.resize(image_rgb, (model.backbone.num_features and 224 or 224, 224))
    overlay = overlay_heatmap(img_resized, heatmap)

    out_path = Path(source_path).with_stem(Path(source_path).stem + "_gradcam")
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Grad-CAM saved: {out_path}")


def print_results(results: dict) -> None:
    width = 57
    bar = "─" * width
    top_label = results["prediction"]
    conf_str = f"{results['confidence']*100:.1f}%"

    print(f"\n┌{bar}┐")
    print(f"│  Prediction : {top_label:<{width - 17}}│")
    print(f"│  Confidence : {conf_str:<{width - 17}}│")
    print(f"│{' ' * width}│")
    print(f"│  Top-{len(results['top_k'])} predictions:{' ' * (width - 20)}│")
    for i, item in enumerate(results["top_k"], 1):
        name = item["class"][:32]
        pct = f"{item['prob']*100:5.1f}%"
        line = f"│    {i}. {name:<33} {pct}  │"
        print(line)
    print(f"└{bar}┘\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict skin disease from a single image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best.pt checkpoint")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--gradcam", action="store_true",
                        help="Save Grad-CAM overlay alongside the input image")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = load_class_names(cfg.data.processed_dir)
    model = load_model_from_checkpoint(args.checkpoint, cfg).to(device)

    results = predict(
        args.image, model, class_names, cfg, device,
        top_k=args.top_k,
        save_gradcam=args.gradcam,
    )
    print_results(results)


if __name__ == "__main__":
    main()
