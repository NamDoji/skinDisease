"""
Main training entry point.

Usage:
    python scripts/train.py
    python scripts/train.py --backbone resnet50
    python scripts/train.py --epochs 20 --batch_size 64
    python scripts/train.py --config configs/default.yaml
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dermnet.config import load_config, override_config
from dermnet.dataset import build_dataloaders, compute_class_weights
from dermnet.evaluate import run_evaluation
from dermnet.model import build_model
from dermnet.trainer import Trainer
from dermnet.transforms import get_train_transforms, get_val_transforms
from dermnet.utils import plot_training_curves, seed_everything, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DermNet skin disease classifier")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backbone", default=None,
                        help="Override backbone (efficientnet_b0, resnet50, convnext_tiny)")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = override_config(cfg, args)

    setup_logging(cfg.paths.log_dir)
    logger = logging.getLogger(__name__)
    seed_everything(cfg.project.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Backbone: {cfg.model.backbone}")

    # ── Load split CSVs ───────────────────────────────────────────────────────
    proc = Path(cfg.data.processed_dir)
    if not (proc / "train.csv").exists():
        logger.error(
            "Processed splits not found. Run: python scripts/prepare_data.py  first."
        )
        sys.exit(1)

    train_df = pd.read_csv(proc / "train.csv")
    val_df = pd.read_csv(proc / "val.csv")
    test_df = pd.read_csv(proc / "test.csv")
    class_names = pd.read_csv(proc / "classes.csv", index_col=0)["class_name"].tolist()

    logger.info(
        f"Dataset  train={len(train_df):,}  val={len(val_df):,}  "
        f"test={len(test_df):,}  classes={len(class_names)}"
    )

    # ── Transforms & DataLoaders ──────────────────────────────────────────────
    train_tf = get_train_transforms(cfg.data.image_size)
    eval_tf = get_val_transforms(cfg.data.image_size)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df,
        train_transform=train_tf,
        eval_transform=eval_tf,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        num_classes=cfg.data.num_classes,
        use_weighted_sampler=cfg.training.use_weighted_sampler,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        cfg.model.backbone,
        cfg.data.num_classes,
        cfg.model.pretrained,
        cfg.model.dropout,
    )

    # ── Loss function ─────────────────────────────────────────────────────────
    class_weights = None
    if cfg.training.use_class_weights:
        class_weights = compute_class_weights(
            train_df["label"].tolist(), cfg.data.num_classes
        ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg.model.label_smoothing,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(model, train_loader, val_loader, criterion, cfg, device)
    history = trainer.fit()

    # ── Save training curves ──────────────────────────────────────────────────
    plot_training_curves(history, save_dir=cfg.paths.figure_dir)

    # ── Final evaluation on held-out test set ─────────────────────────────────
    ckpt_path = Path(cfg.paths.checkpoint_dir) / "best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"Loaded best checkpoint from epoch {ckpt['epoch']}")

    metrics = run_evaluation(
        model, test_loader, device, class_names,
        figure_dir=cfg.paths.figure_dir,
    )

    logger.info("\n" + "=" * 60)
    logger.info("TEST SET RESULTS")
    logger.info(f"  Accuracy    : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    logger.info(f"  Top-5 Acc   : {metrics['top5_accuracy']:.4f}  ({metrics['top5_accuracy']*100:.1f}%)")
    logger.info(f"  Macro F1    : {metrics['macro_f1']:.4f}")
    logger.info(f"  Weighted F1 : {metrics['weighted_f1']:.4f}")
    logger.info("=" * 60)
    logger.info("\n" + metrics["classification_report"])


if __name__ == "__main__":
    main()
