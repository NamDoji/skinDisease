"""
Trainer class: 2-phase fine-tuning, AMP, early stopping, checkpointing.

Phase 1 (epochs 1 → warmup_epochs):
  Backbone frozen. Only the custom head is trained.
  Prevents the large pretrained weights from being destroyed in early steps.

Phase 2 (epochs warmup_epochs+1 → end):
  Backbone unfrozen. Differential LR: head learns 10× faster than backbone.
  Gradient clipping prevents exploding gradients.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .model import DermNetClassifier

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)


class EarlyStopping:
    """Stop training when val_loss stops improving for `patience` epochs."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    def __init__(
        self,
        model: DermNetClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        cfg,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.cfg = cfg
        self.device = device
        self.history = TrainingHistory()
        self.early_stop = EarlyStopping(patience=cfg.training.patience)

        # AMP scaler — no-op if mixed_precision=False
        self.use_amp = cfg.training.mixed_precision and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Phase 1: head-only training
        self.model.freeze_backbone()
        self.optimizer = self._build_optimizer(phase=1)
        self.scheduler = self._build_scheduler(self.optimizer)

        Path(cfg.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_optimizer(self, phase: int) -> AdamW:
        lr = self.cfg.training.learning_rate
        wd = self.cfg.training.weight_decay
        if phase == 1:
            return AdamW(self.model.classifier.parameters(), lr=lr, weight_decay=wd)
        else:
            param_groups = self.model.get_param_groups(
                head_lr=lr, backbone_lr=lr / 10
            )
            return AdamW(param_groups, weight_decay=wd)

    def _build_scheduler(self, optimizer: AdamW) -> SequentialLR:
        warmup_epochs = self.cfg.scheduler.warmup_epochs
        total_epochs = self.cfg.training.epochs
        min_lr = self.cfg.scheduler.min_lr

        warmup = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=min_lr
        )
        return SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )

    def _train_epoch(self) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            bs = images.size(0)
            total_loss += loss.item() * bs
            correct += (logits.argmax(1) == labels).sum().item()
            total += bs

        return total_loss / total, correct / total

    @torch.no_grad()
    def _val_epoch(self) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            bs = images.size(0)
            total_loss += loss.item() * bs
            correct += (logits.argmax(1) == labels).sum().item()
            total += bs

        return total_loss / total, correct / total

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "history": self.history,
            "backbone": self.cfg.model.backbone,
            "num_classes": self.cfg.data.num_classes,
        }
        ckpt_dir = Path(self.cfg.paths.checkpoint_dir)
        torch.save(ckpt, ckpt_dir / "last.pt")
        if is_best:
            torch.save(ckpt, ckpt_dir / "best.pt")
            logger.info(f"  ↑ New best saved (val_loss={val_loss:.4f})")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self) -> TrainingHistory:
        """Run the full training loop and return history."""
        best_val_loss = float("inf")
        warmup_epochs = self.cfg.scheduler.warmup_epochs

        for epoch in range(1, self.cfg.training.epochs + 1):
            t0 = time.time()

            # ── Phase transition ───────────────────────────────────────────────
            if epoch == warmup_epochs + 1:
                logger.info(
                    f"Epoch {epoch}: unfreezing backbone → Phase 2 (full fine-tuning)"
                )
                self.model.unfreeze_backbone()
                self.optimizer = self._build_optimizer(phase=2)
                self.scheduler = self._build_scheduler(self.optimizer)

            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._val_epoch()
            self.scheduler.step()

            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)
            self.history.train_acc.append(train_acc)
            self.history.val_acc.append(val_acc)
            self.history.lr.append(self.optimizer.param_groups[0]["lr"])

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:03d}/{self.cfg.training.epochs}  |  "
                f"train loss={train_loss:.4f} acc={train_acc:.4f}  |  "
                f"val loss={val_loss:.4f} acc={val_acc:.4f}  |  "
                f"lr={self.history.lr[-1]:.2e}  |  {elapsed:.1f}s"
            )

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            self._save_checkpoint(epoch, val_loss, is_best)

            if self.early_stop(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        return self.history
