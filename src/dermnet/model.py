"""
DermNet classifier model using timm pretrained backbones.

Architecture:
  timm backbone (feature extractor, num_classes=0)
    └── Custom head:
          LayerNorm → Dropout(0.4) → Linear(in_features, 512)
          → GELU → Dropout(0.2) → Linear(512, num_classes)

Fine-tuning strategy (2 phases):
  Phase 1 (warmup epochs):  backbone frozen → train head only (fast, stable)
  Phase 2 (remaining):      unfreeze all → differential LR (head 10× backbone)

Supported backbones (change via config):
  - efficientnet_b0   ~5.3M params  (default — best accuracy/speed tradeoff)
  - resnet50          ~25M params   (classic, well-understood)
  - convnext_tiny     ~28M params   (modern, strong regularisation)
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class DermNetClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 23,
        pretrained: bool = True,
        dropout: float = 0.4,
    ):
        """
        Args:
            backbone:    timm model name (see SUPPORTED_BACKBONES above).
            num_classes: number of output classes.
            pretrained:  load ImageNet pretrained weights.
            dropout:     dropout probability in the custom head.
        """
        super().__init__()
        self.backbone_name = backbone

        # timm backbone — num_classes=0 removes original classifier, returns feature vector
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,      # remove original head
            global_pool="avg",
        )
        in_features = self.backbone.num_features

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),          # more stable than BN for fine-tuning
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) image tensor, normalised.

        Returns:
            logits: (B, num_classes) — raw logits (no softmax).
        """
        features = self.backbone(x)      # (B, in_features)
        return self.classifier(features)

    def freeze_backbone(self) -> None:
        """Phase 1: freeze all backbone parameters, train head only."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Phase 2: unfreeze all parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_param_groups(
        self, head_lr: float, backbone_lr: float
    ) -> list[dict]:
        """
        Differential learning rate groups for Phase 2 optimiser.
        Backbone learns 10× slower than the head to avoid destroying pretrained weights.

        Usage:
            optimiser = AdamW(model.get_param_groups(head_lr=3e-4, backbone_lr=3e-5))
        """
        return [
            {"params": self.backbone.parameters(),   "lr": backbone_lr},
            {"params": self.classifier.parameters(), "lr": head_lr},
        ]


def build_model(
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.4,
) -> DermNetClassifier:
    """Factory function — creates model and logs parameter counts."""
    model = DermNetClassifier(backbone, num_classes, pretrained, dropout)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Built {backbone}: {total / 1e6:.1f}M params total, "
          f"{trainable / 1e6:.1f}M trainable")
    return model
