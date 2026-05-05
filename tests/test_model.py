"""Tests for DermNetClassifier and build_model()."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dermnet.model import DermNetClassifier, build_model


class TestModelOutputShape:
    @pytest.mark.parametrize("backbone", ["efficientnet_b0", "resnet50"])
    def test_output_shape(self, backbone, dummy_batch, num_classes):
        model = build_model(backbone, num_classes, pretrained=False)
        images, _ = dummy_batch
        with torch.no_grad():
            logits = model(images)
        assert logits.shape == (images.shape[0], num_classes), (
            f"Expected ({images.shape[0]}, {num_classes}), got {logits.shape}"
        )

    def test_output_is_raw_logits(self, dummy_batch, num_classes):
        """Confirm output is NOT softmax-normalised (rows should not sum to 1)."""
        model = build_model("efficientnet_b0", num_classes, pretrained=False)
        images, _ = dummy_batch
        with torch.no_grad():
            logits = model(images)
        row_sums = logits.sum(dim=1)
        assert not torch.allclose(row_sums, torch.ones_like(row_sums)), (
            "Model output should be raw logits, not probabilities."
        )


class TestFreezeUnfreeze:
    def test_freeze_backbone(self, num_classes):
        model = build_model("efficientnet_b0", num_classes, pretrained=False)
        model.freeze_backbone()
        frozen = [p.requires_grad for p in model.backbone.parameters()]
        assert not any(frozen), "All backbone params should be frozen"

    def test_unfreeze_backbone(self, num_classes):
        model = build_model("efficientnet_b0", num_classes, pretrained=False)
        model.freeze_backbone()
        model.unfreeze_backbone()
        unfrozen = [p.requires_grad for p in model.backbone.parameters()]
        assert all(unfrozen), "All backbone params should be trainable after unfreeze"

    def test_head_always_trainable_after_freeze(self, num_classes):
        """Head parameters should not be affected by freeze_backbone()."""
        model = build_model("efficientnet_b0", num_classes, pretrained=False)
        model.freeze_backbone()
        head_trainable = [p.requires_grad for p in model.classifier.parameters()]
        assert all(head_trainable), "Head should remain trainable when backbone is frozen"


class TestParamGroups:
    def test_two_groups_returned(self, num_classes):
        model = build_model("efficientnet_b0", num_classes, pretrained=False)
        groups = model.get_param_groups(head_lr=1e-3, backbone_lr=1e-4)
        assert len(groups) == 2

    def test_differential_lr(self, num_classes):
        model = build_model("efficientnet_b0", num_classes, pretrained=False)
        groups = model.get_param_groups(head_lr=3e-4, backbone_lr=3e-5)
        assert groups[0]["lr"] != groups[1]["lr"]
        assert groups[1]["lr"] > groups[0]["lr"], (
            "Head group (index 1) should have higher LR than backbone (index 0)"
        )
