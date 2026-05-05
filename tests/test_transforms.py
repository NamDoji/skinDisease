"""Tests for Albumentations transform pipelines."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dermnet.transforms import (
    get_inference_transforms,
    get_train_transforms,
    get_val_transforms,
)


def make_image(h: int = 300, w: int = 300) -> np.ndarray:
    """Create a random uint8 RGB numpy image (H, W, 3)."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


class TestTrainTransforms:
    def test_output_is_tensor(self, image_size):
        t = get_train_transforms(image_size)
        result = t(image=make_image())["image"]
        assert isinstance(result, torch.Tensor)

    def test_output_dtype_float32(self, image_size):
        t = get_train_transforms(image_size)
        result = t(image=make_image())["image"]
        assert result.dtype == torch.float32

    def test_output_shape(self, image_size):
        t = get_train_transforms(image_size)
        result = t(image=make_image())["image"]
        assert result.shape == (3, image_size, image_size)

    def test_values_are_normalised(self, image_size):
        """After ImageNet normalisation values are typically in roughly [-3, 3]."""
        t = get_train_transforms(image_size)
        result = t(image=make_image())["image"]
        assert result.min().item() < 0, "Normalised tensor should have negative values"
        assert result.max().item() < 10, "Normalised tensor should not be large"


class TestValTransforms:
    def test_output_shape(self, image_size):
        t = get_val_transforms(image_size)
        result = t(image=make_image())["image"]
        assert result.shape == (3, image_size, image_size)

    def test_deterministic(self, image_size):
        """Same input must produce the same output every time."""
        t = get_val_transforms(image_size)
        img = make_image()
        out1 = t(image=img.copy())["image"]
        out2 = t(image=img.copy())["image"]
        assert torch.allclose(out1, out2), "Val transform must be deterministic"

    def test_different_from_train_output(self, image_size):
        """Train and val transforms can produce different outputs (stochastic vs. det.)."""
        # Just verify both run without error on the same image
        img = make_image()
        tr = get_train_transforms(image_size)
        vt = get_val_transforms(image_size)
        tr_out = tr(image=img.copy())["image"]
        vt_out = vt(image=img.copy())["image"]
        assert tr_out.shape == vt_out.shape


class TestInferenceTransforms:
    def test_same_as_val(self, image_size):
        """Inference transforms must be identical to val transforms."""
        img = make_image()
        vt = get_val_transforms(image_size)
        it = get_inference_transforms(image_size)
        out_val = vt(image=img.copy())["image"]
        out_inf = it(image=img.copy())["image"]
        assert torch.allclose(out_val, out_inf), (
            "Inference transforms must match val transforms exactly"
        )
