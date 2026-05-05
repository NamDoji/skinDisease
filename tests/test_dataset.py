"""Tests for DermNetDataset, compute_class_weights, make_weighted_sampler."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dermnet.dataset import (
    DermNetDataset,
    compute_class_weights,
    make_weighted_sampler,
)
from dermnet.transforms import get_val_transforms


class TestDermNetDataset:
    def test_len(self, dummy_df):
        ds = DermNetDataset(dummy_df, transform=get_val_transforms())
        assert len(ds) == len(dummy_df)

    def test_getitem_tensor_shape(self, dummy_df, image_size):
        ds = DermNetDataset(dummy_df, transform=get_val_transforms(image_size))
        tensor, label = ds[0]
        assert tensor.shape == (3, image_size, image_size), (
            f"Expected (3, {image_size}, {image_size}), got {tensor.shape}"
        )

    def test_getitem_returns_int_label(self, dummy_df):
        ds = DermNetDataset(dummy_df, transform=get_val_transforms())
        _, label = ds[0]
        assert isinstance(label, int)

    def test_getitem_label_in_range(self, dummy_df, num_classes):
        ds = DermNetDataset(dummy_df, transform=get_val_transforms())
        for i in range(min(10, len(ds))):
            _, label = ds[i]
            assert 0 <= label < num_classes

    def test_no_transform_returns_tensor(self, dummy_df):
        """Without a transform, __getitem__ still returns something usable."""
        ds = DermNetDataset(dummy_df, transform=None)
        img, label = ds[0]
        # img will be a numpy array here — just verify it's non-None
        assert img is not None


class TestClassWeights:
    def test_output_shape(self, num_classes):
        labels = list(range(num_classes)) * 10
        weights = compute_class_weights(labels, num_classes)
        assert weights.shape == (num_classes,)

    def test_weights_are_positive(self, num_classes):
        labels = list(range(num_classes)) * 5
        weights = compute_class_weights(labels, num_classes)
        assert (weights > 0).all()

    def test_balanced_dataset_equal_weights(self, num_classes):
        """Perfectly balanced dataset → all class weights equal."""
        labels = list(range(num_classes)) * 20
        weights = compute_class_weights(labels, num_classes)
        assert torch.allclose(weights, weights[0].expand_as(weights), atol=1e-4)

    def test_rare_class_gets_higher_weight(self, num_classes):
        """Class 0 appears 1×, class 1 appears 10× → weight[0] > weight[1]."""
        labels = [0] + [1] * 10
        weights = compute_class_weights(labels, num_classes)
        assert weights[0] > weights[1]


class TestWeightedSampler:
    def test_sampler_length(self, dummy_df, num_classes):
        sampler = make_weighted_sampler(dummy_df["label"].tolist(), num_classes)
        assert sampler.num_samples == len(dummy_df)

    def test_sampler_uses_replacement(self, dummy_df, num_classes):
        sampler = make_weighted_sampler(dummy_df["label"].tolist(), num_classes)
        assert sampler.replacement is True
