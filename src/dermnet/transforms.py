"""
Albumentations transform pipelines for train, val/test, and inference.

Design notes:
  - Heavy geometric + color augmentation in train to combat overfitting
    on ~19k medical images across 23 classes.
  - Val/test: ONLY resize + CenterCrop + normalize — never augment eval data.
  - Normalize with ImageNet mean/std because we use pretrained ImageNet weights.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """
    Augmentation pipeline for training.
    Simulates natural variation in skin photography: lighting, orientation, focus.
    """
    return A.Compose([
        A.RandomResizedCrop(image_size, image_size, scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=30, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=6, p=1.0),
            A.GridDistortion(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.CoarseDropout(
            max_holes=8,
            max_height=image_size // 10,
            max_width=image_size // 10,
            fill_value=0,
            p=0.3,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """
    Deterministic pipeline for validation and test sets.
    Slight over-resize then center crop matches common eval practice.
    """
    return A.Compose([
        A.Resize(int(image_size * 1.14), int(image_size * 1.14)),
        A.CenterCrop(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_inference_transforms(image_size: int = 224) -> A.Compose:
    """Identical to val transforms — used in predict.py and Grad-CAM."""
    return get_val_transforms(image_size)
