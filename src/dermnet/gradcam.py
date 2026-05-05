"""
Gradient-weighted Class Activation Mapping (Grad-CAM).

Shows WHICH spatial regions drove the model's prediction — critical for
building trust in medical AI and demonstrating explainability in a portfolio.

Implementation is backbone-agnostic: caller specifies the target layer.
Recommended target layers:
  EfficientNet-B0 → model.backbone.blocks[-1]    (last MBConv block)
  ResNet-50       → model.backbone.layer4[-1]     (last residual block)
  ConvNeXt-Tiny   → list(model.backbone.stages)[-1]
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """
    Hooks into a target layer to capture activations and gradients,
    then computes the class-discriminative spatial heatmap.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model:        The DermNetClassifier (or any nn.Module).
            target_layer: The conv layer to hook — last feature map before global pool.
        """
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self._activations = output.detach()      # (B, C, H, W)

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()  # (B, C, H, W)

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single image.

        Args:
            input_tensor: (1, C, H, W) normalised tensor (requires_grad not required).
            target_class: class index to explain; if None uses argmax (predicted class).

        Returns:
            heatmap: (H, W) float32 array in [0, 1].
        """
        self.model.eval()
        # Re-enable gradients locally — needed even in eval mode
        inp = input_tensor.detach().requires_grad_(True)
        logits = self.model(inp)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, target_class].backward()

        # Global-average-pool gradients → channel importance weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1).squeeze()   # (H, W)
        cam = torch.relu(cam).cpu().numpy()

        # Resize to input spatial size and normalise to [0, 1]
        h, w = input_tensor.shape[-2], input_tensor.shape[-1]
        cam = cv2.resize(cam, (w, h))
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.astype(np.float32)

    def remove_hooks(self) -> None:
        """Call when done to avoid memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap onto the original image.

    Args:
        original_image: (H, W, 3) uint8 RGB image.
        heatmap:        (H, W) float32 in [0, 1].
        alpha:          blend weight for heatmap (0 = original, 1 = full heatmap).
        colormap:       OpenCV colormap constant.

    Returns:
        Blended (H, W, 3) uint8 RGB image.
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    if original_image.dtype != np.uint8:
        original_image = np.clip(original_image * 255, 0, 255).astype(np.uint8)

    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_rgb, alpha, 0)
    return overlay


def get_target_layer(model) -> nn.Module:
    """
    Automatically select the appropriate Grad-CAM target layer
    based on the backbone name stored in the model.
    """
    name = model.backbone_name
    if "efficientnet" in name:
        return model.backbone.blocks[-1]
    elif "resnet" in name:
        return model.backbone.layer4[-1]
    elif "convnext" in name:
        stages = list(model.backbone.stages)
        return stages[-1]
    else:
        # Fallback: last named child of backbone
        children = list(model.backbone.children())
        return children[-1]


def visualize_gradcam_grid(
    model: nn.Module,
    target_layer: nn.Module,
    dataset,
    class_names: list[str],
    device: torch.device,
    num_images: int = 8,
    save_path: Path | None = None,
) -> None:
    """
    Create a 2-row grid: original images (top) + Grad-CAM overlays (bottom).
    Picks random images from the dataset.

    Args:
        model:        Trained DermNetClassifier.
        target_layer: Layer to hook for Grad-CAM (use get_target_layer()).
        dataset:      DermNetDataset instance.
        class_names:  List of string class names.
        device:       torch.device.
        num_images:   Number of columns in the grid.
        save_path:    If provided, saves the figure here.
    """
    gradcam = GradCAM(model, target_layer)
    indices = np.random.choice(len(dataset), num_images, replace=False)

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])

    fig = plt.figure(figsize=(num_images * 3, 7))
    gs = gridspec.GridSpec(2, num_images, hspace=0.05, wspace=0.05)

    for col, idx in enumerate(indices):
        tensor, label = dataset[int(idx)]
        tensor_gpu = tensor.unsqueeze(0).to(device)

        heatmap = gradcam(tensor_gpu)

        # Denormalize for display
        img_np = tensor.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
        img_uint8 = (img_np * 255).astype(np.uint8)

        cam_overlay = overlay_heatmap(img_uint8, heatmap)

        ax_top = fig.add_subplot(gs[0, col])
        ax_top.imshow(img_np)
        ax_top.set_title(class_names[label], fontsize=7, pad=2, wrap=True)
        ax_top.axis("off")

        ax_bot = fig.add_subplot(gs[1, col])
        ax_bot.imshow(cam_overlay)
        ax_bot.axis("off")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grad-CAM grid saved to: {save_path}")

    plt.close(fig)
    gradcam.remove_hooks()
