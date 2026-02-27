"""
src/explainability/gradcam.py
──────────────────────────────
Grad-CAM heatmap generation and overlay visualization.
Highlights which regions in the scan influenced the model's prediction.
"""

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_target_layer(model):
    """Auto-detect the last convolutional layer for Grad-CAM."""
    backbone_name = model.backbone_name.lower()

    # EfficientNet family
    if "efficientnet" in backbone_name:
        return [model.backbone.conv_head]

    # ResNet / DenseNet
    elif "resnet" in backbone_name or "densenet" in backbone_name:
        layers = list(model.backbone.children())
        for layer in reversed(layers):
            if isinstance(layer, torch.nn.Sequential):
                return [layer[-1]]

    # ViT — use last attention block
    elif "vit" in backbone_name:
        return [model.backbone.blocks[-1].norm1]

    # Fallback: find last conv layer
    last_conv = None
    for module in model.backbone.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return [last_conv]


class GradCAMVisualizer:
    def __init__(
        self,
        model: torch.nn.Module,
        class_names: List[str],
        device: torch.device,
        output_dir: str = "outputs/grad_cam",
        method: str = "gradcam",
    ):
        self.model = model
        self.class_names = class_names
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        target_layers = get_target_layer(model)
        cam_class = GradCAMPlusPlus if method == "gradcam++" else GradCAM
        self.cam = cam_class(model=model, target_layers=target_layers)

    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        predicted_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap for a single image tensor (C, H, W)."""
        input_tensor = image_tensor.unsqueeze(0).to(self.device)

        targets = [ClassifierOutputTarget(predicted_class)] if predicted_class is not None else None
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        return grayscale_cam[0]  # (H, W)

    def visualize_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        num_samples: int = 12,
        tag: str = "results",
    ):
        """
        Visualize Grad-CAM overlays for a batch of images.
        Saves a grid image showing: original | heatmap overlay | prediction info
        """
        self.model.eval()
        n = min(num_samples, len(images))
        images = images[:n].to(self.device)
        labels = labels[:n]

        with torch.no_grad():
            logits = self.model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

        # Inverse-normalize to get RGB image for overlay
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        cols = 4
        rows = (n + cols - 1) // cols
        fig = plt.figure(figsize=(cols * 4, rows * 4.5))
        gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.3)

        for i in range(n):
            img_np = images[i].cpu().numpy().transpose(1, 2, 0)
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1).astype(np.float32)

            heatmap = self.generate_heatmap(images[i], predicted_class=int(preds[i]))
            overlay = show_cam_on_image(img_np, heatmap, use_rgb=True)

            row, col = divmod(i, cols)
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(overlay)

            true_cls = self.class_names[labels[i].item()]
            pred_cls = self.class_names[preds[i]]
            conf = probs[i][preds[i]] * 100
            correct = "✓" if labels[i].item() == preds[i] else "✗"

            color = "green" if labels[i].item() == preds[i] else "red"
            ax.set_title(
                f"{correct} True: {true_cls}\nPred: {pred_cls} ({conf:.1f}%)",
                fontsize=9,
                color=color,
                fontweight="bold",
            )
            ax.axis("off")

        fig.suptitle("Grad-CAM Visualizations", fontsize=14, fontweight="bold", y=1.01)
        path = self.output_dir / f"gradcam_{tag}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[✓] Grad-CAM saved: {path}")
        return path

    def explain_single(self, image_path: str, transform) -> dict:
        """Explain a single image file. Returns prediction info."""
        img = np.array(Image.open(image_path).convert("RGB"))
        augmented = transform(image=img)
        tensor = augmented["image"]

        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor.unsqueeze(0).to(self.device))
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_class = probs.argmax()
        heatmap = self.generate_heatmap(tensor, predicted_class=int(pred_class))

        return {
            "predicted_class": self.class_names[pred_class],
            "confidence": float(probs[pred_class]),
            "all_probs": {cls: float(p) for cls, p in zip(self.class_names, probs)},
            "heatmap": heatmap,
        }
