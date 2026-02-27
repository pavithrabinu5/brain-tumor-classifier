"""
src/models/classifier.py
─────────────────────────
Transfer learning model using timm backbones.
Supports EfficientNet, ResNet, DenseNet, ViT and more.
"""

import torch
import torch.nn as nn
import timm
from typing import Dict, Optional


class TumorClassifier(nn.Module):
    """
    Transfer learning classifier built on top of pretrained timm backbones.

    Usage:
        model = TumorClassifier(backbone="efficientnet_b3", num_classes=4)
        model = TumorClassifier(backbone="vit_base_patch16_224", num_classes=4)
        model = TumorClassifier(backbone="resnet50", num_classes=4)
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b3",
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        # ── Load pretrained backbone ──────────────────────────
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,          # Remove original head
            global_pool="avg",      # Global average pooling
        )

        # Get feature dimension from the backbone
        self.feature_dim = self.backbone.num_features

        # ── Custom classification head ────────────────────────
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

        # Optionally freeze backbone (useful for early training)
        if freeze_backbone:
            self.freeze_backbone()

        # Weight initialization for the head
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze_backbone(self):
        """Freeze all backbone parameters (train head only)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"[INFO] Backbone frozen. Training head only.")

    def unfreeze_backbone(self, unfreeze_last_n_blocks: int = None):
        """Unfreeze backbone for fine-tuning."""
        if unfreeze_last_n_blocks is None:
            # Unfreeze everything
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("[INFO] Full backbone unfrozen.")
        else:
            # Unfreeze only the last N blocks (layer-wise fine-tuning)
            layers = list(self.backbone.children())
            for layer in layers[-unfreeze_last_n_blocks:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"[INFO] Last {unfreeze_last_n_blocks} blocks unfrozen.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)         # (B, feature_dim)
        logits = self.classifier(features)  # (B, num_classes)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature embeddings (useful for visualization/clustering)."""
        return self.backbone(x)

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }

    def __repr__(self):
        params = self.count_parameters()
        return (
            f"TumorClassifier(\n"
            f"  backbone={self.backbone_name},\n"
            f"  num_classes={self.num_classes},\n"
            f"  features={self.feature_dim},\n"
            f"  total_params={params['total']:,},\n"
            f"  trainable_params={params['trainable']:,}\n"
            f")"
        )


def build_model(config: dict, device: torch.device) -> TumorClassifier:
    """Build model from config dict."""
    model = TumorClassifier(
        backbone=config["model"]["backbone"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
    )
    model = model.to(device)
    print(model)
    return model
