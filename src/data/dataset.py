"""
src/data/dataset.py
────────────────────
PyTorch Dataset class + DataLoader factory with augmentations.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Augmentation Pipelines ───────────────────────────────────

def get_transforms(split: str, image_size: int = 224, config: dict = None) -> A.Compose:
    """Return augmentation pipeline for a given split."""

    normalize = A.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet stats
        std=[0.229, 0.224, 0.225]
    )

    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=30, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.2),
            A.GridDistortion(p=0.2),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),  # Cutout
            normalize,
            ToTensorV2(),
        ])
    else:  # val / test
        return A.Compose([
            A.Resize(image_size, image_size),
            normalize,
            ToTensorV2(),
        ])


# ── Dataset Class ────────────────────────────────────────────

class TumorDataset(Dataset):
    """
    Expects this folder structure:
        root/
          train/  glioma/  meningioma/  ...
          val/    glioma/  meningioma/  ...
          test/   glioma/  meningioma/  ...
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(
        self,
        root: str,
        split: str,                      # "train" | "val" | "test"
        transform: Optional[A.Compose] = None,
        classes: Optional[List[str]] = None,
    ):
        self.root = Path(root) / split
        self.split = split
        self.transform = transform

        # Auto-detect classes from folder names
        if classes:
            self.classes = sorted(classes)
        else:
            self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])

        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}

        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        for cls in self.classes:
            cls_dir = self.root / cls
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in self.EXTENSIONS:
                    self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency weights for imbalanced datasets."""
        counts = np.zeros(len(self.classes))
        for _, label in self.samples:
            counts[label] += 1
        weights = 1.0 / counts
        weights = weights / weights.sum()
        return torch.FloatTensor(weights)

    def get_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        return torch.FloatTensor([class_weights[label] for _, label in self.samples])


# ── DataLoader Factory ───────────────────────────────────────

def build_dataloaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    classes: Optional[List[str]] = None,
    use_weighted_sampler: bool = True,
) -> Dict[str, DataLoader]:
    """
    Returns a dict: {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    datasets = {}
    loaders = {}

    for split in ["train", "val", "test"]:
        transform = get_transforms(split, image_size)
        datasets[split] = TumorDataset(
            root=data_dir,
            split=split,
            transform=transform,
            classes=classes,
        )

    # Use WeightedRandomSampler for training to handle class imbalance
    sampler = None
    if use_weighted_sampler:
        sample_weights = datasets["train"].get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    loaders["train"] = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
    )
    loaders["val"] = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    loaders["test"] = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Print summary
    for split, ds in datasets.items():
        print(f"  {split:<6}: {len(ds):>5} images | classes: {ds.classes}")

    return loaders, datasets
