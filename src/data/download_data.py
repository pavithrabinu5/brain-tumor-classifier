"""
src/data/download_data.py
─────────────────────────
Downloads and organizes the Brain Tumor MRI dataset from Kaggle.
Run: python src/data/download_data.py

Prerequisites:
  pip install kaggle
  Place your kaggle.json in ~/.kaggle/  (from kaggle.com → Account → API)
"""

import os
import shutil
import zipfile
from pathlib import Path
from rich.console import Console
from rich.progress import track

console = Console()


def download_brain_tumor_dataset(raw_dir: str = "data/raw"):
    """Download Brain Tumor MRI dataset from Kaggle."""
    console.print("[bold cyan]Downloading Brain Tumor MRI Dataset...[/]")

    os.makedirs(raw_dir, exist_ok=True)

    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "masoudnickparvar/brain-tumor-mri-dataset",
            path=raw_dir,
            unzip=True
        )
        console.print(f"[green]✓ Dataset downloaded to {raw_dir}[/]")
    except Exception as e:
        console.print(f"[red]Kaggle download failed: {e}[/]")
        console.print("[yellow]Manual download steps:[/]")
        console.print("  1. Go to https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
        console.print("  2. Download and unzip into data/raw/")
        console.print("  3. Re-run this script to organize the files")


def organize_dataset(raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
    """
    Organize raw dataset into:
      data/processed/
        train/  class_a/  class_b/ ...
        val/    class_a/  class_b/ ...
        test/   class_a/  class_b/ ...
    """
    from sklearn.model_selection import train_test_split

    console.print("[bold cyan]Organizing dataset into train/val/test splits...[/]")

    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)

    # Detect class folders (adjust path based on how kaggle unzipped)
    # Common structures: raw/Training/, raw/Testing/
    training_path = raw_path / "Training"
    if not training_path.exists():
        training_path = raw_path  # fallback: classes are directly in raw/

    classes = [d.name for d in training_path.iterdir() if d.is_dir()]
    console.print(f"[green]Found classes: {classes}[/]")

    stats = {}

    for cls in track(classes, description="Processing classes"):
        cls_path = training_path / cls
        images = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.png")) + list(cls_path.glob("*.jpeg"))

        # Split: 70% train, 15% val, 15% test
        train_imgs, temp_imgs = train_test_split(images, test_size=0.30, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.50, random_state=42)

        stats[cls] = {"train": len(train_imgs), "val": len(val_imgs), "test": len(test_imgs)}

        for split, imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            dest = processed_path / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for img_path in imgs:
                shutil.copy2(img_path, dest / img_path.name)

    # Print summary table
    console.print("\n[bold]Dataset Split Summary:[/]")
    console.print(f"{'Class':<20} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    console.print("─" * 50)
    for cls, counts in stats.items():
        total = sum(counts.values())
        console.print(f"{cls:<20} {counts['train']:>8} {counts['val']:>8} {counts['test']:>8} {total:>8}")

    console.print(f"\n[green]✓ Dataset organized at {processed_dir}[/]")
    return stats


if __name__ == "__main__":
    download_brain_tumor_dataset()
    organize_dataset()
