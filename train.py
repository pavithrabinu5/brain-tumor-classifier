"""
train.py
─────────
Main entry point. Runs the full pipeline:
  1. Load config
  2. Build dataloaders
  3. Build model
  4. Train
  5. Evaluate
  6. Generate Grad-CAM visualizations

Usage:
  python train.py
  python train.py --config configs/config.yaml
  python train.py --backbone vit_base_patch16_224 --epochs 30
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from rich.console import Console
from rich.rule import Rule

console = Console()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import build_dataloaders
from src.models.classifier import build_model
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from src.explainability.gradcam import GradCAMVisualizer
from src.data.dataset import get_transforms


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        console.print(f"[green]✓ GPU: {torch.cuda.get_device_name(0)}[/]")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        console.print("[green]✓ Apple Silicon MPS[/]")
    else:
        device = torch.device("cpu")
        console.print("[yellow]⚠ No GPU found, using CPU (training will be slow)[/]")
    return device


def parse_args():
    parser = argparse.ArgumentParser(description="Train Tumor Classifier")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--backbone", default=None, help="Override backbone in config")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_only", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint for eval_only")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Apply CLI overrides
    if args.backbone:   config["model"]["backbone"] = args.backbone
    if args.epochs:     config["training"]["epochs"] = args.epochs
    if args.batch_size: config["training"]["batch_size"] = args.batch_size

    # Setup
    set_seed(config["project"]["seed"])
    device = get_device()

    console.print(Rule(f"[bold cyan]{config['project']['name']}[/]"))
    console.print(f"  Backbone : {config['model']['backbone']}")
    console.print(f"  Classes  : {config['data']['classes']}")
    console.print(f"  Epochs   : {config['training']['epochs']}")

    # ── 1. Data ───────────────────────────────────────────────
    console.print(Rule("[bold]Step 1: Loading Data"))
    loaders, datasets = build_dataloaders(
        data_dir=config["data"]["processed_dir"],
        image_size=config["data"]["image_size"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        classes=config["data"]["classes"],
    )

    # ── 2. Model ──────────────────────────────────────────────
    console.print(Rule("[bold]Step 2: Building Model"))
    model = build_model(config, device)

    # ── 3. Train ──────────────────────────────────────────────
    if not args.eval_only:
        console.print(Rule("[bold]Step 3: Training"))
        trainer = Trainer(model, loaders["train"], loaders["val"], config, device)
        history = trainer.train()
        trainer.load_best_checkpoint()
    else:
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            console.print(f"[green]✓ Loaded checkpoint: {args.checkpoint}[/]")

    # ── 4. Evaluate ───────────────────────────────────────────
    console.print(Rule("[bold]Step 4: Evaluation"))
    evaluator = Evaluator(
        model=model,
        test_loader=loaders["test"],
        class_names=config["data"]["classes"],
        device=device,
        results_dir=config["output"]["results_dir"],
    )
    metrics = evaluator.evaluate()

    # ── 5. Grad-CAM ───────────────────────────────────────────
    console.print(Rule("[bold]Step 5: Grad-CAM Visualizations"))
    explainer = GradCAMVisualizer(
        model=model,
        class_names=config["data"]["classes"],
        device=device,
        output_dir=config["explainability"]["save_dir"],
        method=config["explainability"]["method"],
    )

    # Grab a batch from test set
    test_images, test_labels = next(iter(loaders["test"]))
    explainer.visualize_batch(
        images=test_images,
        labels=test_labels,
        num_samples=config["explainability"]["num_samples"],
        tag="test_set",
    )

    console.print(Rule("[bold green]✓ Pipeline Complete!"))
    console.print(f"  Test Accuracy : {metrics['accuracy']:.4f}")
    console.print(f"  AUC-ROC Macro : {metrics['auc_roc_macro']:.4f}")
    console.print(f"  F1 Weighted   : {metrics['f1_weighted']:.4f}")
    console.print(f"\n  Results saved to: {config['output']['results_dir']}")
    console.print(f"  Grad-CAM saved to: {config['explainability']['save_dir']}")


if __name__ == "__main__":
    main()
