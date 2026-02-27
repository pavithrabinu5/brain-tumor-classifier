"""
src/training/trainer.py
────────────────────────
Full training loop with:
  - Mixed precision (fp16)
  - Learning rate scheduling
  - Early stopping
  - Weights & Biases logging
  - Checkpoint saving
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        improved = (
            value < self.best_value - self.min_delta
            if self.mode == "min"
            else value > self.best_value + self.min_delta
        )
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # ── Optimizer ─────────────────────────────────────────
        self.optimizer = self._build_optimizer()

        # ── Loss ──────────────────────────────────────────────
        self.criterion = nn.CrossEntropyLoss(
            weight=self._get_class_weights() if config["training"]["class_weights"] else None
        )

        # ── Scheduler ─────────────────────────────────────────
        self.scheduler = self._build_scheduler()

        # ── Mixed Precision ───────────────────────────────────
        self.use_amp = config["training"]["mixed_precision"] and device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # ── Early Stopping ────────────────────────────────────
        es_cfg = config["training"]["early_stopping"]
        self.early_stopping = EarlyStopping(patience=es_cfg["patience"])

        # ── Checkpointing ─────────────────────────────────────
        self.checkpoint_dir = Path(config["output"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")

        # ── W&B ───────────────────────────────────────────────
        self.use_wandb = config.get("wandb", {}).get("enabled", False)
        if self.use_wandb:
            import wandb
            wandb.init(
                project=config["wandb"]["project"],
                entity=config["wandb"]["entity"],
                config=config,
                name=f"{config['model']['backbone']}-run",
            )

        # History
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def _build_optimizer(self):
        opt_cfg = self.config["training"]
        params = [
            {"params": self.model.backbone.parameters(), "lr": opt_cfg["learning_rate"] * 0.1},
            {"params": self.model.classifier.parameters(), "lr": opt_cfg["learning_rate"]},
        ]
        if opt_cfg["optimizer"].lower() == "adamw":
            return torch.optim.AdamW(params, weight_decay=opt_cfg["weight_decay"])
        elif opt_cfg["optimizer"].lower() == "adam":
            return torch.optim.Adam(params)
        else:
            return torch.optim.SGD(params, momentum=0.9, nesterov=True)

    def _build_scheduler(self):
        cfg = self.config["training"]
        total_epochs = cfg["epochs"]
        warmup = cfg.get("warmup_epochs", 5)

        if cfg["scheduler"] == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_epochs - warmup, eta_min=1e-6
            )
        elif cfg["scheduler"] == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1)
        else:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=5, factor=0.5
            )

    def _get_class_weights(self) -> Optional[torch.Tensor]:
        try:
            weights = self.train_loader.dataset.get_class_weights()
            return weights.to(self.device)
        except Exception:
            return None

    # ── Single Epoch ──────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, training: bool) -> Dict[str, float]:
        self.model.train(training)
        total_loss, correct, total = 0.0, 0, 0

        desc = "Train" if training else "Val  "
        pbar = tqdm(loader, desc=f"  {desc}", leave=False, ncols=80)

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            if training:
                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }

    # ── Main Train Loop ───────────────────────────────────────

    def train(self):
        epochs = self.config["training"]["epochs"]
        warmup = self.config["training"].get("warmup_epochs", 5)

        console.print(f"\n[bold green]Starting training for {epochs} epochs...[/]\n")

        for epoch in range(1, epochs + 1):
            start = time.time()

            train_metrics = self._run_epoch(self.train_loader, training=True)
            val_metrics = self._run_epoch(self.val_loader, training=False)

            # Scheduler step
            if epoch > warmup:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            elapsed = time.time() - start
            lr = self.optimizer.param_groups[-1]["lr"]

            # Log to history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            # W&B logging
            if self.use_wandb:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "lr": lr,
                })

            # Console output
            console.print(
                f"Epoch [{epoch:>3}/{epochs}]  "
                f"Train Loss: {train_metrics['loss']:.4f}  Acc: {train_metrics['accuracy']:.4f}  │  "
                f"Val Loss: {val_metrics['loss']:.4f}  Acc: {val_metrics['accuracy']:.4f}  │  "
                f"LR: {lr:.2e}  [{elapsed:.1f}s]"
            )

            # Save best checkpoint
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self._save_checkpoint(epoch, val_metrics, tag="best")
                console.print(f"  [green]✓ Best model saved (val_loss={self.best_val_loss:.4f})[/]")

            # Early stopping
            if self.early_stopping(val_metrics["loss"]):
                console.print(f"\n[yellow]Early stopping triggered at epoch {epoch}.[/]")
                break

        # Save final checkpoint
        self._save_checkpoint(epoch, val_metrics, tag="final")
        console.print("\n[bold green]Training complete![/]")
        return self.history

    def _save_checkpoint(self, epoch: int, metrics: dict, tag: str = "best"):
        path = self.checkpoint_dir / f"model_{tag}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": metrics["loss"],
            "val_accuracy": metrics["accuracy"],
            "config": self.config,
        }, path)

    def load_best_checkpoint(self):
        path = self.checkpoint_dir / "model_best.pth"
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        console.print(f"[green]✓ Loaded best checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})[/]")
        return ckpt
