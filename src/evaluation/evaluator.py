"""
src/evaluation/evaluator.py
────────────────────────────
Full evaluation suite:
  - Accuracy, F1, Precision, Recall
  - AUC-ROC (per class + macro)
  - Confusion matrix
  - ROC curves
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)
from rich.console import Console
from rich.table import Table

console = Console()


class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        class_names: List[str],
        device: torch.device,
        results_dir: str = "outputs/results",
    ):
        self.model = model
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def predict(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference, return (true_labels, predicted_labels, probabilities)."""
        self.model.eval()
        all_labels, all_preds, all_probs = [], [], []

        for images, labels in tqdm(self.test_loader, desc="Evaluating", ncols=80):
            images = images.to(self.device)
            logits = self.model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

        return (
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
        )

    def evaluate(self) -> Dict[str, float]:
        """Compute all metrics and save plots."""
        labels, preds, probs = self.predict()

        metrics = {
            "accuracy":  accuracy_score(labels, preds),
            "f1_macro":  f1_score(labels, preds, average="macro"),
            "f1_weighted": f1_score(labels, preds, average="weighted"),
            "precision": precision_score(labels, preds, average="macro", zero_division=0),
            "recall":    recall_score(labels, preds, average="macro", zero_division=0),
        }

        # AUC-ROC (one-vs-rest for multiclass)
        try:
            metrics["auc_roc_macro"] = roc_auc_score(
                labels, probs, multi_class="ovr", average="macro"
            )
        except Exception:
            metrics["auc_roc_macro"] = 0.0

        # Print metrics table
        table = Table(title="Test Set Metrics", style="bold")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for k, v in metrics.items():
            table.add_row(k, f"{v:.4f}")
        console.print(table)

        # Detailed classification report
        report = classification_report(labels, preds, target_names=self.class_names)
        console.print(f"\n[bold]Per-Class Report:[/]\n{report}")

        # Save plots
        self._plot_confusion_matrix(labels, preds)
        self._plot_roc_curves(labels, probs)

        # Save metrics to file
        with open(self.results_dir / "metrics.txt", "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")
            f.write(f"\n{report}")

        return metrics

    def _plot_confusion_matrix(self, labels: np.ndarray, preds: np.ndarray):
        cm = confusion_matrix(labels, preds)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, data, title, fmt in zip(
            axes,
            [cm, cm_norm],
            ["Confusion Matrix (counts)", "Confusion Matrix (normalized)"],
            ["d", ".2%"],
        ):
            sns.heatmap(
                data, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=self.class_names, yticklabels=self.class_names,
                ax=ax, linewidths=0.5,
            )
            ax.set_xlabel("Predicted", fontsize=12)
            ax.set_ylabel("Actual", fontsize=12)
            ax.set_title(title, fontsize=13, fontweight="bold")

        plt.tight_layout()
        path = self.results_dir / "confusion_matrix.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        console.print(f"[green]✓ Confusion matrix saved: {path}[/]")

    def _plot_roc_curves(self, labels: np.ndarray, probs: np.ndarray):
        n_classes = len(self.class_names)
        fig, ax = plt.subplots(figsize=(8, 6))

        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

        for i, (cls_name, color) in enumerate(zip(self.class_names, colors)):
            binary_labels = (labels == i).astype(int)
            if binary_labels.sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(binary_labels, probs[:, i])
            auc = roc_auc_score(binary_labels, probs[:, i])
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls_name} (AUC={auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves (One-vs-Rest)", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        path = self.results_dir / "roc_curves.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        console.print(f"[green]✓ ROC curves saved: {path}[/]")
