"""Evaluation and analysis script for ViT on MNIST.

Provides:
  - Overall and per-class accuracy
  - Top-k accuracy (e.g. top-5)
  - Confusion matrix visualization
  - Per-class precision / recall / F1-score
  - Attention map visualization (which patches the model focuses on)
  - Misclassification analysis (confident wrong predictions)

Usage::

    # Basic evaluation
    python eval.py --checkpoint checkpoints/best.pth

    # Full analysis with visualizations
    python eval.py --checkpoint checkpoints/best.pth --visualize --attn_map --errors

    # Compare EMA weights vs regular weights
    python eval.py --checkpoint checkpoints/best.pth --use_ema

    # See all options
    python eval.py --help
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ViT
from utils import set_seed, get_device, load_checkpoint


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ViT on MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda, mps, cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output directory
    parser.add_argument("--output_dir", type=str, default="./experiments", help="Directory for saving visualization PNGs")

    # Visualization toggles
    parser.add_argument("--visualize", action="store_true", help="Show random prediction grid")
    parser.add_argument("--attn_map", action="store_true", help="Visualize attention maps")
    parser.add_argument("--errors", action="store_true", help="Show worst misclassifications")

    # EMA toggle
    parser.add_argument("--use_ema", action="store_true", help="Use EMA weights from checkpoint")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_test_loader(batch_size: int, workers: int = 4) -> DataLoader:
    """Build MNIST test dataloader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    return DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device, use_ema: bool = False) -> nn.Module:
    """Load ViT model, optionally using EMA weights."""
    model = ViT(
        img_size=28,
        patch_size=7,
        in_ch=1,
        n_classes=10,
        embed_dim=256,
        depth=10,
        n_heads=8,
        mlp_ratio=2.0,
        p=0.1,
        attn_p=0.0,
        drop_path_rate=0.1,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_key = "ema_model_state" if use_ema and "ema_model_state" in ckpt else "model_state"
    if state_key in ckpt:
        model.load_state_dict(ckpt[state_key])
    else:
        model.load_state_dict(ckpt)

    weight_type = "EMA" if (use_ema and "ema_model_state" in ckpt) else "regular"
    print(f"Loaded {weight_type} checkpoint from {checkpoint_path}")
    epoch = ckpt.get("epoch", "?")
    metrics = ckpt.get("metrics", {})
    print(f"  Epoch: {epoch}, Metrics: {metrics}")
    return model


# ---------------------------------------------------------------------------
# Comprehensive evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_comprehensive(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Run full evaluation: acc, top-k acc, confusion matrix, per-class report.

    Returns dict with all computed metrics.
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []
    all_images: list[torch.Tensor] = []

    for images, labels in loader:
        images_dev = images.to(device)
        outputs = model(images_dev)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs)
        all_images.append(images)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_images = torch.cat(all_images)

    # Overall accuracy
    overall_acc = 100.0 * (all_preds == all_labels).sum() / len(all_labels)

    # Per-class accuracy
    class_correct = np.zeros(10, dtype=int)
    class_total = np.zeros(10, dtype=int)
    for label, pred in zip(all_labels, all_preds):
        class_total[label] += 1
        if label == pred:
            class_correct[label] += 1
    per_class_acc = [100.0 * c / t if t > 0 else 0.0 for c, t in zip(class_correct, class_total)]

    # Top-k accuracy
    top5_correct = 0
    for i in range(len(all_labels)):
        top5 = np.argsort(all_probs[i])[-5:]
        if all_labels[i] in top5:
            top5_correct += 1
    top5_acc = 100.0 * top5_correct / len(all_labels)

    # Save misclassified indices for error analysis
    wrong_indices = np.where(all_preds != all_labels)[0]
    wrong_confidences = all_probs[wrong_indices].max(axis=1)
    # Sort by confidence (most confidently wrong first)
    worst_order = np.argsort(wrong_confidences)[::-1]
    worst_indices = wrong_indices[worst_order]

    return {
        "overall_acc": overall_acc,
        "top5_acc": top5_acc,
        "per_class_acc": per_class_acc,
        "class_correct": class_correct,
        "class_total": class_total,
        "predictions": all_preds,
        "labels": all_labels,
        "probs": all_probs,
        "images": all_images,
        "worst_indices": worst_indices,
    }


def print_report(metrics: dict[str, Any]) -> None:
    """Pretty-print evaluation results."""
    print()
    print("=" * 55)
    print(f"  Overall Top-1 Accuracy: {metrics['overall_acc']:.2f}%")
    print(f"  Overall Top-5 Accuracy: {metrics['top5_acc']:.2f}%")
    print("=" * 55)
    print("  Per-class Accuracy:")
    for i in range(10):
        acc = metrics['per_class_acc'][i]
        c = metrics['class_correct'][i]
        t = metrics['class_total'][i]
        print(f"    Digit {i}: {acc:>6.2f}%  ({c:>4}/{t:<5})")
    print("=" * 55)
    print()

    # Classification report (precision, recall, f1)
    print("Classification Report:")
    print(classification_report(
        metrics["labels"],
        metrics["predictions"],
        target_names=[str(i) for i in range(10)],
        digits=4,
    ))


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: str = "confusion_matrix.png",
) -> None:
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10),
        cbar_kws={"label": "Count"},
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix — ViT on MNIST")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved confusion matrix to {save_path}")


def visualize_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 16,
    save_path: str = "predictions_grid.png",
) -> None:
    """Show a grid of random predictions with true/false color coding."""
    model.eval()

    # Collect a few batches
    all_images, all_labels = [], []
    for images, labels in loader:
        all_images.append(images)
        all_labels.append(labels)
        if sum(len(im) for im in all_images) >= 1000:
            break

    images = torch.cat(all_images)[:1000]
    labels = torch.cat(all_labels)[:1000]

    indices = random.sample(range(len(images)), n_samples)
    sample_images = images[indices].to(device)
    true_labels = labels[indices]

    with torch.no_grad():
        outputs = model(sample_images)
        _, preds = outputs.max(1)

    sample_images = sample_images.cpu()

    rows = int(np.sqrt(n_samples))
    cols = n_samples // rows
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        img = sample_images[idx].squeeze().numpy()
        pred = preds[idx].item()
        true = true_labels[idx].item()
        color = "forestgreen" if pred == true else "crimson"

        ax.imshow(img, cmap="gray")
        ax.set_title(f"Pred: {pred}  True: {true}", color=color, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Random Prediction Sample", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved prediction grid to {save_path}")


def visualize_attention_maps(
    model: ViT,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 8,
    save_path: str = "attention_maps.png",
) -> None:
    """Visualize attention maps showing which patches the model focuses on.

    For each sample, overlays the average attention weights (across heads)
    from the last layer onto the original image.
    """
    model.eval()

    # Collect a batch
    for images, labels in loader:
        images = images[:n_samples].to(device)
        labels = labels[:n_samples]
        break

    # Get attention maps from all layers
    attn_maps = model.get_attention_maps(images)  # list of (B, H, N, N)
    # Use last layer
    last_attn = attn_maps[-1]  # (B, H, N, N) where N = n_patches + 1 (cls)

    # Average over heads, take attention FROM cls token TO all patches
    # last_attn[..., 0, :] = cls attending to all tokens (cls first, then patches)
    cls_attn = last_attn.mean(dim=1)[:, 0, 1:]  # (B, N_patches)

    # Reshape to 2D grid (4x4 patches for our 28x28 image with 7x7 patches)
    n_patches_side = int(np.sqrt(cls_attn.shape[1]))  # 4
    cls_attn_2d = cls_attn.reshape(n_samples, n_patches_side, n_patches_side)

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 5))
    for i in range(n_samples):
        # Original image
        img = images[i].cpu().squeeze().numpy()
        ax_img = axes[0, i]
        ax_img.imshow(img, cmap="gray")
        ax_img.set_title(f"True: {labels[i].item()}", fontsize=10)
        ax_img.axis("off")

        # Attention map overlay
        attn = cls_attn_2d[i].detach().cpu().numpy()
        ax_attn = axes[1, i]
        ax_attn.imshow(img, cmap="gray", alpha=0.6)
        # Upsample 4x4 attention to 28x28 using nearest-neighbor
        attn_resized = np.kron(attn, np.ones((7, 7)))
        ax_attn.imshow(attn_resized, cmap="hot", alpha=0.5)
        ax_attn.set_title("Attention", fontsize=10)
        ax_attn.axis("off")

    plt.suptitle(
        "Attention Maps (last layer, averaged over heads)\n"
        "Top: original image  |  Bottom: attention overlay (red = high attn)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved attention maps to {save_path}")


def visualize_errors(
    images: torch.Tensor,
    labels: np.ndarray,
    predictions: np.ndarray,
    probs: np.ndarray,
    worst_indices: np.ndarray,
    n_samples: int = 16,
    save_path: str = "misclassifications.png",
) -> None:
    """Show the most confidently wrong predictions."""
    n_show = min(n_samples, len(worst_indices))
    indices = worst_indices[:n_show]

    rows = int(np.sqrt(n_show))
    cols = (n_show + rows - 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    if n_show == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, ax in zip(indices, axes):
        img = images[idx].squeeze().numpy()
        pred = predictions[idx]
        true = labels[idx]
        conf = probs[idx][pred]

        ax.imshow(img, cmap="gray")
        ax.set_title(
            f"Pred: {pred} ({conf:.1%})\nTrue: {true}",
            color="crimson",
            fontsize=10,
            fontweight="bold",
        )
        ax.axis("off")

    # Hide unused subplots
    for ax in axes[len(indices):]:
        ax.axis("off")

    plt.suptitle(
        f"Top {n_show} Most Confident Misclassifications",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved misclassifications to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device(args.device)
    print(f"Device: {device}\n")

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, device, use_ema=args.use_ema)
    test_loader = get_test_loader(args.batch_size)

    # Evaluate
    print("Running evaluation...")
    metrics = evaluate_comprehensive(model, test_loader, device)
    print_report(metrics)

    # Confusion matrix (always saved)
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(metrics["labels"], metrics["predictions"], save_path=cm_path)

    # Optional visualizations
    if args.visualize:
        vis_path = os.path.join(args.output_dir, "predictions_grid.png")
        visualize_predictions(model, test_loader, device, save_path=vis_path)

    if args.attn_map:
        attn_path = os.path.join(args.output_dir, "attention_maps.png")
        visualize_attention_maps(model, test_loader, device, save_path=attn_path)

    if args.errors:
        n_wrong = len(metrics["worst_indices"])
        if n_wrong > 0:
            print(f"\nFound {n_wrong} misclassifications.")
            err_path = os.path.join(args.output_dir, "misclassifications.png")
            visualize_errors(
                metrics["images"],
                metrics["labels"],
                metrics["predictions"],
                metrics["probs"],
                metrics["worst_indices"],
                save_path=err_path,
            )
        else:
            print("\nPerfect accuracy — no misclassifications to show!")


if __name__ == "__main__":
    main()
