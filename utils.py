"""Utility helpers for MNIST ViT training.

Provides:
  - Reproducibility utilities
  - Optimizer wrappers (EMA, Warmup+Cosine LR)
  - Training metrics tracking (AverageMeter)
  - Early stopping
  - Checkpoint I/O with metadata
  - Model parameter inspection
"""
from __future__ import annotations

import math
import os
import random
import shutil
import sys
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds deterministically across all relevant libraries.

    Note: deterministic CUDA operations may be slower but ensure repeatability.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic mode: forces same algorithm selection each time
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Reproducibility: random seed set to {seed}")


def get_device(prefer: str = "cuda") -> torch.device:
    """Select the best available device."""
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class AverageMeter:
    """Track and compute running averages of scalar values.

    Useful for loss, accuracy, learning rate, etc. over epochs or batches.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0      # last value
        self.avg = 0.0      # running average
        self.sum = 0.0      # accumulated sum
        self.count = 0      # accumulated count

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        return f"{self.avg:.4f}"


# ---------------------------------------------------------------------------
# Learning Rate Scheduler: Warmup + Cosine Annealing
# ---------------------------------------------------------------------------

class WarmupCosineLR(_LRScheduler):
    """Warmup followed by cosine annealing learning rate schedule.

    Combines a linear warmup phase (lr rises from 0 to target) with
    a cosine decay phase (lr decays to eta_min).

    Args:
        optimizer: wrapped optimizer.
        warmup_epochs: number of epochs for linear warmup (0 to skip).
        total_epochs: total number of training epochs.
        eta_min: minimum learning rate after cosine annealing.
        last_epoch: index of last epoch.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.cosine_epochs = total_epochs - warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # Warmup phase: linearly increase from 0 to base_lr
        if self.last_epoch < self.warmup_epochs and self.warmup_epochs > 0:
            ratio = self.last_epoch / self.warmup_epochs
            return [base_lr * ratio for base_lr in self.base_lrs]

        # Cosine annealing phase
        cosine_epoch = self.last_epoch - self.warmup_epochs
        if cosine_epoch >= self.cosine_epochs or self.cosine_epochs <= 0:
            return [self.eta_min for _ in self.base_lrs]

        progress = cosine_epoch / self.cosine_epochs
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * progress))
            / 2
            for base_lr in self.base_lrs
        ]

    def get_last_lr_scaled(self, scale: float = 1.0) -> List[float]:
        """Return LR scaled by a constant factor."""
        return [lr * scale for lr in self.get_lr()]


# ---------------------------------------------------------------------------
# Exponential Moving Average (EMA)
# ---------------------------------------------------------------------------

class ModelEMA:
    """Maintain an exponential moving average of model parameters.

    During training: model weights are normal, but EMA shadow copy
    accumulates a moving average. At inference time, the EMA weights
    often yield better generalization.

    Implementation follows PyTorch best practices:
        ema_param = decay * ema_param + (1 - decay) * model_param
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        # Deep copy so EMA doesn't affect training
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        # Disable gradients for EMA parameters
        for p in self.ema_model.parameters():
            p.requires_grad = False

    def update(self, model: nn.Module) -> None:
        """Update EMA parameters from current model weights."""
        with torch.no_grad():
            for ema_p, model_p in zip(
                self.ema_model.parameters(), model.parameters()
            ):
                if ema_p.dtype in [torch.float32, torch.float16]:
                    ema_p.mul_(self.decay).add_(model_p, alpha=1 - self.decay)

    def state_dict(self) -> Dict[str, Any]:
        """Return EMA model state dict for checkpointing."""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load EMA state from a checkpoint."""
        self.ema_model.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Saves the best model checkpoint and optionally restores
    model weights to the best state at the end.

    Args:
        patience: number of epochs to wait for improvement before stopping.
        min_delta: minimum change to qualify as an improvement.
        mode: 'min' for loss, 'max' for accuracy.
        restore_best: whether to restore model/optimizer to best state at end.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.0001,
        mode: str = "max",
        restore_best: bool = True,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.best_score: float | None = None
        self.counter = 0
        self.early_stop = False
        self.best_state: Dict[str, Any] | None = None
        self.score_str = "+∞" if mode == "min" else "-∞"

    def __call__(self, score: float, model: nn.Module, **kwargs: Any) -> bool:
        """Record score and check whether to stop.

        Returns True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            self._save_best(model, **kwargs)
            return False

        improved = (
            score < self.best_score - self.min_delta
            if self.mode == "min"
            else score > self.best_score + self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
            self._save_best(model, **kwargs)
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best and self.best_state is not None:
                self._restore_best(model, **kwargs)
        return self.early_stop

    def _save_best(self, model: nn.Module, **kwargs: Any) -> None:
        """Store the best model state in memory."""
        self.best_state = {
            "model": deepcopy(model.state_dict()),
            **{k: deepcopy(v.state_dict()) for k, v in kwargs.items()},
        }
        self.score_str = f"{self.best_score:.4f}"

    def _restore_best(self, model: nn.Module, **kwargs: Any) -> None:
        """Restore model and aux states to best recorded state."""
        if self.best_state is None:
            return
        model.load_state_dict(self.best_state["model"])
        for key, obj in kwargs.items():
            if key in self.best_state:
                obj.load_state_dict(self.best_state[key])
        print(f"  >> Early stopping: restored best ({self.score_str}).")

    @property
    def best(self) -> float | None:
        return self.best_score


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: _LRScheduler | None,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
    ema_model: ModelEMA | None = None,
) -> None:
    """Save a training checkpoint with rich metadata.

    Args:
        model: the model to save.
        optimizer: the optimizer state.
        scheduler: the LR scheduler state (optional).
        epoch: current epoch number.
        metrics: dictionary of metric values (e.g. accuracy, loss).
        path: file path to save.
        ema_model: optional EMA model to also checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt: Dict[str, Any] = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        ckpt["scheduler_state"] = scheduler.state_dict()
    if ema_model is not None:
        ckpt["ema_model_state"] = ema_model.state_dict()

    # Save to temp path first, then atomic rename for safety
    tmp_path = path + ".tmp"
    torch.save(ckpt, tmp_path)
    shutil.move(tmp_path, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: _LRScheduler | None = None,
    ema_model: ModelEMA | None = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint and restore model (and optionally optimizer/scheduler/EMA) state.

    Args:
        path: checkpoint file path.
        model: model to load weights into.
        optimizer: optimizer to restore state (optional).
        scheduler: scheduler to restore state (optional).
        ema_model: EMA model to restore (optional).
        strict: whether to strictly enforce key matching.

    Returns:
        Checkpoint metadata dict (epoch, metrics, etc.).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Handle both new and old checkpoint formats
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=strict)
    else:
        # Legacy format: state dict directly
        model.load_state_dict(ckpt, strict=strict)

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    if ema_model is not None and "ema_model_state" in ckpt:
        ema_model.load_state_dict(ckpt["ema_model_state"])

    print(f"Loaded checkpoint: epoch={ckpt.get('epoch', '?')}, "
          f"metrics={ckpt.get('metrics', {})}")
    return ckpt


# ---------------------------------------------------------------------------
# Parameter Inspection
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return the number of model parameters.

    Args:
        trainable_only: if True, count only parameters with requires_grad.
    """
    return sum(
        p.numel()
        for p in model.parameters()
        if not trainable_only or p.requires_grad
    )


def memory_footprint(model: nn.Module) -> str:
    """Estimate GPU memory footprint of model parameters."""
    param_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
    buffer_bytes = sum(
        b.numel() * b.element_size()
        for b in model.buffers()
    )
    total_mb = (param_bytes + buffer_bytes) / (1024 ** 2)
    param_mb = param_bytes / (1024 ** 2)
    return (
        f"  Parameters: {param_mb:.2f} MB  |  Buffers: "
        f"{(buffer_bytes / (1024 ** 2)):.2f} MB  |  Total: {total_mb:.2f} MB"
    )


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def format_epoch_log(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    train_acc: float,
    test_loss: float | None = None,
    test_acc: float | None = None,
    lr: float | None = None,
    time_s: float | None = None,
) -> str:
    """Format a single-line epoch summary for terminal display."""
    parts = [
        f"Epoch {epoch:03d}/{total_epochs}",
        f"train_loss={train_loss:.4f}",
        f"train_acc={train_acc:.2f}%",
    ]
    if test_loss is not None:
        parts.append(f"test_loss={test_loss:.4f}")
    if test_acc is not None:
        parts.append(f"test_acc={test_acc:.2f}%")
    if lr is not None:
        parts.append(f"lr={lr:.2e}")
    if time_s is not None:
        parts.append(f"time={time_s:.1f}s")
    return "  ".join(parts)
