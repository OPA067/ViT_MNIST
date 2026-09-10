"""Training script for ViT on MNIST.

Features:
  - Automatic Mixed Precision (AMP) for faster training
  - Warmup + Cosine annealing LR schedule
  - Exponential Moving Average (EMA) of model weights
  - Early stopping with best-model restoration
  - TensorBoard logging
  - Checkpoint resuming
  - Label smoothing

Usage::

    # Basic training
    python train.py

    # With AMP, custom LR, longer warmup
    python train.py --amp --lr 5e-4 --warmup_epochs 10

    # Resume from checkpoint
    python train.py --resume checkpoints/best.pth

    # See all options
    python train.py --help
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import cast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler  # type: ignore

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None  # type: ignore

from model import ViT
from utils import (
    set_seed,
    get_device,
    count_parameters,
    memory_footprint,
    AverageMeter,
    WarmupCosineLR,
    ModelEMA,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    format_epoch_log,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ViT on MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")

    # Optimizer
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing epsilon")

    # Regularization
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (MLP)")
    parser.add_argument("--attn_dropout", type=float, default=0.0, help="Attention dropout rate")
    parser.add_argument("--drop_path_rate", type=float, default=0.1, help="Stochastic depth rate")

    # LR schedule
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Linear warmup epochs (0 to disable)")
    parser.add_argument("--lr_min", type=float, default=1e-6, help="Minimum LR at end of cosine annealing")

    # Training tricks
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision (fp16)")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate (0 to disable)")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm")

    # Early stopping
    parser.add_argument("--early_stop_patience", type=int, default=15, help="Early stop patience (0 to disable)")

    # Evaluation
    parser.add_argument("--eval_every", type=int, default=10, help="Evaluate every N epochs")
    parser.add_argument("--eval_ema", action="store_true", help="Also evaluate EMA model")

    # System
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda, mps, cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")

    # Paths
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--log_dir", type=str, default="./experiments", help="Log directory (terminal output saved as train.log)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")

    # Model
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=10, help="Number of Transformer blocks")
    parser.add_argument("--n_heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--mlp_ratio", type=float, default=2.0, help="MLP hidden dim / embed_dim")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_dataloaders(batch_size: int, workers: int) -> tuple[DataLoader, DataLoader]:
    """Build MNIST train/test dataloaders."""
    # Standard MNIST normalization values
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    amp_enabled: bool = False,
) -> tuple[float, float]:
    """Evaluate accuracy and loss.

    Returns (avg_loss, accuracy).
    """
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, labels)

        _, predicted = outputs.max(1)
        correct = (predicted == labels).sum().item()
        batch_size = labels.size(0)

        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(100.0 * correct / batch_size, batch_size)

    return loss_meter.avg, acc_meter.avg


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
    scaler: GradScaler | None,
    ema: ModelEMA | None,
    epoch: int,
    writer: SummaryWriter | None,
    global_step: int,
) -> tuple[float, float, int]:
    """Train for one epoch.

    Returns (avg_loss, avg_accuracy, updated_global_step).
    """
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    amp_enabled = scaler is not None

    pbar = tqdm(loader, desc=f"Train E{epoch:03d}", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        # EMA update
        if ema is not None:
            ema.update(model)

        # Metrics
        _, predicted = outputs.max(1)
        correct = (predicted == labels).sum().item()
        batch_size = labels.size(0)

        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(100.0 * correct / batch_size, batch_size)
        global_step += 1

        # TensorBoard per-step logging
        if writer is not None:
            writer.add_scalar("Step/loss", loss.item(), global_step)
            writer.add_scalar("Step/acc", 100.0 * correct / batch_size, global_step)

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "acc": f"{acc_meter.avg:.2f}%",
        })

    return loss_meter.avg, acc_meter.avg, global_step


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str) -> logging.Logger:
    """Configure logging: console + file under log_dir/train.log.

    All terminal output (print) is mirrored to experiments/train.log.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")

    logger = logging.getLogger("vit_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console handler — raw format (no timestamps on terminal)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console_fmt = logging.Formatter("%(message)s")
    console.setFormatter(console_fmt)

    # File handler — with timestamps
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(logging.INFO)
    file_fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # Setup terminal logging: output goes to both console and experiments/train.log
    logger = setup_logging(args.log_dir)

    device = get_device(args.device)
    logger.info(f"Device: {device}")
    logger.info(f"AMP:    {'enabled' if args.amp else 'disabled'}")
    logger.info("")

    # Data
    train_loader, test_loader = get_dataloaders(args.batch_size, args.num_workers)
    logger.info(f"Train samples: {len(train_loader.dataset):,}")
    logger.info(f"Test  samples: {len(test_loader.dataset):,}")

    # Model
    model = ViT(
        img_size=28,
        patch_size=7,
        in_ch=1,
        n_classes=10,
        embed_dim=args.embed_dim,
        depth=args.depth,
        n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        p=args.dropout,
        attn_p=args.attn_dropout,
        drop_path_rate=args.drop_path_rate,
    ).to(device)

    logger.info(model.model_summary())
    logger.info(memory_footprint(model))
    logger.info("")

    # Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = WarmupCosineLR(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        eta_min=args.lr_min,
    )

    # AMP scaler
    scaler = GradScaler() if args.amp else None

    # EMA
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    # Early stopping
    early_stop = (
        EarlyStopping(patience=args.early_stop_patience, mode="max", restore_best=True)
        if args.early_stop_patience > 0
        else None
    )

    # TensorBoard — writes under experiments/tensorboard/
    writer = None
    if HAS_TENSORBOARD:
        tb_dir = os.path.join(args.log_dir, "tensorboard")
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(log_dir=os.path.join(tb_dir, run_name))
        logger.info(f"TensorBoard: logging to {writer.log_dir}")
    else:
        logger.info("TensorBoard: not installed (pip install tensorboard)")
    logger.info("")

    # Resume from checkpoint
    start_epoch = 1
    best_acc = 0.0
    if args.resume:
        ckpt = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema_model=ema,
        )
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("metrics", {}).get("test_acc", 0.0)
        logger.info(f"Resuming from epoch {start_epoch}, best_acc={best_acc:.2f}%")

    global_step = 0
    total_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss, train_acc, global_step = train_epoch(
            model, train_loader, optimizer, criterion, device,
            args.grad_clip, scaler, ema, epoch, writer, global_step,
        )
        scheduler.step()

        epoch_time = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        log_str = format_epoch_log(
            epoch, args.epochs, train_loss, train_acc, lr=lr, time_s=epoch_time,
        )
        logger.info(log_str)

        # TensorBoard epoch logging
        if writer is not None:
            writer.add_scalar("Epoch/train_loss", train_loss, epoch)
            writer.add_scalar("Epoch/train_acc", train_acc, epoch)
            writer.add_scalar("Epoch/lr", lr, epoch)

        # Evaluation
        should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if should_eval:
            test_loss, test_acc = evaluate(model, test_loader, device, criterion, args.amp)
            logger.info(f"  >> Test  loss={test_loss:.4f}  acc={test_acc:.2f}%")

            if writer is not None:
                writer.add_scalar("Epoch/test_loss", test_loss, epoch)
                writer.add_scalar("Epoch/test_acc", test_acc, epoch)

            # Save best
            if test_acc > best_acc:
                best_acc = test_acc
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    {"test_acc": test_acc, "test_loss": test_loss},
                    os.path.join(args.save_dir, "best.pth"),
                    ema_model=ema,
                )
                logger.info(f"  >> New best accuracy: {best_acc:.2f}% — saved.")

            # EMA evaluation
            if ema is not None and args.eval_ema:
                ema_test_loss, ema_test_acc = evaluate(
                    ema.ema_model, test_loader, device, criterion, args.amp
                )
                logger.info(f"  >> EMA   loss={ema_test_loss:.4f}  acc={ema_test_acc:.2f}%")
                if writer is not None:
                    writer.add_scalar("Epoch/ema_test_acc", ema_test_acc, epoch)

            # Early stopping
            if early_stop is not None:
                should_stop = early_stop(test_acc, model, optimizer=optimizer, scheduler=scheduler)
                if should_stop:
                    logger.info(f"\nEarly stopping triggered at epoch {epoch}.")
                    break

    total_time = time.time() - total_start
    logger.info("")
    logger.info("=" * 50)
    logger.info(f"Training finished in {total_time / 60:.1f} min")
    logger.info(f"Best test accuracy: {best_acc:.2f}%")
    logger.info("=" * 50)

    if writer is not None:
        writer.close()

if __name__ == "__main__":
    main()