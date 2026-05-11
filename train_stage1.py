"""
Stage 1: Train the short-term precipitation nowcasting model.

This script trains a SimVP model with Inception-Unet translator to predict
T'_s = 6 future radar frames from T = 13 past frames.

Usage:
    python train_stage1.py --config configs/sevir.yaml
    python train_stage1.py --config configs/sevir.yaml --data_root /path/to/sevir
"""

import os
import sys
import time
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
import yaml

from models import SimVP
from losses import WeightedMSELoss
from metrics import NowcastMetrics
from data import build_sevir_dataloaders


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_dir: str, name: str = "stage1") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    logger: logging.Logger,
    epoch: int,
    log_interval: int = 50,
    grad_clip: float = 1.0,
    grad_accum_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    optimizer.zero_grad()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)    # (B, T_in, 1, H, W)
        targets = targets.to(device)  # (B, T_out, 1, H, W)

        preds = model(inputs)         # (B, T_out, 1, H, W)
        # Scale loss so effective gradient magnitude matches original batch_size
        loss = criterion(preds, targets) / grad_accum_steps
        loss.backward()

        total_loss += loss.item() * grad_accum_steps   # log unscaled loss
        n_batches += 1

        is_last_batch = (batch_idx + 1 == len(loader))
        if (batch_idx + 1) % grad_accum_steps == 0 or is_last_batch:
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / n_batches
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                f"Loss: {avg_loss:.4f} | LR: {lr:.6f}"
            )

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    metrics: NowcastMetrics,
    device: torch.device,
) -> tuple:
    model.eval()
    metrics.reset()
    total_loss = 0.0
    n_batches = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        preds = model(inputs)
        loss = criterion(preds, targets)
        total_loss += loss.item()
        n_batches += 1

        # Update metrics (move to CPU for numpy ops)
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()
        metrics.update(preds_np, targets_np)

    val_loss = total_loss / max(n_batches, 1)
    results = metrics.compute()
    return val_loss, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SimCast Stage 1: Short-term training")
    parser.add_argument("--config", type=str, default="configs/sevir.yaml")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override data_root in config")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    if args.data_root:
        cfg["dataset"]["data_root"] = args.data_root

    # Setup
    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = cfg["checkpoints"]["dir"]
    log_dir = cfg["logging"]["log_dir"]
    logger = setup_logger(log_dir, name="stage1")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {cfg}")

    # Data
    logger.info("Building dataloaders...")
    train_loader, val_loader, _ = build_sevir_dataloaders(
        data_root=cfg["dataset"]["data_root"],
        in_len=cfg["dataset"]["in_len"],
        out_len=cfg["dataset"]["out_len_short"],   # Stage 1: short-term
        batch_size=cfg["dataloader"]["batch_size"],
        num_workers=cfg["dataloader"]["num_workers"],
        pin_memory=cfg["dataloader"]["pin_memory"],
        normalize=cfg["dataset"]["normalize"],
    )
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Model
    T_in = cfg["dataset"]["in_len"]
    T_out_short = cfg["dataset"]["out_len_short"]
    H = W = cfg["dataset"]["resolution"]
    in_shape = (T_in, 1, H, W)

    model = SimVP(
        in_shape=in_shape,
        T_out=T_out_short,
        hid_S=cfg["model"]["hid_S"],
        hid_T=cfg["model"]["hid_T"],
        N_S=cfg["model"]["N_S"],
        N_T=cfg["model"]["N_T"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # Loss
    criterion = WeightedMSELoss(
        tau=cfg["loss"]["tau"],
        wmax=cfg["loss"]["wmax"],
    )

    # Optimizer and scheduler
    optimizer = Adam(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    total_steps = len(train_loader) * cfg["training"]["max_epochs"]
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg["scheduler"]["max_lr"],
        total_steps=total_steps,
        pct_start=cfg["scheduler"]["pct_start"],
    )

    # Metrics
    eval_metrics = NowcastMetrics(
        thresholds=cfg["evaluation"]["csi_thresholds"],
        pool_sizes=cfg["evaluation"]["pool_sizes"],
    )

    # Resume
    start_epoch = 1
    best_csi_m = -1.0
    patience_counter = 0

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_csi_m = ckpt.get("best_csi_m", -1.0)
        logger.info(f"Resumed from epoch {ckpt['epoch']}, best CSI-M: {best_csi_m:.4f}")

    # Training loop
    logger.info(f"Starting Stage 1 training for {cfg['training']['max_epochs']} epochs...")
    max_epochs = cfg["training"]["max_epochs"]
    val_interval = cfg["training"]["val_interval"]
    early_stop = cfg["training"]["early_stop_patience"]

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, logger, epoch,
            log_interval=cfg["logging"]["log_interval"],
            grad_clip=cfg["training"]["grad_clip"],
            grad_accum_steps=cfg.get("grad_accum_steps", 1),
        )
        elapsed = time.time() - t0
        logger.info(f"Epoch {epoch}/{max_epochs} | Train Loss: {train_loss:.4f} | Time: {elapsed:.1f}s")

        # Validation
        if epoch % val_interval == 0 or epoch == max_epochs:
            val_loss, val_results = validate(
                model, val_loader, criterion, eval_metrics, device
            )
            csi_m = val_results.get("CSI_M_POOL1", 0.0)
            ssim = val_results.get("SSIM", 0.0)
            hss = val_results.get("HSS", 0.0)

            logger.info(
                f"[Val] Epoch {epoch} | Loss: {val_loss:.4f} | "
                f"CSI-M: {csi_m:.4f} | SSIM: {ssim:.4f} | HSS: {hss:.4f}"
            )
            logger.info(eval_metrics.summary_str())

            # Save best checkpoint
            if csi_m > best_csi_m:
                best_csi_m = csi_m
                patience_counter = 0
                ckpt_path = os.path.join(ckpt_dir, cfg["checkpoints"]["stage1_name"])
                save_checkpoint({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_csi_m": best_csi_m,
                    "val_results": val_results,
                    "config": cfg,
                }, ckpt_path)
                logger.info(f"Saved best Stage 1 checkpoint: {ckpt_path} (CSI-M={best_csi_m:.4f})")
            else:
                patience_counter += val_interval
                if patience_counter >= early_stop:
                    logger.info(f"Early stopping at epoch {epoch} (patience={early_stop})")
                    break

    logger.info(f"Stage 1 training complete. Best CSI-M: {best_csi_m:.4f}")
    logger.info(f"Checkpoint saved to: {os.path.join(ckpt_dir, cfg['checkpoints']['stage1_name'])}")


if __name__ == "__main__":
    main()
