"""
Stage 2: Train the long-term nowcasting model with short-to-long term
knowledge distillation.

This script:
  1. Loads the Stage 1 short-term model checkpoint
  2. Autoregressively applies the short-term model to augment training data
     (appends T'_l synthetic frames to each training sequence)
  3. Trains a long-term SimVP model (T'_l = 12) on the augmented dataset

The key insight from SimCast: the long-term model learns from both
ground-truth radar data AND short-term model forecasts, enabling it to
capture both short-term and long-term temporal patterns.

Usage:
    python train_stage2.py --config configs/sevir.yaml \
                           --stage1_ckpt checkpoints/sevir/stage1_best.pth
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
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import yaml

from models import SimVP
from losses import WeightedMSELoss
from metrics import NowcastMetrics
from data import SEVIRDataset, build_sevir_dataloaders


# ---------------------------------------------------------------------------
# Utilities (same as Stage 1)
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_dir: str, name: str = "stage2") -> logging.Logger:
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
# Data augmentation: autoregressive inference with Stage 1 model
# ---------------------------------------------------------------------------

@torch.no_grad()
def augment_dataset_with_stage1(
    stage1_model: SimVP,
    train_dataset: SEVIRDataset,
    T_in: int,
    T_out_short: int,
    T_out_long: int,
    device: torch.device,
    logger: logging.Logger,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Use the Stage 1 model autoregressively to extend each training sample.

    For each training sample of shape (T_in + T_out_long, H, W):
      - Take the first T_in frames as input
      - Apply Stage 1 model autoregressively (n_steps = T_out_long // T_out_short)
        to generate T_out_long synthetic frames
      - Append these synthetic frames to the original sample
      - Result: (T_in + T_out_long + T_out_long, H, W)

    This augmented sequence allows the long-term model to be trained with
    random sub-sequence sampling, where some sub-sequences will include
    synthetic frames from the Stage 1 model.

    Args:
        stage1_model:  trained Stage 1 SimVP model
        train_dataset: original training dataset
        T_in:          number of input frames
        T_out_short:   Stage 1 prediction horizon
        T_out_long:    Stage 2 prediction horizon
        device:        compute device
        logger:        logger instance
        batch_size:    batch size for augmentation inference

    Returns:
        augmented_data: (N, T_in + 2*T_out_long, H, W) float32 numpy array
    """
    stage1_model.eval()
    n_steps = (T_out_long + T_out_short - 1) // T_out_short  # ceil division

    logger.info(
        f"Augmenting {len(train_dataset)} training samples with Stage 1 model "
        f"(n_autoregressive_steps={n_steps})..."
    )

    # Temporarily set dataset to use T_in + T_out_long length
    # We need the original samples to get the input context
    augmented_samples = []

    # Process in batches for efficiency
    indices = list(range(len(train_dataset)))
    n_batches = (len(indices) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        batch_indices = indices[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        # Collect input sequences (first T_in frames of each sample)
        batch_inputs = []
        batch_originals = []

        for idx in batch_indices:
            # Read the full seq_len window via lazy loader
            ref = train_dataset.index[idx]
            seq = train_dataset._read_event_frames(
                ref.h5_path, ref.event_idx, ref.frame_start, train_dataset.seq_len
            )  # (seq_len, H, W)
            inp = seq[:T_in]  # (T_in, H, W)
            batch_inputs.append(inp)
            batch_originals.append(seq)

        # Stack and move to device
        inputs_tensor = torch.from_numpy(
            np.stack(batch_inputs)[:, :, np.newaxis, :, :]  # (B, T_in, 1, H, W)
        ).float().to(device)

        # Autoregressive inference
        synthetic_frames = stage1_model.autoregressive_predict(
            inputs_tensor, n_steps=n_steps
        )  # (B, n_steps*T_out_short, 1, H, W)

        # Clip to T_out_long frames
        synthetic_frames = synthetic_frames[:, :T_out_long]  # (B, T_out_long, 1, H, W)
        synthetic_np = synthetic_frames.cpu().numpy()[:, :, 0, :, :]  # (B, T_out_long, H, W)

        # Clamp to valid range [0, 255]
        synthetic_np = np.clip(synthetic_np, 0, 255)

        # Append synthetic frames to original samples
        for i, (orig, synth) in enumerate(zip(batch_originals, synthetic_np)):
            # orig: (seq_len, H, W), synth: (T_out_long, H, W)
            augmented = np.concatenate([orig, synth], axis=0)  # (seq_len + T_out_long, H, W)
            augmented_samples.append(augmented)

        if (batch_idx + 1) % 100 == 0:
            logger.info(f"  Augmented {(batch_idx+1)*batch_size}/{len(indices)} samples")

    augmented_data = np.stack(augmented_samples)  # (N, seq_len+T_out_long, H, W)
    logger.info(f"Augmentation complete. Shape: {augmented_data.shape}")
    return augmented_data


# ---------------------------------------------------------------------------
# Training loop (same structure as Stage 1)
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
    recent_losses = []
    RECENT_WINDOW = 50

    optimizer.zero_grad()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        preds = model(inputs)
        loss = criterion(preds, targets) / grad_accum_steps
        loss.backward()

        unscaled = loss.item() * grad_accum_steps
        total_loss += unscaled
        n_batches += 1
        recent_losses.append(unscaled)
        if len(recent_losses) > RECENT_WINDOW:
            recent_losses.pop(0)

        is_last_batch = (batch_idx + 1 == len(loader))
        if (batch_idx + 1) % grad_accum_steps == 0 or is_last_batch:
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        if (batch_idx + 1) % log_interval == 0:
            recent_avg = sum(recent_losses) / len(recent_losses)
            epoch_avg  = total_loss / n_batches
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                f"Loss(recent/epoch): {recent_avg:.1f}/{epoch_avg:.1f} | LR: {lr:.6f}"
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
    parser = argparse.ArgumentParser(description="SimCast Stage 2: Long-term training with KD")
    parser.add_argument("--config", type=str, default="configs/sevir.yaml")
    parser.add_argument("--stage1_ckpt", type=str, required=True,
                        help="Path to Stage 1 checkpoint")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--skip_augmentation", action="store_true",
                        help="Skip data augmentation (use if already done)")
    parser.add_argument("--aug_cache", type=str, default=None,
                        help="Path to cached augmented data (.npy file)")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Override max_epochs in config (e.g. 15 for a quick trial)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg["dataset"]["data_root"] = args.data_root
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = args.max_epochs

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = cfg["checkpoints"]["dir"]
    log_dir = cfg["logging"]["log_dir"]
    logger = setup_logger(log_dir, name="stage2")
    logger.info(f"Device: {device}")

    T_in = cfg["dataset"]["in_len"]
    T_out_short = cfg["dataset"]["out_len_short"]
    T_out_long = cfg["dataset"]["out_len_long"]
    H = W = cfg["dataset"]["resolution"]

    # -----------------------------------------------------------------------
    # Load Stage 1 model
    # -----------------------------------------------------------------------
    logger.info(f"Loading Stage 1 model from {args.stage1_ckpt}...")
    stage1_ckpt = torch.load(args.stage1_ckpt, map_location=device)

    stage1_model = SimVP(
        in_shape=(T_in, 1, H, W),
        T_out=T_out_short,
        hid_S=cfg["model"]["hid_S"],
        hid_T=cfg["model"]["hid_T"],
        N_S=cfg["model"]["N_S"],
        N_T=cfg["model"]["N_T"],
    ).to(device)
    stage1_model.load_state_dict(stage1_ckpt["model_state"])
    stage1_model.eval()
    logger.info(f"Stage 1 model loaded (best CSI-M: {stage1_ckpt.get('best_csi_m', 'N/A')})")

    # -----------------------------------------------------------------------
    # Build training dataset (long-term)
    # -----------------------------------------------------------------------
    logger.info("Building training dataset...")
    train_ds = SEVIRDataset(
        data_root=cfg["dataset"]["data_root"],
        split="train",
        in_len=T_in,
        out_len=T_out_long,
        seq_len=T_in + T_out_long,
        stride=12,
        normalize=cfg["dataset"]["normalize"],
    )

    # -----------------------------------------------------------------------
    # Data augmentation with Stage 1 model
    # -----------------------------------------------------------------------
    if args.aug_cache and os.path.exists(args.aug_cache):
        logger.info(f"Loading cached augmented data from {args.aug_cache}")
        augmented_data = np.load(args.aug_cache)
    elif not args.skip_augmentation:
        augmented_data = augment_dataset_with_stage1(
            stage1_model=stage1_model,
            train_dataset=train_ds,
            T_in=T_in,
            T_out_short=T_out_short,
            T_out_long=T_out_long,
            device=device,
            logger=logger,
            batch_size=cfg["dataloader"]["batch_size"] * 2,
        )
        # Cache augmented data for future runs
        if args.aug_cache:
            logger.info(f"Saving augmented data to {args.aug_cache}")
            np.save(args.aug_cache, augmented_data)
    else:
        augmented_data = None
        logger.info("Skipping data augmentation")

    # Set augmented data on the training dataset
    if augmented_data is not None:
        train_ds.set_augmented_data(augmented_data)
        logger.info(f"Training dataset size after augmentation: {len(train_ds)}")

    # Build val/test loaders (long-term, no augmentation)
    _, val_loader, _ = build_sevir_dataloaders(
        data_root=cfg["dataset"]["data_root"],
        in_len=T_in,
        out_len=T_out_long,
        batch_size=cfg["dataloader"]["batch_size"],
        num_workers=cfg["dataloader"]["num_workers"],
        pin_memory=cfg["dataloader"]["pin_memory"],
        normalize=cfg["dataset"]["normalize"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["dataloader"]["batch_size"],
        shuffle=True,
        num_workers=cfg["dataloader"]["num_workers"],
        pin_memory=cfg["dataloader"]["pin_memory"],
        drop_last=True,
    )

    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # -----------------------------------------------------------------------
    # Stage 2 model (same architecture, different T_out)
    # -----------------------------------------------------------------------
    model = SimVP(
        in_shape=(T_in, 1, H, W),
        T_out=T_out_long,
        hid_S=cfg["model"]["hid_S"],
        hid_T=cfg["model"]["hid_T"],
        N_S=cfg["model"]["N_S"],
        N_T=cfg["model"]["N_T"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Stage 2 model parameters: {n_params:,}")

    # Loss, optimizer, scheduler
    criterion = WeightedMSELoss(
        tau=cfg["loss"]["tau"],
        wmax=cfg["loss"]["wmax"],
    )
    optimizer = Adam(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    warmup_epochs = cfg["scheduler"].get("warmup_epochs", 5)
    max_epochs = cfg["training"]["max_epochs"]
    min_lr = cfg["scheduler"].get("min_lr", cfg["optimizer"]["lr"] * 0.1)
    # Linear warmup for warmup_epochs, then cosine decay to min_lr
    scheduler_warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    scheduler_cosine = CosineAnnealingLR(
        optimizer,
        T_max=max_epochs - warmup_epochs,
        eta_min=min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_epochs],
    )

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

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    logger.info(f"Starting Stage 2 training for {cfg['training']['max_epochs']} epochs...")
    max_epochs = cfg["training"]["max_epochs"]
    val_interval = cfg["training"]["val_interval"]
    early_stop = cfg["training"]["early_stop_patience"]

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, None, criterion,
            device, logger, epoch,
            log_interval=cfg["logging"]["log_interval"],
            grad_clip=cfg["training"]["grad_clip"],
            grad_accum_steps=cfg.get("grad_accum_steps", 1),
        )
        # Step scheduler once per epoch (cosine annealing is epoch-level)
        scheduler.step()
        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch}/{max_epochs} | Train Loss: {train_loss:.4f} | LR: {cur_lr:.6f} | Time: {elapsed:.1f}s")

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

            if csi_m > best_csi_m:
                best_csi_m = csi_m
                patience_counter = 0
                ckpt_path = os.path.join(ckpt_dir, cfg["checkpoints"]["stage2_name"])
                save_checkpoint({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_csi_m": best_csi_m,
                    "val_results": val_results,
                    "config": cfg,
                }, ckpt_path)
                logger.info(f"Saved best Stage 2 checkpoint: {ckpt_path} (CSI-M={best_csi_m:.4f})")
            else:
                patience_counter += val_interval
                if patience_counter >= early_stop:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

    logger.info(f"Stage 2 training complete. Best CSI-M: {best_csi_m:.4f}")


if __name__ == "__main__":
    main()
