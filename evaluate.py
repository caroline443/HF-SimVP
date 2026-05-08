"""
Evaluation script for SimCast.

Loads a trained model checkpoint and evaluates it on the test set,
reporting CSI, SSIM, HSS, CRPS at multiple thresholds and pool sizes.

Also supports per-timestep temporal analysis and visualization.

Usage:
    # Evaluate Stage 2 (long-term) model
    python evaluate.py --config configs/sevir.yaml \
                       --ckpt checkpoints/sevir/stage2_best.pth

    # Evaluate Stage 1 (short-term) model
    python evaluate.py --config configs/sevir.yaml \
                       --ckpt checkpoints/sevir/stage1_best.pth \
                       --stage short

    # Save predictions for visualization
    python evaluate.py --config configs/sevir.yaml \
                       --ckpt checkpoints/sevir/stage2_best.pth \
                       --save_preds results/preds.npy
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from models import SimVP
from metrics import NowcastMetrics
from data import build_sevir_dataloaders


def setup_logger(name: str = "evaluate") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    loader,
    metrics: NowcastMetrics,
    device: torch.device,
    logger: logging.Logger,
    save_preds: str = None,
    T_out: int = 12,
) -> dict:
    """
    Run full evaluation on a dataloader.

    Returns:
        results dict with all metrics
    """
    model.eval()
    metrics.reset()

    all_preds = [] if save_preds else None
    all_targets = [] if save_preds else None

    n_batches = len(loader)
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)   # (B, T_in, 1, H, W)
        targets = targets.to(device) # (B, T_out, 1, H, W)

        preds = model(inputs)        # (B, T_out, 1, H, W)

        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        # Update overall metrics
        metrics.update(preds_np, targets_np)

        # Update per-timestep metrics
        B, T, C, H, W = preds_np.shape
        for t in range(T):
            metrics.update(
                preds_np[:, t:t+1],
                targets_np[:, t:t+1],
                timestep=t,
            )

        if save_preds is not None:
            all_preds.append(preds_np)
            all_targets.append(targets_np)

        if (batch_idx + 1) % 50 == 0:
            logger.info(f"  Evaluated {batch_idx+1}/{n_batches} batches")

    results = metrics.compute()
    temporal = metrics.compute_temporal()

    if save_preds is not None:
        preds_arr = np.concatenate(all_preds, axis=0)
        targets_arr = np.concatenate(all_targets, axis=0)
        np.save(save_preds, {"preds": preds_arr, "targets": targets_arr})
        logger.info(f"Saved predictions to {save_preds}")

    return results, temporal


def print_results_table(results: dict, thresholds: list, pool_sizes: list):
    """Print results in a formatted table similar to the SimCast paper."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    # Overall metrics
    print(f"\nSSIM:  {results.get('SSIM', float('nan')):.4f}")
    print(f"CRPS:  {results.get('CRPS', float('nan')):.4f}")
    print(f"HSS:   {results.get('HSS', float('nan')):.4f}")

    # CSI table
    print("\nCSI scores:")
    header = f"{'Threshold':>12}" + "".join(f"  POOL{p:>2}" for p in pool_sizes)
    print(header)
    print("-" * len(header))

    for thresh in thresholds:
        row = f"{int(thresh):>12}"
        for pool in pool_sizes:
            key = f"CSI_{int(thresh)}_POOL{pool}"
            val = results.get(key, float("nan"))
            row += f"  {val:>6.4f}"
        print(row)

    # Mean CSI
    print("-" * len(header))
    row = f"{'Mean (CSI-M)':>12}"
    for pool in pool_sizes:
        key = f"CSI_M_POOL{pool}"
        val = results.get(key, float("nan"))
        row += f"  {val:>6.4f}"
    print(row)
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="SimCast Evaluation")
    parser.add_argument("--config", type=str, default="configs/sevir.yaml")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--stage", type=str, default="long",
                        choices=["short", "long"],
                        help="Which stage model to evaluate")
    parser.add_argument("--save_preds", type=str, default=None,
                        help="Path to save predictions (.npy)")
    parser.add_argument("--save_results", type=str, default=None,
                        help="Path to save results JSON")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"])
    args = parser.parse_args()

    logger = setup_logger()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["dataset"]["data_root"] = args.data_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    T_in = cfg["dataset"]["in_len"]
    T_out = (cfg["dataset"]["out_len_short"] if args.stage == "short"
             else cfg["dataset"]["out_len_long"])
    H = W = cfg["dataset"]["resolution"]

    # Load model
    logger.info(f"Loading model from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=device)

    model = SimVP(
        in_shape=(T_in, 1, H, W),
        T_out=T_out,
        hid_S=cfg["model"]["hid_S"],
        hid_T=cfg["model"]["hid_T"],
        N_S=cfg["model"]["N_S"],
        N_T=cfg["model"]["N_T"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"Model loaded (epoch={ckpt.get('epoch', 'N/A')}, "
                f"best CSI-M={ckpt.get('best_csi_m', 'N/A')})")

    # Build dataloader
    logger.info(f"Building {args.split} dataloader...")
    _, val_loader, test_loader = build_sevir_dataloaders(
        data_root=cfg["dataset"]["data_root"],
        in_len=T_in,
        out_len=T_out,
        batch_size=cfg["dataloader"]["batch_size"],
        num_workers=cfg["dataloader"]["num_workers"],
        pin_memory=cfg["dataloader"]["pin_memory"],
        normalize=cfg["dataset"]["normalize"],
    )
    loader = test_loader if args.split == "test" else val_loader
    logger.info(f"Evaluation batches: {len(loader)}")

    # Metrics
    eval_metrics = NowcastMetrics(
        thresholds=cfg["evaluation"]["csi_thresholds"],
        pool_sizes=cfg["evaluation"]["pool_sizes"],
    )

    # Run evaluation
    logger.info("Running evaluation...")
    results, temporal = run_evaluation(
        model=model,
        loader=loader,
        metrics=eval_metrics,
        device=device,
        logger=logger,
        save_preds=args.save_preds,
        T_out=T_out,
    )

    # Print results
    print_results_table(
        results,
        thresholds=cfg["evaluation"]["csi_thresholds"],
        pool_sizes=cfg["evaluation"]["pool_sizes"],
    )

    # Print temporal analysis
    if temporal:
        print("\nPer-timestep CSI-M (POOL1):")
        print(f"{'Timestep':>10}  {'Lead(min)':>10}  {'CSI-M':>8}")
        print("-" * 35)
        for ts in sorted(temporal.keys()):
            lead_min = (ts + 1) * cfg["dataset"]["interval"]
            csi_m = temporal[ts].get("CSI_M_POOL1", float("nan"))
            print(f"{ts:>10}  {lead_min:>10}  {csi_m:>8.4f}")

    # Save results
    if args.save_results:
        os.makedirs(os.path.dirname(args.save_results), exist_ok=True)
        output = {
            "results": {k: float(v) for k, v in results.items()},
            "temporal": {
                str(ts): {k: float(v) for k, v in ts_r.items()}
                for ts, ts_r in temporal.items()
            },
            "config": cfg,
            "checkpoint": args.ckpt,
        }
        with open(args.save_results, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {args.save_results}")


if __name__ == "__main__":
    main()
