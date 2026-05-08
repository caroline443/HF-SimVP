"""
Nowcasting evaluation metrics for precipitation nowcasting.

Implements:
  - CSI  (Critical Success Index) at multiple thresholds and pool sizes
  - HSS  (Heidke Skill Score)
  - SSIM (Structural Similarity Index)
  - CRPS (Continuous Ranked Probability Score, reduces to MAE for deterministic)

Following the evaluation protocol of SimCast / CasCast:
  SEVIR thresholds:   [16, 74, 133, 160, 181, 219]
  HKO-7 thresholds:   [84, 118, 141, 158, 185]
  MeteoNet thresholds:[19, 28, 35, 40, 47]
  Pool sizes:         [1, 4, 16]
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Low-level metric functions (operate on numpy arrays)
# ---------------------------------------------------------------------------

def _pool2d(x: np.ndarray, pool_size: int) -> np.ndarray:
    """
    Apply max pooling with given pool_size.
    x: (..., H, W)
    """
    if pool_size == 1:
        return x
    t = torch.from_numpy(x).float()
    # Flatten leading dims
    orig_shape = t.shape
    h, w = orig_shape[-2], orig_shape[-1]
    t = t.reshape(-1, 1, h, w)
    t = F.max_pool2d(t, kernel_size=pool_size, stride=pool_size)
    new_h, new_w = t.shape[-2], t.shape[-1]
    t = t.reshape(*orig_shape[:-2], new_h, new_w)
    return t.numpy()


def _compute_hits_misses_fas(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> tuple:
    """
    Compute hits, misses, false alarms, correct negatives.

    Args:
        pred:      (...) float array
        target:    (...) float array
        threshold: binarization threshold

    Returns:
        hits, misses, false_alarms, correct_negatives (all scalars)
    """
    pred_bin = pred >= threshold
    tgt_bin = target >= threshold

    hits = np.sum(pred_bin & tgt_bin).item()
    misses = np.sum(~pred_bin & tgt_bin).item()
    false_alarms = np.sum(pred_bin & ~tgt_bin).item()
    correct_neg = np.sum(~pred_bin & ~tgt_bin).item()

    return hits, misses, false_alarms, correct_neg


def compute_csi(hits: float, misses: float, false_alarms: float) -> float:
    denom = hits + misses + false_alarms
    if denom == 0:
        return float("nan")
    return hits / denom


def compute_hss(
    hits: float, misses: float, false_alarms: float, correct_neg: float
) -> float:
    n = hits + misses + false_alarms + correct_neg
    if n == 0:
        return float("nan")
    expected = ((hits + misses) * (hits + false_alarms) +
                (correct_neg + misses) * (correct_neg + false_alarms)) / n
    denom = n - expected
    if denom == 0:
        return float("nan")
    return (hits + correct_neg - expected) / denom


def compute_pod(hits: float, misses: float) -> float:
    denom = hits + misses
    if denom == 0:
        return float("nan")
    return hits / denom


def compute_far(hits: float, false_alarms: float) -> float:
    denom = hits + false_alarms
    if denom == 0:
        return float("nan")
    return false_alarms / denom


def compute_ssim_batch(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute mean SSIM over a batch of images.

    Args:
        pred:   (N, H, W) or (N, 1, H, W) float array in [0, 255]
        target: same shape

    Returns:
        mean SSIM scalar
    """
    if pred.ndim == 4:
        pred = pred[:, 0]
        target = target[:, 0]

    # Normalize to [0, 1]
    pred_n = pred / 255.0
    target_n = target / 255.0

    ssim_vals = []
    for p, t in zip(pred_n, target_n):
        ssim_vals.append(_ssim_single(p, t))
    return float(np.nanmean(ssim_vals))


def _ssim_single(pred: np.ndarray, target: np.ndarray,
                 window_size: int = 11, sigma: float = 1.5) -> float:
    """Compute SSIM for a single pair of images (H, W)."""
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    # Use torch for efficient computation
    p = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    t = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0)

    # Gaussian kernel
    kernel = _gaussian_kernel(window_size, sigma)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1,1,k,k)
    pad = window_size // 2

    mu_p = F.conv2d(p, kernel, padding=pad)
    mu_t = F.conv2d(t, kernel, padding=pad)

    mu_p_sq = mu_p ** 2
    mu_t_sq = mu_t ** 2
    mu_pt = mu_p * mu_t

    sigma_p_sq = F.conv2d(p * p, kernel, padding=pad) - mu_p_sq
    sigma_t_sq = F.conv2d(t * t, kernel, padding=pad) - mu_t_sq
    sigma_pt = F.conv2d(p * t, kernel, padding=pad) - mu_pt

    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2))

    return ssim_map.mean().item()


def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)


def compute_crps_deterministic(pred: np.ndarray, target: np.ndarray) -> float:
    """
    CRPS for deterministic predictions simplifies to MAE.
    Normalized to [0, 1] range.
    """
    return float(np.mean(np.abs(pred - target)) / 255.0)


# ---------------------------------------------------------------------------
# High-level metric accumulator
# ---------------------------------------------------------------------------

class NowcastMetrics:
    """
    Accumulates predictions and targets, then computes all metrics.

    Usage:
        metrics = NowcastMetrics(thresholds=[16, 74, 133, 160, 181, 219],
                                 pool_sizes=[1, 4, 16])
        for pred, target in dataloader:
            metrics.update(pred.numpy(), target.numpy())
        results = metrics.compute()
    """

    def __init__(
        self,
        thresholds: List[float],
        pool_sizes: List[int] = [1, 4, 16],
    ):
        self.thresholds = thresholds
        self.pool_sizes = pool_sizes
        self.reset()

    def reset(self):
        """Reset all accumulators."""
        # For each (threshold, pool_size): accumulate hits/misses/FAs/CNs
        self._hits = {(t, p): 0.0 for t in self.thresholds for p in self.pool_sizes}
        self._misses = {(t, p): 0.0 for t in self.thresholds for p in self.pool_sizes}
        self._fas = {(t, p): 0.0 for t in self.thresholds for p in self.pool_sizes}
        self._cns = {(t, p): 0.0 for t in self.thresholds for p in self.pool_sizes}

        self._ssim_sum = 0.0
        self._ssim_count = 0
        self._crps_sum = 0.0
        self._crps_count = 0

        # Per-timestep accumulators for temporal analysis
        self._t_hits = {}
        self._t_misses = {}
        self._t_fas = {}
        self._t_cns = {}

    def update(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        timestep: Optional[int] = None,
    ):
        """
        Update accumulators with a batch of predictions.

        Args:
            pred:      (B, T, 1, H, W) or (B, T, H, W) float32 in [0, 255]
            target:    same shape as pred
            timestep:  optional timestep index for temporal analysis
        """
        # Normalize shape to (B*T, H, W)
        if pred.ndim == 5:
            B, T, C, H, W = pred.shape
            pred = pred.reshape(B * T, H, W)
            target = target.reshape(B * T, H, W)
        elif pred.ndim == 4:
            B, T, H, W = pred.shape
            pred = pred.reshape(B * T, H, W)
            target = target.reshape(B * T, H, W)

        # SSIM and CRPS
        self._ssim_sum += compute_ssim_batch(pred, target) * pred.shape[0]
        self._ssim_count += pred.shape[0]
        self._crps_sum += compute_crps_deterministic(pred, target) * pred.shape[0]
        self._crps_count += pred.shape[0]

        # CSI / HSS accumulators
        for pool in self.pool_sizes:
            pred_p = _pool2d(pred, pool)
            tgt_p = _pool2d(target, pool)

            for thresh in self.thresholds:
                h, m, fa, cn = _compute_hits_misses_fas(pred_p, tgt_p, thresh)
                self._hits[(thresh, pool)] += h
                self._misses[(thresh, pool)] += m
                self._fas[(thresh, pool)] += fa
                self._cns[(thresh, pool)] += cn

        # Per-timestep (if provided)
        if timestep is not None:
            key = timestep
            if key not in self._t_hits:
                self._t_hits[key] = {(t, p): 0.0 for t in self.thresholds for p in self.pool_sizes}
                self._t_misses[key] = {(t, p): 0.0 for t in self.thresholds for p in self.pool_sizes}
                self._t_fas[key] = {(t, p): 0.0 for t in self.thresholds for p in self.pool_sizes}
                self._t_cns[key] = {(t, p): 0.0 for t in self.thresholds for p in self.pool_sizes}

            for pool in self.pool_sizes:
                pred_p = _pool2d(pred, pool)
                tgt_p = _pool2d(target, pool)
                for thresh in self.thresholds:
                    h, m, fa, cn = _compute_hits_misses_fas(pred_p, tgt_p, thresh)
                    self._t_hits[key][(thresh, pool)] += h
                    self._t_misses[key][(thresh, pool)] += m
                    self._t_fas[key][(thresh, pool)] += fa
                    self._t_cns[key][(thresh, pool)] += cn

    def compute(self) -> Dict:
        """
        Compute all metrics and return as a dictionary.

        Returns a dict with keys like:
          'CSI_M_POOL1', 'CSI_M_POOL4', 'CSI_M_POOL16',
          'CSI_181_POOL1', ..., 'CSI_219_POOL16',
          'HSS_POOL1', ...,
          'SSIM', 'CRPS'
        """
        results = {}

        # SSIM and CRPS
        results["SSIM"] = self._ssim_sum / max(self._ssim_count, 1)
        results["CRPS"] = self._crps_sum / max(self._crps_count, 1)

        # CSI and HSS per threshold and pool size
        for pool in self.pool_sizes:
            pool_key = f"POOL{pool}"
            csi_vals = []

            for thresh in self.thresholds:
                h = self._hits[(thresh, pool)]
                m = self._misses[(thresh, pool)]
                fa = self._fas[(thresh, pool)]
                cn = self._cns[(thresh, pool)]

                csi = compute_csi(h, m, fa)
                hss = compute_hss(h, m, fa, cn)
                pod = compute_pod(h, m)
                far = compute_far(h, fa)

                thresh_key = int(thresh)
                results[f"CSI_{thresh_key}_{pool_key}"] = csi
                results[f"HSS_{thresh_key}_{pool_key}"] = hss
                results[f"POD_{thresh_key}_{pool_key}"] = pod
                results[f"FAR_{thresh_key}_{pool_key}"] = far

                if not np.isnan(csi):
                    csi_vals.append(csi)

            # Mean CSI over all thresholds
            results[f"CSI_M_{pool_key}"] = float(np.mean(csi_vals)) if csi_vals else float("nan")

            # Mean HSS over all thresholds (using pool1 as primary)
            if pool == 1:
                hss_vals = [
                    compute_hss(
                        self._hits[(t, pool)],
                        self._misses[(t, pool)],
                        self._fas[(t, pool)],
                        self._cns[(t, pool)],
                    )
                    for t in self.thresholds
                ]
                results["HSS"] = float(np.nanmean(hss_vals))

        return results

    def compute_temporal(self) -> Dict[int, Dict]:
        """
        Compute per-timestep metrics.

        Returns:
            dict mapping timestep -> metric dict
        """
        temporal = {}
        for ts in sorted(self._t_hits.keys()):
            ts_results = {}
            for pool in self.pool_sizes:
                pool_key = f"POOL{pool}"
                csi_vals = []
                for thresh in self.thresholds:
                    h = self._t_hits[ts][(thresh, pool)]
                    m = self._t_misses[ts][(thresh, pool)]
                    fa = self._t_fas[ts][(thresh, pool)]
                    cn = self._t_cns[ts][(thresh, pool)]
                    csi = compute_csi(h, m, fa)
                    ts_results[f"CSI_{int(thresh)}_{pool_key}"] = csi
                    if not np.isnan(csi):
                        csi_vals.append(csi)
                ts_results[f"CSI_M_{pool_key}"] = float(np.mean(csi_vals)) if csi_vals else float("nan")
            temporal[ts] = ts_results
        return temporal

    def summary_str(self) -> str:
        """Return a human-readable summary of key metrics."""
        r = self.compute()
        lines = [
            "=" * 60,
            "Nowcasting Evaluation Results",
            "=" * 60,
            f"  SSIM:        {r.get('SSIM', float('nan')):.4f}",
            f"  CRPS:        {r.get('CRPS', float('nan')):.4f}",
            f"  HSS:         {r.get('HSS', float('nan')):.4f}",
            "",
            "  CSI-M (mean over all thresholds):",
            f"    POOL1:  {r.get('CSI_M_POOL1', float('nan')):.4f}",
            f"    POOL4:  {r.get('CSI_M_POOL4', float('nan')):.4f}",
            f"    POOL16: {r.get('CSI_M_POOL16', float('nan')):.4f}",
        ]
        # Add per-threshold CSI at POOL1
        lines.append("")
        lines.append("  CSI per threshold (POOL1):")
        for thresh in self.thresholds:
            key = f"CSI_{int(thresh)}_POOL1"
            lines.append(f"    thresh={int(thresh):3d}: {r.get(key, float('nan')):.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)
