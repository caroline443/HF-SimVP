"""
Weighted MSE Loss for precipitation nowcasting.

From SimCast (Eq. 1-2):

  L = sum_{T,H,W} omega(Y) * (f_theta(X) - Y)^2

  omega(x) = 1        if x <= tau
             wmax     otherwise

where tau is the intensity threshold for the highest rainfall category,
and wmax is the weight assigned to heavy rainfall pixels.

For SEVIR: tau=219, wmax=10
For HKO-7: tau=185, wmax=10
For MeteoNet: tau=47, wmax=10
"""

import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    """
    Pixel-wise weighted MSE loss that up-weights high-intensity rainfall regions.

    Args:
        tau:  intensity threshold (pixels above this get weight wmax)
        wmax: weight for high-intensity pixels (default: 10)
    """

    def __init__(self, tau: float, wmax: float = 10.0):
        super().__init__()
        self.tau = tau
        self.wmax = wmax

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, T, C, H, W) — model predictions
            target: (B, T, C, H, W) — ground truth radar frames

        Returns:
            loss: scalar weighted MSE
        """
        # Compute pixel-wise weights based on target intensity
        # omega(x) = wmax if x > tau, else 1
        weights = torch.where(
            target > self.tau,
            torch.full_like(target, self.wmax),
            torch.ones_like(target),
        )

        # Weighted squared error
        sq_err = (pred - target) ** 2
        loss = (weights * sq_err).mean()

        return loss

    def extra_repr(self) -> str:
        return f"tau={self.tau}, wmax={self.wmax}"
