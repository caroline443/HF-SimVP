"""
Inception-Unet Translator for SimVP.

Based on the SimVP paper and the Inception-Unet architecture described in:
  "Going deeper with convolutions" (Szegedy et al., CVPR 2015)

The translator takes the encoded spatiotemporal features of shape
  (B, T*C_hid, H', W')
and outputs refined temporal features of the same shape.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    """Single Inception block with 4 parallel branches."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Each branch outputs out_channels // 4 channels; total = out_channels
        branch_ch = out_channels // 4

        # Branch 1: 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=1),
            nn.GroupNorm(max(1, branch_ch // 4), branch_ch),
            nn.GELU(),
        )

        # Branch 2: 1x1 -> 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=1),
            nn.GroupNorm(max(1, branch_ch // 4), branch_ch),
            nn.GELU(),
            nn.Conv2d(branch_ch, branch_ch, kernel_size=3, padding=1),
            nn.GroupNorm(max(1, branch_ch // 4), branch_ch),
            nn.GELU(),
        )

        # Branch 3: 1x1 -> 5x5 (implemented as two 3x3 for efficiency)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=1),
            nn.GroupNorm(max(1, branch_ch // 4), branch_ch),
            nn.GELU(),
            nn.Conv2d(branch_ch, branch_ch, kernel_size=3, padding=1),
            nn.GroupNorm(max(1, branch_ch // 4), branch_ch),
            nn.GELU(),
            nn.Conv2d(branch_ch, branch_ch, kernel_size=3, padding=1),
            nn.GroupNorm(max(1, branch_ch // 4), branch_ch),
            nn.GELU(),
        )

        # Branch 4: 3x3 max pool -> 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_ch, kernel_size=1),
            nn.GroupNorm(max(1, branch_ch // 4), branch_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class DownBlock(nn.Module):
    """Downsampling block: InceptionBlock + stride-2 conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.inception = InceptionBlock(in_channels, out_channels)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.norm = nn.GroupNorm(max(1, out_channels // 4), out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        feat = self.inception(x)
        down = self.act(self.norm(self.down(feat)))
        return feat, down  # (skip, downsampled)


class UpBlock(nn.Module):
    """Upsampling block: bilinear upsample + concat skip + InceptionBlock."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(max(1, out_channels // 4), out_channels),
            nn.GELU(),
        )
        self.inception = InceptionBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.inception(x)


class InceptionUnet(nn.Module):
    """
    Inception-Unet translator for SimVP.

    Input:  (B, T*C_hid, H', W')
    Output: (B, T*C_hid, H', W')

    The U-Net structure captures multi-scale temporal features.
    """

    def __init__(self, in_channels: int, hid_channels: int = 256, depth: int = 3):
        """
        Args:
            in_channels:  T * C_hid (flattened temporal-channel dim)
            hid_channels: base hidden channel count (must be divisible by 4)
            depth:        number of down/up sampling levels
        """
        super().__init__()
        assert hid_channels % 4 == 0, "hid_channels must be divisible by 4"

        self.depth = depth

        # Initial projection
        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, kernel_size=1),
            nn.GroupNorm(max(1, hid_channels // 4), hid_channels),
            nn.GELU(),
        )

        # Encoder (down path)
        self.down_blocks = nn.ModuleList()
        ch = hid_channels
        self.enc_channels = []
        for i in range(depth):
            out_ch = min(ch * 2, hid_channels * (2 ** depth))
            self.down_blocks.append(DownBlock(ch, out_ch))
            self.enc_channels.append(out_ch)
            ch = out_ch

        # Bottleneck
        self.bottleneck = InceptionBlock(ch, ch)

        # Decoder (up path)
        self.up_blocks = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            skip_ch = self.enc_channels[i]
            out_ch = skip_ch
            self.up_blocks.append(UpBlock(ch, skip_ch, out_ch))
            ch = out_ch

        # Final projection back to in_channels
        self.proj_out = nn.Sequential(
            nn.Conv2d(ch, in_channels, kernel_size=1),
            nn.GroupNorm(max(1, in_channels // 4), in_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T*C_hid, H', W')
        Returns:
            out: (B, T*C_hid, H', W')
        """
        residual = x
        x = self.proj_in(x)

        # Encoder
        skips = []
        for down_block in self.down_blocks:
            skip, x = down_block(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i, up_block in enumerate(self.up_blocks):
            skip = skips[self.depth - 1 - i]
            x = up_block(x, skip)

        x = self.proj_out(x)
        return x + residual  # residual connection
