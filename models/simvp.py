"""
SimVP: Simpler yet Better Video Prediction (CVPR 2022)
with Inception-Unet translator as used in SimCast.

Architecture:
  Encoder:    (B, T, C, H, W) -> (B, T, C_hid, H', W')
  Translator: (B, T*C_hid, H', W') -> (B, T_out*C_hid, H', W')
  Decoder:    (B, T_out, C_hid, H', W') -> (B, T_out, C, H, W)

Reference:
  Gao et al., "SimVP: Simpler yet better video prediction", CVPR 2022.
  Yin et al., "SimCast: Enhancing Precipitation Nowcasting with
               Short-to-Long Term Knowledge Distillation", arXiv 2025.
"""

import torch
import torch.nn as nn
from .inception_unet import InceptionUnet


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2d + GroupNorm + GELU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(max(1, out_ch // 4), out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    """Residual block with two ConvBlocks."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(max(1, channels // 4), channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Spatial encoder: maps each frame independently.
    Input:  (B*T, C, H, W)
    Output: (B*T, C_hid, H/2^N_S, W/2^N_S)
    """

    def __init__(self, in_ch: int, hid_ch: int, N_S: int):
        super().__init__()
        layers = [ConvBlock(in_ch, hid_ch, kernel_size=3, stride=2, padding=1)]
        for _ in range(N_S - 1):
            layers.append(ResBlock(hid_ch))
            layers.append(ConvBlock(hid_ch, hid_ch, kernel_size=3, stride=2, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    Spatial decoder: maps each frame independently.
    Input:  (B*T_out, C_hid, H', W')
    Output: (B*T_out, C, H, W)
    """

    def __init__(self, hid_ch: int, out_ch: int, N_S: int):
        super().__init__()
        layers = []
        for _ in range(N_S - 1):
            layers.append(ResBlock(hid_ch))
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(hid_ch, hid_ch, kernel_size=2, stride=2),
                nn.GroupNorm(max(1, hid_ch // 4), hid_ch),
                nn.GELU(),
            ))
        layers.append(nn.ConvTranspose2d(hid_ch, out_ch, kernel_size=2, stride=2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# SimVP
# ---------------------------------------------------------------------------

class SimVP(nn.Module):
    """
    SimVP with Inception-Unet translator.

    Args:
        in_shape:   (T_in, C, H, W) — input sequence shape
        T_out:      number of output frames to predict
        hid_S:      spatial hidden channels
        hid_T:      temporal hidden channels (used as base for Inception-Unet)
        N_S:        number of spatial encoder/decoder blocks
        N_T:        depth of Inception-Unet translator
    """

    def __init__(
        self,
        in_shape: tuple,
        T_out: int,
        hid_S: int = 64,
        hid_T: int = 512,
        N_S: int = 2,
        N_T: int = 4,
    ):
        super().__init__()
        T_in, C, H, W = in_shape
        self.T_in = T_in
        self.T_out = T_out
        self.C = C
        self.hid_S = hid_S

        # Spatial downsampling factor
        self.ds = 2 ** N_S
        H_enc = H // self.ds
        W_enc = W // self.ds

        # Encoder and Decoder (shared weights across time steps)
        self.encoder = Encoder(C, hid_S, N_S)
        self.decoder = Decoder(hid_S, C, N_S)

        # Translator: operates on (B, T_in * hid_S, H_enc, W_enc)
        # and outputs (B, T_out * hid_S, H_enc, W_enc)
        # We use a projection to handle T_in -> T_out if they differ
        translator_in_ch = T_in * hid_S
        translator_out_ch = T_out * hid_S

        self.translator = InceptionUnet(
            in_channels=translator_in_ch,
            hid_channels=hid_T,
            depth=N_T,
        )

        # If T_in != T_out, add a channel projection after translator
        if translator_in_ch != translator_out_ch:
            self.t_proj = nn.Conv2d(translator_in_ch, translator_out_ch, kernel_size=1)
        else:
            self.t_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T_in, C, H, W)
        Returns:
            pred: (B, T_out, C, H, W)
        """
        B, T_in, C, H, W = x.shape
        assert T_in == self.T_in, f"Expected T_in={self.T_in}, got {T_in}"

        # Encode each frame: (B*T_in, C, H, W) -> (B*T_in, hid_S, H', W')
        x_flat = x.reshape(B * T_in, C, H, W)
        enc = self.encoder(x_flat)                          # (B*T_in, hid_S, H', W')
        _, hid_S, H_enc, W_enc = enc.shape

        # Reshape for translator: (B, T_in*hid_S, H', W')
        enc = enc.reshape(B, T_in * hid_S, H_enc, W_enc)

        # Translate temporal dynamics
        hid = self.translator(enc)                          # (B, T_in*hid_S, H', W')
        hid = self.t_proj(hid)                              # (B, T_out*hid_S, H', W')

        # Decode each output frame: (B*T_out, hid_S, H', W') -> (B*T_out, C, H, W)
        hid = hid.reshape(B * self.T_out, hid_S, H_enc, W_enc)
        out = self.decoder(hid)                             # (B*T_out, C, H, W)
        out = out.reshape(B, self.T_out, C, H, W)

        return out

    @torch.no_grad()
    def autoregressive_predict(self, x: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Autoregressively apply the model to generate n_steps * T_out frames.

        Used in Stage 2 data augmentation: the short-term model is applied
        autoregressively to extend training sequences.

        Args:
            x:       (B, T_in, C, H, W) — initial input sequence
            n_steps: number of autoregressive steps

        Returns:
            preds: (B, n_steps * T_out, C, H, W)
        """
        self.eval()
        preds = []
        current_input = x

        for _ in range(n_steps):
            pred = self.forward(current_input)              # (B, T_out, C, H, W)
            preds.append(pred)
            # Slide window: drop oldest T_out frames, append new predictions
            current_input = torch.cat(
                [current_input[:, self.T_out:], pred], dim=1
            )

        return torch.cat(preds, dim=1)                      # (B, n_steps*T_out, C, H, W)
