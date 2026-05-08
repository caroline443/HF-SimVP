"""
SEVIR Dataset Loader for SimCast.

SEVIR (Storm EVent ImageRy) dataset:
  - VIL (Vertically Integrated Liquid) radar echoes
  - Resolution: 384x384, 5-minute intervals
  - Each event: 49 frames (4 hours)
  - Train/Val/Test split following SimCast paper:
      Train: 35,718 | Val: 9,060 | Test: 12,159

Dataset structure (HDF5 files):
  sevir/
    data/
      vil/
        SEVIR_VIL_STORMEVENTS_2018_0101_0630.h5
        SEVIR_VIL_STORMEVENTS_2018_0701_1231.h5
        ...
        SEVIR_VIL_RANDOMEVENTS_2018_0101_0630.h5
        ...
    CATALOG.csv

Each HDF5 file contains:
  - 'vil': (N_events, 49, 384, 384) uint8 array

Reference:
  Veillette et al., "SEVIR: A Storm Event Imagery Dataset for Deep Learning
  Applications in Radar and Satellite Meteorology", NeurIPS 2020.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEVIR_VIL_MAX = 255.0       # VIL pixel value range [0, 255]
SEVIR_TOTAL_FRAMES = 49     # total frames per event (4 hours at 5-min intervals)

# Train/val/test split by year and month (following SimCast / CasCast)
# Events from 2018-2019 are used; test set = month >= 5 of 2019
SPLIT_RULES = {
    "train": lambda year, month: not (year == 2019 and month >= 5),
    "val":   lambda year, month: (year == 2019 and month in [3, 4]),
    "test":  lambda year, month: (year == 2019 and month >= 5),
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SEVIRDataset(Dataset):
    """
    SEVIR VIL dataset for precipitation nowcasting.

    Each sample is a tuple (input_seq, target_seq) where:
      input_seq:  (T_in, 1, H, W)  float32 in [0, 255]
      target_seq: (T_out, 1, H, W) float32 in [0, 255]

    The dataset supports two modes:
      - 'short': T_out = out_len_short (Stage 1 training)
      - 'long':  T_out = out_len_long  (Stage 2 training / evaluation)

    For Stage 2, augmented samples can be provided via set_augmented_data().
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        in_len: int = 13,
        out_len: int = 12,
        seq_len: int = 25,          # in_len + out_len, used for sliding window
        stride: int = 12,           # stride for sliding window within an event
        normalize: bool = False,
        augmented_data: Optional[np.ndarray] = None,
    ):
        """
        Args:
            data_root:       path to SEVIR dataset root directory
            split:           'train', 'val', or 'test'
            in_len:          number of input frames (T_in)
            out_len:         number of output frames to predict (T_out)
            seq_len:         total sequence length = in_len + out_len
            stride:          sliding window stride within each event
            normalize:       if True, normalize to [0, 1]
            augmented_data:  optional pre-computed augmented sequences
                             shape: (N, seq_len_aug, 1, H, W)
        """
        assert HAS_H5PY, "h5py is required. Install with: pip install h5py"
        assert split in ("train", "val", "test"), f"Invalid split: {split}"

        self.data_root = data_root
        self.split = split
        self.in_len = in_len
        self.out_len = out_len
        self.seq_len = seq_len
        self.stride = stride
        self.normalize = normalize

        # Load all VIL data into memory (or use lazy loading for large datasets)
        self.samples = self._load_samples()

        # Optional augmented data for Stage 2
        self.augmented_data = augmented_data

    def _load_samples(self) -> List[np.ndarray]:
        """
        Load all VIL sequences from HDF5 files.
        Returns a list of arrays, each of shape (N_windows, seq_len, H, W).
        """
        vil_dir = os.path.join(self.data_root, "data", "vil")
        h5_files = sorted(glob.glob(os.path.join(vil_dir, "*.h5")))

        if len(h5_files) == 0:
            raise FileNotFoundError(
                f"No HDF5 files found in {vil_dir}. "
                f"Please download SEVIR from https://registry.opendata.aws/sevir/"
            )

        all_samples = []

        for h5_path in h5_files:
            # Parse year and month from filename
            # e.g., SEVIR_VIL_STORMEVENTS_2018_0101_0630.h5
            fname = os.path.basename(h5_path)
            try:
                parts = fname.split("_")
                year = int(parts[3])
                month_start = int(parts[4][:2])
            except (IndexError, ValueError):
                # Skip files that don't match expected naming
                continue

            # Apply split filter
            split_fn = SPLIT_RULES[self.split]
            if not split_fn(year, month_start):
                continue

            with h5py.File(h5_path, "r") as f:
                if "vil" not in f:
                    continue
                # Shape: (N_events, 49, 384, 384)
                vil_data = f["vil"][:]  # load into memory

            # Extract sliding window samples from each event
            for event_idx in range(vil_data.shape[0]):
                event = vil_data[event_idx]  # (49, 384, 384)
                n_frames = event.shape[0]

                start = 0
                while start + self.seq_len <= n_frames:
                    seq = event[start: start + self.seq_len]  # (seq_len, 384, 384)
                    all_samples.append(seq.astype(np.float32))
                    start += self.stride

        if len(all_samples) == 0:
            raise RuntimeError(
                f"No samples found for split='{self.split}'. "
                f"Check data_root={self.data_root} and split rules."
            )

        return all_samples

    def set_augmented_data(self, augmented_data: np.ndarray):
        """
        Set augmented training data for Stage 2.

        Args:
            augmented_data: (N, seq_len_aug, H, W) float32 array
                            where seq_len_aug = in_len + out_len_long + out_len_long
        """
        self.augmented_data = augmented_data

    def __len__(self) -> int:
        n = len(self.samples)
        if self.augmented_data is not None:
            n += len(self.augmented_data)
        return n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_seq:  (T_in, 1, H, W) float32
            target_seq: (T_out, 1, H, W) float32
        """
        n_orig = len(self.samples)

        if idx < n_orig:
            seq = self.samples[idx]  # (seq_len, H, W)
        else:
            # Augmented sample
            aug_idx = idx - n_orig
            seq = self.augmented_data[aug_idx]  # (seq_len_aug, H, W)

        # Random sub-sequence sampling (as described in SimCast)
        # For training, randomly sample a starting point within the sequence
        # to increase diversity
        total_len = seq.shape[0]
        needed = self.in_len + self.out_len

        if total_len > needed and self.split == "train":
            max_start = total_len - needed
            start = np.random.randint(0, max_start + 1)
        else:
            start = 0

        seq = seq[start: start + needed]  # (in_len + out_len, H, W)

        input_seq = seq[:self.in_len]           # (T_in, H, W)
        target_seq = seq[self.in_len:]          # (T_out, H, W)

        # Add channel dimension
        input_seq = input_seq[:, np.newaxis, :, :]   # (T_in, 1, H, W)
        target_seq = target_seq[:, np.newaxis, :, :]  # (T_out, 1, H, W)

        if self.normalize:
            input_seq = input_seq / SEVIR_VIL_MAX
            target_seq = target_seq / SEVIR_VIL_MAX

        return (
            torch.from_numpy(input_seq),
            torch.from_numpy(target_seq),
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_sevir_dataloaders(
    data_root: str,
    in_len: int = 13,
    out_len: int = 12,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    normalize: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders for SEVIR.

    Returns:
        train_loader, val_loader, test_loader
    """
    seq_len = in_len + out_len

    train_ds = SEVIRDataset(
        data_root=data_root,
        split="train",
        in_len=in_len,
        out_len=out_len,
        seq_len=seq_len,
        stride=12,
        normalize=normalize,
    )
    val_ds = SEVIRDataset(
        data_root=data_root,
        split="val",
        in_len=in_len,
        out_len=out_len,
        seq_len=seq_len,
        stride=seq_len,  # no overlap for val/test
        normalize=normalize,
    )
    test_ds = SEVIRDataset(
        data_root=data_root,
        split="test",
        in_len=in_len,
        out_len=out_len,
        seq_len=seq_len,
        stride=seq_len,
        normalize=normalize,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
