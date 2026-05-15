"""
SEVIR Dataset Loader for SimCast.

SEVIR (Storm EVent ImageRy) dataset:
  - VIL (Vertically Integrated Liquid) radar echoes
  - Resolution: 384x384, 5-minute intervals
  - Each event: 49 frames (4 hours at 5-min intervals)
  - Train/Val/Test split following SimCast paper

Dataset structure (HDF5 files, all directly in data_root):
  sevir_data/
    SEVIR_VIL_STORMEVENTS_2017_0101_0630.h5
    SEVIR_VIL_STORMEVENTS_2017_0701_1231.h5
    SEVIR_VIL_RANDOMEVENTS_2017_0501_0831.h5
    ...

Each HDF5 file contains:
  - 'vil': (N_events, 49, 384, 384) uint8  -- note: some files use (N, H, W, T) layout!

Design: LAZY LOADING — only store (file_path, event_idx, frame_start) tuples
during __init__, and read slices from HDF5 on-the-fly in __getitem__.
This avoids loading 11+ GiB per file into RAM.

Reference:
  Veillette et al., "SEVIR: A Storm Event Imagery Dataset for Deep Learning
  Applications in Radar and Satellite Meteorology", NeurIPS 2020.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, NamedTuple

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
# Available data: 2017-2019
# - test:  2019 month >= 5
# - val:   2019 month  < 5
# - train: 2017 and 2018 (all months)
def _is_test(year, month):  return year == 2019 and month >= 5
def _is_val(year, month):   return year == 2019 and month < 5
def _is_train(year, month): return year < 2019

SPLIT_RULES = {
    "train": _is_train,
    "val":   _is_val,
    "test":  _is_test,
}


# ---------------------------------------------------------------------------
# Sample index entry
# ---------------------------------------------------------------------------

class SampleRef(NamedTuple):
    """Lightweight reference to a single sliding-window sample."""
    h5_path: str      # path to HDF5 file
    event_idx: int    # index of the weather event within the file
    frame_start: int  # starting frame index within the event
    n_frames: int     # total frames in this event (for bounds checking)
    time_last: int    # = frame_start + seq_len  (exclusive end)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SEVIRDataset(Dataset):
    """
    SEVIR VIL dataset with lazy HDF5 loading.

    Each sample is a tuple (input_seq, target_seq) where:
      input_seq:  (T_in,  1, H, W)  float32 in [0, 255]
      target_seq: (T_out, 1, H, W)  float32 in [0, 255]

    Lazy loading: HDF5 files are NOT kept open between __getitem__ calls,
    only per-sample (file, event_idx, frame_start) references are stored.
    This keeps memory usage at ~few MB regardless of dataset size.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        in_len: int = 13,
        out_len: int = 12,
        stride: int = 12,
        normalize: bool = False,
        augmented_data: Optional[np.ndarray] = None,
    ):
        """
        Args:
            data_root:      path to directory containing SEVIR_VIL_*.h5 files
            split:          'train', 'val', or 'test'
            in_len:         number of input frames (T_in)
            out_len:        number of output frames to predict (T_out)
            stride:         sliding window stride within each event
            normalize:      if True, normalize pixel values to [0, 1]
            augmented_data: optional pre-computed augmented sequences for Stage 2
                            shape: (N, seq_len_aug, H, W) float32
        """
        assert HAS_H5PY, "h5py is required. Install with: pip install h5py"
        assert split in ("train", "val", "test"), f"Invalid split: {split}"

        self.data_root = data_root
        self.split = split
        self.in_len = in_len
        self.out_len = out_len
        self.seq_len = in_len + out_len
        self.stride = stride
        self.normalize = normalize

        # Build lightweight index (no data loaded into memory)
        self.index: List[SampleRef] = self._build_index()
        print(f"[SEVIRDataset] split={split}, samples={len(self.index)}")

        # Optional augmented data for Stage 2
        # NOTE: store only the file path (not the memmap object) so that
        # Windows multiprocessing spawn can pickle this Dataset without OOM.
        # The memmap is opened lazily per-worker in __getitem__.
        self._aug_cache_path: Optional[str] = None
        self._aug_synth_only: bool = False
        self._aug_n: int = 0
        # Legacy: accept an in-memory array directly (Linux / num_workers=0)
        self.augmented_data = augmented_data
        if augmented_data is not None:
            self._aug_n = len(augmented_data)
        # Per-worker memmap handle (not pickled, opened on first __getitem__)
        self._aug_mmap: Optional[np.ndarray] = None

    def _build_index(self) -> List[SampleRef]:
        """
        Scan HDF5 files and record (file, event_idx, frame_start) for each
        valid sliding-window sample. No actual data is read here.
        """
        h5_files = sorted(glob.glob(os.path.join(self.data_root, "SEVIR_VIL_*.h5")))

        if len(h5_files) == 0:
            raise FileNotFoundError(
                f"No HDF5 files found in {self.data_root}. "
                f"Expected files matching SEVIR_VIL_*.h5 directly in that directory."
            )

        index = []
        split_fn = SPLIT_RULES[self.split]

        for h5_path in h5_files:
            fname = os.path.basename(h5_path)
            try:
                parts = fname.split("_")
                year = int(parts[3])
                month_start = int(parts[4][:2])
            except (IndexError, ValueError):
                continue

            if not split_fn(year, month_start):
                continue

            # Open file just to read the shape metadata, not the data
            with h5py.File(h5_path, "r") as f:
                if "vil" not in f:
                    continue
                shape = f["vil"].shape  # e.g. (N_events, 49, 384, 384)

            # Detect layout: (N, T, H, W) vs (N, H, W, T)
            # SEVIR official files use (N, 49, 384, 384)
            # Some re-packed files use (N, 384, 384, 49)
            n_events = shape[0]
            # Determine which dim is time (49 frames)
            if shape[1] == SEVIR_TOTAL_FRAMES:
                n_frames = shape[1]   # layout: (N, T, H, W)
            elif shape[3] == SEVIR_TOTAL_FRAMES:
                n_frames = shape[3]   # layout: (N, H, W, T)
            else:
                # fallback: assume dim1 is time
                n_frames = shape[1]

            for event_idx in range(n_events):
                start = 0
                while start + self.seq_len <= n_frames:
                    index.append(SampleRef(
                        h5_path=h5_path,
                        event_idx=event_idx,
                        frame_start=start,
                        n_frames=n_frames,
                        time_last=start + self.seq_len,
                    ))
                    start += self.stride

        if len(index) == 0:
            raise RuntimeError(
                f"No samples found for split='{self.split}'. "
                f"Check data_root={self.data_root} and split rules."
            )

        return index

    def _read_event_frames(self, h5_path: str, event_idx: int,
                           frame_start: int, length: int) -> np.ndarray:
        """
        Read `length` consecutive frames from a single event in an HDF5 file.

        Returns:
            frames: (length, H, W) float32
        """
        with h5py.File(h5_path, "r") as f:
            ds = f["vil"]
            shape = ds.shape
            frame_end = frame_start + length

            # Detect layout
            if shape[1] == SEVIR_TOTAL_FRAMES:
                # (N, T, H, W)
                frames = ds[event_idx, frame_start:frame_end, :, :]  # (T, H, W)
            else:
                # (N, H, W, T)
                frames = ds[event_idx, :, :, frame_start:frame_end]  # (H, W, T)
                frames = frames.transpose(2, 0, 1)                    # (T, H, W)

        return frames.astype(np.float32)

    def set_augmented_data(
        self,
        augmented_data,  # np.ndarray OR str (file path)
        synth_only: bool = False,
    ):
        """
        Set augmented training data for Stage 2 knowledge distillation.

        Args:
            augmented_data: either
              - str: path to a .npy memmap file (recommended on Windows)
              - np.ndarray: in-memory array (only safe with num_workers=0)
            synth_only: cache stores only synthetic frames (T_out_long, H, W)
        """
        self._aug_synth_only = synth_only
        self._aug_mmap = None  # reset per-worker handle

        if isinstance(augmented_data, str):
            # Store path only — memmap opened lazily in __getitem__
            self._aug_cache_path = augmented_data
            # Peek at shape to get N without loading data
            tmp = np.load(augmented_data, mmap_mode="r")
            self._aug_n = len(tmp)
            del tmp
            self.augmented_data = None  # not stored in-memory
        else:
            self._aug_cache_path = None
            self.augmented_data = augmented_data
            self._aug_n = len(augmented_data) if augmented_data is not None else 0

        print(f"[SEVIRDataset] augmented samples added: {self._aug_n}, "
              f"total: {len(self)}, synth_only={synth_only}")

    def _get_aug_mmap(self) -> np.ndarray:
        """Open (or reuse) the per-worker memmap handle."""
        if self._aug_mmap is None:
            if self._aug_cache_path is not None:
                self._aug_mmap = np.load(self._aug_cache_path, mmap_mode="r")
            else:
                self._aug_mmap = self.augmented_data
        return self._aug_mmap

    def __len__(self) -> int:
        return len(self.index) + self._aug_n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_seq:  (T_in,  1, H, W) float32
            target_seq: (T_out, 1, H, W) float32
        """
        n_orig = len(self.index)

        if idx < n_orig:
            ref = self.index[idx]

            # Random sub-sequence sampling (SimCast training trick):
            # The stored seq_len is the base window. For training we can
            # randomly shift the start within the event to boost diversity.
            if self.split == "train":
                max_shift = ref.n_frames - ref.frame_start - self.seq_len
                shift = np.random.randint(0, max(max_shift + 1, 1))
            else:
                shift = 0

            actual_start = ref.frame_start + shift
            seq = self._read_event_frames(
                ref.h5_path, ref.event_idx, actual_start, self.seq_len
            )  # (seq_len, H, W)

        else:
            # Augmented sample — open memmap lazily (safe for Windows spawn)
            aug_idx = idx - n_orig
            mmap = self._get_aug_mmap()

            if self._aug_synth_only:
                # cache stores only synthetic frames (T_out_long, H, W)
                ref = self.index[aug_idx % n_orig]
                orig_seq = self._read_event_frames(
                    ref.h5_path, ref.event_idx, ref.frame_start, self.seq_len
                )  # (seq_len, H, W)
                synth = mmap[aug_idx].astype(np.float32)  # (T_out_long, H, W)
                seq = np.concatenate([orig_seq, synth], axis=0)  # (seq_len+T_out_long, H, W)
            else:
                seq = mmap[aug_idx].astype(np.float32)  # (seq_len_aug, H, W)

            if self.split == "train" and seq.shape[0] > self.seq_len:
                max_start = seq.shape[0] - self.seq_len
                start = np.random.randint(0, max_start + 1)
                seq = seq[start: start + self.seq_len]

        input_seq  = seq[:self.in_len]   # (T_in,  H, W)
        target_seq = seq[self.in_len:]   # (T_out, H, W)

        # Add channel dimension: -> (T, 1, H, W)
        input_seq  = input_seq[:, np.newaxis, :, :]
        target_seq = target_seq[:, np.newaxis, :, :]

        if self.normalize:
            input_seq  = input_seq  / SEVIR_VIL_MAX
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
    train_ds = SEVIRDataset(
        data_root=data_root, split="train",
        in_len=in_len, out_len=out_len, stride=12, normalize=normalize,
    )
    val_ds = SEVIRDataset(
        data_root=data_root, split="val",
        in_len=in_len, out_len=out_len,
        stride=in_len + out_len,  # no overlap for val/test
        normalize=normalize,
    )
    test_ds = SEVIRDataset(
        data_root=data_root, split="test",
        in_len=in_len, out_len=out_len,
        stride=in_len + out_len,
        normalize=normalize,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
