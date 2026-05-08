# SimCast: Precipitation Nowcasting with Short-to-Long Term Knowledge Distillation

Reproduction of the paper:
> **SimCast: Enhancing Precipitation Nowcasting with Short-to-Long Term Knowledge Distillation**
> Yifang Yin, Shengkai Chen, Yiyao Li, Lu Wang, Ruibing Jin, Wei Cui, Shili Xiang
> arXiv:2510.07953, 2025

## Overview

SimCast is a two-stage training pipeline for precipitation nowcasting:
1. **Stage 1**: Train a short-term nowcasting model (T'_s = 6 frames)
2. **Stage 2**: Use the short-term model autoregressively to augment training data, then train a long-term model (T'_l = 12 frames)

Key components:
- **SimVP backbone** with Inception-Unet translator
- **Weighted MSE loss** to prioritize heavy rainfall regions
- **Short-to-long term knowledge distillation** via data augmentation

## Project Structure

```
simcast/
├── configs/
│   ├── sevir.yaml          # SEVIR dataset config
│   ├── hko7.yaml           # HKO-7 dataset config
│   └── meteonet.yaml       # MeteoNet dataset config
├── data/
│   └── sevir_dataset.py    # SEVIR dataloader
├── models/
│   ├── simvp.py            # SimVP model (Encoder + Translator + Decoder)
│   └── inception_unet.py   # Inception-Unet translator
├── losses/
│   └── weighted_mse.py     # Weighted MSE loss
├── metrics/
│   └── nowcast_metrics.py  # CSI, SSIM, HSS, CRPS
├── train_stage1.py         # Stage 1: short-term model training
├── train_stage2.py         # Stage 2: long-term model training with KD
├── evaluate.py             # Evaluation script
└── requirements.txt
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare SEVIR dataset
Download SEVIR from https://registry.opendata.aws/sevir/ and set the path in `configs/sevir.yaml`.

### 3. Train Stage 1 (short-term model)
```bash
python train_stage1.py --config configs/sevir.yaml
```

### 4. Train Stage 2 (long-term model with knowledge distillation)
```bash
python train_stage2.py --config configs/sevir.yaml --stage1_ckpt checkpoints/stage1_best.pth
```

### 5. Evaluate
```bash
python evaluate.py --config configs/sevir.yaml --ckpt checkpoints/stage2_best.pth
```

## Results (SEVIR)

| Method | CSI-M | CSI-181 | CSI-219 | SSIM | HSS |
|--------|-------|---------|---------|------|-----|
| SimVP (baseline) | 0.4153 | 0.2532 | 0.1338 | 0.7772 | 0.5280 |
| SimCast (paper) | 0.4521 | 0.3099 | 0.2007 | 0.7252 | 0.5834 |
| SimCast (ours) | TBD | TBD | TBD | TBD | TBD |

## Citation

```bibtex
@article{yin2025simcast,
  title={SimCast: Enhancing Precipitation Nowcasting with Short-to-Long Term Knowledge Distillation},
  author={Yin, Yifang and Chen, Shengkai and Li, Yiyao and Wang, Lu and Jin, Ruibing and Cui, Wei and Xiang, Shili},
  journal={arXiv preprint arXiv:2510.07953},
  year={2025}
}
```
