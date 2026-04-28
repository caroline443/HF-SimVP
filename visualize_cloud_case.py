import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset_universal import SEVIRDataset
from model import SimVP_Baseline, SimVP_Enhanced


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SEVIR precipitation cloud case")
    parser.add_argument("--data_root", type=str, required=True, help="SEVIR .h5 directory")
    parser.add_argument("--baseline_ckpt", type=str, required=True, help="Baseline checkpoint path")
    parser.add_argument("--enhanced_ckpt", type=str, required=True, help="Enhanced checkpoint path")
    parser.add_argument("--case_idx", type=int, default=78, help="Index in test dataset")
    parser.add_argument("--frame_idx", type=int, default=5, help="Future frame index [0..11], 5 means T+30min")
    parser.add_argument("--out_dir", type=str, default="result", help="Output directory")
    parser.add_argument("--cmap", type=str, default="jet", help="Matplotlib colormap")
    return parser.parse_args()


def load_model(model_ctor, ckpt_path, device):
    model = model_ctor(in_shape=(13, 1, 384, 384)).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    args = parse_args()
    if args.frame_idx < 0 or args.frame_idx > 11:
        raise ValueError("--frame_idx must be in [0, 11]")

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SEVIRDataset(args.data_root, mode="test")
    if args.case_idx < 0 or args.case_idx >= len(dataset):
        raise IndexError(f"case_idx out of range: {args.case_idx}, test size={len(dataset)}")

    inputs, targets = dataset[args.case_idx]
    batch_inputs = inputs.unsqueeze(0).to(device)

    base_model = load_model(SimVP_Baseline, args.baseline_ckpt, device)
    enh_model = load_model(SimVP_Enhanced, args.enhanced_ckpt, device)

    with torch.no_grad():
        pred_base = base_model(batch_inputs)
        pred_enh = enh_model(batch_inputs)

    img_input = inputs[-1, 0].cpu().numpy()
    img_gt = targets[args.frame_idx, 0].cpu().numpy()
    img_base = pred_base[0, args.frame_idx, 0].detach().cpu().numpy()
    img_enh = pred_enh[0, args.frame_idx, 0].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(img_input, cmap=args.cmap, vmin=0.0, vmax=1.0)
    axes[0].set_title("Input (last observed)")
    axes[0].axis("off")

    axes[1].imshow(img_gt, cmap=args.cmap, vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Ground Truth (T+{(args.frame_idx + 1) * 5}m)")
    axes[1].axis("off")

    axes[2].imshow(img_base, cmap=args.cmap, vmin=0.0, vmax=1.0)
    axes[2].set_title("Baseline")
    axes[2].axis("off")

    im = axes[3].imshow(img_enh, cmap=args.cmap, vmin=0.0, vmax=1.0)
    axes[3].set_title("Enhanced")
    axes[3].axis("off")

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    out_path = os.path.join(args.out_dir, f"cloud_case_{args.case_idx}_t{(args.frame_idx + 1) * 5}.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
