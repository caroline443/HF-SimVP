import argparse
import csv
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_universal import SEVIRDataset
from model import SimVP_Baseline, SimVP_Enhanced
from utils_metrics import MetricCalculator, MetricTracker


def parse_args():
    parser = argparse.ArgumentParser(description="HF-SimVP unified benchmark")
    parser.add_argument("--data_root", type=str, required=True, help="SEVIR .h5 文件目录")
    parser.add_argument("--baseline_ckpt", type=str, required=True, help="Baseline 模型权重路径")
    parser.add_argument("--enhanced_ckpt", type=str, required=True, help="Enhanced 模型权重路径")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=0, help="0 表示全量评测")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "log", "benchmark", datetime.now().strftime("%Y%m%d_%H%M%S")),
        help="评测输出目录",
    )
    return parser.parse_args()


def load_model(model_ctor, ckpt_path, device):
    model = model_ctor(in_shape=(13, 1, 384, 384)).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_model(model, loader, calculator, device, max_batches=0):
    tracker = MetricTracker()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc="Evaluating", leave=False), start=1):
            if max_batches > 0 and batch_idx > max_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            batch_metrics = calculator.compute_batch(preds, targets)
            tracker.update(batch_metrics)
    avg_metrics, _ = tracker.result()
    return avg_metrics


def save_table(rows, out_csv):
    fieldnames = [
        "Method",
        "CRPS",
        "SSIM",
        "CSI-M-POOL1",
        "CSI-M-POOL4",
        "CSI-M-POOL16",
        "CSI-H-POOL1",
        "CSI-H-POOL4",
        "CSI-H-POOL16",
        "CSI-E-POOL1",
        "CSI-E-POOL4",
        "CSI-E-POOL16",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    dataset = SEVIRDataset(args.data_root, mode="test")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    calculator = MetricCalculator(device=device)
    models = {
        "Baseline": load_model(SimVP_Baseline, args.baseline_ckpt, device),
        "Enhanced": load_model(SimVP_Enhanced, args.enhanced_ckpt, device),
    }

    rows = []
    for name, model in models.items():
        print(f"Running {name}...")
        avg = evaluate_model(model, loader, calculator, device, max_batches=args.max_batches)
        row = {
            "Method": name,
            "CRPS": f"{avg.get('CRPS', 0.0):.4f}",
            "SSIM": f"{avg.get('SSIM', 0.0):.4f}",
            "CSI-M-POOL1": f"{avg.get('CSI-M-POOL1', 0.0):.4f}",
            "CSI-M-POOL4": f"{avg.get('CSI-M-POOL4', 0.0):.4f}",
            "CSI-M-POOL16": f"{avg.get('CSI-M-POOL16', 0.0):.4f}",
            "CSI-H-POOL1": f"{avg.get('CSI-H-POOL1', 0.0):.4f}",
            "CSI-H-POOL4": f"{avg.get('CSI-H-POOL4', 0.0):.4f}",
            "CSI-H-POOL16": f"{avg.get('CSI-H-POOL16', 0.0):.4f}",
            "CSI-E-POOL1": f"{avg.get('CSI-E-POOL1', 0.0):.4f}",
            "CSI-E-POOL4": f"{avg.get('CSI-E-POOL4', 0.0):.4f}",
            "CSI-E-POOL16": f"{avg.get('CSI-E-POOL16', 0.0):.4f}",
        }
        rows.append(row)

    out_csv = os.path.join(args.save_dir, "benchmark_metrics.csv")
    save_table(rows, out_csv)

    print("=" * 90)
    for row in rows:
        print(row)
    print("=" * 90)
    print(f"Benchmark results saved to: {out_csv}")


if __name__ == "__main__":
    run_benchmark(parse_args())