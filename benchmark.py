import argparse
import csv
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_universal import SEVIRDataset
from model import SimVP_Baseline, SimVP_Enhanced
from utils_metrics import MetricCalculator, MetricTracker


THRESHOLDS = ["M", "H", "E"]
POOL_SCALES = [1, 4, 16]
WEATHER_METRICS = ["CSI", "POD", "FAR"]


def parse_args():
    parser = argparse.ArgumentParser(description="HF-SimVP unified benchmark")
    parser.add_argument("--data_root", type=str, required=True, help="SEVIR .h5 文件目录")
    parser.add_argument("--baseline_ckpt", type=str, default="", help="Baseline 模型权重路径（可选）")
    parser.add_argument("--enhanced_ckpt", type=str, default="", help="Enhanced 模型权重路径（可选）")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=0, help="0 表示全量评测")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="评测输出目录（可选，默认自动保存到权重文件所在目录）",
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
    avg_metrics, curve_metrics = tracker.result()
    return avg_metrics, curve_metrics


def save_table(rows, out_csv):
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_temporal_table(rows, out_csv):
    fieldnames = ["Method", "LeadTimeMin", "Metric", "Threshold", "Pool", "Value"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary_row(name, avg_metrics, curve_metrics):
    row = {
        "Method": name,
        "CRPS": f"{avg_metrics.get('CRPS', 0.0):.4f}",
        "MSE": f"{avg_metrics.get('MSE', 0.0):.4f}",
        "SSIM": f"{avg_metrics.get('SSIM', 0.0):.4f}",
    }

    for metric in WEATHER_METRICS:
        for threshold in THRESHOLDS:
            for scale in POOL_SCALES:
                curve_key = f"{metric}_{threshold}_POOL{scale}"
                curve = curve_metrics.get(curve_key)
                value = float(np.mean(curve)) if curve is not None else 0.0
                row[f"{metric}-{threshold}-POOL{scale}"] = f"{value:.4f}"

    return row


def build_temporal_rows(name, curve_metrics):
    rows = []

    for metric in WEATHER_METRICS:
        for threshold in THRESHOLDS:
            for scale in POOL_SCALES:
                curve_key = f"{metric}_{threshold}_POOL{scale}"
                curve = curve_metrics.get(curve_key)
                if curve is None:
                    continue

                for frame_idx, value in enumerate(curve, start=1):
                    rows.append(
                        {
                            "Method": name,
                            "LeadTimeMin": frame_idx * 5,
                            "Metric": metric,
                            "Threshold": threshold,
                            "Pool": scale,
                            "Value": f"{float(value):.6f}",
                        }
                    )

    return rows


def run_benchmark(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SEVIRDataset(args.data_root, mode="test")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    calculator = MetricCalculator(device=device)
    models = {}

    baseline_ckpt = args.baseline_ckpt.strip()
    enhanced_ckpt = args.enhanced_ckpt.strip()

    if not baseline_ckpt and not enhanced_ckpt:
        raise ValueError("请至少提供一个模型权重：--baseline_ckpt 或 --enhanced_ckpt")

    if baseline_ckpt:
        if not os.path.isfile(baseline_ckpt):
            raise FileNotFoundError(f"Baseline 权重不存在: {baseline_ckpt}")
        models["Baseline"] = load_model(SimVP_Baseline, baseline_ckpt, device)

    if enhanced_ckpt:
        if not os.path.isfile(enhanced_ckpt):
            raise FileNotFoundError(f"Enhanced 权重不存在: {enhanced_ckpt}")
        models["Enhanced"] = load_model(SimVP_Enhanced, enhanced_ckpt, device)

    # 默认把评测结果放到权重所在目录，避免写入仓库目录
    if args.save_dir.strip():
        save_dir = args.save_dir.strip()
    else:
        preferred_ckpt = enhanced_ckpt or baseline_ckpt
        save_dir = os.path.dirname(os.path.abspath(preferred_ckpt))

    if baseline_ckpt and enhanced_ckpt:
        baseline_dir = os.path.dirname(os.path.abspath(baseline_ckpt))
        enhanced_dir = os.path.dirname(os.path.abspath(enhanced_ckpt))
        if baseline_dir != enhanced_dir and not args.save_dir.strip():
            print(
                "[Info] baseline 与 enhanced 权重不在同一路径，"
                f"结果默认保存到: {save_dir}"
            )

    os.makedirs(save_dir, exist_ok=True)

    rows = []
    temporal_rows = []
    for name, model in models.items():
        print(f"Running {name}...")
        avg_metrics, curve_metrics = evaluate_model(model, loader, calculator, device, max_batches=args.max_batches)
        rows.append(build_summary_row(name, avg_metrics, curve_metrics))
        temporal_rows.extend(build_temporal_rows(name, curve_metrics))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(save_dir, f"benchmark_metrics_{timestamp}.csv")
    summary_csv = os.path.join(save_dir, f"benchmark_summary_{timestamp}.csv")
    temporal_csv = os.path.join(save_dir, f"benchmark_temporal_{timestamp}.csv")
    save_table(rows, out_csv)
    save_table(rows, summary_csv)
    save_temporal_table(temporal_rows, temporal_csv)

    print("=" * 90)
    for row in rows:
        print(row)
    print("=" * 90)
    print(f"Benchmark summary saved to: {summary_csv}")
    print(f"Benchmark temporal saved to: {temporal_csv}")
    print(f"Backward-compatible metrics CSV saved to: {out_csv}")


if __name__ == "__main__":
    run_benchmark(parse_args())