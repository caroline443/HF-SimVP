import argparse
import csv
import json
import os
from datetime import datetime

import h5py
import numpy as np
from tqdm import tqdm


THRESHOLDS = ["M", "H", "E"]
POOL_SCALES = [1, 4, 16]
WEATHER_METRICS = ["CSI", "POD", "FAR"]


def _format_threshold_label(value):
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def parse_thresholds(text):
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("--thresholds 至少需要一个阈值")
    # 去重并保持顺序
    seen = set()
    ordered = []
    for v in values:
        if v not in seen:
            ordered.append(v)
            seen.add(v)
    return ordered


def inspect_raw_data_distribution(data_root, max_files=4, max_events_per_file=32, max_points_per_event=20000):
    h5_files = sorted([f for f in os.listdir(data_root) if f.endswith(".h5")])
    if not h5_files:
        raise ValueError(f"在 {data_root} 未找到 .h5 文件")

    selected_files = h5_files[:max_files]
    sampled_values = []
    per_file = []

    for filename in selected_files:
        f_path = os.path.join(data_root, filename)
        with h5py.File(f_path, "r") as hf:
            keys = list(hf.keys())
            data_key = "vil" if "vil" in keys else keys[0]
            arr = hf[data_key]
            event_count = min(arr.shape[0], max_events_per_file)

            file_samples = []
            for event_idx in range(event_count):
                event = arr[event_idx]
                flat = np.asarray(event).reshape(-1)
                step = max(len(flat) // max_points_per_event, 1)
                sampled = flat[::step]
                file_samples.append(sampled)
                sampled_values.append(sampled)

            if file_samples:
                file_concat = np.concatenate(file_samples)
                per_file.append(
                    {
                        "file": filename,
                        "key": data_key,
                        "events_sampled": event_count,
                        "min": float(np.min(file_concat)),
                        "max": float(np.max(file_concat)),
                        "mean": float(np.mean(file_concat)),
                        "p50": float(np.percentile(file_concat, 50)),
                        "p95": float(np.percentile(file_concat, 95)),
                        "p99": float(np.percentile(file_concat, 99)),
                    }
                )

    if not sampled_values:
        raise ValueError("值域检查失败：未采样到任何数据")

    global_concat = np.concatenate(sampled_values)
    return {
        "data_root": data_root,
        "files_sampled": len(selected_files),
        "events_per_file": max_events_per_file,
        "global": {
            "min": float(np.min(global_concat)),
            "max": float(np.max(global_concat)),
            "mean": float(np.mean(global_concat)),
            "p50": float(np.percentile(global_concat, 50)),
            "p95": float(np.percentile(global_concat, 95)),
            "p99": float(np.percentile(global_concat, 99)),
        },
        "per_file": per_file,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="HF-SimVP unified benchmark")
    parser.add_argument("--data_root", type=str, required=True, help="SEVIR .h5 文件目录")
    parser.add_argument("--baseline_ckpt", type=str, default="", help="Baseline 模型权重路径（可选）")
    parser.add_argument("--enhanced_ckpt", type=str, default="", help="Enhanced 模型权重路径（可选）")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=0, help="0 表示全量评测")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="16,74,133,160,181,219",
        help="逗号分隔阈值，默认与主流论文一致",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="评测输出目录（可选，默认自动保存到权重文件所在目录）",
    )
    parser.add_argument("--inspect_data", action="store_true", help="输出原始数据值域统计")
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="仅做值域检查，不加载模型",
    )
    parser.add_argument("--inspect_files", type=int, default=4, help="值域检查采样文件数")
    parser.add_argument("--inspect_events", type=int, default=32, help="每个文件采样事件数")
    return parser.parse_args()


def load_model(model_ctor, ckpt_path, device):
    model = model_ctor(in_shape=(13, 1, 384, 384)).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_model(model, loader, calculator, device, max_batches=0):
    from utils_metrics import MetricTracker

    tracker = MetricTracker(
        threshold_labels=calculator.threshold_labels,
        pool_scales=calculator.pool_scales,
    )
    batch_avg_curves = {}
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc="Evaluating", leave=False), start=1):
            if max_batches > 0 and batch_idx > max_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            batch_metrics = calculator.compute_batch(preds, targets)
            tracker.update(batch_metrics)

            # 备选聚合口径：先按 batch 算比率，再跨 batch 平均
            for threshold in calculator.threshold_labels:
                for scale in calculator.pool_scales:
                    suffix = f"{threshold}_POOL{scale}"
                    tp = batch_metrics.get(f"TP_{suffix}")
                    fn = batch_metrics.get(f"FN_{suffix}")
                    fp = batch_metrics.get(f"FP_{suffix}")
                    if tp is None or fn is None or fp is None:
                        continue
                    eps = 1e-6
                    csi = tp / (tp + fn + fp + eps)
                    pod = tp / (tp + fn + eps)
                    far = fp / (tp + fp + eps)

                    for metric_name, curve in (("CSI", csi), ("POD", pod), ("FAR", far)):
                        key = f"{metric_name}_{suffix}"
                        batch_avg_curves.setdefault(key, []).append(curve)

    avg_metrics, curve_metrics = tracker.result()

    batch_avg_curve_metrics = {}
    for key, curves in batch_avg_curves.items():
        batch_avg_curve_metrics[key] = np.mean(np.stack(curves, axis=0), axis=0)

    return avg_metrics, curve_metrics, batch_avg_curve_metrics


def save_table(rows, out_csv):
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_temporal_table(rows, out_csv):
    fieldnames = ["Method", "LeadTimeMin", "Aggregation", "Metric", "Threshold", "Pool", "Value"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary_row(name, avg_metrics, curve_metrics, batch_avg_curve_metrics, threshold_labels):
    row = {
        "Method": name,
        "CRPS": f"{avg_metrics.get('CRPS', 0.0):.4f}",
        "MSE": f"{avg_metrics.get('MSE', 0.0):.4f}",
        "SSIM": f"{avg_metrics.get('SSIM', 0.0):.4f}",
    }

    for metric in WEATHER_METRICS:
        for threshold in threshold_labels:
            for scale in POOL_SCALES:
                curve_key = f"{metric}_{threshold}_POOL{scale}"
                curve = curve_metrics.get(curve_key)
                value = float(np.mean(curve)) if curve is not None else 0.0
                row[f"{metric}-{threshold}-POOL{scale}"] = f"{value:.4f}"

                curve_batch = batch_avg_curve_metrics.get(curve_key)
                value_batch = float(np.mean(curve_batch)) if curve_batch is not None else 0.0
                row[f"{metric}-{threshold}-POOL{scale}-BATCHAVG"] = f"{value_batch:.4f}"
                row[f"{metric}-{threshold}-POOL{scale}-DELTA"] = f"{(value - value_batch):.4f}"

    return row


def build_temporal_rows(name, curve_metrics, batch_avg_curve_metrics, threshold_labels):
    rows = []

    for metric in WEATHER_METRICS:
        for threshold in threshold_labels:
            for scale in POOL_SCALES:
                curve_key = f"{metric}_{threshold}_POOL{scale}"
                curve = curve_metrics.get(curve_key)
                curve_batch = batch_avg_curve_metrics.get(curve_key)

                if curve is not None:
                    for frame_idx, value in enumerate(curve, start=1):
                        rows.append(
                            {
                                "Method": name,
                                "LeadTimeMin": frame_idx * 5,
                                "Aggregation": "global_count",
                                "Metric": metric,
                                "Threshold": threshold,
                                "Pool": scale,
                                "Value": f"{float(value):.6f}",
                            }
                        )

                if curve_batch is not None:
                    for frame_idx, value in enumerate(curve_batch, start=1):
                        rows.append(
                            {
                                "Method": name,
                                "LeadTimeMin": frame_idx * 5,
                                "Aggregation": "batch_avg",
                                "Metric": metric,
                                "Threshold": threshold,
                                "Pool": scale,
                                "Value": f"{float(value):.6f}",
                            }
                        )

    return rows


def run_benchmark(args):
    threshold_values = parse_thresholds(args.thresholds)
    threshold_labels = [_format_threshold_label(v) for v in threshold_values]

    if args.inspect_only:
        inspect_dir = args.save_dir.strip() or os.path.abspath(args.data_root)
        os.makedirs(inspect_dir, exist_ok=True)
        inspect_stats = inspect_raw_data_distribution(
            args.data_root,
            max_files=args.inspect_files,
            max_events_per_file=args.inspect_events,
        )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        inspect_json = os.path.join(inspect_dir, f"data_inspect_{ts}.json")
        with open(inspect_json, "w", encoding="utf-8") as f:
            json.dump(inspect_stats, f, ensure_ascii=False, indent=2)
        print(f"Data inspection saved to: {inspect_json}")
        print(json.dumps(inspect_stats["global"], ensure_ascii=False, indent=2))
        return

    import torch
    from torch.utils.data import DataLoader

    from dataset_universal import SEVIRDataset
    from model import SimVP_Baseline, SimVP_Enhanced
    from utils_metrics import MetricCalculator

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SEVIRDataset(args.data_root, mode="test")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    calculator = MetricCalculator(
        device=device,
        thresholds=threshold_values,
        threshold_labels=threshold_labels,
        pool_scales=POOL_SCALES,
    )
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

    if args.inspect_data:
        inspect_stats = inspect_raw_data_distribution(
            args.data_root,
            max_files=args.inspect_files,
            max_events_per_file=args.inspect_events,
        )
        inspect_path = os.path.join(save_dir, "data_inspect_latest.json")
        with open(inspect_path, "w", encoding="utf-8") as f:
            json.dump(inspect_stats, f, ensure_ascii=False, indent=2)
        print(f"Data inspection saved to: {inspect_path}")
        print(f"Global raw stats: {inspect_stats['global']}")

    rows = []
    temporal_rows = []
    for name, model in models.items():
        print(f"Running {name}...")
        avg_metrics, curve_metrics, batch_avg_curve_metrics = evaluate_model(
            model,
            loader,
            calculator,
            device,
            max_batches=args.max_batches,
        )
        rows.append(build_summary_row(name, avg_metrics, curve_metrics, batch_avg_curve_metrics, threshold_labels))
        temporal_rows.extend(build_temporal_rows(name, curve_metrics, batch_avg_curve_metrics, threshold_labels))

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