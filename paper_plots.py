import argparse
import csv
import glob
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


THRESHOLDS = ["M", "H", "E"]
POOL_SCALES = [1, 4, 16]
WEATHER_METRICS = ["CSI", "POD", "FAR"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper-ready figures from benchmark CSV files")
    parser.add_argument("--results_dir", type=str, default="", help="目录下自动寻找最新 benchmark_summary/temporal CSV")
    parser.add_argument("--summary_csv", type=str, default="", help="benchmark_summary_*.csv 路径")
    parser.add_argument("--temporal_csv", type=str, default="", help="benchmark_temporal_*.csv 路径")
    parser.add_argument("--out_dir", type=str, default="", help="图片输出目录，默认放在 CSV 同级目录")
    parser.add_argument("--dpi", type=int, default=400, help="导出分辨率")
    parser.add_argument("--pool", type=int, default=1, choices=[1, 4, 16], help="用于时序图的池化尺度")
    parser.add_argument("--style", type=str, default="seaborn-v0_8-whitegrid", help="matplotlib style")
    return parser.parse_args()


def _latest_file(results_dir, pattern):
    files = glob.glob(os.path.join(results_dir, pattern))
    if not files:
        return ""
    files.sort(key=os.path.getmtime)
    return files[-1]


def resolve_inputs(args):
    summary_csv = args.summary_csv.strip()
    temporal_csv = args.temporal_csv.strip()

    if args.results_dir.strip():
        results_dir = args.results_dir.strip()
        if not summary_csv:
            summary_csv = _latest_file(results_dir, "benchmark_summary_*.csv")
        if not temporal_csv:
            temporal_csv = _latest_file(results_dir, "benchmark_temporal_*.csv")

    if not summary_csv or not os.path.isfile(summary_csv):
        raise FileNotFoundError("未找到 summary CSV，请传 --summary_csv 或 --results_dir")
    if not temporal_csv or not os.path.isfile(temporal_csv):
        raise FileNotFoundError("未找到 temporal CSV，请传 --temporal_csv 或 --results_dir")

    if args.out_dir.strip():
        out_dir = args.out_dir.strip()
    else:
        out_dir = os.path.dirname(os.path.abspath(summary_csv))

    os.makedirs(out_dir, exist_ok=True)
    return summary_csv, temporal_csv, out_dir


def read_summary_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {"Method": row["Method"]}
            for k, v in row.items():
                if k == "Method":
                    continue
                parsed[k] = float(v)
            rows.append(parsed)
    if not rows:
        raise ValueError("summary CSV 为空")
    return rows


def read_temporal_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "Method": row["Method"],
                    "LeadTimeMin": int(float(row["LeadTimeMin"])),
                    "Metric": row["Metric"],
                    "Threshold": row["Threshold"],
                    "Pool": int(float(row["Pool"])),
                    "Value": float(row["Value"]),
                }
            )
    if not rows:
        raise ValueError("temporal CSV 为空")
    return rows


def method_order(summary_rows):
    preferred = ["Baseline", "Enhanced"]
    names = [row["Method"] for row in summary_rows]
    ordered = [name for name in preferred if name in names]
    ordered.extend([name for name in names if name not in ordered])
    return ordered


def method_colors(methods):
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    return {m: palette[i % len(palette)] for i, m in enumerate(methods)}


def get_summary_value(summary_rows, method, metric, threshold, pool):
    key = f"{metric}-{threshold}-POOL{pool}"
    for row in summary_rows:
        if row["Method"] == method:
            return row.get(key, np.nan)
    return np.nan


def save_figure(fig, out_dir, basename, dpi):
    png_path = os.path.join(out_dir, f"{basename}.png")
    pdf_path = os.path.join(out_dir, f"{basename}.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_weather_thresholds(summary_rows, methods, colors, out_dir, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)

    x = np.arange(len(THRESHOLDS))
    width = 0.8 / max(1, len(methods))

    for ax_idx, metric in enumerate(WEATHER_METRICS):
        ax = axes[ax_idx]
        for m_idx, method in enumerate(methods):
            ys = [get_summary_value(summary_rows, method, metric, th, 1) for th in THRESHOLDS]
            offset = (m_idx - (len(methods) - 1) / 2) * width
            ax.bar(x + offset, ys, width=width, label=method, color=colors[method], alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(THRESHOLDS)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{metric} @ POOL1")
        ax.set_xlabel("Threshold")
        if ax_idx == 0:
            ax.set_ylabel("Score")

    axes[0].legend(frameon=True, fontsize=9)
    fig.suptitle("Weather Skill by Threshold (SEVIR)", fontsize=12)
    return save_figure(fig, out_dir, "fig_weather_threshold_pool1", dpi)


def plot_weather_pools(summary_rows, methods, colors, out_dir, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)

    x = np.arange(len(POOL_SCALES))
    width = 0.8 / max(1, len(methods))

    for ax_idx, threshold in enumerate(THRESHOLDS):
        ax = axes[ax_idx]
        for m_idx, method in enumerate(methods):
            ys = [get_summary_value(summary_rows, method, "CSI", threshold, p) for p in POOL_SCALES]
            offset = (m_idx - (len(methods) - 1) / 2) * width
            ax.bar(x + offset, ys, width=width, label=method, color=colors[method], alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in POOL_SCALES])
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"CSI-{threshold} across Pools")
        ax.set_xlabel("Pool Scale")
        if ax_idx == 0:
            ax.set_ylabel("CSI")

    axes[0].legend(frameon=True, fontsize=9)
    fig.suptitle("Spatial Tolerance Sensitivity (CSI)", fontsize=12)
    return save_figure(fig, out_dir, "fig_csi_pool_sensitivity", dpi)


def temporal_curve(rows, method, metric, threshold, pool):
    selected = [
        r for r in rows
        if r["Method"] == method
        and r["Metric"] == metric
        and r["Threshold"] == threshold
        and r["Pool"] == pool
    ]
    selected.sort(key=lambda z: z["LeadTimeMin"])
    xs = [r["LeadTimeMin"] for r in selected]
    ys = [r["Value"] for r in selected]
    return xs, ys


def plot_temporal_csi(rows, methods, colors, out_dir, dpi, pool):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)

    for ax_idx, threshold in enumerate(THRESHOLDS):
        ax = axes[ax_idx]
        for method in methods:
            xs, ys = temporal_curve(rows, method, "CSI", threshold, pool)
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=4.5, color=colors[method], label=method)

        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"CSI-{threshold} Temporal Decay")
        ax.set_xlabel("Lead Time (min)")
        if ax_idx == 0:
            ax.set_ylabel("CSI")

    axes[0].legend(frameon=True, fontsize=9)
    fig.suptitle(f"Temporal Decay Curves @ POOL{pool}", fontsize=12)
    return save_figure(fig, out_dir, f"fig_temporal_decay_pool{pool}", dpi)


def plot_temporal_multi_metric(rows, methods, colors, out_dir, dpi, pool):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)

    for ax_idx, metric in enumerate(WEATHER_METRICS):
        ax = axes[ax_idx]
        for method in methods:
            xs, ys = temporal_curve(rows, method, metric, "M", pool)
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=4.5, color=colors[method], label=method)

        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{metric}-M Temporal")
        ax.set_xlabel("Lead Time (min)")
        if ax_idx == 0:
            ax.set_ylabel("Score")

    axes[0].legend(frameon=True, fontsize=9)
    fig.suptitle(f"Temporal Curves (Threshold=M, POOL{pool})", fontsize=12)
    return save_figure(fig, out_dir, f"fig_temporal_metrics_pool{pool}", dpi)


def write_manifest(out_dir, files):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    manifest = os.path.join(out_dir, "paper_figures_manifest.txt")
    with open(manifest, "w", encoding="utf-8") as f:
        f.write(f"Generated at: {ts}\n")
        for p in files:
            f.write(p + "\n")
    return manifest


def main():
    args = parse_args()
    plt.style.use(args.style)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    summary_csv, temporal_csv, out_dir = resolve_inputs(args)
    summary_rows = read_summary_csv(summary_csv)
    temporal_rows = read_temporal_csv(temporal_csv)

    methods = method_order(summary_rows)
    colors = method_colors(methods)

    exported = []
    exported.extend(plot_weather_thresholds(summary_rows, methods, colors, out_dir, args.dpi))
    exported.extend(plot_weather_pools(summary_rows, methods, colors, out_dir, args.dpi))
    exported.extend(plot_temporal_csi(temporal_rows, methods, colors, out_dir, args.dpi, args.pool))
    exported.extend(plot_temporal_multi_metric(temporal_rows, methods, colors, out_dir, args.dpi, args.pool))
    manifest = write_manifest(out_dir, exported)

    print("=" * 90)
    print(f"Summary CSV : {summary_csv}")
    print(f"Temporal CSV: {temporal_csv}")
    print(f"Output Dir  : {out_dir}")
    print("Exported files:")
    for path in exported:
        print(path)
    print(manifest)
    print("=" * 90)


if __name__ == "__main__":
    main()
