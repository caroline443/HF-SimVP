# benchmark_temporal.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd

# 导入本地模块
from dataset_universal import SEVIRDataset
from model import SimVP_Baseline, SimVP_Enhanced
from utils_metrics import MetricCalculator

# --- ⚙️ 配置中心 ---
# 🔥🔥🔥 请务必修改为你刚才训练出来的最佳模型路径 🔥🔥🔥
BASELINE_PATH = "/root/autodl-tmp/radar/checkpoints_v3/sevir_baseline_best_1211_2305.pth"
ENHANCED_PATH = "/root/autodl-tmp/radar/checkpoints_v3/sevir_enhanced_best_1211_2321.pth"

# 数据路径
H5_PATH = "./sevir_data/SEVIR_VIL_STORMEVENTS_2018_0101_0630.h5"
SAVE_DIR = "./paper_results"  # 图表保存位置

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

def run_temporal_benchmark():
    print("🚀 启动时序稳定性评测 (Temporal Decay Analysis)...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. 准备数据
    dataset = SEVIRDataset(H5_PATH)
    # 为了快一点，只测前 100 个 batch (论文足够了)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. 加载模型
    models = {}
    
    if os.path.exists(BASELINE_PATH):
        print(f"📦 Loading Baseline: {BASELINE_PATH}")
        m_base = SimVP_Baseline(in_shape=(13, 1, 384, 384)).to(DEVICE)
        m_base.load_state_dict(torch.load(BASELINE_PATH, map_location=DEVICE))
        m_base.eval()
        models["Baseline"] = m_base
    
    if os.path.exists(ENHANCED_PATH):
        print(f"📦 Loading Enhanced: {ENHANCED_PATH}")
        m_ours = SimVP_Enhanced(in_shape=(13, 1, 384, 384)).to(DEVICE)
        m_ours.load_state_dict(torch.load(ENHANCED_PATH, map_location=DEVICE))
        m_ours.eval()
        models["Enhanced"] = m_ours
        
    if not models:
        print("❌ 未找到模型文件，请检查路径配置！")
        return

    # 3. 初始化统计容器
    # 我们要记录每一帧 (T=0 到 T=11) 的 CSI
    # 结构: {"Baseline": [csi_t0, csi_t1, ...], "Enhanced": [...]}
    temporal_stats = {name: np.zeros(12) for name in models}
    frame_counts = 0
    max_batches = 50 # 测 50 个 batch 足够画趋势了
    
    calculator = MetricCalculator(DEVICE)
    threshold = 0.3 # 强对流阈值
    
    # 4. 循环测试
    with torch.no_grad():
        for i, (inputs, targets) in tqdm(enumerate(loader), total=max_batches, desc="Testing"):
            if i >= max_batches: break
            
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            for name, model in models.items():
                preds = model(inputs) # [B, 12, 1, H, W]
                
                # 针对每一帧分别计算 CSI
                for t in range(12):
                    # 取出第 t 帧: [B, 1, H, W]
                    pred_t = preds[:, t, ...]
                    target_t = targets[:, t, ...]
                    
                    # 计算该帧的指标
                    metrics = calculator.compute(pred_t, target_t, threshold=threshold)
                    temporal_stats[name][t] += metrics["CSI"]
            
            frame_counts += 1
            
    # 取平均
    for name in models:
        temporal_stats[name] /= frame_counts

    # 5. 打印数据 & 保存 CSV
    print("\n📊 时序衰减数据 (CSI per Frame):")
    print(f"{'Frame':<6} | {'Baseline':<10} | {'Enhanced':<10} | {'Improvement':<10}")
    print("-" * 45)
    
    frames = list(range(1, 13)) # 1-12帧 (即 5min - 60min)
    csv_data = {"Frame": frames}
    
    for t in range(12):
        base_score = temporal_stats.get("Baseline", np.zeros(12))[t]
        ours_score = temporal_stats.get("Enhanced", np.zeros(12))[t]
        imp = (ours_score - base_score) / (base_score + 1e-6) * 100
        
        print(f"T+{t+1:<2}   | {base_score:.4f}     | {ours_score:.4f}     | +{imp:.1f}%")
    
    # 保存 CSV
    for name in models:
        csv_data[name] = temporal_stats[name]
    pd.DataFrame(csv_data).to_csv(os.path.join(SAVE_DIR, "temporal_metrics.csv"), index=False)

    # 6. 画折线图 (重点！)
    print("\n🎨 正在绘制时序衰减曲线...")
    plt.figure(figsize=(10, 6))
    
    # 定义时间轴 (每帧5分钟)
    time_axis = [t * 5 for t in range(1, 13)] 
    
    # 画线
    if "Baseline" in models:
        plt.plot(time_axis, temporal_stats["Baseline"], 'b--o', label='Baseline (SimVP)', linewidth=2, markersize=6)
    if "Enhanced" in models:
        plt.plot(time_axis, temporal_stats["Enhanced"], 'r-^', label='Ours (Enhanced)', linewidth=2, markersize=6)
    
    plt.xlabel('Forecast Time (minutes)', fontsize=12)
    plt.ylabel(f'CSI (Threshold={threshold})', fontsize=12)
    plt.title('Performance Decay over Time', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # 保存
    save_path = os.path.join(SAVE_DIR, "temporal_decay_curve.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ 折线图已保存至: {save_path}")

if __name__ == "__main__":
    run_temporal_benchmark()