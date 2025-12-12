# benchmark_final.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd # 需要 pip install pandas 用于生成漂亮表格

# 导入本地模块
from dataset import SEVIRDataset
from model import SimVP_Baseline, SimVP_Enhanced
from utils_metrics import MetricCalculator

# --- ⚙️ 配置中心 ---
# 请修改为你的实际路径
BASELINE_PATH = "/root/autodl-tmp/radar/checkpoints_v3/sevir_baseline_best_1211_2305.pth" 
ENHANCED_PATH = "/root/autodl-tmp/radar/checkpoints_v3/sevir_enhanced_best_1211_2321.pth"
H5_PATH = "./sevir_data/SEVIR_VIL_STORMEVENTS_2018_0101_0630.h5"
SAVE_DIR = "./paper_results" # 所有图表保存到这里

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8

os.makedirs(SAVE_DIR, exist_ok=True)

def visualize_case(inputs, target, pred_base, pred_ours, idx, save_dir):
    """
    生成论文用的四联对比图
    """
    # 取 Batch 中的第一个样本，取最后一帧 (T+12)
    in_img = inputs[0, -1, 0].cpu().numpy()
    gt_img = target[0, -1, 0].cpu().numpy()
    base_img = pred_base[0, -1, 0].cpu().numpy()
    ours_img = pred_ours[0, -1, 0].cpu().numpy()
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    cmap = 'jet' # 气象雷达标准色标
    
    # 1. Input
    axes[0].imshow(in_img, cmap=cmap, vmin=0, vmax=1)
    axes[0].set_title("Input (Last Frame)", fontsize=14)
    axes[0].axis('off')
    
    # 2. Ground Truth
    axes[1].imshow(gt_img, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("Ground Truth (T+1h)", fontsize=14)
    axes[1].axis('off')
    
    # 3. Baseline
    axes[2].imshow(base_img, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title("Baseline (SimVP)", fontsize=14)
    axes[2].axis('off')
    
    # 4. Ours
    im = axes[3].imshow(ours_img, cmap=cmap, vmin=0, vmax=1)
    axes[3].set_title("Ours (Enhanced)", fontsize=14, fontweight='bold', color='red')
    axes[3].axis('off')
    
    # Colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    plt.savefig(os.path.join(save_dir, f"vis_case_{idx}.png"), bbox_inches='tight', dpi=150)
    plt.close()

def run_full_benchmark():
    print("🚀 启动论文级全套评测...")
    
    # 1. 准备
    dataset = SEVIRDataset(H5_PATH)
    # 选取一部分数据进行测试，避免跑太久
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    calculator = MetricCalculator(DEVICE)
    
    # 2. 加载模型
    models = {}
    try:
        base = SimVP_Baseline(in_shape=(13, 1, 384, 384)).to(DEVICE)
        base.load_state_dict(torch.load(BASELINE_PATH, map_location=DEVICE))
        base.eval()
        models['Baseline'] = base
    except: print("⚠️ Baseline 模型未找到")

    try:
        ours = SimVP_Enhanced(in_shape=(13, 1, 384, 384)).to(DEVICE)
        ours.load_state_dict(torch.load(ENHANCED_PATH, map_location=DEVICE))
        ours.eval()
        models['Enhanced'] = ours
    except: print("⚠️ Enhanced 模型未找到")
    
    if not models: return

    # 3. 统计容器
    stats = {name: {"CSI": 0, "HSS": 0, "MSE": 0, "PSNR": 0} for name in models}
    psd_records = {name: [] for name in models}
    gt_psd_records = []
    
    # 4. 循环测试
    max_batches = 30 # 测 30 个 batch
    vis_interval = 10 # 每 10 个 batch 保存一张可视化的图
    
    with torch.no_grad():
        for i, (inputs, targets) in tqdm(enumerate(test_loader), total=max_batches):
            if i >= max_batches: break
            
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # 记录 GT PSD
            gt_last = targets[:, -1, 0].cpu().numpy()
            for img in gt_last:
                gt_psd_records.append(calculator.compute_rapsd(img))
            
            # 模型推理
            preds_map = {}
            for name, model in models.items():
                preds = model(inputs)
                preds_map[name] = preds
                
                # 计算标量指标
                metrics = calculator.compute(preds, targets, threshold=0.3)
                stats[name]["CSI"] += metrics["CSI"]
                stats[name]["HSS"] += metrics["HSS"]
                stats[name]["MSE"] += metrics["MSE"]
                stats[name]["PSNR"] += metrics["PSNR"]
                
                # 记录 PSD
                pred_last = preds[:, -1, 0].cpu().numpy()
                for img in pred_last:
                    psd_records[name].append(calculator.compute_rapsd(img))
            
            # 生成可视化图 (每隔几次保存一张，用于挑选 Case)
            if i % vis_interval == 0 and 'Baseline' in preds_map and 'Enhanced' in preds_map:
                visualize_case(inputs, targets, preds_map['Baseline'], preds_map['Enhanced'], i, SAVE_DIR)

    # 5. 结果汇总与输出
    
    # A. 打印 Excel 表格
    print("\n📊 最终指标对比 (Average over dataset):")
    df_data = []
    for name in models:
        row = {k: v/max_batches for k, v in stats[name].items()}
        row['Model'] = name
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    # 调整列顺序
    df = df[['Model', 'CSI', 'HSS', 'MSE', 'PSNR']]
    print(df.to_markdown(index=False))
    # 保存 CSV
    df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"), index=False)

    # B. 画 PSD 曲线
    print("\n🎨 绘制 PSD 曲线...")
    plt.figure(figsize=(10, 8))
    
    # GT
    avg_gt_psd = np.mean(gt_psd_records, axis=0)
    freq = np.arange(len(avg_gt_psd))
    plt.loglog(freq, avg_gt_psd, 'k-', label='Ground Truth', linewidth=2)
    
    colors = {'Baseline': 'blue', 'Enhanced': 'red'}
    for name in models:
        avg_psd = np.mean(psd_records[name], axis=0)
        min_len = min(len(freq), len(avg_psd))
        plt.loglog(freq[:min_len], avg_psd[:min_len], 
                   color=colors.get(name, 'green'), label=name, linewidth=2)
    
    plt.xlabel('Frequency')
    plt.ylabel('Power Spectral Density')
    plt.title('PSD Analysis')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(os.path.join(SAVE_DIR, "final_psd_curve.png"), dpi=300)
    
    print(f"✅ 所有结果已保存至: {SAVE_DIR}")

if __name__ == "__main__":
    run_full_benchmark()