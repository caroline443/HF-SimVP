import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# 导入两个模型
from model import SimVP_Enhanced, SimVP_Baseline
from dataset import SEVIRDataset

# --- 配置 ---
# benchmark.py 中的配置部分
ENHANCED_PATH = "checkpoints/enhance_epoch_20.pth"  # 你的优化模型
BASELINE_PATH = "checkpoints/baseline_epoch_20.pth" # <--- 修改这里！你刚才跑出来的最新版
H5_PATH = "./sevir_data/SEVIR_VIL_STORMEVENTS_2018_0101_0630.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8

def calculate_csi(pred, target, threshold=0.1):
    """计算气象 CSI (Critical Success Index)"""
    # 二值化
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    
    hits = (pred_bin * target_bin).sum()
    misses = ((1 - pred_bin) * target_bin).sum()
    false_alarms = (pred_bin * (1 - target_bin)).sum()
    
    # CSI = Hits / (Hits + Misses + FalseAlarms)
    csi = hits / (hits + misses + false_alarms + 1e-6)
    return csi.item()

def evaluate_model(model, loader, name="Model"):
    model.eval()
    total_mse = 0
    total_csi = 0
    total_ssim = 0
    count = 0
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=f"Testing {name}"):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # 推理
            preds = model(inputs)
            
            # 1. 计算 MSE
            total_mse += criterion(preds, targets).item()
            
            # 2. 计算 CSI (阈值取 0.2，对应约 50/255 的 VIL 值，属于显著对流)
            total_csi += calculate_csi(preds, targets, threshold=0.4)
            
            # 3. 计算 SSIM (需要转回 CPU numpy)
            # 为了速度，只计算 Batch 中第一张图的最后一帧
            pred_np = preds[0, -1, 0].cpu().numpy()
            target_np = targets[0, -1, 0].cpu().numpy()
            
            # SSIM 需要数据范围在 [0, 1] 或指定 data_range
            score = ssim(target_np, pred_np, data_range=1.0)
            total_ssim += score
            
            count += 1
            if count >= 50: break # 为了快速测试，只跑50个Batch，正式跑可以去掉
            
    return {
        "MSE (↓)": total_mse / count,
        "CSI (↑)": total_csi / count,
        "SSIM (↑)": total_ssim / count
    }

def run_benchmark():
    print("🚀 开始对比评测...")
    dataset = SEVIRDataset(H5_PATH)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 1. 评估 Baseline
    print("\n[1/2] Loading Baseline...")
    try:
        model_base = SimVP_Baseline(in_shape=(13, 1, 384, 384)).to(DEVICE)
        model_base.load_state_dict(torch.load(BASELINE_PATH, map_location=DEVICE))
        metrics_base = evaluate_model(model_base, test_loader, name="Baseline")
    except FileNotFoundError:
        print("❌ 找不到 Baseline 模型文件，请先训练 SimVP_Baseline！")
        metrics_base = {"MSE (↓)": 0, "CSI (↑)": 0, "SSIM (↑)": 0}

    # 2. 评估 Enhanced (Ours)
    print("\n[2/2] Loading Enhanced (Ours)...")
    try:
        model_ours = SimVP_Enhanced(in_shape=(13, 1, 384, 384)).to(DEVICE)
        model_ours.load_state_dict(torch.load(ENHANCED_PATH, map_location=DEVICE))
        metrics_ours = evaluate_model(model_ours, test_loader, name="Enhanced")
    except FileNotFoundError:
        print("❌ 找不到 Enhanced 模型文件！")
        return

    # 3. 打印对比表格
    print("\n" + "="*40)
    print(f"{'Metric':<15} | {'Baseline':<10} | {'Ours (Enhanced)':<15}")
    print("-" * 45)
    for key in metrics_base.keys():
        val_base = metrics_base[key]
        val_ours = metrics_ours[key]
        
        # 判断好坏
        if "(↓)" in key:
            better = val_ours < val_base
        else:
            better = val_ours > val_base
        mark = "✅" if better else " "
        
        print(f"{key:<15} | {val_base:.4f}     | {val_ours:.4f} {mark}")
    print("="*40)
    print("解读：")
    print("1. MSE 越低越好 -> 代表整体像素误差小")
    print("2. CSI 越高越好 -> 代表强对流(暴雨/冰雹)抓得准 (关键指标!)")
    print("3. SSIM 越高越好 -> 代表图像清晰度高，不模糊")

if __name__ == "__main__":
    run_benchmark()