import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# 导入两个模型类
from model import SimVP_Enhanced, SimVP_Baseline
from dataset import SEVIRDataset

# --- 核心配置 ---
BASELINE_PATH = "/root/autodl-tmp/radar/checkpoints_v3/sevir_baseline_best_1211_2305.pth" 
ENHANCED_PATH = "/root/autodl-tmp/radar/checkpoints_v3/sevir_enhanced_best_1211_2321.pth"
H5_PATH = "./sevir_data/SEVIR_VIL_STORMEVENTS_2018_0101_0630.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def visualize_comparison():
    print("🚀 启动终极对比可视化...")
    
    # 1. 准备数据 (随机取一个样本)
    dataset = SEVIRDataset(H5_PATH)
    # 建议多试几个 index，找到一个强对流明显的例子 (比如 idx=10, 25, 42 等)
    idx = 78
    print(f"正在读取第 {idx} 个事件...")
    input_seq, target_seq = dataset[idx]
    
    # 转为 Batch 格式
    input_tensor = input_seq.unsqueeze(0).to(DEVICE)
    
    # 2. 加载 Baseline 模型并推理
    print("正在加载 Baseline...")
    try:
        model_base = SimVP_Baseline(in_shape=(13, 1, 384, 384)).to(DEVICE)
        model_base.load_state_dict(torch.load(BASELINE_PATH, map_location=DEVICE))
        model_base.eval()
        with torch.no_grad():
            pred_base = model_base(input_tensor)
        print("✅ Baseline 推理完成")
    except Exception as e:
        print(f"❌ Baseline 加载失败: {e}")
        pred_base = None

    # 3. 加载 Enhanced (Ours) 模型并推理
    print("正在加载 Enhanced (Ours)...")
    try:
        model_ours = SimVP_Enhanced(in_shape=(13, 1, 384, 384)).to(DEVICE)
        model_ours.load_state_dict(torch.load(ENHANCED_PATH, map_location=DEVICE))
        model_ours.eval()
        with torch.no_grad():
            pred_ours = model_ours(input_tensor)
        print("✅ Enhanced 推理完成")
    except Exception as e:
        print(f"❌ Enhanced 加载失败: {e}")
        pred_ours = None

    # 4. 绘图准备
    # 取未来第 6 帧 (T+30min) 进行对比
    frame_idx = 5 
    
    # 获取 Numpy 数组
    last_input = input_seq[-1, 0].numpy()
    target_img = target_seq[frame_idx, 0].numpy()
    
    if pred_base is not None:
        img_base = pred_base[0, frame_idx, 0].cpu().numpy()
    else:
        img_base = np.zeros_like(target_img)

    if pred_ours is not None:
        img_ours = pred_ours[0, frame_idx, 0].cpu().numpy()
    else:
        img_ours = np.zeros_like(target_img)

    # 5. 画四联图
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    cmap = 'jet' # 气象常用色标
    
    # (1) Input
    axes[0].imshow(last_input, cmap=cmap, vmin=0, vmax=1)
    axes[0].set_title("Input (T=0)", fontsize=14)
    axes[0].axis('off')
    
    # (2) Ground Truth
    axes[1].imshow(target_img, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("Ground Truth (T+30m)", fontsize=14)
    axes[1].axis('off')
    
    # # (3) Baseline
    axes[2].imshow(img_base, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title("Baseline", fontsize=14)
    axes[2].axis('off')
    
    # (4) Ours
    im = axes[3].imshow(img_ours, cmap=cmap, vmin=0, vmax=1)
    axes[3].set_title("Ours (Enhanced)", fontsize=14, fontweight='bold', color='red')
    axes[3].axis('off')
    
    # Colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    save_path = "comparison_result.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"\n✨ 对比图已生成: {save_path}")
    print("请打开图片，重点观察 'Baseline' 和 'Ours' 在红色强回波区域的区别！")

if __name__ == "__main__":
    visualize_comparison()