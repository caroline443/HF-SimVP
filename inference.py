import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import colors

# 导入你的模型和数据类
from model import SimVP_Enhanced, SimVP_Baseline
from dataset import SEVIRDataset

# --- ⚙️ 配置 ---
BASELINE_PATH = "/root/autodl-tmp/radar/checkpoints_v3/sevir_baseline_best_1211_2305.pth" 
ENHANCED_PATH = "/root/autodl-tmp/radar/checkpoints_v3/sevir_enhanced_best_1211_2321.pth"
H5_PATH = "./sevir_data/SEVIR_VIL_STORMEVENTS_2018_0101_0630.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "./paper_results"
os.makedirs(SAVE_DIR, exist_ok=True)

def visualize_sequence_comparison():
    print("🚀 正在生成序列对比图 (Fig 4 Style)...")
    
    # 1. 挑选一个好的 Case
    # 建议挑选：10, 78, 42, 25 (找一个强回波移动明显的)
    CASE_IDX = 78 
    dataset = SEVIRDataset(H5_PATH)
    input_seq, target_seq = dataset[CASE_IDX] # target_seq shape: [12, 1, 384, 384]
    
    input_tensor = input_seq.unsqueeze(0).to(DEVICE)
    
    # 2. 推理
    models = {}
    try:
        base = SimVP_Baseline(in_shape=(13, 1, 384, 384)).to(DEVICE)
        base.load_state_dict(torch.load(BASELINE_PATH, map_location=DEVICE))
        base.eval()
        with torch.no_grad(): models['SimVP'] = base(input_tensor)
    except: print("⚠️ Baseline 加载失败")

    try:
        ours = SimVP_Enhanced(in_shape=(13, 1, 384, 384)).to(DEVICE)
        ours.load_state_dict(torch.load(ENHANCED_PATH, map_location=DEVICE))
        ours.eval()
        with torch.no_grad(): models['Ours'] = ours(input_tensor)
    except: print("⚠️ Ours 加载失败")

    # 3. 绘图配置
    # 我们选取 6 个时间点: T+10, T+20, T+30, T+40, T+50, T+60
    # 假设数据间隔 5min，对应索引: 1, 3, 5, 7, 9, 11
    time_indices = [1, 3, 5, 7, 9, 11]
    time_labels = ["10 min", "20 min", "30 min", "40 min", "50 min", "60 min"]
    
    rows = ['Ground Truth', 'SimVP', 'Ours']
    nrows = 3
    ncols = 6
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10), constrained_layout=True)
    
    # 定义色标 (使用自定义的离散色标，类似 SimCast，看起来更清晰)
    # 0-255 映射到 0-1
    cmap_data = [
        (0.0, 0.0, 0.0, 0.0), # 透明/黑
        (0.1, 0.1, 0.1, 1.0), # 深灰
        (0.0, 0.0, 1.0, 1.0), # 蓝
        (0.0, 1.0, 1.0, 1.0), # 青
        (0.0, 1.0, 0.0, 1.0), # 绿
        (1.0, 1.0, 0.0, 1.0), # 黄
        (1.0, 0.6, 0.0, 1.0), # 橙
        (1.0, 0.0, 0.0, 1.0), # 红
        (1.0, 0.0, 1.0, 1.0)  # 紫 (极强)
    ]
    # 如果不想用离散的，可以直接用 'jet' 或 'nipy_spectral'
    cmap = 'jet' 
    
    # 4. 填充数据
    for col, t_idx in enumerate(time_indices):
        # --- Row 1: Ground Truth ---
        gt_img = target_seq[t_idx, 0].numpy()
        axes[0, col].imshow(gt_img, cmap=cmap, vmin=0, vmax=1)
        axes[0, col].set_title(f"{time_labels[col]}", fontsize=14)
        if col == 0: axes[0, col].set_ylabel("Ground Truth", fontsize=16, fontweight='bold')
        
        # --- Row 2: SimVP ---
        if 'SimVP' in models:
            pred_img = models['SimVP'][0, t_idx, 0].cpu().numpy()
            axes[1, col].imshow(pred_img, cmap=cmap, vmin=0, vmax=1)
            if col == 0: axes[1, col].set_ylabel("SimVP", fontsize=16, fontweight='bold')
        
        # --- Row 3: Ours ---
        if 'Ours' in models:
            pred_img = models['Ours'][0, t_idx, 0].cpu().numpy()
            axes[2, col].imshow(pred_img, cmap=cmap, vmin=0, vmax=1)
            if col == 0: axes[2, col].set_ylabel("Ours (Enhanced)", fontsize=16, fontweight='bold', color='red')

    # 5. 美化
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        # 给 Ours 加个红框表示强调
        if ax in axes[2, :]:
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(1)

    save_path = os.path.join(SAVE_DIR, f"sequence_comparison_case_{CASE_IDX}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✨ 序列对比图已保存: {save_path}")

if __name__ == "__main__":
    visualize_sequence_comparison()