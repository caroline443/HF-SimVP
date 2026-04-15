import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.colors as mcolors

# --- 导入模块 ---
from utils_metrics import MetricCalculator, MetricTracker
from dataset_universal import SEVIRDataset
from model import SimVP_Baseline, SimVP_Enhanced

# --- ⚙️ 全局配置 ---
BASELINE_PATH = r"F:\zyx\HF-SimVP\checkpoints_v4_full\sevir_baseline_best_loss_1213_2026.pth" 
ENHANCED_PATH = r"F:\zyx\HF-SimVP\checkpoints_v4_full\sevir_enhanced_best_csi_1219_0916.pth"
H5_PATH = r"F:\zyx\HF-SimVP\dataset\sevir_data"
SAVE_DIR = "./paper_results_final_v5" # 改个名字防止覆盖

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_TEST_BATCHES = None # None 表示跑全量

os.makedirs(SAVE_DIR, exist_ok=True)

# --- 🎨 绘图函数 (逻辑基本不变，只需确认Key) ---
def plot_simcast_curves(trackers, save_dir):
    print("📈 正在绘制指标折线图...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = ['CSI', 'POD', 'FAR']
    # 注意: 这里使用 utils_metrics 里生成的兼容 Key
    thresholds = {'M': 'Moderate (VIL>74)', 'E': 'Extreme (VIL>219)'}
    
    time_steps = np.arange(1, 13) * 5 
    styles = {
        'Baseline': {'fmt': 'b--o', 'label': 'Baseline'},
        'Enhanced': {'fmt': 'r-^', 'label': 'Ours (HF-SimVP)'}
    }

    for row_idx, (suffix, title_suffix) in enumerate(thresholds.items()):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            # 这里对应 MetricTracker 返回的 metrics_curve 里的 Key，例如 CSI_M
            key = f"{metric}_{suffix}" 
            
            for name, tracker in trackers.items():
                _, curves = tracker.result()
                if key in curves:
                    data = curves[key]
                    style = styles.get(name, {'fmt': 'g-x', 'label': name})
                    ax.plot(time_steps, data, style['fmt'], label=style['label'], linewidth=2)
            
            ax.set_title(f"{metric} - {title_suffix}", fontsize=12)
            ax.set_xlabel("Lead Time (min)")
            ax.grid(True, linestyle='--', alpha=0.5)
            if row_idx == 0 and col_idx == 0: ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_curves.png"), dpi=300)
    print("✅ 折线图绘制完成")

# --- 序列可视化函数 (保持不变) ---
def visualize_sequence_strip(model_base, model_ours, dataset, case_idx, save_dir):
    """
    绘制对标 SimCast Fig.4 的长条序列对比图
    """
    print(f"🎨 正在绘制 Case {case_idx} 的序列可视化...")
    
    input_seq, target_seq = dataset[case_idx]
    input_tensor = input_seq.unsqueeze(0).to(DEVICE)
    
    # 推理
    with torch.no_grad():
        pred_base = model_base(input_tensor)
        pred_ours = model_ours(input_tensor)
    
    # 定义 SimCast 风格离散色标
    bounds = np.array([0, 16, 31, 59, 74, 100, 133, 160, 181, 255]) / 255.0
    colors_hex = ['#FFFFFF', '#C0C0C0', '#00FF00', '#008000', '#FFFF00', '#FFA500', '#FF0000', '#8B0000', '#FF00FF']
    cmap = mcolors.ListedColormap(colors_hex)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # 选取 6 个时间点 (10min ... 60min)
    time_indices = [1, 3, 5, 7, 9, 11]
    time_labels = ["10 min", "20 min", "30 min", "40 min", "50 min", "60 min"]
    
    fig, axes = plt.subplots(3, 6, figsize=(20, 8), constrained_layout=True)
    
    data_map = [
        ('Ground Truth', target_seq),
        ('SimVP', pred_base[0].cpu()),
        ('Ours', pred_ours[0].cpu())
    ]
    
    for row, (name, seq_data) in enumerate(data_map):
        for col, t_idx in enumerate(time_indices):
            ax = axes[row, col]
            img = seq_data[t_idx, 0].numpy()
            
            ax.imshow(img, cmap=cmap, norm=norm)
            ax.set_xticks([]); ax.set_yticks([])
            
            if row == 0: ax.set_title(time_labels[col], fontsize=14)
            if col == 0: 
                color = 'red' if name == 'Ours' else 'black'
                weight = 'bold' if name == 'Ours' else 'normal'
                ax.set_ylabel(name, fontsize=16, fontweight=weight, color=color)
            
            # 给 Ours 加红框
            if name == 'Ours':
                for spine in ax.spines.values():
                    spine.set_edgecolor('red'); spine.set_linewidth(2)

    save_path = os.path.join(save_dir, f"vis_sequence_case_{case_idx}.png")
    plt.savefig(save_path, dpi=200)
    print(f"✅ 序列图已保存: {save_path}")

# --- 🚀 主程序 (表格生成逻辑大改) ---
def run_benchmark():
    print("🚀 启动最终 Benchmark...")
    
    # 1. 数据与模型
    dataset = SEVIRDataset(H5_PATH, mode='test')
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    models = {}
    # 加载 Baseline
    try:
        base = SimVP_Baseline(in_shape=(13, 1, 384, 384)).to(DEVICE)
        base.load_state_dict(torch.load(BASELINE_PATH, map_location=DEVICE))
        base.eval()
        models['Baseline'] = base
    except: print("⚠️ Baseline 加载失败")

    # 加载 Enhanced
    try:
        ours = SimVP_Enhanced(in_shape=(13, 1, 384, 384)).to(DEVICE)
        ours.load_state_dict(torch.load(ENHANCED_PATH, map_location=DEVICE))
        ours.eval()
        models['Enhanced'] = ours
    except: print("⚠️ Enhanced 加载失败")

    # 2. 推理
    trackers = {name: MetricTracker() for name in models}
    calculator = MetricCalculator(device=DEVICE)
    
    print(f"🔄 开始推理 (Batches: {len(test_loader)})...")
    with torch.no_grad():
        for i, (inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader)):
            if NUM_TEST_BATCHES and i >= NUM_TEST_BATCHES: break
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            for name, model in models.items():
                preds = model(inputs)
                batch_res = calculator.compute_batch(preds, targets)
                trackers[name].update(batch_res)

    # 3. 🔥 生成论文表格 (SimCast 风格) 🔥
    print("\n📊 生成最终对比表格...")
    df_data = []
    
    for name, tracker in trackers.items():
        avg, _ = tracker.result()
        
        row = {
            'Method': name,
            'CRPS': f"{avg['CRPS']:.4f}",
            'SSIM': f"{avg['SSIM']:.4f}",
            # CSI-M (Moderate) Pool 1/4/16
            'CSI-M-1': f"{avg['CSI-M-POOL1']:.4f}",
            'CSI-M-4': f"{avg['CSI-M-POOL4']:.4f}",
            'CSI-M-16': f"{avg['CSI-M-POOL16']:.4f}",
            # CSI-181 (High) Pool 1/4/16 (对应你的 CSI-H)
            'CSI-H-1': f"{avg['CSI-H-POOL1']:.4f}",
            'CSI-H-4': f"{avg['CSI-H-POOL4']:.4f}",
            'CSI-H-16': f"{avg['CSI-H-POOL16']:.4f}",
            # CSI-219 (Extreme) Pool 1/4/16
            'CSI-E-1': f"{avg['CSI-E-POOL1']:.4f}",
            'CSI-E-4': f"{avg['CSI-E-POOL4']:.4f}",
            'CSI-E-16': f"{avg['CSI-E-POOL16']:.4f}",
        }
        df_data.append(row)
        
    df = pd.DataFrame(df_data)
    # 调整列顺序，使其更像论文
    cols = ['Method', 'CRPS', 'SSIM', 
            'CSI-M-1', 'CSI-M-4', 'CSI-M-16', 
            'CSI-H-1', 'CSI-H-4', 'CSI-H-16',
            'CSI-E-1', 'CSI-E-4', 'CSI-E-16']
    df = df[cols]
    
    print(df.to_string(index=False))
    df.to_csv(os.path.join(SAVE_DIR, "final_paper_table.csv"), index=False)

    # 4. 绘图
    plot_simcast_curves(trackers, SAVE_DIR)
    
    # 5. 可视化 Case (找个好的 Case 比如 78)
    if 'Baseline' in models and 'Enhanced' in models:
        visualize_sequence_strip(models['Baseline'], models['Enhanced'], dataset, case_idx=78, save_dir=SAVE_DIR)
        
    print(f"\n✨ 所有结果已保存至: {SAVE_DIR}")

if __name__ == "__main__":
    run_benchmark()