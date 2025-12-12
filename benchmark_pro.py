import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.colors as mcolors

# --- 导入你的模块 ---
# 确保 utils_metrics.py 就在同一目录下，且内容是你刚才发给我的最新版
from utils_metrics import MetricCalculator, MetricTracker
from dataset import SEVIRDataset
from model import SimVP_Baseline, SimVP_Enhanced

# --- ⚙️ 全局配置 (请确认路径无误) ---
BASELINE_PATH = "/root/autodl-tmp/radar/checkpoints_v3/sevir_baseline_best_1211_2305.pth" 
ENHANCED_PATH = "/root/autodl-tmp/radar/checkpoints_v3/sevir_enhanced_best_1211_2321.pth"
H5_PATH = "./sevir_data/SEVIR_VIL_STORMEVENTS_2018_0101_0630.h5"
SAVE_DIR = "./paper_results_final" # 结果保存路径

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_TEST_BATCHES = None # 测试多少个 Batch (全量测试可设为 None 或 len(loader))

# 确保保存目录存在
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 🎨 绘图辅助函数 ---

def plot_simcast_curves(trackers, save_dir):
    """
    绘制对标 SimCast 的折线图 (CSI, POD, FAR 随时间变化)
    重点展示 Extreme (219) 和 Moderate (74) 阈值
    """
    print("📈 正在绘制指标折线图...")
    
    # 准备画布: 2行3列 (第一行 Moderate, 第二行 Extreme)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = ['CSI', 'POD', 'FAR']
    thresholds = {'M': 'Moderate (VIL>74)', 'E': 'Extreme (VIL>219)'}
    
    # 假设预测 12 帧，间隔 5 分钟 -> 60 分钟
    time_steps = np.arange(1, 13) * 5 
    
    # 样式配置
    styles = {
        'Baseline': {'fmt': 'b--o', 'label': 'SimVP (Baseline)'},
        'Enhanced': {'fmt': 'r-^', 'label': 'Ours (Enhanced)'}
    }

    for row_idx, (suffix, title_suffix) in enumerate(thresholds.items()):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            key = f"{metric}_{suffix}" # 例如 CSI_M
            
            # 遍历所有模型 (Baseline, Enhanced)
            for name, tracker in trackers.items():
                # 获取该模型的曲线数据 (result() 返回的第二个值是 curves)
                _, curves = tracker.result()
                
                if key in curves:
                    data = curves[key]
                    # 绘制曲线
                    style = styles.get(name, {'fmt': 'g-x', 'label': name})
                    ax.plot(time_steps, data, style['fmt'], label=style['label'], linewidth=2)
            
            ax.set_title(f"{metric} - {title_suffix}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Lead Time (min)")
            ax.set_ylabel(metric)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # 只在第一个子图显示图例
            if row_idx == 0 and col_idx == 0:
                ax.legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, "metrics_curves_simcast.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ 折线图已保存: {save_path}")

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

# --- 🚀 主程序 ---

def run_benchmark():
    print("🚀 启动论文级 Benchmark...")
    
    # 1. 准备数据
    dataset = SEVIRDataset(H5_PATH)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. 加载模型
    models = {}
    try:
        base = SimVP_Baseline(in_shape=(13, 1, 384, 384)).to(DEVICE)
        base.load_state_dict(torch.load(BASELINE_PATH, map_location=DEVICE))
        base.eval()
        models['Baseline'] = base
        print("Load Baseline: OK")
    except Exception as e: print(f"Load Baseline Failed: {e}")

    try:
        ours = SimVP_Enhanced(in_shape=(13, 1, 384, 384)).to(DEVICE)
        ours.load_state_dict(torch.load(ENHANCED_PATH, map_location=DEVICE))
        ours.eval()
        models['Enhanced'] = ours
        print("Load Enhanced: OK")
    except Exception as e: print(f"Load Enhanced Failed: {e}")
    
    if not models: return

    # 3. 初始化评估工具
    # 每个模型一个独立的 Tracker
    trackers = {name: MetricTracker() for name in models}
    calculator = MetricCalculator(device=DEVICE)
    
    # 4. 循环测试
    print(f"开始推理测试集 (Max Batches: {NUM_TEST_BATCHES})...")
    with torch.no_grad():
        for i, (inputs, targets) in tqdm(enumerate(test_loader), total=NUM_TEST_BATCHES):
            if NUM_TEST_BATCHES and i >= NUM_TEST_BATCHES: break
            
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            for name, model in models.items():
                # 推理
                preds = model(inputs)
                # 计算当前 batch 的所有指标
                batch_metrics = calculator.compute_batch(preds, targets)
                # 累积到 Tracker
                trackers[name].update(batch_metrics)

    # 5. 生成结果表格
    print("\n📊 最终指标对比 (Table Comparison):")
    df_data = []
    
    for name, tracker in trackers.items():
        # 获取平均值 (avg) 和 曲线数据 (curves)
        avg_scores, _ = tracker.result()
        
        # 整理成一行数据
        row = {
            'Model': name,
            'CRPS (MAE)': avg_scores['CRPS'],
            'SSIM': avg_scores['SSIM'],
            'HSS': avg_scores['HSS_M'],    # 默认展示 Moderate
            'CSI-M (Pool1)': avg_scores['CSI_M'],
            'CSI-H (133)': avg_scores['CSI_H'],
            'CSI-E (219)': avg_scores['CSI_E'],
            'FAR-M': avg_scores['FAR_M']
        }
        df_data.append(row)

    df = pd.DataFrame(df_data)
    print(df.to_string(index=False)) # 打印到终端
    df.to_csv(os.path.join(SAVE_DIR, "final_metrics_table.csv"), index=False)
    
    # 6. 生成折线图 (CSI/POD/FAR 随时间变化)
    plot_simcast_curves(trackers, SAVE_DIR)
    
    # 7. 生成序列可视化 (Case Study)
    # 建议多跑几个 case_idx 挑个好的，比如 10, 25, 42, 78
    if 'Baseline' in models and 'Enhanced' in models:
        visualize_sequence_strip(models['Baseline'], models['Enhanced'], dataset, case_idx=78, save_dir=SAVE_DIR)
        
    print(f"\n✨ 所有结果已保存至: {SAVE_DIR}")

if __name__ == "__main__":
    run_benchmark()