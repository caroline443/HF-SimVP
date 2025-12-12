import numpy as np
import matplotlib.pyplot as plt
from dataset import SEVIRDataset
from torch.utils.data import DataLoader
import torch

def compute_rapsd(image_array):
    """
    计算图像的径向平均功率谱密度 (RAPSD)
    image_array: [H, W] 的 numpy 数组
    """
    H, W = image_array.shape
    # 1. 二维傅里叶变换
    fft = np.fft.fft2(image_array)
    # 2. 将零频分量移到中心
    fft_shift = np.fft.fftshift(fft)
    # 3. 计算功率谱
    magnitude = np.abs(fft_shift) ** 2
    
    # 4. 计算径向平均
    center = (H // 2, W // 2)
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)
    
    # 统计每个半径下的平均能量
    tbin = np.bincount(r.ravel(), magnitude.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / np.maximum(nr, 1) # 避免除以0
    
    return radial_profile

def plot_psd_comparison(model_baseline, model_ours, dataset, device='cuda'):
    """
    画出 Ground Truth vs Baseline vs Ours 的 PSD 曲线
    """
    print("正在计算 PSD 曲线 (这能证明你的图像不模糊)...")
    
    # 取一个 Batch 数据
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)
    
    # 预测
    with torch.no_grad():
        pred_base = model_baseline(inputs)
        pred_ours = model_ours(inputs)
        
    # 我们只看最后一帧 (T+12)
    idx_frame = -1 
    
    psd_gt = []
    psd_base = []
    psd_ours = []
    
    # 对 Batch 里每一张图算 PSD 然后取平均
    for i in range(inputs.shape[0]):
        gt_img = targets[i, idx_frame, 0].cpu().numpy()
        base_img = pred_base[i, idx_frame, 0].cpu().numpy()
        ours_img = pred_ours[i, idx_frame, 0].cpu().numpy()
        
        psd_gt.append(compute_rapsd(gt_img))
        psd_base.append(compute_rapsd(base_img))
        psd_ours.append(compute_rapsd(ours_img))
        
    # 取平均
    avg_psd_gt = np.mean(psd_gt, axis=0)
    avg_psd_base = np.mean(psd_base, axis=0)
    avg_psd_ours = np.mean(psd_ours, axis=0)
    
    # 画图
    plt.figure(figsize=(8, 6))
    freq = np.arange(len(avg_psd_gt))
    
    # 使用对数坐标，这是标准做法
    plt.loglog(freq, avg_psd_gt, 'k-', label='Ground Truth (Real)', linewidth=2)
    plt.loglog(freq, avg_psd_base, 'b--', label='SimVP Baseline (Blurry)')
    plt.loglog(freq, avg_psd_ours, 'r-', label='Ours (Enhanced)', linewidth=2)
    
    plt.xlabel('Frequency (Wavenumber)')
    plt.ylabel('Power Spectral Density')
    plt.title('Analysis of Image Sharpness (PSD)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig('psd_comparison.png')
    print("✅ PSD 对比图已生成: psd_comparison.png")