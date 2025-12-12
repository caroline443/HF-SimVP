# utils_metrics.py
import torch
import numpy as np

class MetricCalculator:
    def __init__(self, device='cuda'):
        self.device = device

    def compute(self, pred, target, threshold=0.3):
        """
        计算标量指标 (CSI, HSS, MSE, PSNR)
        用于训练时的快速验证或 Benchmark 表格生成
        """
        # 确保输入是 tensor
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred).to(self.device)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).to(self.device)

        # 二值化
        pred_bin = (pred > threshold).float()
        target_bin = (target > threshold).float()
        
        # 混淆矩阵
        TP = (pred_bin * target_bin).sum().item()
        FN = ((1 - pred_bin) * target_bin).sum().item()
        FP = (pred_bin * (1 - target_bin)).sum().item()
        TN = ((1 - pred_bin) * (1 - target_bin)).sum().item()
        
        # 1. 气象指标
        csi = TP / (TP + FN + FP + 1e-6)
        pod = TP / (TP + FN + 1e-6)
        far = FP / (TP + FP + 1e-6)
        
        # HSS
        numerator = 2 * (TP * TN - FN * FP)
        denominator = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
        hss = numerator / (denominator + 1e-6)
        
        # 2. 图像指标
        mse = torch.nn.functional.mse_loss(pred, target).item()
        psnr = 10 * np.log10(1.0 / (mse + 1e-8))
        
        return {
            "CSI": csi,
            "HSS": hss,
            "POD": pod,
            "FAR": far,
            "MSE": mse,
            "PSNR": psnr
        }

    @staticmethod
    def compute_rapsd(image_array):
        """
        计算单张图像的径向平均功率谱密度 (RAPSD)
        image_array: numpy array, shape [H, W]
        返回: 1D array (频率对应的能量值)
        """
        # 1. 傅里叶变换
        fft = np.fft.fft2(image_array)
        # 2. 移频，把低频移到中心
        fft_shift = np.fft.fftshift(fft)
        # 3. 计算功率谱 (幅度平方)
        magnitude = np.abs(fft_shift) ** 2
        
        # 4. 径向平均 (把 2D 频谱压扁成 1D 曲线)
        H, W = image_array.shape
        center = (H // 2, W // 2)
        y, x = np.ogrid[:H, :W]
        # 计算每个像素到中心的距离
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(int)
        
        # 统计每个半径 r 下的平均能量
        # bincount 计算每个 r 值对应的 magnitude 之和
        tbin = np.bincount(r.ravel(), magnitude.ravel())
        # 计算每个 r 值出现的次数
        nr = np.bincount(r.ravel())
        
        # 避免除以 0
        radial_profile = tbin / np.maximum(nr, 1)
        
        return radial_profile