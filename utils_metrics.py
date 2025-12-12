import torch
import torch.nn.functional as F
import numpy as np
from math import exp

class MetricCalculator:
    def __init__(self, device='cuda'):
        self.device = device
        # 阈值配置 (归一化到 0-1)
        # M: Moderate (SimCast CSI-M)
        # H: Heavy
        # E: Extreme (SimCast CSI-219)
        self.thresholds = {
            'M': 74.0 / 255.0,   
            'H': 133.0 / 255.0,
            'E': 219.0 / 255.0
        }
        #以此类推，初始化高斯核用于SSIM计算
        self.window = self.create_window(11, 1).to(device)

    def compute_batch(self, pred, target):
        """
        全能指标计算
        """
        # 确保数据在 GPU
        if isinstance(pred, np.ndarray): pred = torch.from_numpy(pred).to(self.device)
        if isinstance(target, np.ndarray): target = torch.from_numpy(target).to(self.device)
        
        results = {}

        # --- 1. 回归/图像指标 (CRPS, SSIM, MSE) ---
        
        # CRPS (对于确定性模型 ≈ MAE/L1)
        mae = torch.mean(torch.abs(pred - target)).item()
        results['CRPS'] = mae # 在论文中注明 CRPS (MAE)

        # MSE
        mse = torch.nn.functional.mse_loss(pred, target).item()
        results['MSE'] = mse

        # SSIM (结构相似性)
        # 确保输入是 [B, C, H, W]，如果 T 维度存在，需要合并 B*T
        if pred.ndim == 5: # [B, T, C, H, W]
            b, t, c, h, w = pred.shape
            pred_s = pred.reshape(-1, c, h, w)
            target_s = target.reshape(-1, c, h, w)
        else:
            pred_s, target_s = pred, target
            
        ssim_val = self._ssim(pred_s, target_s, self.window, window_size=11, channel=1)
        results['SSIM'] = ssim_val.item()

        # --- 2. 分类指标 (CSI, POD, FAR, HSS) ---
        for name, thresh in self.thresholds.items():
            pred_bin = (pred > thresh).float()
            target_bin = (target > thresh).float()

            # 按时间步(dim=1)分别统计，保留时间维度 [T]
            # 假设输入是 [B, T, C, H, W] -> sum over (0, 2, 3, 4)
            dims = (0, 2, 3, 4) if pred.ndim == 5 else (0, 2, 3)
            
            TP = (pred_bin * target_bin).sum(dim=dims)
            FN = ((1 - pred_bin) * target_bin).sum(dim=dims)
            FP = (pred_bin * (1 - target_bin)).sum(dim=dims)
            TN = ((1 - pred_bin) * (1 - target_bin)).sum(dim=dims)

            results[f'TP_{name}'] = TP.cpu().numpy()
            results[f'FN_{name}'] = FN.cpu().numpy()
            results[f'FP_{name}'] = FP.cpu().numpy()
            results[f'TN_{name}'] = TN.cpu().numpy()

        return results

    # --- SSIM 辅助函数 (无需外部库) ---
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.counts = {} 
        self.scalars = {'MSE': [], 'SSIM': [], 'CRPS': []}

    def update(self, batch_results):
        for k, v in batch_results.items():
            if 'TP' in k or 'FN' in k or 'FP' in k or 'TN' in k:
                if k not in self.counts: self.counts[k] = v
                else: self.counts[k] += v
            elif k in self.scalars:
                self.scalars[k].append(v)

    def result(self):
        metrics_curve = {}
        metrics_avg = {}

        # 1. 标量平均
        for k in self.scalars:
            if len(self.scalars[k]) > 0:
                metrics_avg[k] = np.mean(self.scalars[k])

        # 2. 分类指标计算
        for suffix in ['M', 'H', 'E']:
            TP = self.counts.get(f'TP_{suffix}', 0)
            FN = self.counts.get(f'FN_{suffix}', 0)
            FP = self.counts.get(f'FP_{suffix}', 0)
            TN = self.counts.get(f'TN_{suffix}', 0)
            
            # CSI, POD, FAR, HSS
            eps = 1e-6
            csi = TP / (TP + FN + FP + eps)
            pod = TP / (TP + FN + eps)
            far = FP / (TP + FP + eps)
            
            num = 2 * (TP * TN - FN * FP)
            den = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
            hss = num / (den + eps)

            # 存曲线 (Array)
            metrics_curve[f'CSI_{suffix}'] = csi
            metrics_curve[f'POD_{suffix}'] = pod
            metrics_curve[f'FAR_{suffix}'] = far
            metrics_curve[f'HSS_{suffix}'] = hss
            
            # 存平均值 (Scalar)
            metrics_avg[f'CSI_{suffix}'] = csi.mean()
            metrics_avg[f'POD_{suffix}'] = pod.mean()
            metrics_avg[f'FAR_{suffix}'] = far.mean()
            metrics_avg[f'HSS_{suffix}'] = hss.mean()

        return metrics_avg, metrics_curve