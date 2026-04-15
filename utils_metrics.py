import torch
import torch.nn.functional as F
import numpy as np
from math import exp

class MetricCalculator:
    def __init__(self, device='cuda'):
        self.device = device
        # 对应论文的阈值
        self.thresholds = {
            'M': 74.0 / 255.0,    
            'H': 133.0 / 255.0, # 对应 High/133
            'E': 219.0 / 255.0  # 对应 Extreme/219
        }
        self.pool_scales = [1, 4, 16]
        self.window = self.create_window(11, 1).to(device)

    def compute_batch(self, pred, target):
        if isinstance(pred, np.ndarray): pred = torch.from_numpy(pred).to(self.device)
        if isinstance(target, np.ndarray): target = torch.from_numpy(target).to(self.device)
        
        results = {}

        # 1. 标量指标 (CRPS/MSE/SSIM) - 取 Batch 平均
        mae = torch.mean(torch.abs(pred - target)).item()
        results['CRPS'] = mae 
        
        mse = torch.nn.functional.mse_loss(pred, target).item()
        results['MSE'] = mse

        # SSIM 计算 (为了速度，SSIM通常计算 Batch 平均，不画曲线)
        b, t, c, h, w = pred.shape
        pred_s = pred.reshape(-1, c, h, w)
        target_s = target.reshape(-1, c, h, w)
        
        ssim_val = self._ssim(torch.clamp(pred_s, 0, 1), torch.clamp(target_s, 0, 1), 
                              self.window, window_size=11, channel=1)
        results['SSIM'] = ssim_val.item()

        # 2. 分类指标 (CSI等) - 保留时间维度 [T] 以绘制曲线
        # 预处理：将 Tensor 重塑为 [B*T, C, H, W] 进行池化，然后再变回 [B, T, ...]
        
        for scale in self.pool_scales:
            if scale > 1:
                # MaxPool 空间下采样
                p_pooled = F.max_pool2d(pred_s, kernel_size=scale, stride=scale)
                t_pooled = F.max_pool2d(target_s, kernel_size=scale, stride=scale)
            else:
                p_pooled, t_pooled = pred_s, target_s

            # 恢复时间维度: [B*T, C, H', W'] -> [B, T, C, H', W']
            h_new, w_new = p_pooled.shape[-2:]
            p_restored = p_pooled.view(b, t, c, h_new, w_new)
            t_restored = t_pooled.view(b, t, c, h_new, w_new)

            for name, thresh in self.thresholds.items():
                pred_bin = (p_restored > thresh).float()
                target_bin = (t_restored > thresh).float()

                # 关键：只在 (Batch, H, W) 维度求和，保留 Time (dim=1)
                # 结果形状为 [T] (在当前 Batch 上的累积)
                dims = (0, 2, 3, 4) 
                
                TP = (pred_bin * target_bin).sum(dim=dims)
                FN = ((1 - pred_bin) * target_bin).sum(dim=dims)
                FP = (pred_bin * (1 - target_bin)).sum(dim=dims)
                TN = ((1 - pred_bin) * (1 - target_bin)).sum(dim=dims)

                suffix = f"{name}_POOL{scale}"
                results[f'TP_{suffix}'] = TP.cpu().numpy() # numpy array [T]
                results[f'FN_{suffix}'] = FN.cpu().numpy()
                results[f'FP_{suffix}'] = FP.cpu().numpy()
                results[f'TN_{suffix}'] = TN.cpu().numpy()

        return results

    # --- SSIM 辅助函数 (保持不变) ---
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
        mu1_sq = mu1.pow(2); mu2_sq = mu2.pow(2); mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        C1 = 0.01**2; C2 = 0.03**2
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.counts = {} # 存储 numpy array [T]
        self.scalars = {'MSE': [], 'SSIM': [], 'CRPS': []}

    def update(self, batch_results):
        for k, v in batch_results.items():
            if k.startswith(('TP', 'FN', 'FP', 'TN')):
                if k not in self.counts: 
                    self.counts[k] = v # Initialize with first batch array
                else: 
                    self.counts[k] += v # Accumulate arrays
            elif k in self.scalars:
                self.scalars[k].append(v)

    def result(self):
        metrics_avg = {}
        metrics_curve = {} # 专门存 [T] 维度的曲线数据

        # 1. 标量平均
        for k in self.scalars:
            if len(self.scalars[k]) > 0:
                metrics_avg[k] = np.mean(self.scalars[k])

        # 2. 计算 CSI 曲线和平均值
        threshold_names = ['M', 'H', 'E'] # Moderate, High, Extreme
        pool_scales = [1, 4, 16]

        for name in threshold_names:
            for scale in pool_scales:
                suffix = f"{name}_POOL{scale}"
                
                # 获取累积后的 [T] 数组
                TP = self.counts.get(f'TP_{suffix}', np.array([0]))
                FN = self.counts.get(f'FN_{suffix}', np.array([0]))
                FP = self.counts.get(f'FP_{suffix}', np.array([0]))
                TN = self.counts.get(f'TN_{suffix}', np.array([0]))
                
                eps = 1e-6
                # 计算曲线 (Vector operations)
                csi_curve = TP / (TP + FN + FP + eps)
                pod_curve = TP / (TP + FN + eps)
                far_curve = FP / (TP + FP + eps)
                
                # 保存曲线数据 (供绘图用)
                # 命名规范: CSI_M_POOL1 (这是原本的 Key 风格)
                # 如果 scale=1，我们也保留 CSI_M 这种简写方便老代码兼容
                metrics_curve[f'CSI_{suffix}'] = csi_curve
                metrics_curve[f'POD_{suffix}'] = pod_curve
                metrics_curve[f'FAR_{suffix}'] = far_curve
                
                if scale == 1:
                    metrics_curve[f'CSI_{name}'] = csi_curve
                    metrics_curve[f'POD_{name}'] = pod_curve
                    metrics_curve[f'FAR_{name}'] = far_curve

                # 计算全时段平均值 (用于表格)
                metrics_avg[f'CSI-{name}-POOL{scale}'] = csi_curve.mean()
                # 为了兼容你的老代码，如果是 POOL1，增加一个简写 Key
                if scale == 1:
                    metrics_avg[f'CSI_{name}'] = csi_curve.mean()
                    metrics_avg[f'FAR_{name}'] = far_curve.mean()

        return metrics_avg, metrics_curve