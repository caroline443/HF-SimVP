import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import argparse
import logging
from datetime import datetime
from pytorch_msssim import MS_SSIM 

# --- 导入本地模块 ---
from model import SimVP_Baseline, SimVP_Enhanced
from dataset_universal import SEVIRDataset
from utils_metrics import MetricCalculator, MetricTracker

# --- 全局配置 (V3.3 CSI-219修复版) ---
CONFIG = {
    "DATASET_TYPE": "sevir", 
    "PATH_SEVIR": r"F:\zyx\dataset\sevir_data", # 确认路径
    
    "BATCH_SIZE": 8,
    "LR": 0.0002,       # Baseline 默认学习率
    "LR_ENHANCED": 0.0001,  # Enhanced 更保守，降低失稳风险
    "EPOCHS": 50,       # 跑满 50 轮
    "PATIENCE": 50,     # 禁用 Early Stopping
    "GRAD_CLIP": 1.0,   # 梯度裁剪阈值
    "GRAD_NORM_GUARD": 10.0,  # 梯度范数过大时跳过更新
    "LOSS_GUARD": 5.0,  # loss 异常上限
    "LOSS_ALPHA_ENHANCED": 0.6,  # Enhanced 损失组合权重
    "LOSS_WARMUP_EPOCHS": 5,      # warm-up 缩短到 5 epoch，让极端权重更早生效
    "LOSS_EXT_MAX_WEIGHT": 6.0,   # 极端像素权重，足够强以突破均值回归
    "LOSS_FOCAL_WEIGHT": 0.20,    # Focal 分支权重，提高以驱动模型预测极端降水
    "LOSS_FOCAL_ALPHA": 0.75,     # 正样本（漏报）权重，1-alpha 为负样本（误报）权重
    "BEST_MODEL_FAR_PENALTY": 0.3, # best model 判据：score = CSI_219 - FAR_PENALTY * FAR_219
    "LOG_EVERY": 100,
    "NUM_WORKERS": 4,     
    "CKPT_DIR": os.path.join(r"F:\zyx\result", datetime.now().strftime("%Y%m%d")),
    "LOG_DIR": os.path.join(os.path.dirname(__file__), "log"),
    "SEED": 42
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_logger(name, save_dir):
    if logging.getLogger(name).hasHandlers(): return logging.getLogger(name)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    os.makedirs(save_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(save_dir, f'{name}.log'), encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())
    return logger

# --- 动态预热混合损失 V3 (只用于 Enhanced) ---
class HybridLoss(nn.Module):
    """
    三分支混合损失：
      1. loss_structure : MS-SSIM，保结构
      2. loss_intensity : 分级加权 L1，强化中/高/极端降水
      3. loss_focal     : 双向 Focal BCE，同时约束漏报（FN）和误报（FP）

    V3 改动（修复 FAR 失控）：
      - ext_weight: 8.0 -> 4.0，避免过度偏向极端像素导致模型倾向于过度预测
      - Focal Loss 改为双向：
          正样本（target > thresh_ext）：惩罚漏报，权重 focal_alpha
          负样本（target <= thresh_ext 且 pred > thresh_ext）：惩罚误报，权重 1-focal_alpha
        focal_alpha=0.75 表示漏报惩罚略重于误报，在 CSI 和 FAR 之间取得平衡
      - focal_weight: 0.15 -> 0.10，降低 focal 分支整体强度，避免主导训练
    """
    def __init__(self, device, alpha=0.6, warmup_epochs=10, max_weight=4.0,
                 focal_weight=0.10, focal_alpha=0.75):
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.warmup_epochs = warmup_epochs
        self.max_weight = max_weight
        self.focal_weight = focal_weight
        self.focal_alpha = focal_alpha  # 正样本（漏报）权重，1-focal_alpha 为负样本（误报）权重
        self.focal_gamma = 2.0
        self.l1_loss = nn.L1Loss(reduction='none')
        self.ssim_module = MS_SSIM(data_range=1.0, size_average=True, channel=1, win_size=3).to(device)

        self.thresh_mid  = 74.0  / 255.0
        self.thresh_high = 133.0 / 255.0
        self.thresh_ext  = 219.0 / 255.0

        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, pred, target):
        pred   = pred.float()
        target = target.float()

        # --- 1. Warm-up：极端权重从 1.0 线性增长到 max_weight ---
        if self.current_epoch < self.warmup_epochs:
            ext_weight = 1.0 + (self.max_weight - 1.0) * (self.current_epoch / self.warmup_epochs)
        else:
            ext_weight = self.max_weight

        # --- 2. 分级加权 L1 ---
        weights = torch.ones_like(target)
        weights[target > self.thresh_mid]  = 2.0
        weights[target > self.thresh_high] = 5.0
        weights[target > self.thresh_ext]  = ext_weight

        l1_pixel = self.l1_loss(pred, target)
        loss_intensity = torch.mean(weights * l1_pixel)

        # --- 3. MS-SSIM 结构损失（加保护，防止全黑 batch 导致 nan/inf）---
        b, t, c, h, w = pred.shape
        try:
            if self.training:
                pred_n   = pred   + torch.rand_like(pred)   * 1e-6
                target_n = target + torch.rand_like(target) * 1e-6
            else:
                pred_n, target_n = pred, target
            loss_structure = 1 - self.ssim_module(
                pred_n.reshape(-1, c, h, w),
                target_n.reshape(-1, c, h, w)
            )
            if not torch.isfinite(loss_structure):
                loss_structure = torch.tensor(0.0, dtype=pred.dtype, device=self.device)
        except Exception:
            loss_structure = torch.tensor(0.0, dtype=pred.dtype, device=self.device)

        # --- 4. 双向 Focal BCE ---
        # 正样本：target > thresh_ext 的位置，直接对 pred 计算 BCE(pred, 1)
        #   → 无论 pred 当前多低，梯度都会推动 pred 往 1 走（突破均值回归）
        # 负样本：target <= thresh_ext 但 pred > thresh_ext 的位置，计算 BCE(pred, 0)
        #   → 惩罚误报，防止 FAR 失控
        # 两路都用 focal 调制：难样本获得更大梯度
        pred_flat = pred.clamp(1e-6, 1 - 1e-6)
        pos_mask  = (target > self.thresh_ext)                        # 真正极端降水
        neg_mask  = (~pos_mask) & (pred_flat > self.thresh_ext)       # 误报位置

        loss_focal = torch.tensor(0.0, dtype=pred.dtype, device=self.device)
        n_terms = 0

        if pos_mask.any():
            p_pos = pred_flat[pos_mask]
            # focal 调制：p 越低（越难预测），(1-p)^gamma 越大，梯度越强
            fl_pos = ((1 - p_pos) ** self.focal_gamma) * (-torch.log(p_pos))
            loss_focal = loss_focal + self.focal_alpha * fl_pos.mean()
            n_terms += 1

        if neg_mask.any():
            p_neg = pred_flat[neg_mask]
            # focal 调制：p 越高（越确信误报），p^gamma 越大，梯度越强
            fl_neg = (p_neg ** self.focal_gamma) * (-torch.log(1 - p_neg))
            loss_focal = loss_focal + (1 - self.focal_alpha) * fl_neg.mean()
            n_terms += 1

        if n_terms > 1:
            loss_focal = loss_focal / n_terms

        # --- 5. 三分支加权求和 ---
        loss_main = (1 - self.alpha) * loss_structure + self.alpha * loss_intensity
        return loss_main + self.focal_weight * loss_focal

def train_pipeline(mode):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = (device == "cuda")
    timestamp = datetime.now().strftime("%m%d_%H%M")
    logger = get_logger(f"{CONFIG['DATASET_TYPE']}_{mode}_{timestamp}", CONFIG["LOG_DIR"])
    logger.info(f"🚀 启动 V3.1 训练 | 模式: {mode} | Clip: {CONFIG['GRAD_CLIP']}")
    
    # 1. 数据集
    train_set = SEVIRDataset(CONFIG["PATH_SEVIR"], mode='train')
    val_set = SEVIRDataset(CONFIG["PATH_SEVIR"], mode='test')
    
    train_loader = DataLoader(train_set, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, 
                              num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, 
                            num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
    
    logger.info(f"Train Samples: {len(train_set)} | Val Samples: {len(val_set)}")

    # 2. 模型初始化
    in_len, out_len = 13, 12
    if mode == 'baseline':
        logger.info("⚡ 正在初始化 Baseline (SimVP + MSELoss)...")
        model = SimVP_Baseline(in_shape=(in_len, 1, 384, 384)).to(device)
        criterion = nn.MSELoss() 
        train_lr = CONFIG["LR"]
    else: 
        logger.info(
            f"🔥 正在初始化 Enhanced (SpatialAttn + HybridLoss) | "
            f"alpha={CONFIG['LOSS_ALPHA_ENHANCED']} | warmup={CONFIG['LOSS_WARMUP_EPOCHS']} | "
            f"ext_weight={CONFIG['LOSS_EXT_MAX_WEIGHT']} | focal_weight={CONFIG['LOSS_FOCAL_WEIGHT']}"
        )
        model = SimVP_Enhanced(in_shape=(in_len, 1, 384, 384)).to(device)
        criterion = HybridLoss(
            device,
            alpha=CONFIG["LOSS_ALPHA_ENHANCED"],
            warmup_epochs=CONFIG["LOSS_WARMUP_EPOCHS"],
            max_weight=CONFIG["LOSS_EXT_MAX_WEIGHT"],
            focal_weight=CONFIG["LOSS_FOCAL_WEIGHT"],
            focal_alpha=CONFIG["LOSS_FOCAL_ALPHA"],
        ).to(device)
        train_lr = CONFIG["LR_ENHANCED"]
    
    optimizer = optim.AdamW(model.parameters(), lr=train_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=train_lr, steps_per_epoch=len(train_loader), epochs=CONFIG["EPOCHS"],
        pct_start=0.2, div_factor=25, final_div_factor=100
    )
    scaler = GradScaler('cuda', enabled=amp_enabled)
    metrics_calc = MetricCalculator(device)
    logger.info(
        f"🔧 Optimizer配置 | lr={train_lr:.6f} | amp={amp_enabled} | "
        f"loss_guard={CONFIG['LOSS_GUARD']} | grad_guard={CONFIG['GRAD_NORM_GUARD']}"
    )
    
    best_csi = 0.0

    # 3. 训练循环
    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        train_loss = 0
        valid_steps = 0
        skip_loss_steps = 0
        skip_grad_steps = 0
        grad_norm_sum = 0.0
        grad_norm_max = 0.0
        grad_norm_count = 0
        
        # 🔥 [关键修复] 只有 Enhanced 模式的 HybridLoss 才需要更新 Epoch
        if mode == 'enhanced' and hasattr(criterion, 'set_epoch'):
            criterion.set_epoch(epoch)
            
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{CONFIG['EPOCHS']}")
        
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda', enabled=amp_enabled):
                outputs = model(inputs)
            # 在 FP32 下计算损失，减少 MS-SSIM 数值不稳定
            loss = criterion(outputs.float(), targets.float())
            
            # 熔断机制
            if loss.item() > CONFIG["LOSS_GUARD"] or (not torch.isfinite(loss)):
                skip_loss_steps += 1
                if skip_loss_steps <= 5:
                    logger.warning(f"⚠️ Abnormal Loss ({loss.item():.6f}), skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["GRAD_CLIP"])

            if not torch.isfinite(grad_norm) or grad_norm > CONFIG["GRAD_NORM_GUARD"]:
                skip_grad_steps += 1
                if skip_grad_steps <= 5:
                    logger.warning(f"⚠️ Abnormal GradNorm ({float(grad_norm):.6f}), skipping step")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            valid_steps += 1
            grad_norm_value = float(grad_norm)
            grad_norm_sum += grad_norm_value
            grad_norm_max = max(grad_norm_max, grad_norm_value)
            grad_norm_count += 1

            if valid_steps % CONFIG["LOG_EVERY"] == 0:
                loop.set_postfix(loss=f"{loss.item():.4f}", grad=f"{grad_norm_value:.3f}")
            
        # --- 验证 ---
        model.eval()
        tracker = MetricTracker()
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast('cuda', enabled=amp_enabled):
                    outputs = model(inputs)
                batch_metrics = metrics_calc.compute_batch(outputs.float(), targets.float())
                tracker.update(batch_metrics)

        avg_metrics, _, _ = tracker.result()
        # Enhanced 用 CSI-219 + FAR 约束作为主指标，Baseline 仍用 CSI-74
        if mode == 'enhanced':
            avg_csi_219 = avg_metrics.get('CSI-219-POOL1', 0.0)
            avg_far_219 = avg_metrics.get('FAR-219-POOL1', 0.0)
            # score = CSI_219 - penalty * FAR_219
            # penalty=0.3 表示 FAR 每上升 0.1，score 下降 0.03，约束误报不能过高
            far_penalty = CONFIG["BEST_MODEL_FAR_PENALTY"]
            val_score = avg_csi_219 - far_penalty * avg_far_219
        else:
            avg_csi_219 = avg_metrics.get('CSI-219-POOL1', 0.0)
            avg_far_219 = avg_metrics.get('FAR-219-POOL1', 0.0)
            val_score = avg_metrics.get('CSI-74-POOL1', 0.0)
        avg_csi_74 = avg_metrics.get('CSI-74-POOL1', 0.0)
        avg_ssim = avg_metrics.get('SSIM', 0.0)
        avg_loss = train_loss / max(1, valid_steps)
        avg_grad = grad_norm_sum / max(1, grad_norm_count)
        current_lr = optimizer.param_groups[0]["lr"]
        
        logger.info(
            f"Ep {epoch+1} | Loss: {avg_loss:.4f} | "
            f"CSI-74: {avg_csi_74:.4f} | CSI-219: {avg_csi_219:.4f} | FAR-219: {avg_far_219:.4f} | "
            f"Score: {val_score:.4f} | SSIM: {avg_ssim:.4f} | "
            f"Grad(avg/max): {avg_grad:.4f}/{grad_norm_max:.4f} | "
            f"Skip(loss/grad): {skip_loss_steps}/{skip_grad_steps} | "
            f"ValidSteps: {valid_steps}/{len(train_loader)} | LR: {current_lr:.6e}"
        )
        
        if val_score > best_csi:
            best_csi = val_score
            os.makedirs(CONFIG["CKPT_DIR"], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(CONFIG["CKPT_DIR"], f"{mode}_best_csi.pth"))
            logger.info(f"🔥 New Best Score (CSI_219 - {far_penalty}*FAR_219): {best_csi:.4f} "
                        f"[CSI_219={avg_csi_219:.4f}, FAR_219={avg_far_219:.4f}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认跑 enhanced，你可以通过命令行改
    parser.add_argument('--mode', type=str, default='enhanced', choices=['baseline', 'enhanced'])
    parser.add_argument('--data', type=str, default='sevir')
    args = parser.parse_args()
    
    CONFIG['DATASET_TYPE'] = args.data
    set_seed(CONFIG["SEED"])
    
    # 开始训练
    train_pipeline(args.mode)