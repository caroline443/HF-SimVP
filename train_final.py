import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
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

# --- 全局配置 (V3.1 修复版) ---
CONFIG = {
    "DATASET_TYPE": "sevir", 
    "PATH_SEVIR": r"F:\zyx\HF-SimVP\dataset\sevir_data", # 确认路径
    
    "BATCH_SIZE": 8,
    "LR": 0.0005,       # 5e-4: 配合 Gradient Clip 使用
    "EPOCHS": 50,       # 跑满 50 轮
    "PATIENCE": 50,     # 禁用 Early Stopping
    "GRAD_CLIP": 0.1,   # 防爆阀
    "NUM_WORKERS": 4,     
    "SAVE_DIR": "./checkpoints_v3_stable",
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

# --- 动态预热混合损失 (只用于 Enhanced) ---
class HybridLoss(nn.Module):
    def __init__(self, device, alpha=0.7):
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.l1_loss = nn.L1Loss(reduction='none')
        self.ssim_module = MS_SSIM(data_range=1.0, size_average=True, channel=1, win_size=3).to(device)
        
        self.thresh_mid = 74.0 / 255.0
        self.thresh_high = 133.0 / 255.0
        self.thresh_ext = 210.0 / 255.0 
        
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()

        # Warm-up 策略
        warmup_epochs = 5
        max_weight = 10.0
        
        if self.current_epoch < warmup_epochs:
            ext_weight = 1.0 + (max_weight - 1.0) * (self.current_epoch / warmup_epochs)
        else:
            ext_weight = max_weight
            
        weights = torch.ones_like(target)
        weights[target > self.thresh_mid] = 2.0
        weights[target > self.thresh_high] = 5.0
        weights[target > self.thresh_ext] = ext_weight 

        l1_pixel = self.l1_loss(pred, target)
        loss_intensity = torch.mean(weights * l1_pixel)
        
        if self.training:
            pred_n = pred + torch.rand_like(pred) * 1e-6
            target_n = target + torch.rand_like(target) * 1e-6
        else:
            pred_n, target_n = pred, target
            
        b, t, c, h, w = pred.shape
        loss_structure = 1 - self.ssim_module(pred_n.reshape(-1, c, h, w), target_n.reshape(-1, c, h, w))
        
        return (1 - self.alpha) * loss_structure + self.alpha * loss_intensity

def train_pipeline(mode):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().strftime("%m%d_%H%M")
    logger = get_logger(f"{CONFIG['DATASET_TYPE']}_{mode}_{timestamp}", CONFIG["SAVE_DIR"])
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
    else: 
        logger.info("🔥 正在初始化 Enhanced (SpatialAttn + HybridLoss)...")
        model = SimVP_Enhanced(in_shape=(in_len, 1, 384, 384)).to(device)
        criterion = HybridLoss(device, alpha=0.7).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CONFIG["LR"], steps_per_epoch=len(train_loader), epochs=CONFIG["EPOCHS"],
        pct_start=0.1, div_factor=10, final_div_factor=100
    )
    scaler = GradScaler() 
    metrics_calc = MetricCalculator(device)
    
    best_csi = 0.0

    # 3. 训练循环
    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        train_loss = 0
        
        # 🔥 [关键修复] 只有 Enhanced 模式的 HybridLoss 才需要更新 Epoch
        if mode == 'enhanced' and hasattr(criterion, 'set_epoch'):
            criterion.set_epoch(epoch)
            
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{CONFIG['EPOCHS']}")
        
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs.float(), targets.float())
            
            # 熔断机制
            if loss.item() > 20.0 or torch.isnan(loss) or torch.isinf(loss):
                # Baseline 的 MSE 初始值可能较大，稍微放宽一点阈值到 20
                print(f"⚠️ Warning: Abnormal Loss ({loss.item()}), skipping batch!")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["GRAD_CLIP"])

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
            
        # --- 验证 ---
        model.eval()
        tracker = MetricTracker()
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                batch_metrics = metrics_calc.compute_batch(outputs.float(), targets.float())
                tracker.update(batch_metrics)

        avg_metrics, _ = tracker.result()
        avg_csi = avg_metrics['CSI_M']
        avg_ssim = avg_metrics['SSIM']
        
        logger.info(f"Ep {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | CSI-M: {avg_csi:.4f} | SSIM: {avg_ssim:.4f}")
        
        if avg_csi > best_csi:
            best_csi = avg_csi
            torch.save(model.state_dict(), os.path.join(CONFIG["SAVE_DIR"], f"{mode}_best_csi.pth"))
            logger.info(f"🔥 New Best CSI: {best_csi:.4f}")

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