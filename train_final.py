import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import multiprocessing
import numpy as np
import copy
from pytorch_msssim import MS_SSIM # 引入结构相似性损失

# 导入本地模块
from dataset import SEVIRDataset
from model import SimVP_Enhanced 

# --- ⚙️ 全局配置 ---
H5_PATH = "./sevir_data/SEVIR_VIL_STORMEVENTS_2018_0101_0630.h5"
BATCH_SIZE = 8 
LR = 0.0005  # 混合 Loss 比较复杂，学习率稍微调低一点点更稳
EPOCHS = 50
PATIENCE = 10
VAL_RATIO = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "enhanced_hybrid" # 最终版模型名字

# --- 🔥 核心创新：混合损失函数 (Hybrid Loss) ---
class HybridLoss(nn.Module):
    def __init__(self, weight_l1=5.0, threshold=0.3, alpha=0.8):
        super().__init__()
        self.weight_l1 = weight_l1
        self.threshold = threshold
        self.alpha = alpha 
        
        self.l1_loss = nn.L1Loss(reduction='none')
        # MS-SSIM: 多尺度结构相似性
        self.ssim_module = MS_SSIM(data_range=1.0, size_average=True, channel=1)

    def forward(self, pred, target):
        # 1. 计算加权 L1 部分 (不需要改维度，直接算)
        l1 = self.l1_loss(pred, target)
        weights = torch.ones_like(target)
        weights[target > self.threshold] = self.weight_l1
        loss_intensity = torch.mean(weights * l1)
        
        # 2. 计算 SSIM 部分 (🔥🔥🔥 核心修复：维度变形 🔥🔥🔥)
        # 输入 pred 形状: [Batch, Time, Channel, H, W]
        # SSIM 需要: [N, Channel, H, W]
        # 解决方案：把 Batch 和 Time 维度合并
        b, t, c, h, w = pred.shape
        pred_2d = pred.reshape(b * t, c, h, w)
        target_2d = target.reshape(b * t, c, h, w)
        
        # 现在 pred_2d 变成了 [96, 1, 384, 384] (假设Batch=8)，符合 SSIM 要求
        loss_structure = 1 - self.ssim_module(pred_2d, target_2d)
        
        # 3. 组合
        total_loss = (1 - self.alpha) * loss_structure + self.alpha * loss_intensity
        return total_loss

# --- 早停工具 (保持不变) ---
class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
    def get_best_model(self): return self.best_model_state

def train():
    print(f"🚀 启动终极训练: {MODEL_NAME}")
    print("✨ 策略: SimVP + Spatial Attention + Hybrid Loss (Weighted L1 + MS-SSIM)")
    
    # 1. 数据
    full_dataset = SEVIRDataset(H5_PATH)
    val_size = int(len(full_dataset) * VAL_RATIO)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 2. 模型
    model = SimVP_Enhanced(in_shape=(13, 1, 384, 384)).to(DEVICE)
    
    # 3. 优化器 & 混合 Loss
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 🔥 启用混合 Loss
    criterion = HybridLoss(weight_l1=5.0, threshold=0.3, alpha=0.7).to(DEVICE)
    
    early_stopping = EarlyStopping(patience=PATIENCE)

    # 4. 循环
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"   Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        
        scheduler.step(avg_val)
        
        if avg_val < early_stopping.best_loss if early_stopping.best_loss else True:
            torch.save(model.state_dict(), f"checkpoints/{MODEL_NAME}_best.pth")
            print("   🌟 Best Model Saved!")
            
        early_stopping(avg_val, model)
        if early_stopping.early_stop:
            print("🛑 Early Stopping!")
            break

if __name__ == "__main__":
    train()