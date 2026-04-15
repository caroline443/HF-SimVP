# train_pro.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler # 混合精度
from tqdm import tqdm
import os
import argparse
import logging
from datetime import datetime
import copy
from pytorch_msssim import MS_SSIM 

# --- 导入本地模块 ---
# 请确保 dataset_universal.py 和 model.py, utils_metrics.py 在同一目录
from model import SimVP_Baseline, SimVP_Enhanced
from dataset_universal import SEVIRDataset, ImageFolderDataset 
from utils_metrics import MetricCalculator

# --- 全局配置中心 ---
CONFIG = {
    # 默认数据集 (可通过命令行覆盖)
    "DATASET_TYPE": "sevir", 
    
    # === 路径配置 (请修改为你服务器的真实路径) ===
    "PATH_SEVIR": "./sevir_data/SEVIR_VIL_STORMEVENTS_2018_0101_0630.h5",
    "PATH_HKO": "./hko_data/train",         
    "PATH_METEONET": "./meteonet_data/train", 
    
    # === 训练参数 ===
    "BATCH_SIZE": 8,
    "LR": 0.001,
    "EPOCHS": 50,
    "PATIENCE": 10,
    "VAL_RATIO": 0.1,     # 验证集比例
    "NUM_WORKERS": 4,     #数据加载线程数
    "SAVE_DIR": "./checkpoints_v3",
    "SEED": 42
}

# --- 辅助函数: 随机种子 ---
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f" 随机种子已设置为: {seed}")

# --- 辅助函数: 日志系统 ---
def get_logger(name, save_dir):
    if logging.getLogger(name).hasHandlers():
        return logging.getLogger(name)
        
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 文件日志
    file_handler = logging.FileHandler(os.path.join(save_dir, f'{name}.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台日志
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

# --- 混合损失函数 (Hybrid Loss) ---
class HybridLoss(nn.Module):
    def __init__(self, device, weight_l1=5.0, threshold=0.3, alpha=0.5):
        super().__init__()
        self.weight_l1 = weight_l1
        self.threshold = threshold
        self.alpha = alpha
        self.l1_loss = nn.L1Loss(reduction='none')
        # win_size=3 适应小图，防止报错
        self.ssim_module = MS_SSIM(data_range=1.0, size_average=True, channel=1, win_size=3).to(device)

    def forward(self, pred, target):
        # 1. Weighted L1: 针对强回波加权
        l1 = self.l1_loss(pred, target)
        weights = torch.ones_like(target)
        weights[target > self.threshold] = self.weight_l1
        loss_intensity = torch.mean(weights * l1)
        
        # 2. MS-SSIM: 结构相似性
        # 需要将 (B, T, C, H, W) -> (B*T, C, H, W)
        b, t, c, h, w = pred.shape
        pred_flat = pred.reshape(-1, c, h, w)
        target_flat = target.reshape(-1, c, h, w)
        loss_structure = 1 - self.ssim_module(pred_flat, target_flat)
        
        # 3. 组合
        return (1 - self.alpha) * loss_structure + self.alpha * loss_intensity

# --- 主训练管线 ---
def train_pipeline(mode):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().strftime("%m%d_%H%M")
    
    # 获取日志记录器
    logger = get_logger(f"{CONFIG['DATASET_TYPE']}_{mode}_{timestamp}", CONFIG["SAVE_DIR"])
    logger.info(f"启动训练 | 模式: {mode} | 数据集: {CONFIG['DATASET_TYPE']}")
    
    # -------------------------------------------
    # 1. 动态加载数据集 (根据配置切换)
    # -------------------------------------------
    dataset_type = CONFIG['DATASET_TYPE']
    
    if dataset_type == 'sevir':
        dataset = SEVIRDataset(CONFIG["PATH_SEVIR"])
        in_len, out_len = 13, 12
        threshold = 0.3  # VIL 阈值
        
    elif dataset_type == 'hko':
        # HKO 输入10帧，预测10帧
        dataset = ImageFolderDataset(CONFIG["PATH_HKO"], input_len=10, pred_len=10)
        in_len, out_len = 10, 10
        threshold = 0.4  # dBZ 阈值
        
    elif dataset_type == 'meteonet':
        # MeteoNet 假设12帧预测12帧
        dataset = ImageFolderDataset(CONFIG["PATH_METEONET"], input_len=12, pred_len=12)
        in_len, out_len = 12, 12
        threshold = 0.25 
    else:
        raise ValueError(f"未知的数据集类型: {dataset_type}")
        
    logger.info(f"数据集加载完成: {len(dataset)} 样本 | Input: {in_len} -> Output: {out_len}")

    # 划分训练/验证集
    val_size = int(len(dataset) * CONFIG["VAL_RATIO"])
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], 
                                      generator=torch.Generator().manual_seed(CONFIG["SEED"]))
    
    train_loader = DataLoader(train_set, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, 
                              num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, 
                            num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)

    # -------------------------------------------
    # 2. 模型与优化器初始化
    # -------------------------------------------
    logger.info(f" 构建模型: {mode} (In_Shape: {in_len}, 1, 384, 384)")
    
    if mode == 'baseline':
        model = SimVP_Baseline(in_shape=(in_len, 1, 384, 384)).to(device)
        criterion = nn.MSELoss()
        desc = "Baseline (MSE)"
    else: # enhanced
        model = SimVP_Enhanced(in_shape=(in_len, 1, 384, 384)).to(device)
        # Enhanced 使用混合 Loss
        criterion = HybridLoss(device, weight_l1=5.0, threshold=threshold, alpha=0.5).to(device)
        desc = "Enhanced (Hybrid Loss)"
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler() # 混合精度缩放器
    metrics_calc = MetricCalculator(device) # 指标计算工具
    
    # -------------------------------------------
    # 3. 训练循环
    # -------------------------------------------
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{CONFIG['EPOCHS']} [{dataset_type}]")
        
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # 🚀 混合精度前向
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # 🚀 混合精度反向
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- 验证阶段 ---
        model.eval()
        val_loss = 0
        metrics_sum = {"CSI": 0, "HSS": 0}
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # 计算验证集指标 (抽样监控 CSI)
                res = metrics_calc.compute(outputs, targets, threshold=threshold)
                metrics_sum["CSI"] += res["CSI"]
                metrics_sum["HSS"] += res["HSS"]

        avg_val_loss = val_loss / len(val_loader)
        avg_csi = metrics_sum["CSI"] / len(val_loader)
        avg_hss = metrics_sum["HSS"] / len(val_loader)
        
        # 记录日志
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | CSI: {avg_csi:.4f} | HSS: {avg_hss:.4f}")
        
        # 更新学习率
        scheduler.step(avg_val_loss)

        # --- 保存策略 ---
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            # 文件名包含：数据集_模式_best_时间.pth
            save_name = f"{dataset_type}_{mode}_best_{timestamp}.pth"
            save_path = os.path.join(CONFIG["SAVE_DIR"], save_name)
            torch.save(model.state_dict(), save_path)
            logger.info(f"   New Best Model Saved: {save_name}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["PATIENCE"]:
                logger.info("Early Stopping Triggered!")
                break
    
    # 清理显存
    del model, optimizer, scaler
    torch.cuda.empty_cache()

# --- 入口函数 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='both', choices=['baseline', 'enhanced', 'both'], help='训练模式')
    parser.add_argument('--data', type=str, default='sevir', choices=['sevir', 'hko', 'meteonet'], help='选择数据集')
    args = parser.parse_args()
    
    # 覆盖默认配置
    CONFIG['DATASET_TYPE'] = args.data
    set_seed(CONFIG["SEED"])
    
    if args.mode in ['baseline', 'both']:
        train_pipeline('baseline')
        
    if args.mode in ['enhanced', 'both']:
        train_pipeline('enhanced')