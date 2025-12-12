import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import multiprocessing
import numpy as np
import copy

# 导入本地模块
from dataset import SEVIRDataset
from model import SimVP_Baseline, SimVP_Enhanced

# --- ⚙️ 全局配置参数 ---
H5_PATH = "./sevir_data/SEVIR_VIL_STORMEVENTS_2018_0101_0630.h5"
BATCH_SIZE = 8
LR = 0.001
EPOCHS = 50           # 设置最大轮数，由早停机制决定何时停止
PATIENCE = 10         # 早停耐心值：如果10轮验证集Loss不下降，就停止
VAL_RATIO = 0.1       # 验证集比例 (10% 用于验证，90% 用于训练)
NUM_WORKERS = min(4, multiprocessing.cpu_count())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 是否尝试断点续训 (如果目录里有 .pth 会自动加载继续跑)
RESUME = True 

class EarlyStopping:
    """早停工具类"""
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
            print(f'   ⚠️ EarlyStopping 计数: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            
    def get_best_model(self):
        return self.best_model_state


def train_one_model(model_class, model_name):
    """
    通用训练函数
    model_class: SimVP_Baseline 或 SimVP_Enhanced 类
    model_name: "baseline" 或 "enhanced" (用于保存文件名)
    """
    print("\n" + "="*50)
    print(f"🚀 启动训练任务: {model_name.upper()}")
    print("="*50)

    # 1. 准备数据 (划分 训练集 vs 验证集)
    full_dataset = SEVIRDataset(H5_PATH)
    total_size = len(full_dataset)
    val_size = int(total_size * VAL_RATIO)
    train_size = total_size - val_size
    
    # 随机切割
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], 
                                              generator=torch.Generator().manual_seed(42))
    
    print(f"📊 数据集划分: 总数 {total_size} | 训练集 {train_size} | 验证集 {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    # 验证集不需要 shuffle
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)

    # 2. 初始化模型
    model = model_class(in_shape=(13, 1, 384, 384)).to(DEVICE)
    
    # --- 断点续训逻辑 ---
    start_epoch = 0
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 优先加载 Best，其次加载最新的 Epoch
    best_path = f"{checkpoint_dir}/{model_name}_best.pth"
    
    if RESUME and os.path.exists(best_path):
        print(f"🔄 发现最优模型存档 {best_path}，正在加载并继续微调...")
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    
    # 3. 优化器 & Loss
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    # 暂时使用标准 L1 Loss 确保收敛
    class WeightedL1Loss(nn.Module):
        def __init__(self, weight=5.0, threshold=0.3):
            super().__init__()
            self.weight = weight
            self.threshold = threshold
            self.mae = nn.L1Loss(reduction='none') 
        def forward(self, pred, target):
            loss = self.mae(pred, target)
            weights = torch.ones_like(target)
            weights[target > self.threshold] = self.weight
            return torch.mean(weights * loss)

    criterion = WeightedL1Loss(weight=5.0, threshold=0.3).to(DEVICE)
    
    # 🔥🔥🔥 修复点：删除了 verbose=True 参数 🔥🔥🔥
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 初始化早停
    early_stopping = EarlyStopping(patience=PATIENCE, delta=0.0001)

    # 4. 训练循环
    for epoch in range(start_epoch, EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"[{model_name.upper()}] Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase (关键！用于判断是否过拟合) ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            # 验证集不更新梯度，速度快
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # 打印日志
        print(f"   Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # 更新学习率
        scheduler.step(avg_val_loss)

        # --- 保存策略 1: 发现最优模型 (Loss 创新低) ---
        if avg_val_loss < early_stopping.best_loss if early_stopping.best_loss else True:
            save_path = f"{checkpoint_dir}/{model_name}_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"   🌟 恭喜！发现新的最优模型，已保存: {save_path}")

        # --- 保存策略 2: 定期保存 (每5轮) ---
        if (epoch + 1) % 5 == 0:
            save_path = f"{checkpoint_dir}/{model_name}_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"   💾 定期存档已保存: {save_path}")

        # --- 早停检查 ---
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"\n🛑 早停触发！验证集 Loss 连续 {PATIENCE} 轮不再下降。")
            print("训练提前结束，防止过拟合。")
            break

    print(f"✅ {model_name} 训练流程结束！")
    
    # 清理显存，为下一个模型腾地方
    del model
    del optimizer
    torch.cuda.empty_cache()

def main():
    # 1. 运行 Baseline
    # try:
    #     train_one_model(SimVP_Baseline, "baseline")
    # except Exception as e:
    #     print(f"❌ Baseline 训练出错: {e}")
    #     # 为了调试，如果 Baseline 挂了，我们还是让 Enhanced 试着跑一下
    #     import traceback
    #     traceback.print_exc()

    # 2. 运行 Enhanced (Ours)
    try:
        train_one_model(SimVP_Enhanced, "enhanced")
    except Exception as e:
        print(f"❌ Enhanced 训练出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()