import torch
from torch.utils.data import DataLoader
import h5py
import os
from dataset_universal import SEVIRDataset  # 导入你的数据集类

# --- 配置 (保持和你 benchmark_pro.py 一致) ---
H5_PATH = "./sevir_data/SEVIR_VIL_STORMEVENTS_2018_0101_0630.h5"
BATCH_SIZE = 8

def diagnose():
    print("🔍 开始数据诊断...")

    # 1. 检查文件是否存在
    if not os.path.exists(H5_PATH):
        print(f"❌ 致命错误: 文件路径不存在 -> {H5_PATH}")
        print("   -> 请检查路径是否写错？是否在上一级目录？")
        return

    # 2. 检查 H5 文件内部结构
    try:
        with h5py.File(H5_PATH, 'r') as hf:
            print(f"✅ H5 文件打开成功。")
            print(f"   Keys found: {list(hf.keys())}")
            # 假设 SEVIR 通常有 'vil' 这个 key
            if 'vil' in hf:
                print(f"   'vil' shape: {hf['vil'].shape}")
                print(f"   预计样本数 (Series): {hf['vil'].shape[0]}")
            else:
                print("⚠️ 警告: 未在 H5 中找到 'vil' 键，你的 Dataset 类可能无法读取。")
    except Exception as e:
        print(f"❌ H5 文件损坏或无法读取: {e}")
        return

    # 3. 检查 Dataset 类加载情况
    print("\nAttempting to initialize SEVIRDataset...")
    try:
        dataset = SEVIRDataset(H5_PATH)
        print(f"✅ Dataset 初始化成功。")
        print(f"📏 len(dataset) = {len(dataset)}")
        
        if len(dataset) == 0:
            print("❌ 严重问题: Dataset 长度为 0！")
            print("   -> 这意味着 dataset.__len__() 返回了 0。")
            print("   -> 请检查 dataset_universal.py 里的过滤逻辑或索引生成逻辑。")
            return
    except Exception as e:
        print(f"❌ Dataset 初始化崩溃: {e}")
        return

    # 4. 检查 DataLoader 迭代
    print("\nAttempting to iterate DataLoader...")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"📏 len(loader) = {len(loader)}")

    try:
        first_batch = next(iter(loader))
        inputs, targets = first_batch
        print(f"✅ 成功读取第一个 Batch!")
        print(f"   Input shape: {inputs.shape}")
        print(f"   Target shape: {targets.shape}")
    except StopIteration:
        print("❌ DataLoader 是空的 (StopIteration)！无法取出一个 Batch。")
    except Exception as e:
        print(f"❌ 读取 Batch 时报错: {e}")

if __name__ == "__main__":
    diagnose()