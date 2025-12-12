# dataset_universal.py
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os
from PIL import Image
import glob

class SEVIRDataset(Dataset):
    """SEVIR (H5格式) 加载器"""
    def __init__(self, h5_file_path, input_len=13, pred_len=12):
        super().__init__()
        self.h5_file_path = h5_file_path
        self.total_len = input_len + pred_len
        self.input_len = input_len
        # 预先读取长度，避免反复打开文件
        with h5py.File(h5_file_path, 'r') as hf:
            self.keys = list(hf.keys())
            # 自动寻找 key
            self.data_source = 'vil' if 'vil' in self.keys else self.keys[0]
            self.num_events = hf[self.data_source].shape[0]

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        with h5py.File(self.h5_file_path, 'r') as hf:
            raw_data = hf[self.data_source][idx] 
        
        # 维度统一化: (T, H, W)
        # SEVIR 原始可能是 (H, W, T=49)
        if raw_data.shape[-1] == 49: 
            raw_data = np.transpose(raw_data, (2, 0, 1))
            
        seq_data = raw_data[:self.total_len]
        # 归一化 (0-255 -> 0-1)
        norm_data = seq_data.astype(np.float32) / 255.0
        # 增加 Channel: (T, 1, H, W)
        norm_data = np.expand_dims(norm_data, axis=1)
        
        return torch.from_numpy(norm_data[:self.input_len]), torch.from_numpy(norm_data[self.input_len:])

class ImageFolderDataset(Dataset):
    """
    通用图片文件夹加载器 (适用于 HKO-7 和 MeteoNet)
    假设目录结构: root/sample_001/1.png, 2.png...
    """
    def __init__(self, root_dir, input_len=10, pred_len=10, file_ext='.png'):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.seq_len = input_len + pred_len
        
        self.samples = []
        if not os.path.exists(root_dir):
            print(f"⚠️ Warning: 路径不存在 {root_dir}")
            return

        # 扫描子文件夹
        subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
        
        for sf in subfolders:
            files = sorted(glob.glob(os.path.join(sf, f"*{file_ext}")))
            # 如果文件夹内图片数量足够，则作为一个样本
            if len(files) >= self.seq_len:
                # 简单起见，只取前 seq_len 张
                self.samples.append(files[:self.seq_len])
                
        print(f"📊 {root_dir} 加载完毕: 共 {len(self.samples)} 个序列样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_paths = self.samples[idx]
        frames = []
        for p in file_paths:
            # 读取灰度图并 Resize 为 384x384
            img = Image.open(p).convert('L').resize((384, 384))
            frames.append(np.array(img))
            
        frames_np = np.stack(frames, axis=0) # (T, H, W)
        norm_data = frames_np.astype(np.float32) / 255.0 # 归一化
        norm_data = np.expand_dims(norm_data, axis=1) # (T, 1, H, W)
        
        return torch.from_numpy(norm_data[:self.input_len]), torch.from_numpy(norm_data[self.input_len:])