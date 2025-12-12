import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class SEVIRDataset(Dataset):
    """
    针对 SEVIR VIL 数据集的专用加载器
    功能：读取H5 -> 维度修正 -> 归一化 -> 切分输入/输出
    """
    def __init__(self, h5_file_path, input_len=13, pred_len=12):
        super().__init__()
        self.h5_file_path = h5_file_path
        self.input_len = input_len
        self.pred_len = pred_len
        self.total_len = input_len + pred_len
        
        # 打开文件读取元数据
        with h5py.File(h5_file_path, 'r') as hf:
            self.keys = list(hf.keys())
            # SEVIR 官方数据的 key 通常是 'vil' 或者一串数字ID
            # 这里做一个简单的自动查找
            if 'vil' in self.keys:
                self.data_source = 'vil'
                self.num_events = hf['vil'].shape[0]
            else:
                # 如果不是 'vil'，默认取第一个 key
                self.data_source = self.keys[0]
                self.num_events = hf[self.data_source].shape[0]

    def _normalize(self, data):
        """将 0-255 映射到 0-1"""
        return data.astype(np.float32) / 255.0

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        with h5py.File(self.h5_file_path, 'r') as hf:
            # 读取原始数据
            raw_data = hf[self.data_source][idx] 
        
        # --- 核心修正: 维度检查与转置 ---
        # 目标格式: (Time, Height, Width)
        # 如果原始格式是 (H, W, T) 即最后一维是时间(通常是49)
        if raw_data.shape[-1] == 49:
            raw_data = np.transpose(raw_data, (2, 0, 1))
        
        # 截取我们需要的时间长度
        seq_data = raw_data[:self.total_len]
        
        # 归一化
        norm_data = self._normalize(seq_data)
        
        # 增加 Channel 维度: [Time, H, W] -> [Time, Channel, H, W]
        norm_data = np.expand_dims(norm_data, axis=1)
        
        # 拆分 Input (过去) 和 Target (未来)
        input_seq = norm_data[:self.input_len]
        target_seq = norm_data[self.input_len:]
        
        return torch.from_numpy(input_seq), torch.from_numpy(target_seq)

# dataset.py 更新
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from PIL import Image
import os

class HKODataset(Dataset):
    """
    新增：针对香港天文台 (HKO-7) 数据的加载器
    数据通常是 .png 图片序列或 .npy 文件
    """
    def __init__(self, data_dir, input_len=10, pred_len=10):
        # HKO 标准是输入10帧，预测10帧
        super().__init__()
        self.data_dir = data_dir
        self.input_len = input_len
        self.pred_len = pred_len
        # 假设你已经处理成了 npy 列表
        self.samples = self._make_dataset(data_dir)

    def _make_dataset(self, dir):
        # 这里写简单的读取逻辑，返回文件路径列表
        # 实际 HKO 数据通常需要简单的预处理脚本转成 npy
        pass 

    def __getitem__(self, idx):
        # 读取逻辑...
        # 重点：HKO 是 dBZ (0-70)，需要归一化到 0-1
        # output shape: [Time, 1, H, W]
        pass