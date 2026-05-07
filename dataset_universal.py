import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os
import glob

class SEVIRDataset(Dataset):
    """
    [V3.1] 全量 SEVIR 加载器 (含滑窗扩充策略)

    测试集文件选取规则（与 EarthFormer/SimCast/PreDiff 对齐）：
      - 仅使用 RANDOMEVENTS 文件，排除 STORMEVENTS
      - STORMEVENTS 是专门挑选的强对流事件，极端降水像素比例远高于 RANDOM，
        混入测试集会导致高阈值 CSI（133/160/181/219）系统性虚高，
        无法与文献数字直接对比
      - 训练集保留全部 RANDOM + STORM 文件（有助于学习极端降水特征）
    """
    def __init__(self, data_root, mode='train', input_len=13, pred_len=12):
        super().__init__()
        self.data_root = data_root
        self.mode = mode
        self.input_len = input_len
        self.pred_len = pred_len
        self.total_len = input_len + pred_len
        
        # --- 滑窗步长 (Stride) ---
        # 训练时：stride=12（每 10 分钟滑动一次），扩充样本量，提升泛化
        # 测试时：stride=49（等于事件总帧数，完全不重叠）
        #
        # [文献对齐说明]
        # EarthFormer / SimCast / PreDiff 等论文的测试集均采用不重叠切分：
        # 每个 49 帧事件只取一个 [0:25] 窗口（input_len=13, pred_len=12 共 25 帧）。
        # stride=49 保证每个事件最多产生 1 个测试样本（49-25+1=25 < 49，
        # 实际只有 t_start=0 满足条件），与文献口径一致。
        # 若改为 stride<25 则会产生重叠测试样本，导致指标虚高，不可与文献直接对比。
        if mode == 'train':
            self.stride = 12
        else:
            self.stride = 49  # 不重叠，与文献评测口径一致
        
        # 1. 扫描所有 .h5 文件
        all_files = sorted(glob.glob(os.path.join(data_root, '*.h5')))
        self.files = []
        
        print(f"[{mode.upper()}] 正在扫描数据文件: {data_root}")
        
        # 2. 根据年份划分 Train/Test (SimCast/EarthFormer 标准)
        for f_path in all_files:
            filename = os.path.basename(f_path)
            try:
                parts = filename.split('_')
                year_idx = -1
                for i, part in enumerate(parts):
                    if part in ['2017', '2018', '2019']:
                        year_idx = i
                        break
                
                if year_idx == -1: continue 

                year = int(parts[year_idx])
                start_date = parts[year_idx+1] 
                month = int(start_date[:2]) 
                
                is_train_file = True
                if year > 2019:
                    is_train_file = False
                elif year == 2019:
                    if month >= 5: # 5月及以后是测试集（与 EarthFormer/SimCast 标准一致）
                        is_train_file = False
                
                if mode == 'train' and is_train_file:
                    self.files.append(f_path)
                elif mode == 'test' and not is_train_file:
                    # 测试集只保留 RANDOMEVENTS，排除 STORMEVENTS
                    # 与 EarthFormer/SimCast/PreDiff 文献评测口径一致
                    if 'STORMEVENTS' in filename:
                        print(f"  [TEST] 跳过 STORM 文件（与文献口径对齐）: {filename}")
                        continue
                    self.files.append(f_path)
                    
            except Exception as e:
                print(f"跳过文件 {filename}: {e}")
                continue

        if len(self.files) == 0:
            raise ValueError(f" 在 {data_root} 下未找到符合 {mode} 模式的文件！")

        # 3. 构建全局滑窗索引
        self.sample_indices = [] 
        print(f"[{mode.upper()}] 正在构建滑窗索引 (Stride={self.stride})...")
        
        for f_path in self.files:
            try:
                with h5py.File(f_path, 'r') as hf:
                    keys = list(hf.keys())
                    data_source = 'vil' if 'vil' in keys else keys[0]
                    
                    # 获取 shape, 假设是 (N, T, H, W) 或 (N, H, W, T)
                    # SEVIR VIL 通常是 (N, 49, 384, 384)
                    shape = hf[data_source].shape
                    event_count = shape[0]
                    # T_max 从数据实际 shape 读取，兼容非标准文件
                    # SEVIR VIL 标准为 49 帧；若文件维度为 (N, H, W, T) 则取 shape[-1]
                    if shape[-1] == 49:
                        T_max = shape[-1]   # (N, H, W, T) 格式
                    else:
                        T_max = shape[1]    # (N, T, H, W) 格式，标准 SEVIR

                    # 滑窗切分
                    # 测试时 stride=49，每个事件只产生 1 个样本（t_start=0），
                    # 与文献不重叠评测口径一致
                    for i in range(event_count):
                        for t_start in range(0, T_max - self.total_len + 1, self.stride):
                            self.sample_indices.append({
                                'path': f_path,
                                'key': data_source,
                                'event_idx': i,
                                'time_offset': t_start
                            })
                            
            except Exception as e:
                print(f" 读取 {f_path} 失败: {e}")

        print(f" [{mode.upper()}] 索引构建完成: 共 {len(self.sample_indices)} 个序列样本")

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        info = self.sample_indices[idx]
        
        with h5py.File(info['path'], 'r') as hf:
            raw_event = hf[info['key']][info['event_idx']] 
        
        # 维度统一化: (T, H, W)
        if raw_event.shape[-1] == 49: 
            raw_event = np.transpose(raw_event, (2, 0, 1))
        
        # 根据滑窗 offset 切片
        t_start = info['time_offset']
        t_end = t_start + self.total_len
        seq_data = raw_event[t_start:t_end] 
        
        # 归一化 (0-255 -> 0-1)
        norm_data = seq_data.astype(np.float32) / 255.0
        
        # 增加 Channel: (T, 1, H, W)
        norm_data = np.expand_dims(norm_data, axis=1)
        
        return torch.from_numpy(norm_data[:self.input_len]), torch.from_numpy(norm_data[self.input_len:])