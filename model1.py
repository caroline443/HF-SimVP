import torch
import torch.nn as nn

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SpatialAttention(nn.Module):
    """
    【创新点 1】空间注意力机制
    用于让模型聚焦于高 VIL 值的强对流核心区域
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv1(x_cat)
        return x * self.sigmoid(scale)

class InceptionAdapter(nn.Module):
    """
    SimVP 的核心 Translator 层
    结合了 Inception (多尺度感受野) 和 Spatial Attention
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 2
        # 分支 1: 小卷积核
        self.branch1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels)
        # 分支 2: 大卷积核
        self.branch2 = nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2, groups=mid_channels)
        
        self.conv1x1 = nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1)
        self.attn = SpatialAttention() # 嵌入注意力

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat([x1, x2], dim=1)
        out = self.conv1x1(out)
        out = self.attn(out) # 注意力加权
        return x + out  # 残差连接

class SimVP_Enhanced(nn.Module):
    def __init__(self, in_shape, hid_S=64, hid_T=256, N_S=4, N_T=4):
        """
        SimVP 增强版主架构
        in_shape: (Time, Channel, H, W)
        """
        super().__init__()
        T, C, H, W = in_shape
        self.enc_in_channels = T * C
        self.dec_out_channels = 12 * C # 预测未来12帧
        
        # 1. Encoder (提取空间特征)
        self.enc = nn.Sequential(
            BasicConv2d(self.enc_in_channels, hid_S, 3, 1, 1),
            *[BasicConv2d(hid_S, hid_S, 3, 1, 1) for _ in range(N_S)]
        )
        
        # 2. Translator (演变推断)
        layers = []
        for _ in range(N_T):
            layers.append(InceptionAdapter(hid_S, hid_S))
            layers.append(nn.BatchNorm2d(hid_S))
            layers.append(nn.ReLU(inplace=True))
        self.translator = nn.Sequential(*layers)
        
        # 3. Decoder (还原图像)
        self.dec = nn.Sequential(
            *[BasicConv2d(hid_S, hid_S, 3, 1, 1) for _ in range(N_S)],
            nn.Conv2d(hid_S, self.dec_out_channels, 3, 1, 1)
        )

    def forward(self, x):
        # x: [Batch, Time, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B, T*C, H, W) # 压缩时间维
        
        feat = self.enc(x)
        feat = self.translator(feat)
        out = self.dec(feat)
        
        # Output: [Batch, 12, C, H, W]
        out = out.view(B, 12, C, H, W)
        return torch.sigmoid(out)
# --- 追加在 model.py 文件末尾 ---

class InceptionOriginal(nn.Module):
    """原始 SimVP 的 Inception 模块 (无注意力机制)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 2
        self.branch1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels)
        self.branch2 = nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2, groups=mid_channels)
        self.conv1x1 = nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat([x1, x2], dim=1)
        out = self.conv1x1(out)
        return x + out

class SimVP_Baseline(nn.Module):
    """
    【基线模型】原始 SimVP (Vanilla Version)
    用于对比，证明 Enhanced 版本的有效性
    """
    def __init__(self, in_shape, hid_S=64, hid_T=256, N_S=4, N_T=4):
        super().__init__()
        T, C, H, W = in_shape
        self.enc_in_channels = T * C
        self.dec_out_channels = 12 * C
        
        # 1. Encoder
        self.enc = nn.Sequential(
            BasicConv2d(self.enc_in_channels, hid_S, 3, 1, 1),
            *[BasicConv2d(hid_S, hid_S, 3, 1, 1) for _ in range(N_S)]
        )
        
        # 2. Translator (原始版本，无 Attention)
        layers = []
        for _ in range(N_T):
            layers.append(InceptionOriginal(hid_S, hid_S))
            layers.append(nn.BatchNorm2d(hid_S))
            layers.append(nn.ReLU(inplace=True))
        self.translator = nn.Sequential(*layers)
        
        # 3. Decoder
        self.dec = nn.Sequential(
            *[BasicConv2d(hid_S, hid_S, 3, 1, 1) for _ in range(N_S)],
            nn.Conv2d(hid_S, self.dec_out_channels, 3, 1, 1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B, T*C, H, W)
        feat = self.enc(x)
        feat = self.translator(feat)
        out = self.dec(feat)
        out = out.view(B, 12, C, H, W)
        return torch.sigmoid(out)