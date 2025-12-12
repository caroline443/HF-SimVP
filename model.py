# model.py
import torch
import torch.nn as nn

# --- 基础组件 ---
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# --- 创新点: 空间注意力机制 (Spatial Attention) ---
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿着通道维度求平均和最大值 -> [Batch, 1, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接 -> [Batch, 2, H, W]
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # 卷积 + Sigmoid 生成权重图
        scale = self.conv1(x_cat)
        #以此增强原始特征
        return x * self.sigmoid(scale)

# --- 模块 A: 原始 Inception (用于 Baseline) ---
class InceptionOriginal(nn.Module):
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
        return x + self.conv1x1(out)

# --- 模块 B: 改进版 Inception (用于 Enhanced) ---
class InceptionEnhanced(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 2
        self.branch1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels)
        self.branch2 = nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2, groups=mid_channels)
        self.conv1x1 = nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1)
        
        # 🔥 插入空间注意力
        self.attn = SpatialAttention()

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat([x1, x2], dim=1)
        out = self.conv1x1(out)
        
        # 🔥 先融合，再注意力加权
        out = self.attn(out) 
        return x + out

# --- 模型主架构 (SimVP) ---
class SimVP_Base(nn.Module):
    def __init__(self, in_shape, hid_S=64, hid_T=256, N_S=4, N_T=4, model_type='baseline'):
        super().__init__()
        T, C, H, W = in_shape
        self.enc_in_channels = T * C
        self.dec_out_channels = 12 * 1 # 预测未来12帧，1个通道
        
        # 1. Encoder
        self.enc = nn.Sequential(
            BasicConv2d(self.enc_in_channels, hid_S, 3, 1, 1),
            *[BasicConv2d(hid_S, hid_S, 3, 1, 1) for _ in range(N_S)]
        )
        
        # 2. Translator
        layers = []
        for _ in range(N_T):
            if model_type == 'baseline':
                layers.append(InceptionOriginal(hid_S, hid_S))
            else:
                layers.append(InceptionEnhanced(hid_S, hid_S)) # 使用带注意力的模块
            
            layers.append(nn.BatchNorm2d(hid_S))
            layers.append(nn.ReLU(inplace=True))
        self.translator = nn.Sequential(*layers)
        
        # 3. Decoder
        self.dec = nn.Sequential(
            *[BasicConv2d(hid_S, hid_S, 3, 1, 1) for _ in range(N_S)],
            nn.Conv2d(hid_S, self.dec_out_channels, 3, 1, 1)
        )

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B, T*C, H, W)
        
        feat = self.enc(x)
        feat = self.translator(feat)
        out = self.dec(feat)
        
        out = out.view(B, 12, 1, H, W)
        return torch.sigmoid(out) # 归一化到 0-1

# --- 供外部调用的两个类 ---
def SimVP_Baseline(in_shape, **kwargs):
    return SimVP_Base(in_shape, model_type='baseline', **kwargs)

def SimVP_Enhanced(in_shape, **kwargs):
    return SimVP_Base(in_shape, model_type='enhanced', **kwargs)