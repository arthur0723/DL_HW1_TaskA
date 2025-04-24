import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    通道注意力機制，根據輸入通道的數量和組合動態調整特徵重要性
    """
    def __init__(self, in_channels, reduction_ratio=4):
        super(ChannelAttention, self).__init__()
        # 確保輸入通道數至少為 1
        in_channels = max(in_channels, 1)
        
        # 定義注意力機制
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 定義共享的多層感知機
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, max(in_channels // reduction_ratio, 1), 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(max(in_channels // reduction_ratio, 1), in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 應用平均池化和最大池化
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        
        # 結合並應用 sigmoid 函數
        out = self.sigmoid(avg_out + max_out)
        return out

class ChannelAdaptiveLayer(nn.Module):
    """
    通道自適應層：將任意數量的輸入通道映射到固定數量的輸出通道
    """
    def __init__(self, out_channels=64):
        super(ChannelAdaptiveLayer, self).__init__()
        self.out_channels = out_channels
        
        # 為每個可能的單通道輸入創建一個 1x1 卷積
        self.r_conv = nn.Conv2d(1, out_channels, kernel_size=1)
        self.g_conv = nn.Conv2d(1, out_channels, kernel_size=1)
        self.b_conv = nn.Conv2d(1, out_channels, kernel_size=1)
        
        # 為雙通道組合創建 1x1 卷積
        self.rg_conv = nn.Conv2d(2, out_channels, kernel_size=1)
        self.rb_conv = nn.Conv2d(2, out_channels, kernel_size=1)
        self.gb_conv = nn.Conv2d(2, out_channels, kernel_size=1)
        
        # 為標準 RGB 通道創建 1x1 卷積
        self.rgb_conv = nn.Conv2d(3, out_channels, kernel_size=1)
        
        # 用於初始化權重的策略
        self._init_weights()
    
    def _init_weights(self):
        """初始化權重以維持輸出方差"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        根據輸入通道數選擇適當的卷積
        
        Args:
            x (Tensor): 形狀為 [B, C, H, W] 的輸入張量，C 可以是 1, 2 或 3
            
        Returns:
            Tensor: 形狀為 [B, out_channels, H, W] 的輸出張量
        """
        # 獲取輸入通道數
        in_channels = x.size(1)
        
        if in_channels == 1:
            # 假設是單通道 (R or G or B)
            # 我們可以檢查批次中的第一個圖像來確定它是哪個通道
            sample = x[0].mean().item()
            
            # 一個簡單的啟發式方法來猜測通道 (可根據實際情況調整)
            if sample > 0.6:  # 紅色通道通常值較高
                return F.relu(self.r_conv(x))
            elif sample < 0.4:  # 藍色通道通常值較低
                return F.relu(self.b_conv(x))
            else:  # 綠色通道通常在中間
                return F.relu(self.g_conv(x))
                
        elif in_channels == 2:
            # 假設是雙通道 (RG or RB or GB)
            # 同樣，我們可以檢查批次中的第一個圖像來確定它是哪種組合
            ch0_mean = x[:, 0].mean().item()
            ch1_mean = x[:, 1].mean().item()
            
            # 另一個啟發式方法來猜測通道組合
            if ch0_mean > 0.5 and ch1_mean > 0.4:  # 可能是 RG
                return F.relu(self.rg_conv(x))
            elif ch0_mean > 0.5 and ch1_mean < 0.4:  # 可能是 RB
                return F.relu(self.rb_conv(x))
            else:  # 可能是 GB
                return F.relu(self.gb_conv(x))
                
        elif in_channels == 3:
            # 標準 RGB 輸入
            return F.relu(self.rgb_conv(x))
            
        else:
            # 不支持的通道數
            raise ValueError(f"不支持的輸入通道數: {in_channels}")

class DynamicConvolution(nn.Module):
    """
    動態卷積模組：能夠處理可變數量的輸入通道，並應用動態選擇的卷積核
    """
    def __init__(self, out_channels=64, mid_channels=64, kernel_size=3, stride=1, padding=1):
        super(DynamicConvolution, self).__init__()
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 通道自適應層
        self.channel_adaptive = ChannelAdaptiveLayer(mid_channels)
        
        # 通道注意力機制
        self.channel_attention = ChannelAttention(mid_channels)
        
        # 標準卷積層
        self.conv = nn.Conv2d(mid_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        # 應用通道自適應層
        x = self.channel_adaptive(x)
        
        # 應用通道注意力
        attn = self.channel_attention(x)
        x = x * attn
        
        # 應用標準卷積
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        
        return x