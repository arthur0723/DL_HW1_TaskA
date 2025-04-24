import torch
import torch.nn as nn
import torch.nn.functional as F
from .dynamic_conv import DynamicConvolution

class DynamicConvNet(nn.Module):
    """
    動態卷積神經網絡：使用動態卷積模組處理可變通道輸入
    """
    def __init__(self, num_classes=100):
        super(DynamicConvNet, self).__init__()
        
        # 第一個卷積塊 - 使用動態卷積模組處理可變通道輸入
        self.dynamic_conv = DynamicConvolution(out_channels=64, mid_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二個卷積塊 - 標準卷積
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三個卷積塊 - 標準卷積
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全連接層
        # 假設輸入圖像大小為224x224，經過3次下採樣後為28x28
        self.fc = nn.Linear(256 * 28 * 28, num_classes)
        
    def forward(self, x):
        # 第一個動態卷積塊
        x = self.dynamic_conv(x)
        x = self.pool1(x)
        
        # 第二個卷積塊
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # 第三個卷積塊
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # 扁平化並通過全連接層
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

def create_dynamic_net(num_classes=100):
    """
    創建一個動態卷積網絡模型實例
    
    Args:
        num_classes (int, optional): 分類類別數量，默認為100
        
    Returns:
        nn.Module: 動態卷積網絡模型
    """
    model = DynamicConvNet(num_classes)
    return model