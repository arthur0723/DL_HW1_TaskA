import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardConvNet(nn.Module):
    """
    標準卷積神經網絡，用作基礎比較模型
    簡單的三層卷積網絡，使用固定的RGB輸入
    """
    def __init__(self, num_classes=100):
        super(StandardConvNet, self).__init__()
        
        # 第一個卷積塊 - 固定為3通道(RGB)輸入
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二個卷積塊
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三個卷積塊
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全連接層
        # 假設輸入圖像大小為224x224，經過3次下採樣後為28x28
        self.fc = nn.Linear(256 * 28 * 28, num_classes)
        
    def forward(self, x):
        # 第一個卷積塊
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # 第二個卷積塊
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # 第三個卷積塊
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # 扁平化並通過全連接層
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

def create_standard_net(num_classes=100):
    """
    創建一個標準卷積網絡模型實例
    
    Args:
        num_classes (int, optional): 分類類別數量，默認為100
        
    Returns:
        nn.Module: 標準卷積網絡模型
    """
    model = StandardConvNet(num_classes)
    return model