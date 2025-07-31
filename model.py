import torch.nn as nn
import torch.nn.functional as F

class MFCC_CNN(nn.Module):
    def __init__(self, num_classes=11):
        super(MFCC_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # 输入 1 个 channel（MFCC）
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # 展平后的大小根据输入时间轴变化，这里做个近似假设
        self.fc1 = nn.Linear(9920, 128)  # 修改为合适的值（按输入尺寸调）
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [16, ?, ?]
        x = self.pool(F.relu(self.conv2(x)))  # -> [32, ?, ?]
        x = self.pool(F.relu(self.conv3(x)))  # -> [64, ?, ?]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
