import torch.nn as nn
import torch.nn.functional as F

class AudioCNN(nn.Module):
    def __init__(self, num_classes=11):  # 11类乐器
        super(AudioCNN, self).__init__()
        # 第一层卷积：输入通道1（梅尔图像），输出通道16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # 最大池化层：降低时频分辨率
        self.pool = nn.MaxPool2d(2, 2)

        # 第二、三层卷积
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # 全连接层输入维度根据池化后大小而来（这里假设输入 [1, 64, 64]）
        self.fc1 = nn.Linear(15872, 128)  # 中间特征层
        self.fc2 = nn.Linear(128, num_classes)  # 输出为11类

    def forward(self, x):
        # CNN + ReLU + Pooling 三层堆叠
        x = self.pool(F.relu(self.conv1(x)))  # [B, 1, 64, T] -> [B, 16, 32, T/2]
        x = self.pool(F.relu(self.conv2(x)))  # -> [B, 32, 16, T/4]
        x = self.pool(F.relu(self.conv3(x)))  # -> [B, 64, 8, T/8]

        # Flatten 成一维向量
        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # 输出未经过softmax的 logits（交叉熵损失自带 softmax）
