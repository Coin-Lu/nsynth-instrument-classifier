import os
import torch
from torch.utils.data import DataLoader
from dataset import NSynthMFCCDataset
from model import MFCC_CNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置路径
root = os.path.expanduser('~/datasets/nsynth')
train_json = os.path.join(root, 'nsynth-train/examples.json')
train_audio = os.path.join(root, 'nsynth-train/audio')
valid_json = os.path.join(root, 'nsynth-valid/examples.json')
valid_audio = os.path.join(root, 'nsynth-valid/audio')

# 数据加载（使用 MFCC 数据集类）
train_set = NSynthMFCCDataset(train_json, train_audio)
valid_set = NSynthMFCCDataset(valid_json, valid_audio)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=32)

# 初始化模型
model = MFCC_CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# 训练循环
for epoch in range(10):
    model.train()
    total_loss = 0
    for mfcc, label in train_loader:
        mfcc, label = mfcc.to(device), label.to(device)
        out = model(mfcc)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Train Loss: {total_loss/len(train_loader):.4f}")

    # 验证部分
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for mfcc, label in valid_loader:
            mfcc, label = mfcc.to(device), label.to(device)
            out = model(mfcc)
            pred = torch.argmax(out, dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    print(f"Validation Accuracy: {correct / total * 100:.2f}%\n")

# 保存模型
torch.save(model.state_dict(), 'mfcc_audio_model.pth')
