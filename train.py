import os
import torch
from torch.utils.data import DataLoader
from dataset import NSynthDataset
from model import AudioCNN

# 使用 GPU 或 CPU
device = torch.device('cuda' if torch.cuda.is_available() else """ 'cpu' """)

# 设置路径（自动展开 ~）
root = os.path.expanduser('~/datasets/nsynth')
train_json = os.path.join(root, 'nsynth-train/examples.json')
train_audio = os.path.join(root, 'nsynth-train/audio')
valid_json = os.path.join(root, 'nsynth-valid/examples.json')
valid_audio = os.path.join(root, 'nsynth-valid/audio')

# 加载训练/验证数据集
train_set = NSynthDataset(train_json, train_audio)
valid_set = NSynthDataset(valid_json, valid_audio)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False)

# 初始化模型和优化器
model = AudioCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# 训练过程
for epoch in range(10):
    model.train()
    total_loss = 0
    for mel, label in train_loader:
        mel, label = mel.to(device), label.to(device)
        output = model(mel)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Train Loss: {total_loss/len(train_loader):.4f}")

    # 验证部分
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for mel, label in valid_loader:
            mel, label = mel.to(device), label.to(device)
            out = model(mel)
            pred = torch.argmax(out, dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    acc = correct / total * 100
    print(f"Validation Accuracy: {acc:.2f}%\n")

    torch.save(model.state_dict(), 'audio_model.pth')
