import torch
from dataset import NSynthDataset
from model import AudioCNN
from torch.utils.data import DataLoader
import os

# 判断是否有 CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 测试集加载
root = os.path.expanduser('~/datasets/nsynth')
test_set = NSynthDataset(
    json_path=os.path.join(root, 'nsynth-test/examples.json'),
    audio_dir=os.path.join(root, 'nsynth-test/audio')
)
test_loader = DataLoader(test_set, batch_size=32)

# 加载模型权重
model = AudioCNN().to(device)
model.load_state_dict(torch.load('audio_model.pth'))  # 训练保存的模型文件
model.eval()

# 评估准确率
correct = total = 0
with torch.no_grad():
    for mel, label in test_loader:
        mel, label = mel.to(device), label.to(device)
        out = model(mel)
        pred = torch.argmax(out, dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)

print(f"Test Accuracy: {correct / total * 100:.2f}%")
