import os
import torch
from torch.utils.data import DataLoader
from dataset import NSynthMFCCDataset
from model import MFCC_CNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 测试集路径
root = os.path.expanduser('~/datasets/nsynth')
test_set = NSynthMFCCDataset(
    json_path=os.path.join(root, 'nsynth-test/examples.json'),
    audio_dir=os.path.join(root, 'nsynth-test/audio')
)
test_loader = DataLoader(test_set, batch_size=32)

# 加载训练好的模型
model = MFCC_CNN().to(device)
model.load_state_dict(torch.load('mfcc_audio_model.pth'))
model.eval()

# 准确率计算
correct = total = 0
with torch.no_grad():
    for mfcc, label in test_loader:
        mfcc, label = mfcc.to(device), label.to(device)
        out = model(mfcc)
        pred = torch.argmax(out, dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)

print(f"Test Accuracy: {correct / total * 100:.2f}%")