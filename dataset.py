import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset

class NSynthDataset(Dataset):
    def __init__(self, json_path, audio_dir, transform=None):
        # 读取 JSON 文件中包含的所有样本元数据
        with open(json_path, 'r') as f:
            self.metadata = json.load(f)

        # 音频文件所在的文件夹路径
        self.audio_dir = audio_dir

        # 获取所有样本的唯一 ID 列表
        self.ids = list(self.metadata.keys())

        # 使用 torchaudio 提供的 MelSpectrogram 转换器
        self.transform = transform or torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,  # NSynth 采样率为 16kHz
            n_fft=1024,         # FFT窗口大小
            hop_length=256,     # 每帧步长
            n_mels=64           # 提取64个Mel滤波器频带
        )

    def __len__(self):
        # 返回数据集总样本数
        return len(self.ids)

    def __getitem__(self, idx):
        # 通过索引获取一个样本的元数据
        uid = self.ids[idx]
        info = self.metadata[uid]

        # 获取类别标签（0-10）代表11类乐器家族
        label = info['instrument_family']

        # 拼接成完整路径并加载音频
        audio_path = os.path.join(self.audio_dir, f"{uid}.wav")
        waveform, _ = torchaudio.load(audio_path)  # shape: [1, T]

        # 转换为梅尔频谱图（形状为 [1, 64, time]）
        mel_spec = self.transform(waveform)
        mel_spec = mel_spec.squeeze(0)             # 去掉 batch 中的 channel 维度 [64, time]
        return mel_spec.unsqueeze(0), torch.tensor(label, dtype=torch.long)
        # 最终返回维度 [1, 64, time] 和 int64 标签
