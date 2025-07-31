import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset

class NSynthMFCCDataset(Dataset):
    def __init__(self, json_path, audio_dir, transform=None):
        with open(json_path, 'r') as f:
            self.metadata = json.load(f)
        self.audio_dir = audio_dir
        self.ids = list(self.metadata.keys())

        # 默认使用 torchaudio 的 MFCC 提取器
        self.transform = transform or torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=40,                   # 提取40维 MFCC
            melkwargs={
                'n_fft': 1024,
                'hop_length': 256,
                'n_mels': 64
            }
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        uid = self.ids[idx]
        info = self.metadata[uid]
        label = info['instrument_family']  # 0~10

        audio_path = os.path.join(self.audio_dir, f"{uid}.wav")
        waveform, _ = torchaudio.load(audio_path)

        mfcc = self.transform(waveform)  # [1, n_mfcc, time]
        mfcc = mfcc.squeeze(0)           # [n_mfcc, time]
        return mfcc.unsqueeze(0), torch.tensor(label, dtype=torch.long)  # [1, n_mfcc, time], label
