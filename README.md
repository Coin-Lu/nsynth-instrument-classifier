# 🎵 使用 MFCC + CNN 进行 NSynth 乐器分类

一个基于 PyTorch 的深度学习项目，使用 MFCC 特征和卷积神经网络（CNN）从原始音频中分类乐器。在 Debian 12 系统上，使用 Anaconda 虚拟环境进行构建与测试。

---

## 📁 数据集：[NSynth](https://magenta.tensorflow.org/datasets/nsynth)

> Google Magenta 提供的神经合成器数据集  
> 超过 1000 种乐器的 30 万多个音符  
> 每个音符是一个 4 秒的 16kHz WAV 文件

### ✅ 解压后的文件夹结构：

```
~/datasets/nsynth/
├── nsynth-train/
│   ├── audio/            # WAV 文件（约 6 万个）
│   └── examples.json     # 元数据
├── nsynth-valid/
│   ├── audio/
│   └── examples.json
└── nsynth-test/
    ├── audio/
    └── examples.json
```

---

## 💻 环境配置（Debian 12 + Anaconda）

```bash
# 1. 创建并激活虚拟环境
conda create -n nsynth python=3.10 -y
conda activate nsynth

# 2. 安装 PyTorch 和 torchaudio（CPU 或 CUDA）
conda install pytorch torchaudio -c pytorch

# 3. 可选：安装其它工具
pip install matplotlib tqdm
```

---

## 🧠 模型概览

我们使用 **MFCC**（梅尔频率倒谱系数）作为输入特征，并将其输入到一个 **三层卷积神经网络（CNN）**，通过全连接层最终分类为 11 个乐器家族之一。

```
WAV (16kHz) → MFCC (40x时间步) → CNN → 全连接层 → Softmax (11 类)
```

---

## 🗂 项目结构

```
nsynth_project_mfcc/
├── dataset.py         # 加载 WAV，计算 MFCC
├── model.py           # 定义 CNN 模型结构
├── train.py           # 训练循环
├── eval.py            # 在测试集上评估
├── utils.py           # 工具函数脚本（可选）
├── mfcc_audio_model.pth   # 训练完成后保存的模型
└── README.md
```

---

## 🧾 dataset.py

使用 `torchaudio.transforms.MFCC` 加载 NSynth 音频并计算 MFCC，返回形如 `(1, 40, T)` 的 MFCC 张量，以及整数标签（范围为 0~10）。

---

## 🧱 model.py

一个为 MFCC 输入设计的轻量 CNN 模型：

- Conv2d(1 → 16 → 32 → 64)
- ReLU 激活 + 最大池化
- 展平 → 全连接 → 输出 11 类

```python
x → [B, 1, 40, Time] → CNN → [B, 64, H, W] → FC → [B, 11]
```

---

## 🚀 train.py

使用交叉熵损失函数（CrossEntropyLoss）和 Adam 优化器进行训练。

```bash
python train.py
```

每轮训练会输出 loss 和验证准确率，最终模型保存为：

```bash
mfcc_audio_model.pth
```

---

## 🧪 eval.py

加载已保存的模型并在 **测试集** 上进行评估：

```bash
python eval.py
```

示例输出：

```
Test Accuracy: 72.56%
```

---

## 📊 结果

| 输入特征         | 验证集准确率 | 测试集准确率 | 轮数 |
|------------------|---------------|----------------|------|
| 原始 MFCC (40xT) | ~69–74%       | ~70–73%        | 10   |

---

## 📌 未来改进方向

- 使用对数梅尔频谱图（Log-Mel）或增量 MFCC（delta-MFCC）
- 尝试更深的 CNN / ResNet / Transformer 架构
- 在自定义乐器数据上进行微调
- 添加训练曲线和混淆矩阵等可视化

---

## 📚 参考资料

- [NSynth 数据集](https://magenta.tensorflow.org/datasets/nsynth)
- [Torchaudio MFCC 文档](https://pytorch.org/audio/stable/transforms.html#mfcc)
- [PyTorch 官方文档](https://pytorch.org)
