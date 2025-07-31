
# 🎵 NSynth Instrument Classification with MFCC + CNN

A PyTorch-based deep learning project that classifies musical instruments from raw audio using MFCC features and Convolutional Neural Networks (CNNs). Built and tested in Debian 12 using Anaconda virtual environments.

---

## 📁 Dataset: [NSynth](https://magenta.tensorflow.org/datasets/nsynth)

> Neural Synthesizer Dataset by Google Magenta  
> 300k+ musical notes from over 1,000 instruments  
> Each note is a 4-second, 16kHz WAV file

### ✅ Folder Structure After Extraction:

```
~/datasets/nsynth/
├── nsynth-train/
│   ├── audio/            # WAV files (~60k)
│   └── examples.json     # metadata
├── nsynth-valid/
│   ├── audio/
│   └── examples.json
└── nsynth-test/
    ├── audio/
    └── examples.json
```

---

## 💻 Environment Setup (Debian 12 + Anaconda)

```bash
# 1. Create and activate environment
conda create -n nsynth python=3.10 -y
conda activate nsynth

# 2. Install PyTorch and torchaudio (CPU or CUDA)
conda install pytorch torchaudio -c pytorch

# 3. Optional: Other utilities
pip install matplotlib tqdm
```

---

## 🧠 Model Overview

We use **MFCCs** (Mel-Frequency Cepstral Coefficients) as input features and feed them into a **3-layer CNN**, followed by fully connected layers for final classification into 11 instrument families.

```
WAV (16kHz) → MFCC (40xTime) → CNN → FC → Softmax (11 classes)
```

---

## 🗂 Project Structure

```
nsynth_project_mfcc/
├── dataset.py         # Load WAVs, compute MFCCs
├── model.py           # CNN model definition
├── train.py           # Training loop
├── eval.py            # Final test set evaluation
├── mfcc_audio_model.pth   # Saved model (after training)
└── README.md
```

---

## 🧾 dataset.py

Loads NSynth audio and computes MFCCs (using `torchaudio.transforms.MFCC`).  
Returns `(1, 40, T)` MFCC tensors + integer labels (`0~10`).

---

## 🧱 model.py

A lightweight CNN model for MFCC inputs:

- Conv2d(1 → 16 → 32 → 64)
- ReLU + MaxPooling
- Flatten → FC → Output (11)

```python
x → [B, 1, 40, Time] → CNN → [B, 64, H, W] → FC → [B, 11]
```

---

## 🚀 train.py

Trains the model with CrossEntropyLoss + Adam optimizer for N epochs.

```bash
python train.py
```

It will print loss and validation accuracy for each epoch and save the model to:

```bash
mfcc_audio_model.pth
```

---

## 🧪 eval.py

Loads the saved model and evaluates performance on the **test set**:

```bash
python eval.py
```

Example output:

```
Test Accuracy: 72.89%
```

---

## 📊 Results

| Input Features | Accuracy (valid) | Accuracy (test) | Epochs |
|----------------|------------------|------------------|--------|
| Raw MFCC (40xTime) | ~68–72%        | ~68–73%          | 10     |

---

## 📌 Future Improvements

- Use log-mel spectrogram or delta-MFCC
- Try deeper CNN / ResNet / Transformer
- Fine-tune on custom instruments
- Add training plots & confusion matrix

---

## 📚 Reference

- [NSynth Dataset](https://magenta.tensorflow.org/datasets/nsynth)
- [Torchaudio MFCC Docs](https://pytorch.org/audio/stable/transforms.html#mfcc)
- [PyTorch Official Docs](https://pytorch.org)

---

## 🔒 License

MIT License
