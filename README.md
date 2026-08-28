# 🎙️ Modern Voice Cloner (Flow Matching DiT & Voice-to-Voice)

A modern, high-fidelity voice cloning and speech conversion system based on **Continuous Normalizing Flows (Optimal Transport Flow Matching - OT-CFM)**, **Diffusion Transformers (DiT)** with AdaLN modulation, and a **24kHz Neural Vocoder (Vocos)**. Accelerated via hardware on **NVIDIA GeForce RTX CUDA GPUs**.

Compared to traditional tools like **RVC** and **so-vits-svc**, this system delivers major advancements:
- **True Zero-Shot Voice Cloning**: Clone any voice using just 3 to 10 seconds of reference audio—no prior training required.
- **Dual Operating Modes**: Full support for both **Text-to-Speech (TTS)** and **Audio-to-Audio (Voice-to-Voice Conversion)** with GPU-accelerated Whisper ASR.
- **Artifact-Free Speech**: Eliminates the robotic, metallic timbre characteristic of older neural vocoders.
- **Native Brazilian Portuguese (PT-BR) & Multilingual Support**: Specialized foundation weights for natural pronunciation of accents and nasal vowels (`ã`, `õ`, `é`, `ó`, `ç`), plus English/multilingual base models.
- **High Definition Audio**: 24,000 Hz sample rate powered by the Vocos neural vocoder.
- **Fast Fine-Tuning**: Support for training custom voice models with local audio datasets, Exponential Moving Average (EMA), and Automatic Mixed Precision (AMP).

---

## 📁 Project Structure

```
voice_cloner/
├── configs/
│   └── default_config.yaml         # Model, audio, and training hyperparameters
├── models/
│   ├── dit.py                      # Diffusion Transformer with AdaLN-Zero & Multi-Head Attention
│   ├── flow_matching.py            # OT-CFM (Optimal Transport Flow Matching) & ODE samplers
│   ├── text_encoder.py             # UTF-8 tokenizer & ConvNeXt text encoder
│   └── vocoder.py                  # 24kHz Vocos neural vocoder with Griffin-Lim fallback
├── dataset/
│   ├── preprocessor.py             # 24kHz resampling, Mel-spectrogram extraction & normalization
│   └── audio_dataset.py            # PyTorch Dataset & Collate function with dynamic padding
├── training/
│   └── train.py                    # Complete training/fine-tuning pipeline with EMA & AMP
├── inference/
│   ├── cloner.py                   # Main inference engine (TTS, Voice Conversion & Whisper ASR)
│   └── infer.py                    # Command-Line Interface (CLI)
├── gui_pyqt.py                     # Native desktop GUI built with PyQt6 (Dark Theme)
├── start_gui.bat                   # 1-Click launcher script for Windows
├── DOCUMENTACAO_E_HISTORICO.md     # Detailed architecture documentation & issue resolution log
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 🚀 Installation & Virtual Environment (`venv`)

### 1. Create and Activate Virtual Environment

On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install PyTorch with CUDA & Dependencies

For NVIDIA GPUs (CUDA 12.x):
```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

---

## 🔊 How to Use

### Option A: Native Desktop GUI (PyQt6) - Recommended

Simply **double-click** the [`start_gui.bat`](file:///f:/workspace/voice_cloner/start_gui.bat) file or run in terminal:
```powershell
.\start_gui.bat
# or: .\venv\Scripts\python.exe gui_pyqt.py
```

The PyQt6 Desktop window opens with a dark theme, providing:
1. **💬 Text-to-Speech (TTS) Tab**: Enter any text, select a 3-10s reference audio, select the base model/language (`🇧🇷 PT-BR` or `🇺🇸 English`), synthesize in < 2 seconds, and listen/export to WAV.
2. **🔄 Audio-to-Audio (Voice-to-Voice) Tab**: Select any spoken or sung source audio, choose the target voice sample, and automatically transform the speech into the target voice using Whisper ASR on GPU.
3. **🏋️ Training / Fine-Tuning Tab**: Pick a local dataset directory, configure epochs/batch size/learning rate, and monitor live logs and progress bars asynchronously without UI freezing.
4. **ℹ️ Info & Hardware Tab**: Live status detection of NVIDIA CUDA GPU acceleration (e.g. RTX 4090).

---

### Option B: Command-Line Interface (CLI)

#### 1. Text-to-Speech (TTS Voice Cloning):
```powershell
.\venv\Scripts\python.exe inference/infer.py `
    --ref_audio "data/demo_speaker/audio_01.wav" `
    --text "Hello! This is a voice cloning demonstration using Flow Matching DiT." `
    --output "output_tts.wav" `
    --language "pt-br" `
    --steps 32 `
    --cfg 2.0
```

#### 2. Audio-to-Audio (Voice-to-Voice Conversion):
```powershell
.\venv\Scripts\python.exe inference/infer.py `
    --source_audio "my_recording.wav" `
    --ref_audio "target_voice.wav" `
    --output "output_converted.wav" `
    --language "pt-br"
```

---

### Option C: Direct Python API Usage

```python
from inference.cloner import VoiceCloner

# 1. Initialize cloner (defaults to PT-BR foundation weights on CUDA)
cloner = VoiceCloner(language="pt-br")

# 2. Mode 1: Text-to-Speech Voice Cloning
waveform, sr = cloner.clone_voice(
    ref_audio_path="target_speaker.wav",
    text="Texto sintetizado com a voz clonada em português nativo.",
    output_path="tts_result.wav",
    speed=1.0,
    n_steps=32,
    cfg_strength=2.0
)

# 3. Mode 2: Audio-to-Audio Voice Conversion
waveform, sr, transcribed_text = cloner.convert_voice(
    source_audio_path="source_speech.wav",
    target_ref_audio="target_speaker.wav",
    output_path="v2v_result.wav"
)
print("Recognized speech:", transcribed_text)
```

---

## 🏋️ Training & Fine-Tuning Custom Voices

### Dataset Preparation
Place audio files (`.wav`, `.mp3`, `.flac`) inside a directory (e.g., `data/my_speaker/`). Optionally, add companion text files with identical basenames (`audio01.wav` and `audio01.txt`) containing transcripts.

### Starting Training:
```powershell
.\venv\Scripts\python.exe training/train.py `
    --data_dir "data/my_speaker" `
    --epochs 50 `
    --batch_size 8 `
    --lr 0.0002
```

The training engine utilizes:
- **Optimal Transport CFM**: Continuous probability flow matching for fast convergence.
- **Exponential Moving Average (EMA)**: Weight smoothing for crystal-clear auditory synthesis.
- **Automatic Mixed Precision (AMP)**: High speed and low VRAM footprint on NVIDIA RTX GPUs.
- **Checkpoints**: Saved periodically in `checkpoints/`.
