# 🎙️ Modern Voice Cloner (Flow Matching DiT)

Sistema moderno de clonagem de voz baseado em **Flow Matching** (Continuous Normalizing Flows com Optimal Transport) e **Diffusion Transformers (DiT)** com modulação AdaLN e vocoder neural a 24kHz.

Comparado ao **RVC** e **so-vits-svc**, este sistema traz avanços significativos:
- **Zero-Shot Real**: Clona qualquer voz com apenas 3 a 10 segundos de áudio de referência, sem necessidade de treinar previamente.
- **Fine-Tuning Rápido**: Suporte a treinamento com poucos minutos de áudio para fidelidade máxima de timbre e prosódia.
- **Livre de artefatos robóticos**: Elimina os sons metálicos característicos de vocoders antigos.
- **Alta Definição**: Áudio a 24kHz com suporte ao vocoder neural Vocos.

---

## 📁 Estrutura do Projeto

```
voice_cloner/
├── configs/
│   └── default_config.yaml     # Configurações do modelo, áudio e treino
├── models/
│   ├── dit.py                  # Diffusion Transformer com AdaLN e Multi-Head Attention
│   ├── flow_matching.py        # OT-CFM (Optimal Transport Flow Matching) e amostragem ODE
│   ├── text_encoder.py         # Tokenizer UTF-8 multilingue e encoder ConvNeXt
│   └── vocoder.py              # Vocoder neural Vocos (24kHz) com fallback Griffin-Lim
├── dataset/
│   ├── preprocessor.py         # Resampling a 24kHz, extração de Mel e normalização
│   └── audio_dataset.py        # PyTorch Dataset e Collate com padding dinâmico
├── training/
│   └── train.py                # Script completo de treino/fine-tuning com EMA e AMP
├── inference/
│   ├── cloner.py               # Motor de inferência Zero-Shot e carregador de pesos
│   └── infer.py                # Interface de linha de comando (CLI)
├── app.py                      # Interface Web interativa em Gradio
├── requirements.txt            # Dependências Python
└── README.md
```

---

## 🚀 Instalação e Ambiente Virtual (`venv`)

### 1. Criar e Ativar o Ambiente Virtual

No Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Instalar as Dependências

```powershell
pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 🔊 Como Usar - Inferência e Clonagem

### Opção A: Interface Desktop Nativa (PyQt6)

Basta dar **dois cliques** no arquivo [`start_gui.bat`](file:///f:/workspace/voice_cloner/start_gui.bat) ou executar no terminal:
```powershell
.\start_gui.bat
# ou: .\venv\Scripts\python.exe gui_pyqt.py
```
A janela nativa Desktop do PyQt abrirá com tema moderno escuro (Dark Theme), permitindo:
1. **Aba de Clonagem / Síntese**: Escolher áudio de referência, digitar o texto, ouvir o áudio gerado diretamente pelo app e salvar em WAV.
2. **Aba de Treinamento**: Selecionar pasta de dataset, definir épocas/batch size e acompanhar logs e progresso em tempo real sem travar a interface.
3. **Aba de Informações**: Status da aceleração de hardware (NVIDIA CUDA / GPU).

*(Obs: Se desejar a interface Web Gradio antiga no navegador, execute `.\venv\Scripts\python.exe app.py`)*

---

### Opção B: Linha de Comando (CLI)

Clone uma voz rapidamente através do terminal:
```powershell
.\venv\Scripts\python.exe inference/infer.py `
    --ref_audio "caminho/para/audio_referencia.wav" `
    --text "Olá! Esta é uma demonstração de clonagem de voz moderna com Flow Matching." `
    --output "resultado.wav" `
    --steps 32 `
    --cfg 2.0
```

---

### Opção C: Uso Direto no Código Python

```python
from inference.cloner import VoiceCloner

# Inicializa o clonador (com ou sem checkpoint treinado)
cloner = VoiceCloner(checkpoint_path="checkpoints/best_model.pt")

# Clona a voz a partir de uma amostra de referência
waveform, sr = cloner.clone_voice(
    ref_audio_path="amostra_voz.wav",
    text="Texto sintetizado com a voz clonada.",
    output_path="saida_clonada.wav",
    speed=1.0,
    n_steps=32,
    cfg_strength=2.0
)
```

---

## 🏋️ Como Treinar / Fazer Fine-Tuning

### Preparação do Dataset
Coloque seus arquivos de áudio (`.wav`, `.mp3`) dentro de uma pasta (ex: `data/`), opcionalmente acompanhados por arquivos de texto com o mesmo nome contendo a transcrição (`exemplo01.wav` e `exemplo01.txt`), ou passe um arquivo `metadata.json` / `metadata.csv`.

### Executando o Treinamento:
```powershell
.\venv\Scripts\python.exe training/train.py `
    --data_dir "data" `
    --epochs 50 `
    --batch_size 8 `
    --lr 0.0002
```

O treinamento utiliza:
- **Optimal Transport CFM**: Aprendizado por fluxo vetorial contínuo.
- **EMA (Exponential Moving Average)**: Suavização exponencial dos pesos para geração auditiva límpida.
- **Automatic Mixed Precision (AMP)**: Alta velocidade e economia de VRAM em GPUs NVIDIA RTX.
- **Checkpoints**: Salvos periodicamente no diretório `checkpoints/`.
