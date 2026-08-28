import torch
import torch.nn as nn
import torchaudio


class VocoderWrapper:
    """
    Wrapper para Vocoder Neural de alta fidelidade (Vocos) com fallback automático.
    Converte espectrogramas Mel em forma de onda WAV a 24kHz.
    """
    def __init__(self, sample_rate: int = 24000, n_mels: int = 100, device: str = "cuda"):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.device = device if torch.cuda.is_available() else "cpu"
        self.vocos = None
        self._load_vocoder()

    def _load_vocoder(self):
        try:
            from vocos import Vocos
            print(f"[Vocoder] Carregando modelo neural Vocos (charactr/vocos-mel-24khz) no dispositivo {self.device}...")
            self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
            self.vocos.to(self.device)
            self.vocos.eval()
            print("[Vocoder] Vocos carregado com sucesso!")
        except Exception as e:
            print(f"[Vocoder] Aviso: Vocos não pôde ser carregado diretamente ({e}). Usando fallback de inversão mel...")
            self.vocos = None

    @torch.no_grad()
    def mel_to_audio(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: Tensor com formato (1, T, n_mels) ou (B, n_mels, T)
        Retorna: waveform (1, num_samples) a 24kHz
        """
        # Normaliza dimensões: queremos (B, n_mels, T)
        if mel.ndim == 3 and mel.shape[-1] == self.n_mels:
            mel = mel.transpose(1, 2)  # (B, T, n_mels) -> (B, n_mels, T)
        
        mel = mel.to(self.device)

        if self.vocos is not None:
            try:
                # Vocos aceita (B, n_mels, T)
                audio = self.vocos.decode(mel)
                return audio.cpu()
            except Exception as e:
                print(f"[Vocoder] Erro na decodificação com Vocos ({e}). Usando sintetizador fallback...")

        # Fallback: Transformada Inversa com Griffin-Lim
        return self._griffin_lim_fallback(mel)

    def _griffin_lim_fallback(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Fallback matemático robusto utilizando Griffin-Lim de alta precisão.
        """
        mel_cpu = mel.cpu().float()
        # Converte dB/log-mel para escala de amplitude linear
        mel_linear = torch.exp(mel_cpu)
        
        inv_mel_scale = torchaudio.transforms.InverseMelScale(
            n_stft=1024 // 2 + 1,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            f_min=0.0,
            f_max=self.sample_rate / 2.0
        )
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_iter=32
        )
        
        linear_spec = inv_mel_scale(mel_linear)
        audio = griffin_lim(linear_spec)
        return audio
