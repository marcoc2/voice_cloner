import os
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import numpy as np


class AudioPreprocessor:
    """
    Pré-processador de áudio para treinamento e inferência.
    Converte arquivos de áudio para mono a 24kHz e extrai espectrogramas Mel normalizados.
    """
    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 100,
        f_min: float = 0.0,
        f_max: float = 12000.0
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=1.0,
            norm="slaney",
            mel_scale="slaney"
        )

    def load_audio(self, audio_path: str | Path) -> torch.Tensor:
        """
        Carrega áudio de qualquer formato (.wav, .mp3, .flac, .ogg), converte para mono e resampleia para 24kHz.
        Retorna: tensor (1, T_samples)
        """
        audio_path_str = str(audio_path)
        try:
            data, sr = sf.read(audio_path_str)
            if data.ndim == 1:
                waveform = torch.from_numpy(data).float().unsqueeze(0)
            else:
                waveform = torch.from_numpy(data.T).float()
        except Exception:
            waveform, sr = torchaudio.load(audio_path_str)
        
        # Converte para mono se for multicanal
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample para 24kHz se necessário
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Normalização de volume (-1.0 a 1.0 com margem)
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = (waveform / max_val) * 0.95

        return waveform

    def wav_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extrai o espectrograma Mel em escala logarítmica.
        waveform: (1, T_samples) ou (B, 1, T_samples)
        Retorna: mel com formato (B, T_frames, n_mels)
        """
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)  # (1, 1, T_samples)
        
        mel = self.mel_transform(waveform)  # (B, 1, n_mels, T_frames)
        mel = torch.log(torch.clamp(mel, min=1e-5))  # Escala log-mel
        mel = mel.squeeze(1).transpose(1, 2)  # (B, T_frames, n_mels)
        return mel

    def process_file(self, audio_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Processa um arquivo individual e retorna tanto a waveform quanto o mel.
        """
        wav = self.load_audio(audio_path)
        mel = self.wav_to_mel(wav)
        return wav, mel

    def save_audio(self, waveform: torch.Tensor, output_path: str | Path):
        """
        Salva waveform em arquivo WAV a 24kHz usando soundfile.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if waveform.ndim == 2:
            wav_np = waveform.squeeze(0).cpu().float().numpy()
        elif waveform.ndim == 1:
            wav_np = waveform.cpu().float().numpy()
        else:
            wav_np = waveform.squeeze().cpu().float().numpy()

        # Garante amplitude válida
        wav_np = np.clip(wav_np, -1.0, 1.0)
        sf.write(str(output_path), wav_np, self.sample_rate)
