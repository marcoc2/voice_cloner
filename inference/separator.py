"""
Separação de voz e fundo, para converter só a fala.

Por que importa: a conversão de voz degrada quando há música ou plateia por
baixo da fala. Foi medido neste projeto — num trecho de TV com música ao fundo,
a correlação de F0 entre a fala original e a convertida caiu de 0,78 (fala
limpa) para perto de zero. O extrator de F0 e o encoder de conteúdo do Seed-VC
sofrem com o fundo.

Separando antes, a conversão recebe só a voz, e a música volta intacta na
remontagem — ela nunca passa pelo modelo.

O Demucs devolve quatro fontes (voz, baixo, bateria, resto), mas elas somam a
mistura. Então não é preciso separar as quatro e remixar três à mão:

    resto = mistura - voz

que é exatamente o que o `--two-stems` do Demucs faz por baixo.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

MODELO_PADRAO = "htdemucs"


class VocalSeparator:
    """
    Separa a voz do resto com o Demucs.

    O modelo trabalha em 44,1 kHz estéreo; a conversão de taxa e de canais é
    feita aqui, e o resultado volta na taxa que entrou.
    """

    def __init__(self, device: str | None = None, model_name: str = MODELO_PADRAO):
        try:
            from demucs.pretrained import get_model
        except ImportError as e:
            raise ImportError(
                "demucs nao esta instalado. Instale com:\n"
                "  pip install demucs\n"
                "(nao mexe em torch, torchaudio nem numpy)"
            ) from e

        self.device_str = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        print(f"[Separator] Carregando modelo de separacao ({model_name})...")
        self.model = get_model(model_name)
        self.model.to(self.device_str)
        self.model.eval()

        self.sample_rate = int(getattr(self.model, "samplerate", 44100))
        self.fontes = list(getattr(self.model, "sources", ["drums", "bass", "other", "vocals"]))
        print(f"[Separator] Pronto em {self.device_str.upper()} "
              f"({self.sample_rate} Hz, fontes: {', '.join(self.fontes)}).")

    def separar(
        self,
        audio: np.ndarray,
        sample_rate: int,
        shifts: int = 0,
        overlap: float = 0.25
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Devolve (voz, resto), ambos mono e na mesma taxa e comprimento da entrada.

        `shifts` > 0 melhora um pouco a separação repetindo a inferência com
        deslocamentos e tirando a média — custa proporcionalmente mais tempo.
        """
        from demucs.apply import apply_model

        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        n_original = len(audio)
        if n_original == 0:
            return audio.copy(), audio.copy()

        # O Demucs espera 44,1 kHz estereo.
        if sample_rate != self.sample_rate:
            import librosa
            trabalho = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
        else:
            trabalho = audio
        estereo = np.stack([trabalho, trabalho], axis=0)

        # Normaliza como o Demucs faz internamente e desfaz depois.
        tensor = torch.from_numpy(estereo).float()
        media = tensor.mean()
        desvio = tensor.std()
        if desvio < 1e-8:
            desvio = torch.tensor(1.0)
        tensor = (tensor - media) / desvio

        with torch.inference_mode():
            fontes = apply_model(
                self.model,
                tensor.unsqueeze(0).to(self.device_str),
                shifts=int(shifts),
                overlap=float(overlap),
                progress=False,
                device=self.device_str,
            )[0]
        fontes = fontes * desvio + media

        indice_voz = self.fontes.index("vocals") if "vocals" in self.fontes else -1
        voz_estereo = fontes[indice_voz].cpu().numpy()
        voz = voz_estereo.mean(axis=0).astype(np.float32)

        # As fontes somam a mistura: o resto e o que sobra tirando a voz.
        resto = trabalho[:len(voz)] - voz if len(voz) <= len(trabalho) else trabalho - voz[:len(trabalho)]
        resto = np.ascontiguousarray(resto, dtype=np.float32)

        if sample_rate != self.sample_rate:
            import librosa
            voz = librosa.resample(voz, orig_sr=self.sample_rate, target_sr=sample_rate)
            resto = librosa.resample(resto, orig_sr=self.sample_rate, target_sr=sample_rate)

        voz = self._ajustar(voz, n_original)
        resto = self._ajustar(resto, n_original)
        return voz, resto

    @staticmethod
    def _ajustar(x: np.ndarray, n: int) -> np.ndarray:
        """Reamostragem muda o comprimento por alguns quadros; alinha de volta."""
        if len(x) < n:
            return np.pad(x, (0, n - len(x)))
        return np.ascontiguousarray(x[:n], dtype=np.float32)

    @staticmethod
    def proporcao_de_fundo(voz: np.ndarray, resto: np.ndarray) -> float:
        """
        Quanto do sinal e fundo, em 0..1. Serve para avisar quando separar vale
        a pena: numa gravacao ja limpa, o resto e quase nada.
        """
        energia_voz = float(np.sum(voz.astype(np.float64) ** 2))
        energia_resto = float(np.sum(resto.astype(np.float64) ** 2))
        total = energia_voz + energia_resto
        return (energia_resto / total) if total > 1e-12 else 0.0
