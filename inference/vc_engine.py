"""
Motor de Voice Conversion quadro a quadro (Seed-VC).

Diferença fundamental para o caminho F5-TTS deste projeto:

  F5-TTS  : áudio -> Whisper -> TEXTO -> TTS -> áudio
            Entre os dois passos, tudo que não é texto é descartado: duração,
            curva de pitch, ênfase, pausas. O TTS reinventa. Resultado: o áudio
            muda de tamanho, a inflexão é outra e palavras que o ASR não ouviu
            simplesmente somem.

  Seed-VC : áudio -> features de conteúdo + F0 por quadro -> áudio
            O texto nunca existe. A duração e a entoação vêm da fonte; só o
            timbre vem da referência. Saída e entrada têm o mesmo comprimento,
            então o lip sync se mantém.

Medido neste projeto (florinda -> voz do silvio, 7,68s):
    erro de duração   F5-TTS 5,0%   | Seed-VC 0,1%
    correlação de F0  F5-TTS -0,15  | Seed-VC 0,78
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
import torch

from inference.hf_compat import apply_all


class SeedVCEngine:
    """
    Envelopa o SeedVCWrapper com uma interface direta: entra áudio, sai áudio
    do mesmo tamanho com o timbre trocado.
    """

    def __init__(self, device: str | None = None, f0_condition: bool = True):
        # Os remendos precisam vir antes de importar o seed_vc: o BigVGAN
        # embutido nele quebra com o huggingface_hub 1.x já no carregamento.
        apply_all()

        try:
            from seed_vc.seed_vc_wrapper import SeedVCWrapper
        except ImportError as e:
            raise ImportError(
                "seed-vc nao esta instalado. Instale com:\n"
                "  pip install seed-vc\n"
                "(nao toca em torch, torchaudio, numpy nem torchcodec)"
            ) from e

        # patch_bigvgan so funciona depois que o modulo dele existe
        from inference.hf_compat import patch_bigvgan
        patch_bigvgan()

        self.device_str = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.f0_condition = f0_condition

        # 44.1kHz no modo com F0, 22.05kHz sem
        self.sample_rate = 44100 if f0_condition else 22050

        print(f"[SeedVC] Carregando modelos de voice conversion "
              f"({'com F0' if f0_condition else 'sem F0'}, {self.sample_rate}Hz)...")
        self._wrapper = SeedVCWrapper(device=torch.device(self.device_str))
        print(f"[SeedVC] Motor pronto em {self.device_str.upper()}.")

    def convert_file(
        self,
        source_path: str | Path,
        target_ref_path: str | Path,
        diffusion_steps: int = 25,
        inference_cfg_rate: float = 0.7,
        auto_f0_adjust: bool = True,
        pitch_shift: int = 0
    ) -> tuple[np.ndarray, int]:
        """
        Converte um arquivo inteiro para o timbre da referência.
        A duração é preservada (length_adjust=1.0).
        """
        # convert_voice tem `yield` no corpo, então é sempre uma função geradora:
        # mesmo com stream_output=False o áudio vem no valor do StopIteration.
        gerador = self._wrapper.convert_voice(
            source=str(source_path),
            target=str(target_ref_path),
            diffusion_steps=int(diffusion_steps),
            length_adjust=1.0,
            inference_cfg_rate=float(inference_cfg_rate),
            f0_condition=self.f0_condition,
            auto_f0_adjust=auto_f0_adjust,
            pitch_shift=int(pitch_shift),
            stream_output=False,
        )

        audio = None
        try:
            while True:
                next(gerador)
        except StopIteration as parada:
            audio = parada.value

        if audio is None:
            raise RuntimeError("Seed-VC nao retornou audio para este trecho.")

        return np.ascontiguousarray(audio, dtype=np.float32), self.sample_rate

    def convert_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_ref_path: str | Path,
        **kwargs
    ) -> tuple[np.ndarray, int]:
        """
        Mesma coisa para um trecho já em memória. O Seed-VC lê de arquivo, então
        gravamos um temporário — o custo é irrelevante perto da difusão.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            caminho = f.name
        try:
            sf.write(caminho, audio, int(sample_rate))
            return self.convert_file(caminho, target_ref_path, **kwargs)
        finally:
            try:
                Path(caminho).unlink()
            except OSError:
                pass
