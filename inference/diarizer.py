"""
Diarização de falantes: descobre quantas pessoas falam num áudio e marca
quem fala em cada intervalo.

Usado pelo modo "trocar a voz de um personagem só" numa conversa, sem precisar
recortar os trechos à mão.

Os pesos do pyannote são gated no HuggingFace: é preciso aceitar os termos uma
vez por conta em
  https://huggingface.co/pyannote/speaker-diarization-3.1
  https://huggingface.co/pyannote/segmentation-3.0
e ter o token salvo (`huggingface-cli login`) ou em HF_TOKEN.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
import torch

try:
    from pyannote.audio import Pipeline as PyannotePipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False


def _aplicar_compatibilidade():
    """
    Os remendos de compatibilidade com o huggingface_hub 1.x e com o
    `weights_only` do torch 2.6 vivem em inference/hf_compat.py, porque o
    Seed-VC precisa exatamente dos mesmos.
    """
    if not PYANNOTE_AVAILABLE:
        return
    from inference.hf_compat import patch_hf_hub_download, patch_torch_load
    patch_hf_hub_download()
    patch_torch_load()


_aplicar_compatibilidade()


DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

GATED_HELP = (
    "Os pesos de diarizacao sao gated no HuggingFace. Aceite os termos (uma vez por conta) em:\n"
    "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
    "  https://huggingface.co/pyannote/segmentation-3.0\n"
    "e garanta um token valido (`huggingface-cli login` ou variavel HF_TOKEN)."
)


@dataclass
class SpeechSegment:
    """Um trecho contínuo de fala atribuído a um falante."""
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


class SpeakerDiarizer:
    """
    Wrapper do pyannote.audio. Recebe o áudio de uma conversa e devolve os
    trechos de fala já rotulados por falante (SPEAKER_00, SPEAKER_01, ...).
    """

    def __init__(
        self,
        device: str | None = None,
        model_name: str = DIARIZATION_MODEL,
        hf_token: str | None = None
    ):
        if not PYANNOTE_AVAILABLE:
            raise ImportError(
                "pyannote.audio nao esta instalado. Instale com:\n"
                "  pip install \"pyannote.audio==3.3.2\" \"numpy<2\"\n"
                "(a versao 4.x exige torch>=2.8 e puxaria torchcodec, quebrando o resto do projeto)"
            )

        self.device_str = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        if hf_token is None:
            try:
                from huggingface_hub import get_token
                hf_token = get_token()
            except Exception:
                hf_token = None

        print(f"[Diarizer] Carregando modelo de diarizacao ({model_name})...")
        try:
            self.pipeline = PyannotePipeline.from_pretrained(model_name, use_auth_token=hf_token)
        except Exception as e:
            raise RuntimeError(f"Falha ao carregar '{model_name}': {e}\n\n{GATED_HELP}") from e

        if self.pipeline is None:
            # O pyannote devolve None (sem levantar excecao) quando o acesso e negado.
            raise RuntimeError(f"Acesso negado a '{model_name}'.\n\n{GATED_HELP}")

        self.pipeline.to(torch.device(self.device_str))
        print(f"[Diarizer] Diarizacao pronta em {self.device_str.upper()}.")

    def diarize(
        self,
        audio_path: str | Path,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        merge_gap: float = 0.4,
        min_duration: float = 0.35
    ) -> list[SpeechSegment]:
        """
        Detecta os falantes do áudio.

        num_speakers força a quantidade exata; min/max apenas limitam a busca.
        Deixe tudo em None para o modelo decidir sozinho quantas pessoas há.

        merge_gap junta trechos vizinhos do mesmo falante separados por pausas
        curtas — a diarização fatia em respirações, e trechos de meio segundo
        dão contexto ruim para o TTS.
        min_duration descarta fragmentos curtos demais para transcrever.
        """
        audio_path = str(audio_path)
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = int(num_speakers)
        else:
            if min_speakers is not None:
                kwargs["min_speakers"] = int(min_speakers)
            if max_speakers is not None:
                kwargs["max_speakers"] = int(max_speakers)

        print(f"[Diarizer] Analisando '{Path(audio_path).name}'...")
        annotation = self.pipeline(audio_path, **kwargs)

        raw = [
            SpeechSegment(float(turn.start), float(turn.end), str(speaker))
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        raw.sort(key=lambda s: s.start)

        segments = self._merge_adjacent(raw, merge_gap)
        segments = [s for s in segments if s.duration >= min_duration]

        speakers = sorted({s.speaker for s in segments})
        total = sum(s.duration for s in segments)
        print(f"[Diarizer] {len(speakers)} falante(s) em {len(segments)} trechos ({total:.1f}s de fala):")
        for spk in speakers:
            spk_segs = [s for s in segments if s.speaker == spk]
            spk_dur = sum(s.duration for s in spk_segs)
            share = (spk_dur / total * 100) if total else 0.0
            print(f"[Diarizer]   {spk}: {spk_dur:6.1f}s em {len(spk_segs):3d} trechos ({share:4.1f}%)")

        return segments

    @staticmethod
    def _merge_adjacent(segments: list[SpeechSegment], merge_gap: float) -> list[SpeechSegment]:
        """Funde trechos consecutivos do mesmo falante separados por pausas curtas."""
        if not segments:
            return []

        merged = [SpeechSegment(segments[0].start, segments[0].end, segments[0].speaker)]
        for seg in segments[1:]:
            last = merged[-1]
            if seg.speaker == last.speaker and seg.start - last.end <= merge_gap:
                last.end = max(last.end, seg.end)
            else:
                merged.append(SpeechSegment(seg.start, seg.end, seg.speaker))
        return merged

    @staticmethod
    def speakers(segments: list[SpeechSegment]) -> list[str]:
        """IDs de falante ordenados do que mais fala para o que menos fala."""
        totals: dict[str, float] = {}
        for s in segments:
            totals[s.speaker] = totals.get(s.speaker, 0.0) + s.duration
        return sorted(totals, key=lambda k: totals[k], reverse=True)

    @staticmethod
    def export_speaker_samples(
        audio_path: str | Path,
        segments: list[SpeechSegment],
        output_dir: str | Path,
        max_seconds: float = 6.0
    ) -> dict[str, str]:
        """
        Salva uma amostra de cada falante para você ouvir e descobrir quem é quem.
        Usa o trecho contínuo mais longo de cada um, que é também o melhor
        candidato a áudio de referência.
        """
        audio, sr = sf.read(str(audio_path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        samples: dict[str, str] = {}
        for spk in {s.speaker for s in segments}:
            best = max((s for s in segments if s.speaker == spk), key=lambda s: s.duration)
            start = int(best.start * sr)
            end = int(min(best.end, best.start + max_seconds) * sr)
            out = output_dir / f"{spk}.wav"
            sf.write(str(out), audio[start:end], sr)
            samples[spk] = str(out)
            print(f"[Diarizer] Amostra de {spk}: {out} ({(end - start) / sr:.1f}s)")

        return samples
