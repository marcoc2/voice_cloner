import json
import csv
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
try:
    from dataset.preprocessor import AudioPreprocessor
    from models.text_encoder import SimpleTokenizer
except ImportError:
    from .preprocessor import AudioPreprocessor
    from ..models.text_encoder import SimpleTokenizer


class VoiceCloningDataset(Dataset):
    """
    Dataset para treinamento e fine-tuning do clonador de voz.
    Suporta metadados em JSON, CSV ou pares de arquivos (audio.wav + audio.txt).
    """
    def __init__(
        self,
        data_dir: str | Path,
        metadata_file: str | Path | None = None,
        max_duration_sec: float = 15.0,
        min_duration_sec: float = 1.0,
        prompt_duration_sec: float = 3.0,
        sample_rate: int = 24000
    ):
        self.data_dir = Path(data_dir)
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate)
        self.tokenizer = SimpleTokenizer()
        self.max_duration_sec = max_duration_sec
        self.min_duration_sec = min_duration_sec
        self.prompt_duration_sec = prompt_duration_sec
        self.samples = []

        self._load_samples(metadata_file)

    def _load_samples(self, metadata_file: str | Path | None):
        if metadata_file and Path(metadata_file).exists():
            meta_path = Path(metadata_file)
            if meta_path.suffix == ".json":
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        self.samples.append({
                            "audio_path": self._resolve_path(item["audio_path"]),
                            "text": item.get("text", "")
                        })
            elif meta_path.suffix in [".csv", ".tsv"]:
                delimiter = "\t" if meta_path.suffix == ".tsv" else "|"
                with open(meta_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f, delimiter=delimiter)
                    for row in reader:
                        if len(row) >= 2:
                            self.samples.append({
                                "audio_path": self._resolve_path(row[0]),
                                "text": row[1]
                            })
        else:
            # Varredura automática no diretório por arquivos de áudio
            audio_exts = [".wav", ".mp3", ".flac", ".ogg"]
            for ext in audio_exts:
                for audio_path in self.data_dir.glob(f"**/*{ext}"):
                    txt_path = audio_path.with_suffix(".txt")
                    text = ""
                    if txt_path.exists():
                        with open(txt_path, "r", encoding="utf-8") as f:
                            text = f.read().strip()
                    self.samples.append({
                        "audio_path": audio_path,
                        "text": text
                    })

        print(f"[Dataset] Total de amostras carregadas: {len(self.samples)}")

    def _resolve_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if not p.is_absolute():
            p = self.data_dir / p
        return p

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.samples[idx]
        wav, full_mel = self.preprocessor.process_file(item["audio_path"])
        # full_mel: (1, T_frames, mel_dim) -> (T_frames, mel_dim)
        full_mel = full_mel.squeeze(0)
        
        T_frames = full_mel.shape[0]
        # Frame rate aproximado = 24000 / 256 ~= 93.75 frames/segundo
        fps = self.preprocessor.sample_rate / self.preprocessor.hop_length
        prompt_frames = int(self.prompt_duration_sec * fps)

        # Cria o tensor prompt_mel com a mesma dimensão temporal do target_mel
        # Os frames do prompt contêm o áudio real e os frames restantes são mascarados com zero
        prompt_mel = torch.zeros_like(full_mel)
        if T_frames > prompt_frames + 10:
            prompt_mel[:prompt_frames, :] = full_mel[:prompt_frames, :]
        else:
            prompt_mel = full_mel.clone()

        target_mel = full_mel

        # Tokenização do texto
        text_tokens = torch.tensor(self.tokenizer.encode(item["text"]), dtype=torch.long)

        return {
            "target_mel": target_mel,
            "prompt_mel": prompt_mel,
            "text_tokens": text_tokens
        }


def voice_cloning_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Collate function para agrupar amostras de diferentes comprimentos em lotes com padding.
    """
    target_mels = [b["target_mel"] for b in batch]
    prompt_mels = [b["prompt_mel"] for b in batch]
    text_tokens = [b["text_tokens"] for b in batch]

    # Padding de mel-spectrograms ao longo da dimensão de tempo
    target_mel_padded = pad_sequence(target_mels, batch_first=True, padding_value=0.0)
    prompt_mel_padded = pad_sequence(prompt_mels, batch_first=True, padding_value=0.0)

    # Padding de tokens de texto
    text_tokens_padded = pad_sequence(text_tokens, batch_first=True, padding_value=0)

    # Criação de máscara booleana para os frames válidos do target mel
    lengths = torch.tensor([m.shape[0] for m in target_mels], dtype=torch.long)
    max_len = target_mel_padded.shape[1]
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)

    return {
        "target_mel": target_mel_padded,
        "prompt_mel": prompt_mel_padded,
        "text_tokens": text_tokens_padded,
        "mask": mask
    }
