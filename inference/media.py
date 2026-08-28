"""
Vídeo: extrair o áudio para trabalhar e devolver o vídeo com a trilha nova.

Trabalhar direto do MP4 evita o vaivém de extrair o áudio à mão antes e casar
de novo depois. O vídeo em si nunca é recodificado — o fluxo de imagem é
copiado como está, então não há perda nem espera de encoding.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

EXTENSOES_VIDEO = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".flv", ".mpg", ".mpeg"}

# No Windows, sem isto uma janela de console pisca a cada chamada.
_SEM_JANELA = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}

_ffmpeg_cache: str | None = None
_ffprobe_cache: str | None = None


def _procurar(nome: str) -> str | None:
    """Procura o executável no PATH, nos locais usuais e no pacote imageio."""
    achado = shutil.which(nome)
    if achado:
        return achado

    candidatos = [
        rf"C:\Program Files\FFMPEG\bin\{nome}.exe",
        rf"C:\Program Files\ffmpeg\bin\{nome}.exe",
        rf"C:\ffmpeg\bin\{nome}.exe",
    ]
    for c in candidatos:
        if os.path.exists(c):
            return c

    if nome == "ffmpeg":
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    return None


def caminho_ffmpeg() -> str | None:
    global _ffmpeg_cache
    if _ffmpeg_cache is None:
        _ffmpeg_cache = _procurar("ffmpeg") or ""
    return _ffmpeg_cache or None


def caminho_ffprobe() -> str | None:
    global _ffprobe_cache
    if _ffprobe_cache is None:
        _ffprobe_cache = _procurar("ffprobe") or ""
    return _ffprobe_cache or None


def ffmpeg_disponivel() -> bool:
    return caminho_ffmpeg() is not None


AJUDA_FFMPEG = (
    "O ffmpeg nao foi encontrado. Ele e necessario para abrir video e para gravar o MP4 de saida.\n"
    "Instale de https://www.gyan.dev/ffmpeg/builds/ e deixe o ffmpeg.exe no PATH, "
    "ou rode: pip install imageio-ffmpeg"
)


def eh_video(caminho: str | Path) -> bool:
    return Path(caminho).suffix.lower() in EXTENSOES_VIDEO


def _rodar(argumentos: list[str], descricao: str):
    resultado = subprocess.run(
        argumentos, capture_output=True, text=True, encoding="utf-8", errors="replace", **_SEM_JANELA
    )
    if resultado.returncode != 0:
        # O ffmpeg escreve tudo em stderr; as ultimas linhas dizem o que houve.
        cauda = "\n".join((resultado.stderr or "").strip().splitlines()[-6:])
        raise RuntimeError(f"{descricao} falhou (codigo {resultado.returncode}).\n{cauda}")
    return resultado


def extrair_audio(video: str | Path, saida_wav: str | Path, sample_rate: int | None = None) -> str:
    """
    Extrai a trilha de áudio do vídeo para WAV mono.

    O WAV é gravado sem compressão para não somar perda antes da conversão de
    voz — o arquivo é temporário, o tamanho não importa.
    """
    ffmpeg = caminho_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(AJUDA_FFMPEG)

    video, saida_wav = str(video), str(saida_wav)
    Path(saida_wav).parent.mkdir(parents=True, exist_ok=True)

    # Checar antes da conversao para dar um recado util em vez do erro cru do ffmpeg.
    if not tem_audio(video):
        raise RuntimeError(
            f"O arquivo '{Path(video).name}' nao tem trilha sonora — nao ha audio para trabalhar."
        )

    argumentos = [ffmpeg, "-y", "-i", video, "-vn", "-ac", "1"]
    if sample_rate:
        argumentos += ["-ar", str(int(sample_rate))]
    argumentos += ["-c:a", "pcm_s16le", saida_wav]

    _rodar(argumentos, "Extracao do audio")
    if not os.path.exists(saida_wav) or os.path.getsize(saida_wav) == 0:
        raise RuntimeError(
            f"O video '{Path(video).name}' nao gerou audio. "
            f"Ele tem trilha sonora?"
        )
    return saida_wav


def juntar_audio_video(video: str | Path, audio: str | Path, saida: str | Path,
                       bitrate_audio: str = "192k") -> str:
    """
    Devolve o vídeo original com a trilha de áudio trocada.

    `-c:v copy` copia o fluxo de imagem sem recodificar: é rápido e não perde
    qualidade nenhuma. Só o áudio é codificado (AAC, o que o MP4 espera).
    Como a conversão de voz preserva a duração, imagem e som continuam em sincronia.
    """
    ffmpeg = caminho_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(AJUDA_FFMPEG)

    video, audio, saida = str(video), str(audio), str(saida)
    Path(saida).parent.mkdir(parents=True, exist_ok=True)

    argumentos = [
        ffmpeg, "-y",
        "-i", video,
        "-i", audio,
        "-map", "0:v:0",       # imagem do original
        "-map", "1:a:0",       # audio novo
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", bitrate_audio,
        "-shortest",
        saida,
    ]
    _rodar(argumentos, "Gravacao do MP4")
    return saida


def tem_audio(caminho: str | Path) -> bool:
    """Confere se o arquivo tem trilha sonora."""
    ffprobe = caminho_ffprobe()
    if ffprobe is None:
        return True  # sem ffprobe, deixa o ffmpeg reclamar depois
    try:
        resultado = _rodar(
            [ffprobe, "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type",
             "-of", "default=noprint_wrappers=1:nokey=1", str(caminho)],
            "Leitura dos fluxos"
        )
        return "audio" in (resultado.stdout or "")
    except RuntimeError:
        return False


def duracao(caminho: str | Path) -> float | None:
    """Duração em segundos, via ffprobe. None se não der para descobrir."""
    ffprobe = caminho_ffprobe()
    if ffprobe is None:
        return None
    try:
        resultado = _rodar(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(caminho)],
            "Leitura da duracao"
        )
        return float((resultado.stdout or "").strip())
    except (RuntimeError, ValueError):
        return None


def tem_video(caminho: str | Path) -> bool:
    """Confere se o arquivo tem mesmo um fluxo de imagem (extensão pode mentir)."""
    ffprobe = caminho_ffprobe()
    if ffprobe is None:
        return eh_video(caminho)
    try:
        resultado = _rodar(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_type",
             "-of", "default=noprint_wrappers=1:nokey=1", str(caminho)],
            "Leitura dos fluxos"
        )
        return "video" in (resultado.stdout or "")
    except RuntimeError:
        return False
