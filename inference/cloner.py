import os
import sys
from pathlib import Path

# Garante acesso aos módulos do projeto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
import soundfile as sf
import numpy as np

# Suporte ao motor Flow Matching Foundation Model pré-treinado
try:
    from f5_tts.api import F5TTS
    from huggingface_hub import hf_hub_download
    F5_AVAILABLE = True
except ImportError:
    F5_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from models.dit import DiT
from models.text_encoder import TextEncoder, SimpleTokenizer
from models.flow_matching import ConditionalFlowMatching
from models.vocoder import VocoderWrapper
from dataset.preprocessor import AudioPreprocessor


# Repositórios oficiais de modelos pré-treinados
PT_BR_REPO = "traderpedroso/F5-TTS-BRAZILIAN-PORTUGUESE"
PT_BR_FILENAME = "model_stable.safetensors"

# Arquiteturas do F5-TTS.
# ATENÇÃO: o checkpoint PT-BR é um fine-tune do F5TTS_Base (v0), não do v1.
# As duas arquiteturas têm exatamente os mesmos shapes de tensores, então
# carregar o v0 como v1 passa silenciosamente no load_state_dict — mas o v1 usa
# pe_attn_head=None (RoPE em todas as cabeças) e text_mask_padding=True,
# enquanto o v0 foi treinado com pe_attn_head=1 e text_mask_padding=False.
# O resultado é o alinhamento texto/áudio quebrado: o timbre sai correto e a
# fala vira fonemas sem nexo. Por isso a arquitetura é amarrada ao checkpoint.
PT_BR_ARCH = "F5TTS_Base"
DEFAULT_ARCH = "F5TTS_v1_Base"

# Whisper usado para transcrever áudio de origem e referência.
ASR_MODEL = "openai/whisper-large-v3-turbo"
ASR_FALLBACK_MODEL = "openai/whisper-small"


class VoiceCloner:
    """
    Motor principal de clonagem de voz e conversão acústica.
    Suporta:
    1. Text-to-Speech Voice Cloning (Texto -> Áudio com Voz Clonada)
    2. Voice-to-Voice Conversion (Áudio de Origem -> Áudio com Voz Clonada)
    """
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        language: str = "pt-br",  # 'pt-br' ou 'en'
        config_path: str | Path = "configs/default_config.yaml",
        device: str | None = None,
        model_arch: str | None = None  # 'F5TTS_Base' (v0) ou 'F5TTS_v1_Base' (v1)
    ):
        self.device_str = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device_str)
        self.language = language.lower()
        self.checkpoint_path = checkpoint_path
        self.model_arch = model_arch
        self.f5_engine = None
        self.custom_cfm = None
        self.asr_pipeline = None
        self._diarizer = None
        self._vc_engine = None

        print(f"[VoiceCloner] Inicializando motor de clonagem (Idioma: {self.language.upper()}) no dispositivo: {self.device}")

        # Carrega configuração padrão
        cfg_file = Path(config_path)
        if not cfg_file.is_absolute():
            cfg_file = Path(__file__).resolve().parent.parent / config_path
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.sample_rate = self.config["audio"]["sample_rate"]
        self.preprocessor = AudioPreprocessor(sample_rate=self.sample_rate)

        self._init_models()

    def _init_models(self):
        if F5_AVAILABLE:
            try:
                ckpt = ""
                # A arquitetura precisa ser a mesma em que o checkpoint foi
                # treinado: os shapes são idênticos entre v0 e v1, então um
                # descasamento não gera erro, só fala ininteligível.
                arch = DEFAULT_ARCH
                is_pt_br = False

                # 1. Se o usuário passou um checkpoint customizado
                if self.checkpoint_path and Path(self.checkpoint_path).exists():
                    ckpt = str(self.checkpoint_path)
                    print(f"[VoiceCloner] Carregando checkpoint customizado: {ckpt}")
                # 2. Se o idioma for Português Brasileiro (PT-BR)
                elif self.language in ["pt", "pt-br", "portugues", "portuguese"]:
                    print(f"[VoiceCloner] Carregando Foundation Model especialista em Português Brasileiro (PT-BR)...")
                    try:
                        ckpt = hf_hub_download(repo_id=PT_BR_REPO, filename=PT_BR_FILENAME)
                        arch = PT_BR_ARCH
                        is_pt_br = True
                        print(f"[VoiceCloner] Pesos PT-BR carregados: {ckpt}")
                    except Exception as e:
                        print(f"[VoiceCloner] Aviso ao baixar PT-BR do HuggingFace ({e}). Usando base padrão...")
                        ckpt = ""

                # Escolha explícita do usuário tem prioridade sobre o padrão.
                if self.model_arch:
                    arch = self.model_arch

                self.f5_engine = F5TTS(
                    model=arch,
                    ckpt_file=ckpt,
                    device=self.device_str
                )
                self.model_arch = arch
                print(f"[VoiceCloner] Motor F5-TTS pronto na GPU (arquitetura {arch} | {'PT-BR Nativo' if is_pt_br else 'Multilíngue/EN'})!")
                return
            except Exception as e:
                print(f"[VoiceCloner] Aviso: Falha ao carregar F5TTS ({e}). Inicializando modelo nativo...")

        # Fallback para modelo nativo DiT
        m_cfg = self.config["model"]
        self.tokenizer = SimpleTokenizer()
        self.dit = DiT(
            dim=m_cfg["dim"],
            depth=m_cfg["depth"],
            heads=m_cfg["heads"],
            dim_head=m_cfg["dim_head"],
            ff_mult=m_cfg["ff_mult"],
            mel_dim=m_cfg["mel_dim"],
            text_dim=m_cfg["text_dim"],
            dropout=0.0
        )
        self.text_encoder = TextEncoder(
            vocab_size=m_cfg.get("vocab_size", 256),
            embed_dim=m_cfg["text_dim"],
            depth=4
        )
        self.custom_cfm = ConditionalFlowMatching(
            transformer=self.dit,
            text_encoder=self.text_encoder,
            sigma_min=float(m_cfg.get("sigma_min", 1e-4))
        ).to(self.device)

        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            self.load_checkpoint(self.checkpoint_path)

        self.vocoder = VocoderWrapper(sample_rate=self.sample_rate, device=self.device_str)

    def _get_asr(self):
        """
        Inicializa o Whisper ASR para reconhecimento e conversão de áudio para áudio.
        No modo Áudio -> Áudio a transcrição vira o texto que será falado, então
        erro de ASR sai direto no áudio final: vale usar o modelo maior.
        chunk_length_s habilita o modo long-form (sem ele o Whisper corta em 30s).
        """
        if self.asr_pipeline is None and TRANSFORMERS_AVAILABLE:
            device_arg = 0 if self.device_str == "cuda" else -1
            for model_name in (ASR_MODEL, ASR_FALLBACK_MODEL):
                try:
                    print(f"[VoiceCloner] Carregando Whisper ASR ({model_name}) para reconhecimento do audio de origem...")
                    self.asr_pipeline = pipeline(
                        "automatic-speech-recognition",
                        model=model_name,
                        chunk_length_s=30,
                        device=device_arg
                    )
                    print("[VoiceCloner] Whisper ASR carregado com sucesso!")
                    break
                except Exception as e:
                    print(f"[VoiceCloner] Erro ao carregar {model_name}: {e}")
        return self.asr_pipeline

    def transcribe_audio(self, audio_path: str | Path) -> str:
        """
        Transcreve o conteúdo de fala de um áudio (suporta mono, estéreo, WAV, MP3, etc.).
        """
        # Se existir arquivo .txt com o mesmo nome, usa-o diretamente
        txt_path = Path(audio_path).with_suffix(".txt")
        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.read().strip()

        try:
            audio_data, sr = sf.read(str(audio_path))
        except Exception as e:
            print(f"[VoiceCloner] Erro ao ler '{audio_path}': {e}")
            return ""

        return self.transcribe_array(audio_data, sr)

    def transcribe_array(self, audio_data: np.ndarray, sample_rate: int, verbose: bool = True) -> str:
        """
        Transcreve um trecho de áudio já carregado em memória.
        Usado pela conversão por falante, que fatia a conversa e transcreve
        cada trecho separadamente, sem passar por arquivos temporários.
        """
        asr = self._get_asr()
        if asr is None:
            return ""

        try:
            # Converte estéreo (ou múltiplos canais) para mono
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=-1)

            audio_data = audio_data.astype(np.float32)
            lang = "portuguese" if self.language in ["pt", "pt-br", "portugues", "portuguese"] else "english"
            res = asr(
                {"raw": audio_data, "sampling_rate": int(sample_rate)},
                generate_kwargs={"language": lang, "task": "transcribe"}
            )
            text = res.get("text", "").strip()
            if verbose:
                print(f"[VoiceCloner] Audio transcrito ({len(text)} caracteres): '{text}'")
            return text
        except Exception as e:
            print(f"[VoiceCloner] Erro na transcricao com Whisper: {e}")
        return ""

    def _prepare_reference(self, ref_path: str | Path, ref_text: str = "") -> tuple[str, str]:
        """
        Alinha o áudio de referência com a sua transcrição.

        O F5-TTS recorta internamente a referência em ~12s, mas usa o ref_text
        recebido por inteiro. Se o texto descreve 19s de áudio e o modelo só
        escuta 11s, o alinhamento texto/áudio do prompt fica errado e a fala
        gerada sai como fonemas sem nexo (com o timbre correto).

        Aqui o recorte é feito primeiro e a transcrição é do trecho que o modelo
        realmente vai ouvir.
        """
        ref_path = str(ref_path)
        user_text = (ref_text or "").strip()

        # Transcrição fornecida à mão ou via arquivo .txt irmão do áudio
        if not user_text:
            txt_path = Path(ref_path).with_suffix(".txt")
            if txt_path.exists():
                with open(txt_path, "r", encoding="utf-8") as f:
                    user_text = f.read().strip()

        try:
            from f5_tts.infer.utils_infer import preprocess_ref_audio_text
        except ImportError:
            return ref_path, user_text

        try:
            original_dur = sf.info(ref_path).duration
        except Exception:
            original_dur = 0.0

        # O "." é um placeholder: evita que o F5 dispare o ASR interno dele aqui,
        # que não recebe o idioma e transcreveria em inglês.
        clipped_path, _ = preprocess_ref_audio_text(ref_path, user_text or ".", show_info=lambda *a, **k: None)

        try:
            clipped_dur = sf.info(clipped_path).duration
        except Exception:
            clipped_dur = original_dur

        was_clipped = original_dur > 0 and clipped_dur < original_dur - 0.5

        if user_text and not was_clipped:
            return clipped_path, user_text

        if user_text and was_clipped:
            print(
                f"[VoiceCloner] Aviso: referencia de {original_dur:.1f}s foi recortada para "
                f"{clipped_dur:.1f}s pelo F5-TTS; a transcricao informada nao corresponde mais "
                f"ao trecho usado. Retranscrevendo o recorte (use uma referencia de ate ~10s "
                f"para aproveitar sua propria transcricao)."
            )

        text = self.transcribe_audio(clipped_path)
        return clipped_path, text

    def load_checkpoint(self, checkpoint_path: str | Path):
        print(f"[VoiceCloner] Carregando pesos do checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device)
        if "ema_shadow" in state and state["ema_shadow"]:
            print("[VoiceCloner] Aplicando pesos da Média Móvel Exponencial (EMA)...")
            model_dict = self.custom_cfm.state_dict()
            for k, v in state["ema_shadow"].items():
                if k in model_dict:
                    model_dict[k].copy_(v)
            self.custom_cfm.load_state_dict(model_dict)
        elif "state_dict" in state:
            self.custom_cfm.load_state_dict(state["state_dict"])
        else:
            self.custom_cfm.load_state_dict(state)
        print("[VoiceCloner] Pesos customizados carregados com sucesso!")

    def clone_voice(
        self,
        ref_audio_path: str | Path,
        text: str,
        ref_text: str = "",
        output_path: str | Path | None = None,
        speed: float = 1.0,
        n_steps: int = 32,
        cfg_strength: float = 2.0,
        solver: str = "euler"
    ) -> tuple[np.ndarray | torch.Tensor, int]:
        """
        MODO 1: Text-to-Speech Voice Cloning (Texto -> Áudio com Voz Clonada)
        """
        ref_p = Path(ref_audio_path)
        if not ref_p.exists():
            raise FileNotFoundError(f"Áudio de referência '{ref_audio_path}' não encontrado.")

        print(f"[VoiceCloner] Sintetizando voz com referência '{ref_p.name}' | Texto: '{text}'...")

        if self.f5_engine is not None:
            # Recorta a referência e transcreve o recorte, para que ref_text e
            # ref_audio descrevam exatamente o mesmo trecho de fala.
            ref_file, ref_text = self._prepare_reference(ref_p, ref_text)
            wav_out, sr, _ = self.f5_engine.infer(
                ref_file=ref_file,
                ref_text=ref_text.strip() if ref_text else "",
                gen_text=text.strip(),
                speed=float(speed),
                nfe_step=int(n_steps),
                cfg_strength=float(cfg_strength),
                file_wave=str(output_path) if output_path else None
            )
            return wav_out, sr

        # Fallback para síntese CFM nativa
        if not ref_text or not ref_text.strip():
            ref_text = self.transcribe_audio(ref_p)
        _, prompt_mel = self.preprocessor.process_file(ref_p)
        prompt_mel = prompt_mel.to(self.device)

        token_ids = self.tokenizer.encode(text)
        text_tokens = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        chars_count = max(1, len(text.strip()))
        estimated_duration_sec = max(1.5, (chars_count / 13.0) / max(0.2, speed) + 0.5)
        fps = self.sample_rate / 256
        target_len = int(estimated_duration_sec * fps)

        generated_mel = self.custom_cfm.sample(
            prompt_mel=prompt_mel,
            text_tokens=text_tokens,
            target_len=target_len,
            n_steps=n_steps,
            cfg_strength=cfg_strength,
            solver=solver,
            device=self.device
        )
        waveform = self.vocoder.mel_to_audio(generated_mel)

        if output_path:
            self.preprocessor.save_audio(waveform, output_path)

        return waveform, self.sample_rate

    def _get_vc_engine(self, f0_condition: bool = True):
        """Carrega o Seed-VC sob demanda (leva ~2 min na primeira vez)."""
        if self._vc_engine is None or self._vc_engine.f0_condition != f0_condition:
            from inference.vc_engine import SeedVCEngine
            self._vc_engine = SeedVCEngine(device=self.device_str, f0_condition=f0_condition)
        return self._vc_engine

    def _get_diarizer(self):
        """Carrega o diarizador sob demanda (os pesos são pesados e gated)."""
        if self._diarizer is None:
            from inference.diarizer import SpeakerDiarizer
            self._diarizer = SpeakerDiarizer(device=self.device_str)
        return self._diarizer

    @staticmethod
    def _predicted_duration(ref_seconds: float, ref_text: str, gen_text: str, speed: float) -> float:
        """
        Reproduz a conta de duração do F5-TTS para saber, antes de gerar, quanto
        tempo ele daria a este texto.

        Serve para avisar quando o trecho original é curto demais para o que foi
        dito. Repare no `0.3`: o F5 força velocidade 0.3 para texto com menos de
        10 bytes, o que faz uma interjeição de meio segundo virar quase dois.
        """
        ref_bytes = max(1, len(ref_text.encode("utf-8")))
        gen_bytes = max(1, len(gen_text.encode("utf-8")))
        local_speed = 0.3 if gen_bytes < 10 else max(1e-3, speed)
        return ref_seconds / ref_bytes * gen_bytes / local_speed

    @staticmethod
    def _trim_silence(wav: np.ndarray, top_db: int = 30) -> np.ndarray:
        """Remove silêncio das pontas do trecho gerado."""
        try:
            import librosa
            trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
            return trimmed if len(trimmed) else wav
        except Exception:
            return wav

    @staticmethod
    def _fit_to_slot(wav: np.ndarray, slot_samples: int, fit_mode: str) -> np.ndarray:
        """
        Encaixa a fala gerada no buraco deixado pelo trecho original.

        'stretch' estica/comprime no tempo para casar exatamente com o slot,
        preservando o sincronismo do restante da conversa.
        'pad' preserva a velocidade natural da fala: sobra vira silêncio, e o
        que passar do slot empurra o resto da linha do tempo para frente.
        """
        if slot_samples <= 0 or len(wav) == slot_samples:
            return wav

        if fit_mode == "stretch":
            try:
                import librosa
                wav = librosa.effects.time_stretch(
                    np.ascontiguousarray(wav, dtype=np.float32),
                    rate=len(wav) / slot_samples
                )
            except Exception as e:
                print(f"[VoiceCloner] Aviso: time-stretch falhou ({e}); usando corte/silencio.")

        if len(wav) < slot_samples:
            return np.pad(wav, (0, slot_samples - len(wav)))
        return wav[:slot_samples]

    def convert_speaker(
        self,
        source_audio_path: str | Path,
        target_ref_audio: str | Path,
        speaker: str | None = None,
        target_ref_text: str = "",
        output_path: str | Path | None = None,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        speed: float = 1.0,
        n_steps: int = 32,
        cfg_strength: float = 2.0,
        fit_mode: str = "stretch",  # 'stretch' mantem a sincronia | 'pad' mantem o ritmo natural
        crossfade: float = 0.015,
        seed: int | None = None,
        progress_cb=None,
        engine: str = "f5",          # 'f5' (ASR->TTS) ou 'seedvc' (VC quadro a quadro)
        f0_condition: bool = True,   # so no seedvc: preserva a curva de pitch
        diffusion_steps: int = 25    # so no seedvc
    ) -> tuple[np.ndarray, int, list[dict]]:
        """
        MODO 3: troca a voz de UM falante numa conversa, numa passada só.

        Diariza o áudio, escolhe o falante alvo, regenera apenas os trechos dele
        com a voz de referência e remonta a conversa — os demais falantes seguem
        com o áudio original, sem recorte manual.

        speaker: qual ID substituir ('SPEAKER_01'). None usa quem mais fala.
        seed: fixa a amostragem do Flow Matching. O F5 sorteia uma seed nova a
        cada chamada, então um trecho que saiu ruim pode sair bom na rodada
        seguinte — fixar a seed torna o resultado reproduzível.
        progress_cb: chamado como progress_cb(feito, total, texto) a cada trecho,
        para a interface gráfica acompanhar sem depender do stdout.

        engine: 'f5' passa por Whisper e ressintetiza com o F5-TTS — serve para
        trocar também o que é dito, mas a duração e a entoação são reinventadas.
        'seedvc' faz voice conversion quadro a quadro: não existe texto no meio,
        a duração é idêntica à do trecho original e a inflexão é preservada, que
        é o que mantém o lip sync. Use 'seedvc' para dublar personagem.
        Retorna (waveform, sample_rate, relatório trecho a trecho).
        """
        source_p = Path(source_audio_path)
        target_p = Path(target_ref_audio)
        if not source_p.exists():
            raise FileNotFoundError(f"Áudio de origem '{source_audio_path}' não encontrado.")
        if not target_p.exists():
            raise FileNotFoundError(f"Áudio da voz alvo '{target_ref_audio}' não encontrado.")
        engine = (engine or "f5").lower()
        if engine not in ("f5", "seedvc"):
            raise ValueError(f"engine deve ser 'f5' ou 'seedvc', recebido '{engine}'.")
        if engine == "f5" and self.f5_engine is None:
            raise RuntimeError("A troca por falante com F5 exige o motor F5-TTS; o fallback nativo nao suporta este modo.")

        vc_engine = self._get_vc_engine(f0_condition) if engine == "seedvc" else None

        from inference.diarizer import SpeakerDiarizer

        segments = self._get_diarizer().diarize(
            source_p,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        if not segments:
            raise ValueError("Nenhuma fala detectada no audio de origem.")

        ranking = SpeakerDiarizer.speakers(segments)
        if speaker is None:
            speaker = ranking[0]
            print(f"[VoiceCloner] Nenhum falante informado; usando o que mais fala: {speaker}")
        elif speaker not in ranking:
            raise ValueError(f"Falante '{speaker}' nao encontrado. Detectados: {', '.join(ranking)}")

        targets = [s for s in segments if s.speaker == speaker]
        print(f"[VoiceCloner] Substituindo {len(targets)} trecho(s) de {speaker} pela voz de '{target_p.name}'.")

        # A montagem acontece na taxa nativa do motor escolhido: 24 kHz do Vocos
        # no F5, 44,1 kHz do Seed-VC com F0. Reamostrar a saída do Seed-VC para
        # 24 kHz jogaria fora metade da banda sem motivo.
        sr = 24000 if engine == "f5" else vc_engine.sample_rate
        audio, orig_sr = sf.read(str(source_p), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if orig_sr != sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

        # A referência é preparada uma vez só: recortar e transcrever a cada
        # trecho seria desperdício, e o F5 já cacheia o recorte por hash.
        # O Seed-VC não usa texto nenhum: dispensa transcrever a referência.
        if engine == "f5":
            ref_file, ref_text = self._prepare_reference(target_p, target_ref_text)
            try:
                ref_seconds = float(sf.info(ref_file).duration)
            except Exception:
                ref_seconds = 0.0
        else:
            ref_file, ref_text, ref_seconds = str(target_p), "", 0.0
        apertados = 0

        # Estratégia do Seed-VC: uma única chamada, mas contendo APENAS a fala do
        # falante alvo, emendada.
        #
        # Chamar uma vez por trecho seria lento — o Seed-VC reprocessa a
        # referência inteira (features do Whisper, mel, embedding de locutor) a
        # cada chamada. Já converter a conversa inteira é pior ainda: as
        # estatísticas de F0 passam a ser calculadas sobre música, plateia e as
        # outras vozes, e a conversão degrada.
        #
        # Medido em 3 min de TV (7 trechos): converter tudo levou 248s com
        # similaridade 0.415 com a voz alvo; converter só o falante levou 6s com
        # similaridade 0.627.
        trechos_convertidos: dict[int, np.ndarray] = {}
        if engine == "seedvc":
            silencio = np.zeros(int(0.25 * sr), dtype=np.float32)
            pedacos, offsets, pos = [], [], 0
            for indice, seg in enumerate(targets):
                ini = max(0, int(seg.start * sr))
                fim = min(len(audio), int(seg.end * sr))
                if fim <= ini:
                    continue
                pedaco = audio[ini:fim]
                pedacos.append(pedaco)
                offsets.append((indice, pos, pos + len(pedaco)))
                pos += len(pedaco) + len(silencio)
                pedacos.append(silencio)

            if pedacos:
                emendado = np.concatenate(pedacos[:-1])
                print(f"[VoiceCloner] Convertendo {len(offsets)} trecho(s) de {speaker} "
                      f"({len(emendado)/sr:.1f}s de fala) numa passada...")
                convertido, vc_sr = vc_engine.convert_array(
                    emendado, sr, ref_file, diffusion_steps=diffusion_steps
                )
                if vc_sr != sr:
                    import librosa
                    convertido = librosa.resample(convertido, orig_sr=vc_sr, target_sr=sr)
                # O comprimento pode variar em alguns milissegundos; alinhamos.
                if len(convertido) < len(emendado):
                    convertido = np.pad(convertido, (0, len(emendado) - len(convertido)))
                convertido = convertido[:len(emendado)]
                for indice, ini, fim in offsets:
                    trechos_convertidos[indice] = np.ascontiguousarray(
                        convertido[ini:fim], dtype=np.float32
                    )
                print(f"[VoiceCloner] Conversao pronta; remontando a conversa.")

        fade = max(0, int(crossfade * sr))
        pieces: list[np.ndarray] = []
        report: list[dict] = []
        cursor = 0  # amostra do original já copiada

        for indice_seg, seg in enumerate(targets):
            start = max(0, int(seg.start * sr))
            end = min(len(audio), int(seg.end * sr))
            if end <= start:
                continue

            # Trecho dos outros falantes (ou silêncio) antes deste
            if start > cursor:
                pieces.append(audio[cursor:start].copy())

            slot = audio[start:end]

            if engine == "seedvc":
                recorte = trechos_convertidos.get(indice_seg)
                if recorte is None or not len(recorte):
                    print(f"[VoiceCloner]   {seg.start:7.2f}s-{seg.end:7.2f}s  sem conversao, mantendo original.")
                    pieces.append(slot.copy())
                    report.append({"start": seg.start, "end": seg.end, "text": "", "status": "mantido"})
                    cursor = end
                    if progress_cb is not None:
                        progress_cb(len(report), len(targets), "")
                    continue

                # Sem ASR e sem texto: o trecho já está convertido e alinhado.
                usa_fade = bool(fade and pieces and len(pieces[-1]) >= fade)
                alvo = len(slot) + (fade if usa_fade else 0)
                fitted = self._fit_to_slot(np.ascontiguousarray(recorte, dtype=np.float32), alvo, "pad")

                orig_rms = float(np.sqrt(np.mean(slot ** 2)))
                gen_rms = float(np.sqrt(np.mean(fitted ** 2)))
                if gen_rms > 1e-6 and orig_rms > 1e-6:
                    fitted = fitted * (orig_rms / gen_rms)
                    pico = float(np.abs(fitted).max())
                    if pico > 0.99:
                        fitted = fitted * (0.99 / pico)

                if usa_fade and len(fitted) > fade:
                    rampa = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                    cauda = pieces[-1][-fade:] * (1.0 - rampa) + fitted[:fade] * rampa
                    pieces[-1] = np.concatenate([pieces[-1][:-fade], cauda])
                    fitted = fitted[fade:]
                if fade and len(fitted) > fade:
                    fitted = fitted.copy()
                    fitted[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

                pieces.append(fitted)
                slot_dur = len(slot) / sr
                print(f"[VoiceCloner]   {seg.start:7.2f}s-{seg.end:7.2f}s  "
                      f"({slot_dur:5.2f}s convertidos quadro a quadro)")
                report.append({
                    "start": seg.start, "end": seg.end, "text": "",
                    "slot_seconds": slot_dur, "natural_seconds": slot_dur,
                    "tight": False, "status": "substituido"
                })
                cursor = end
                if progress_cb is not None:
                    progress_cb(len(report), len(targets), "")
                continue

            text = self.transcribe_array(slot, sr, verbose=False).strip()

            if not text:
                print(f"[VoiceCloner]   {seg.start:7.2f}s-{seg.end:7.2f}s  sem fala reconhecida, mantendo original.")
                pieces.append(slot.copy())
                report.append({"start": seg.start, "end": seg.end, "text": "", "status": "mantido"})
                cursor = end
                if progress_cb is not None:
                    progress_cb(len(report), len(targets), "")
                continue

            slot_dur = len(slot) / sr
            natural_est = self._predicted_duration(ref_seconds, ref_text, text, speed)

            # No modo 'stretch' pedimos a duração exata do trecho original ao
            # próprio F5 (fix_duration tem prioridade sobre a estimativa dele, e
            # também sobre o hack de velocidade 0.3 para texto curto). Gerar já
            # no tempo certo soa muito melhor do que gerar solto e comprimir
            # depois no phase vocoder.
            fix_duration = (ref_seconds + slot_dur) if fit_mode == "stretch" and ref_seconds else None

            wav_out, gen_sr, _ = self.f5_engine.infer(
                ref_file=ref_file,
                ref_text=ref_text,
                gen_text=text,
                speed=float(speed),
                nfe_step=int(n_steps),
                cfg_strength=float(cfg_strength),
                fix_duration=fix_duration,
                seed=seed,
                show_info=lambda *a, **k: None
            )
            wav_out = np.ascontiguousarray(wav_out, dtype=np.float32)
            natural = len(wav_out) / gen_sr

            if fit_mode == "pad":
                wav_out = self._trim_silence(wav_out)

            # `+ fade`: o crossfade abaixo consome essas amostras na sobreposição
            # com o trecho anterior. Sem essa folga, cada emenda encurtaria a
            # saída e o erro acumularia, dessincronizando os outros falantes.
            usa_fade = bool(fade and pieces and len(pieces[-1]) >= fade)
            alvo = len(slot) + (fade if usa_fade else 0)
            fitted = self._fit_to_slot(wav_out, alvo, fit_mode)

            # Avisa quando o trecho original é curto demais para o que foi dito:
            # o resultado sai acelerado, e o caminho é ajustar a fronteira ou
            # usar --fit_mode pad.
            aperto = natural_est / slot_dur if slot_dur > 0 else 1.0
            apertado = aperto > 1.6 or aperto < 0.55
            if apertado:
                apertados += 1

            # Casa o volume com o do trecho original, senão a troca salta ao
            # ouvido numa conversa com níveis desiguais.
            orig_rms = float(np.sqrt(np.mean(slot ** 2)))
            gen_rms = float(np.sqrt(np.mean(fitted ** 2)))
            if gen_rms > 1e-6 and orig_rms > 1e-6:
                fitted = fitted * (orig_rms / gen_rms)
                peak = float(np.abs(fitted).max())
                if peak > 0.99:
                    fitted = fitted * (0.99 / peak)

            # Crossfade na emenda de entrada: a cauda do trecho anterior e a
            # cabeça do gerado ocupam o mesmo intervalo, então a soma ponderada
            # não muda a duração total.
            if usa_fade and len(fitted) > fade:
                ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                tail = pieces[-1][-fade:] * (1.0 - ramp) + fitted[:fade] * ramp
                pieces[-1] = np.concatenate([pieces[-1][:-fade], tail])
                fitted = fitted[fade:]

            # Emenda de saída: micro fade-out para o áudio original não entrar
            # com estalo. Não altera a duração.
            if fade and len(fitted) > fade:
                fitted = fitted.copy()
                fitted[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

            pieces.append(fitted)
            marca = "  <-- APERTADO" if apertado else ""
            print(f"[VoiceCloner]   {seg.start:7.2f}s-{seg.end:7.2f}s  "
                  f"(slot {slot_dur:5.2f}s / natural {natural_est:5.2f}s / x{aperto:4.2f})"
                  f"  '{text[:50]}'{marca}")
            report.append({
                "start": seg.start, "end": seg.end, "text": text,
                "slot_seconds": slot_dur, "natural_seconds": natural,
                "estimated_seconds": natural_est, "squeeze": aperto,
                "tight": apertado, "status": "substituido"
            })
            cursor = end

            if progress_cb is not None:
                progress_cb(len(report), len(targets), text)

        if cursor < len(audio):
            pieces.append(audio[cursor:].copy())

        result = np.concatenate(pieces) if pieces else audio
        peak = float(np.abs(result).max())
        if peak > 0.99:
            result = result * (0.99 / peak)

        if output_path:
            sf.write(str(output_path), result, sr)
            print(f"[VoiceCloner] Conversa remontada salva em: {output_path}")

        trocados = sum(1 for r in report if r["status"] == "substituido")
        motor = "Seed-VC (quadro a quadro)" if engine == "seedvc" else "F5-TTS (ASR->TTS)"
        print(f"[VoiceCloner] Concluido: {trocados}/{len(targets)} trechos de {speaker} trocados com {motor}.")
        if apertados:
            print(f"[VoiceCloner] Atencao: {apertados} trecho(s) marcados APERTADO — o tempo original nao "
                  f"comporta bem o texto falado, entao a fala sai acelerada ou arrastada. "
                  f"Considere --fit_mode pad para preservar o ritmo natural.")
        return result, sr, report

    def convert_voice(
        self,
        source_audio_path: str | Path,
        target_ref_audio: str | Path,
        target_ref_text: str = "",
        output_path: str | Path | None = None,
        speed: float = 1.0,
        n_steps: int = 32,
        cfg_strength: float = 2.0,
        solver: str = "euler",
        engine: str = "seedvc",      # 'seedvc' (quadro a quadro) ou 'f5' (ASR->TTS)
        f0_condition: bool = True,   # so no seedvc
        diffusion_steps: int = 25    # so no seedvc
    ) -> tuple[np.ndarray | torch.Tensor, int, str]:
        """
        MODO 2: Voice-to-Voice Conversion (Áudio de Origem -> Áudio com a Voz Alvo Clonada)
        Converte uma gravação de áudio de uma pessoa na voz da pessoa alvo.

        engine='seedvc' (padrão) faz voice conversion quadro a quadro: a saída
        tem o mesmo comprimento da entrada e mantém a entoação — não passa por
        texto, então nenhuma palavra se perde no caminho. É o caso de uso direto
        de quem vem do RVC ou So-VITS, e não exige diarizar nada.

        engine='f5' transcreve com o Whisper e ressintetiza com o F5-TTS. Só vale
        quando o objetivo é mudar o que é dito; a duração e a inflexão mudam.

        O terceiro item do retorno é o texto reconhecido — vazio no modo seedvc,
        que não transcreve.
        """
        source_p = Path(source_audio_path)
        target_p = Path(target_ref_audio)

        if not source_p.exists():
            raise FileNotFoundError(f"Áudio de origem '{source_audio_path}' não encontrado.")
        if not target_p.exists():
            raise FileNotFoundError(f"Áudio da voz alvo '{target_ref_audio}' não encontrado.")

        engine = (engine or "seedvc").lower()
        if engine not in ("f5", "seedvc"):
            raise ValueError(f"engine deve ser 'f5' ou 'seedvc', recebido '{engine}'.")

        print(f"[VoiceCloner] Conversao Audio -> Audio iniciada: '{source_p.name}' -> Voz '{target_p.name}' "
              f"(motor {'Seed-VC quadro a quadro' if engine == 'seedvc' else 'F5-TTS ASR->TTS'})...")

        if engine == "seedvc":
            # Caminho direto: nada de ASR, nada de diarizacao. Entra audio, sai
            # audio do mesmo tamanho com o timbre trocado.
            vc = self._get_vc_engine(f0_condition)
            wav_out, sr = vc.convert_file(
                source_p, target_p, diffusion_steps=diffusion_steps
            )
            if output_path:
                sf.write(str(output_path), wav_out, sr)
                print(f"[VoiceCloner] Audio convertido salvo em: {output_path}")
            try:
                dur_in = float(sf.info(str(source_p)).duration)
                print(f"[VoiceCloner] Duracao: entrada {dur_in:.2f}s -> saida {len(wav_out)/sr:.2f}s")
            except Exception:
                pass
            return wav_out, sr, ""

        # 1. Extrai o conteúdo falado do áudio de origem via Whisper ASR
        transcribed_text = self.transcribe_audio(source_p)
        if not transcribed_text:
            raise ValueError("Não foi possível reconhecer fala no áudio de origem. Verifique se o áudio contém voz nítida.")

        print(f"[VoiceCloner] Conteúdo detectado no áudio de origem: '{transcribed_text}'")

        # 2. Sintetiza a fala transcrita com a voz alvo clonada
        wav_out, sr = self.clone_voice(
            ref_audio_path=target_p,
            text=transcribed_text,
            ref_text=target_ref_text,
            output_path=output_path,
            speed=speed,
            n_steps=n_steps,
            cfg_strength=cfg_strength,
            solver=solver
        )

        return wav_out, sr, transcribed_text
