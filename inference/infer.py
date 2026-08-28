import sys
import argparse
from pathlib import Path

# Garante acesso aos módulos do projeto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.cloner import VoiceCloner


def main():
    parser = argparse.ArgumentParser(description="CLI de Inferência e Conversão de Voz (Flow Matching)")
    parser.add_argument("--ref_audio", type=str, default="", help="Caminho para a voz alvo/referência a ser clonada")
    parser.add_argument("--character", type=str, default="",
                        help="Nome de um personagem da biblioteca (aba 👥 Personagens), em vez de --ref_audio. "
                             "Com vários áudios, eles são emendados numa referência única.")
    parser.add_argument("--list_characters", action="store_true",
                        help="Lista os personagens salvos na biblioteca e sai.")
    parser.add_argument("--ref_text", type=str, default="", help="Transcrição do áudio de referência da voz alvo (opcional)")
    parser.add_argument("--text", type=str, default="", help="Texto para a voz falar (Modo TTS)")
    parser.add_argument("--source_audio", type=str, default="", help="Áudio de origem a ser convertido (Modo Áudio para Áudio / Voice-to-Voice)")
    parser.add_argument("--output", type=str, default="output.wav", help="Caminho do arquivo WAV de saída")
    parser.add_argument("--checkpoint", type=str, default=None, help="Caminho opcional para checkpoint (.pt)")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Arquivo de configuração")
    parser.add_argument("--language", type=str, default="pt-br", choices=["pt-br", "en"], help="Idioma do modelo base ('pt-br' ou 'en')")
    parser.add_argument("--steps", type=int, default=32, help="Número de passos de integração ODE (16-64)")
    parser.add_argument("--cfg", type=float, default=2.0, help="Força do Classifier-Free Guidance (1.0-3.5)")
    parser.add_argument("--speed", type=float, default=1.0, help="Velocidade da fala (1.0 = normal)")
    parser.add_argument("--solver", type=str, default="euler", choices=["euler", "midpoint"], help="Solver ODE")
    parser.add_argument("--model_arch", type=str, default=None, choices=["F5TTS_Base", "F5TTS_v1_Base"],
                        help="Arquitetura do F5-TTS. Precisa bater com a do checkpoint (PT-BR usa F5TTS_Base). "
                             "Deixe vazio para a escolha automática.")

    # --- Conversa com varios falantes (diarizacao) --------------------------
    parser.add_argument("--list_speakers", action="store_true",
                        help="Só diariza o --source_audio: lista os falantes detectados e salva uma amostra "
                             "de cada um para você identificar quem é quem.")
    parser.add_argument("--speaker", type=str, default=None,
                        help="Troca a voz de UM falante da conversa, mantendo os demais. Passe o ID "
                             "(ex.: SPEAKER_01) ou 'auto' para usar quem mais fala.")
    parser.add_argument("--num_speakers", type=int, default=None, help="Força a quantidade exata de falantes")
    parser.add_argument("--min_speakers", type=int, default=None, help="Mínimo de falantes na busca")
    parser.add_argument("--max_speakers", type=int, default=None, help="Máximo de falantes na busca")
    parser.add_argument("--fit_mode", type=str, default="stretch", choices=["stretch", "pad"],
                        help="'stretch' encaixa a fala gerada no tempo original (mantém a sincronia); "
                             "'pad' mantém o ritmo natural e empurra a linha do tempo.")
    parser.add_argument("--samples_dir", type=str, default="speaker_samples",
                        help="Pasta onde salvar as amostras de --list_speakers")
    parser.add_argument("--seed", type=int, default=None,
                        help="Fixa a seed da geração. Sem ela cada rodada sorteia uma nova, "
                             "então um trecho ruim pode melhorar só rodando de novo. (só no motor f5)")
    parser.add_argument("--engine", type=str, default="seedvc", choices=["seedvc", "f5"],
                        help="Motor da troca de falante. 'seedvc' faz voice conversion quadro a quadro: "
                             "mesma duração, mesma inflexão, lip sync preservado (padrão). "
                             "'f5' passa por Whisper e ressintetiza — só use se quiser mudar o que é dito.")
    parser.add_argument("--no_f0", action="store_true",
                        help="Desliga o condicionamento de F0 do Seed-VC (22kHz em vez de 44.1kHz). "
                             "Preserva menos a entoação; use se o resultado com F0 sair instável.")
    parser.add_argument("--diffusion_steps", type=int, default=25,
                        help="Passos de difusão do Seed-VC (10-50). Mais passos = melhor e mais lento.")
    
    args = parser.parse_args()

    if args.list_characters:
        from inference.voice_library import VoiceLibrary
        biblioteca = VoiceLibrary()
        nomes = biblioteca.nomes()
        if not nomes:
            print("[CLI] Nenhum personagem salvo ainda. Crie na aba 'Personagens' da GUI.")
        else:
            print(f"[CLI] {len(nomes)} personagem(ns) na biblioteca:")
            for nome in nomes:
                print(f"  - {nome:24s} {biblioteca.resumo(nome)}")
        return

    # O personagem resolve para um arquivo de referencia; dai em diante o
    # pipeline nao sabe que a biblioteca existe.
    if args.character:
        from inference.voice_library import VoiceLibrary
        biblioteca = VoiceLibrary()
        if biblioteca.obter(args.character) is None:
            disponiveis = ", ".join(biblioteca.nomes()) or "(nenhum)"
            parser.error(f"personagem '{args.character}' nao encontrado. Disponiveis: {disponiveis}")
        try:
            args.ref_audio = biblioteca.referencia(args.character)
        except ValueError as e:
            parser.error(str(e))
        personagem = biblioteca.obter(args.character)
        if personagem is not None and personagem.ref_text and not args.ref_text:
            args.ref_text = personagem.ref_text
        print(f"[CLI] Personagem '{args.character}': {biblioteca.resumo(args.character)} -> {args.ref_audio}")

    if not args.ref_audio and not args.list_speakers:
        parser.error("--ref_audio (ou --character) e obrigatorio (exceto com --list_speakers)")

    # --list_speakers so diariza; carregar o F5-TTS seria desperdicio.
    cloner = None
    if not args.list_speakers:
        cloner = VoiceCloner(
            checkpoint_path=args.checkpoint,
            language=args.language,
            config_path=args.config,
            model_arch=args.model_arch
        )

    if args.list_speakers:
        if not args.source_audio:
            print("[ERRO] --list_speakers precisa de --source_audio.")
            return
        from inference.diarizer import SpeakerDiarizer
        diarizer = SpeakerDiarizer()
        segments = diarizer.diarize(
            args.source_audio,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers
        )
        SpeakerDiarizer.export_speaker_samples(args.source_audio, segments, args.samples_dir)
        print(f"[CLI] Ouca as amostras em '{args.samples_dir}' e rode de novo com --speaker <ID>.")

    elif args.speaker:
        print(f"[CLI] Modo Troca de Falante ativado em '{args.source_audio}'...")
        if not args.source_audio:
            print("[ERRO] --speaker precisa de --source_audio (a conversa completa).")
            return
        cloner.convert_speaker(
            source_audio_path=args.source_audio,
            target_ref_audio=args.ref_audio,
            speaker=None if args.speaker.lower() == "auto" else args.speaker,
            target_ref_text=args.ref_text,
            output_path=args.output,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            speed=args.speed,
            n_steps=args.steps,
            cfg_strength=args.cfg,
            fit_mode=args.fit_mode,
            seed=args.seed,
            engine=args.engine,
            f0_condition=not args.no_f0,
            diffusion_steps=args.diffusion_steps
        )

    elif args.source_audio:
        print(f"[CLI] Modo Audio -> Audio (Voice Conversion) ativado com origem: '{args.source_audio}'...")
        cloner.convert_voice(
            source_audio_path=args.source_audio,
            target_ref_audio=args.ref_audio,
            target_ref_text=args.ref_text,
            output_path=args.output,
            speed=args.speed,
            n_steps=args.steps,
            cfg_strength=args.cfg,
            solver=args.solver,
            engine=args.engine,
            f0_condition=not args.no_f0,
            diffusion_steps=args.diffusion_steps
        )
    elif args.text:
        print(f"[CLI] Modo Texto -> Audio (TTS) ativado...")
        cloner.clone_voice(
            ref_audio_path=args.ref_audio,
            text=args.text,
            ref_text=args.ref_text,
            output_path=args.output,
            speed=args.speed,
            n_steps=args.steps,
            cfg_strength=args.cfg,
            solver=args.solver
        )
    else:
        print("[ERRO] Forneca um texto com --text para TTS ou um audio de origem com --source_audio para conversao de voz.")


if __name__ == "__main__":
    main()
