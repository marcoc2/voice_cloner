"""
Quick Demo Script:
Gera um dataset de demonstração na pasta `data/demo_speaker/`,
treina o modelo por algumas épocas e sintetiza uma frase clonada.
"""
import os
import sys
from pathlib import Path

# Garante acesso aos módulos do projeto
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import soundfile as sf

from training.train import Trainer
from inference.cloner import VoiceCloner
import yaml


def create_demo_data(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    sr = 24000
    
    # Frases de exemplo para o dataset de demonstração
    samples = [
        ("audio_01.wav", "Olá, bem-vindo ao novo clonador de voz baseado em Flow Matching e Diffusion Transformers."),
        ("audio_02.wav", "Em 2026 a síntese de voz alcançou alta fidelidade com vocoders neurais a vinte e quatro kilohertz."),
        ("audio_03.wav", "Este modelo suporta clonagem zero-shot com poucos segundos de áudio de referência."),
        ("audio_04.wav", "Treinamento rápido e convergência com Optimal Transport Conditional Flow Matching."),
    ]

    print(f"[Demo] Gerando {len(samples)} amostras de áudio de exemplo em '{data_dir}'...")
    for idx, (filename, text) in enumerate(samples):
        # Gera uma onda harmônica com variação de frequência simulando voz/tom
        duration = 3.5
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        base_freq = 150 + (idx * 20)
        # Tom fundamental + harmônicos com modulação suave
        signal = (
            0.4 * np.sin(2 * np.pi * base_freq * t) +
            0.2 * np.sin(2 * np.pi * (base_freq * 2) * t) +
            0.1 * np.sin(2 * np.pi * (base_freq * 3) * t)
        )
        # Envelope suave
        envelope = np.ones_like(signal)
        fade_len = int(0.1 * sr)
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)
        signal = signal * envelope

        sf.write(str(data_dir / filename), signal, sr)

        txt_file = data_dir / filename.replace(".wav", ".txt")
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(text)

    print("[Demo] Dados de demonstração criados com sucesso!")


def main():
    demo_dir = Path("data/demo_speaker")
    create_demo_data(demo_dir)

    print("\n[Demo] Carregando configurações...")
    with open("configs/default_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Configuração rápida para demo
    config["training"]["max_epochs"] = 10
    config["training"]["batch_size"] = 2
    config["training"]["save_every_epochs"] = 5

    print("\n[Demo] Iniciando treinamento rápido de demonstração (10 épocas)...")
    trainer = Trainer(config)
    trainer.fit(data_dir=demo_dir)

    print("\n[Demo] Realizando inferência de teste...")
    cloner = VoiceCloner(
        checkpoint_path="checkpoints/checkpoint_epoch_10.pt",
        config_path="configs/default_config.yaml"
    )

    out_file = "demo_output.wav"
    cloner.clone_voice(
        ref_audio_path=demo_dir / "audio_01.wav",
        text="A clonagem de voz moderna foi realizada com sucesso pelo pipeline de Flow Matching!",
        output_path=out_file,
        speed=1.0,
        n_steps=32,
        cfg_strength=2.0
    )

    print(f"\n[Sucesso] Demonstracao concluida! O audio sintetizado foi salvo em: {out_file}")


if __name__ == "__main__":
    main()
