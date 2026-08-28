import os
import sys
import shutil
import tempfile
from pathlib import Path

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torchaudio
import numpy as np

from models.dit import DiT
from models.text_encoder import TextEncoder, SimpleTokenizer
from models.flow_matching import ConditionalFlowMatching
from models.vocoder import VocoderWrapper
from dataset.preprocessor import AudioPreprocessor
from dataset.audio_dataset import VoiceCloningDataset, voice_cloning_collate_fn
from training.train import Trainer
from inference.cloner import VoiceCloner


def test_tokenizer_and_encoder():
    print("[-] Testando Tokenizer e TextEncoder...")
    tokenizer = SimpleTokenizer()
    text = "Olá mundo! Teste de clonagem de voz 2026."
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert len(tokens) > 0, "Token list está vazia!"
    print(f"    Texto original: '{text}'")
    print(f"    Tokens: {tokens[:10]}... (total {len(tokens)})")
    print(f"    Decodificado: '{decoded}'")

    encoder = TextEncoder(vocab_size=tokenizer.vocab_size, embed_dim=128, depth=2)
    x = torch.tensor([tokens, tokens], dtype=torch.long)
    out = encoder(x)
    assert out.shape == (2, len(tokens), 128)
    
    # Teste de alinhamento temporal
    aligned = encoder.align_to_mel_length(out, target_mel_len=150)
    assert aligned.shape == (2, 150, 128)
    print("    [OK] Tokenizer e TextEncoder passaram com sucesso!")


def test_dit_and_flow_matching():
    print("[-] Testando DiT e Conditional Flow Matching...")
    dit = DiT(dim=128, depth=2, heads=4, dim_head=32, mel_dim=100, text_dim=128)
    text_enc = TextEncoder(vocab_size=256, embed_dim=128, depth=2)
    cfm = ConditionalFlowMatching(transformer=dit, text_encoder=text_enc)

    B, T, mel_dim = 2, 64, 100
    target_mel = torch.randn(B, T, mel_dim)
    prompt_mel = torch.randn(B, T, mel_dim)
    text_tokens = torch.randint(1, 200, (B, 20))
    mask = torch.ones((B, T), dtype=torch.bool)

    # Teste de cálculo de perda
    loss_dict = cfm.compute_loss(target_mel, prompt_mel, text_tokens, mask)
    assert "loss" in loss_dict and loss_dict["loss"].item() > 0
    print(f"    Loss OT-CFM: {loss_dict['loss'].item():.4f}")

    # Teste de amostragem ODE Euler
    sampled_mel = cfm.sample(
        prompt_mel=prompt_mel[:1],
        text_tokens=text_tokens[:1],
        target_len=50,
        n_steps=4,
        device="cpu"
    )
    assert sampled_mel.shape == (1, 50, 100)
    print("    [OK] DiT e Flow Matching passaram com sucesso!")


def test_audio_preprocessor_and_dataset():
    print("[-] Testando AudioPreprocessor e Dataset...")
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        # Gera áudio senoidal sintético para teste
        sr = 24000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        sin_wave = 0.5 * np.sin(2 * np.pi * 440 * t)
        wav_tensor = torch.from_numpy(sin_wave).float().unsqueeze(0)

        audio_file = tmp_dir / "test_sample.wav"
        txt_file = tmp_dir / "test_sample.txt"
        import soundfile as sf
        sf.write(str(audio_file), sin_wave, sr)
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("Esta é uma amostra sintética de teste.")

        prep = AudioPreprocessor(sample_rate=sr, n_mels=100)
        loaded_wav, mel = prep.process_file(audio_file)
        assert mel.shape[-1] == 100
        print(f"    Áudio carregado: {loaded_wav.shape}, Mel: {mel.shape}")

        dataset = VoiceCloningDataset(data_dir=tmp_dir, sample_rate=sr)
        assert len(dataset) == 1
        item = dataset[0]
        assert "target_mel" in item and "text_tokens" in item

        batch = voice_cloning_collate_fn([item, item])
        assert batch["target_mel"].shape[0] == 2
        print("    [OK] Preprocessor e Dataset passaram com sucesso!")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_training_and_inference_pipeline():
    print("[-] Testando ciclo completo de Treino e Inferência...")
    tmp_dir = Path(tempfile.mkdtemp())
    ckpt_dir = tmp_dir / "checkpoints"
    try:
        sr = 24000
        t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
        sin_wave = 0.3 * np.sin(2 * np.pi * 220 * t)

        for i in range(2):
            import soundfile as sf
            sf.write(str(tmp_dir / f"sample_{i}.wav"), sin_wave, sr)
            with open(tmp_dir / f"sample_{i}.txt", "w", encoding="utf-8") as f:
                f.write(f"Frase de teste número {i}.")

        config = {
            "model": {
                "dim": 64,
                "depth": 2,
                "heads": 2,
                "dim_head": 32,
                "ff_mult": 2,
                "mel_dim": 100,
                "text_dim": 64,
                "vocab_size": 256,
                "dropout": 0.0,
                "sigma_min": 1e-4
            },
            "audio": {
                "sample_rate": 24000,
                "n_fft": 1024,
                "hop_length": 256,
                "win_length": 1024,
                "n_mels": 100,
                "f_min": 0,
                "f_max": 12000
            },
            "training": {
                "batch_size": 2,
                "learning_rate": 1e-3,
                "max_epochs": 2,
                "save_every_epochs": 1,
                "ema_decay": 0.9,
                "checkpoint_dir": str(ckpt_dir)
            },
            "inference": {
                "n_steps": 4,
                "cfg_strength": 1.5,
                "solver": "euler"
            }
        }

        # Executa 2 épocas de treino
        trainer = Trainer(config)
        trainer.fit(data_dir=tmp_dir)

        ckpt_file = ckpt_dir / "checkpoint_epoch_2.pt"
        assert ckpt_file.exists(), "Checkpoint não foi salvo!"
        print(f"    Checkpoint de teste gerado: {ckpt_file}")

        # Executa inferência com o checkpoint
        config_path = tmp_dir / "test_config.yaml"
        import yaml
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        cloner = VoiceCloner(checkpoint_path=ckpt_file, config_path=config_path)
        out_wav = tmp_dir / "output_cloned.wav"
        cloner.clone_voice(
            ref_audio_path=tmp_dir / "sample_0.wav",
            text="Testando clonagem bem-sucedida!",
            output_path=out_wav,
            n_steps=4
        )
        assert out_wav.exists() and out_wav.stat().st_size > 0
        print(f"    [OK] Áudio clonado gerado com sucesso: {out_wav.stat().st_size} bytes!")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("=== INICIANDO SUÍTE DE TESTES DO CLONADOR DE VOZ ===")
    test_tokenizer_and_encoder()
    test_dit_and_flow_matching()
    test_audio_preprocessor_and_dataset()
    test_training_and_inference_pipeline()
    print("=== TODOS OS TESTES PASSARAM COM SUCESSO! ===")
