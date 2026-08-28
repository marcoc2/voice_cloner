import os
import sys
from pathlib import Path

# Garante acesso aos módulos do projeto
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import gradio as gr
import torch
import soundfile as sf
import tempfile

from inference.cloner import VoiceCloner
from training.train import Trainer
import yaml

# Carrega configuração padrão
config_file = BASE_DIR / "configs" / "default_config.yaml"
with open(config_file, "r", encoding="utf-8") as f:
    default_config = yaml.safe_load(f)

# Instância global do cloner (lazy loaded)
cloner_instance = None


def get_cloner(checkpoint_path: str | None = None) -> VoiceCloner:
    global cloner_instance
    if cloner_instance is None or checkpoint_path:
        cloner_instance = VoiceCloner(
            checkpoint_path=checkpoint_path if checkpoint_path and Path(checkpoint_path).exists() else None
        )
    return cloner_instance


def synthesize_voice(
    audio_input,
    target_text: str,
    checkpoint_path: str,
    speed: float,
    n_steps: int,
    cfg_strength: float,
    solver: str
):
    if not audio_input:
        return None, "Erro: Por favor envie um áudio de referência ou grave sua voz pelo microfone."
    if not target_text or not target_text.strip():
        return None, "Erro: Digite o texto que a voz clonada deve falar."

    try:
        cloner = get_cloner(checkpoint_path if checkpoint_path and checkpoint_path.strip() else None)
        
        # Cria arquivo temporário de saída
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            output_file = tmp_out.name

        waveform, sr = cloner.clone_voice(
            ref_audio_path=audio_input,
            text=target_text.strip(),
            output_path=output_file,
            speed=speed,
            n_steps=int(n_steps),
            cfg_strength=float(cfg_strength),
            solver=solver
        )
        return output_file, f"Voz clonada e sintetizada com sucesso ({n_steps} passos ODE)!"
    except Exception as e:
        return None, f"Erro durante a síntese: {str(e)}"


def run_training_job(
    data_directory: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    resume_checkpoint: str
):
    if not data_directory or not Path(data_directory).exists():
        return f"Erro: O diretório de dados '{data_directory}' não existe."

    try:
        config = default_config.copy()
        config["training"]["max_epochs"] = int(epochs)
        config["training"]["batch_size"] = int(batch_size)
        config["training"]["learning_rate"] = float(learning_rate)

        trainer = Trainer(config)
        if resume_checkpoint and Path(resume_checkpoint).exists():
            trainer.load_checkpoint(resume_checkpoint)

        trainer.fit(data_dir=data_directory)
        return f"Treinamento concluído com sucesso por {epochs} épocas! Checkpoints salvos na pasta 'checkpoints/'."
    except Exception as e:
        return f"Erro durante o treinamento: {str(e)}"


def create_ui():
    custom_css = """
    .gradio-container { max-width: 950px !important; margin: 0 auto !important; }
    .header-box { text-align: center; margin-bottom: 20px; }
    .status-box { font-weight: bold; }
    """

    with gr.Blocks(title="Clonador de Voz Moderno (Flow Matching DiT)", css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎙️ Clonador de Voz Moderno
            ### Arquitetura Flow Matching DiT (Diffusion Transformer) com Síntese Zero-Shot & Fine-Tuning
            """,
            elem_classes="header-box"
        )

        with gr.Tabs():
            # Aba 1: Inferência / Síntese
            with gr.TabItem("✨ Clonar Voz (Inferência Zero-Shot)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 1. Voz de Referência (3 a 10 segundos)")
                        audio_input = gr.Audio(
                            label="Envie um arquivo de áudio ou grave pelo microfone",
                            type="filepath"
                        )
                        
                        gr.Markdown("### 2. Texto Desejado")
                        text_input = gr.Textbox(
                            label="Texto que a voz clonada deve falar",
                            placeholder="Olá! Este é um exemplo de clonagem de voz moderna utilizando Flow Matching e Diffusion Transformers.",
                            lines=4
                        )

                        with gr.Accordion("⚙️ Parâmetros Avançados", open=False):
                            checkpoint_input = gr.Textbox(
                                label="Caminho para Checkpoint Treinado (Opcional)",
                                placeholder="ex: checkpoints/best_model.pt"
                            )
                            speed_slider = gr.Slider(
                                label="Velocidade da Fala",
                                minimum=0.5,
                                maximum=2.0,
                                value=1.0,
                                step=0.1
                            )
                            steps_slider = gr.Slider(
                                label="Passos de Integração ODE",
                                minimum=8,
                                maximum=64,
                                value=32,
                                step=4
                            )
                            cfg_slider = gr.Slider(
                                label="Classifier-Free Guidance (CFG)",
                                minimum=1.0,
                                maximum=4.0,
                                value=2.0,
                                step=0.2
                            )
                            solver_dropdown = gr.Dropdown(
                                label="Solver ODE",
                                choices=["euler", "midpoint"],
                                value="euler"
                            )

                        synthesize_btn = gr.Button("🚀 Clonar e Sintetizar Voz", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("### 3. Áudio Gerado")
                        audio_output = gr.Audio(label="Resultado da Clonagem", type="filepath")
                        status_output = gr.Textbox(label="Status da Execução", interactive=False)

                synthesize_btn.click(
                    fn=synthesize_voice,
                    inputs=[
                        audio_input,
                        text_input,
                        checkpoint_input,
                        speed_slider,
                        steps_slider,
                        cfg_slider,
                        solver_dropdown
                    ],
                    outputs=[audio_output, status_output]
                )

            # Aba 2: Treinamento / Fine-Tuning
            with gr.TabItem("🏋️ Treinamento / Fine-Tuning"):
                gr.Markdown("### Treinar ou Refinar o Modelo com Seu Próprio Conjunto de Vozes")
                with gr.Row():
                    with gr.Column():
                        dataset_dir_input = gr.Textbox(
                            label="Diretório do Dataset de Áudio",
                            placeholder="ex: data/minhas_vozes/ ou f:/meu_dataset",
                            value="data"
                        )
                        epochs_input = gr.Number(label="Número de Épocas", value=20, precision=0)
                        batch_size_input = gr.Number(label="Batch Size", value=8, precision=0)
                        lr_input = gr.Number(label="Taxa de Aprendizado (LR)", value=0.0002)
                        resume_ckpt_input = gr.Textbox(
                            label="Continuar de Checkpoint (Opcional)",
                            placeholder="ex: checkpoints/checkpoint_epoch_10.pt"
                        )

                        train_btn = gr.Button("⚡ Iniciar Treinamento", variant="secondary")

                    with gr.Column():
                        training_log_output = gr.Textbox(
                            label="Status do Treinamento",
                            lines=10,
                            interactive=False
                        )

                train_btn.click(
                    fn=run_training_job,
                    inputs=[
                        dataset_dir_input,
                        epochs_input,
                        batch_size_input,
                        lr_input,
                        resume_ckpt_input
                    ],
                    outputs=[training_log_output]
                )

            # Aba 3: Informações e Guia
            with gr.TabItem("📖 Informações e Arquitetura"):
                gr.Markdown(
                    """
                    ### Sobre a Arquitetura Flow Matching DiT
                    
                    Ao contrário de métodos mais antigos baseados puramente em VITS ou RVC:
                    
                    - **Continuous Normalizing Flows (CNF)**: Modela o caminho ideal (Optimal Transport) entre uma distribuição gaussiana simples e o espectrograma de áudio.
                    - **Diffusion Transformer (DiT)**: Blocos Transformer com modulação AdaLN que aprendem a prever o vetor do campo de velocidade acústico.
                    - **Zero-Shot Real**: A voz de referência é injetada diretamente como condicionamento, permitindo replicar timbres instantaneamente.
                    - **Vocoder Neural de 24kHz**: Reconstrução rápida da forma de onda a 24.000 Hz.
                    """
                )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)
