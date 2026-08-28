import os
import sys
import time
import winsound
from pathlib import Path

# Garante inclusão do diretório raiz no PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import yaml
import torch
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QIcon, QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QLineEdit, QTextEdit,
    QFileDialog, QSlider, QSpinBox, QDoubleSpinBox, QComboBox,
    QProgressBar, QGroupBox, QGridLayout, QMessageBox, QFrame, QSplitter,
    QListWidget, QListWidgetItem, QScrollArea, QSizePolicy
)

from inference.cloner import VoiceCloner
from inference.voice_library import VoiceLibrary
from training.train import Trainer, EMA
from dataset.audio_dataset import VoiceCloningDataset, voice_cloning_collate_fn
from torch.utils.data import DataLoader


DARK_STYLESHEET = """
QMainWindow {
    background-color: #121214;
}
QWidget {
    background-color: #121214;
    color: #E1E1E6;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #29292E;
    background-color: #18181B;
    border-radius: 8px;
    top: -1px;
}
QTabBar::tab {
    background-color: #1F1F23;
    color: #A1A1AA;
    padding: 10px 18px;
    margin-right: 4px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: bold;
}
QTabBar::tab:selected {
    background-color: #27272A;
    color: #00E676;
    border-bottom: 2px solid #00E676;
}
QTabBar::tab:hover {
    color: #FFFFFF;
}
QGroupBox {
    background-color: #18181B;
    border: 1px solid #29292E;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 14px;
    font-weight: bold;
    color: #00E676;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}
QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #202024;
    border: 1px solid #323238;
    border-radius: 6px;
    padding: 8px;
    color: #FFFFFF;
}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #00E676;
}
QListWidget {
    background-color: #202024;
    border: 1px solid #323238;
    border-radius: 6px;
    color: #FFFFFF;
    padding: 2px;
}
QListWidget::item {
    padding: 5px 6px;
    border-radius: 4px;
}
QListWidget::item:selected {
    background-color: #00E676;
    color: #121214;
    font-weight: bold;
}
QListWidget::item:hover:!selected {
    background-color: #29292E;
}
QScrollArea {
    background-color: transparent;
    border: none;
}
QScrollArea > QWidget > QWidget {
    background-color: transparent;
}
QScrollBar:vertical {
    background-color: #18181B;
    width: 10px;
    margin: 0;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background-color: #3F3F46;
    min-height: 30px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background-color: #52525B;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}
QPushButton {
    background-color: #29292E;
    border: 1px solid #3F3F46;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
    color: #FFFFFF;
}
QPushButton:hover {
    background-color: #3F3F46;
    border-color: #52525B;
}
QPushButton:pressed {
    background-color: #18181B;
}
QPushButton#primaryBtn {
    background-color: #00C853;
    color: #000000;
    border: none;
    font-size: 14px;
    padding: 12px;
}
QPushButton#primaryBtn:hover {
    background-color: #00E676;
}
QPushButton#primaryBtn:disabled {
    background-color: #2E5C38;
    color: #888888;
}
QProgressBar {
    background-color: #202024;
    border: 1px solid #323238;
    border-radius: 6px;
    text-align: center;
    color: #FFFFFF;
    font-weight: bold;
}
QProgressBar::chunk {
    background-color: #00E676;
    border-radius: 5px;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #27272A;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #00E676;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #FFFFFF;
    border: 1px solid #00E676;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}
"""


class InferenceWorker(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, cloner: VoiceCloner, ref_audio: str, text: str, ref_text: str, output_path: str, speed: float, n_steps: int, cfg: float, solver: str):
        super().__init__()
        self.cloner = cloner
        self.ref_audio = ref_audio
        self.text = text
        self.ref_text = ref_text
        self.output_path = output_path
        self.speed = speed
        self.n_steps = n_steps
        self.cfg = cfg
        self.solver = solver

    def run(self):
        try:
            self.progress_signal.emit(f"Sintetizando voz com Foundation Model...")
            waveform, sr = self.cloner.clone_voice(
                ref_audio_path=self.ref_audio,
                text=self.text,
                ref_text=self.ref_text,
                output_path=self.output_path,
                speed=self.speed,
                n_steps=self.n_steps,
                cfg_strength=self.cfg,
                solver=self.solver
            )
            self.finished_signal.emit(str(self.output_path))
        except Exception as e:
            self.error_signal.emit(str(e))


class VoiceConversionWorker(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str, str)
    error_signal = pyqtSignal(str)

    def __init__(self, cloner: VoiceCloner, source_audio: str, target_audio: str, target_ref_text: str, output_path: str, speed: float, n_steps: int, cfg: float, solver: str, engine: str = "seedvc", diffusion_steps: int = 25):
        super().__init__()
        self.cloner = cloner
        self.source_audio = source_audio
        self.target_audio = target_audio
        self.target_ref_text = target_ref_text
        self.output_path = output_path
        self.speed = speed
        self.n_steps = n_steps
        self.cfg = cfg
        self.solver = solver
        self.engine = engine
        self.diffusion_steps = diffusion_steps

    def run(self):
        try:
            if self.engine == "seedvc":
                self.progress_signal.emit(
                    "Convertendo a voz quadro a quadro com o Seed-VC "
                    "(na primeira vez baixa os modelos, pode levar alguns minutos)..."
                )
            else:
                self.progress_signal.emit("Reconhecendo fala do áudio de origem com Whisper ASR...")
            wav_out, sr, text = self.cloner.convert_voice(
                source_audio_path=self.source_audio,
                target_ref_audio=self.target_audio,
                target_ref_text=self.target_ref_text,
                output_path=self.output_path,
                speed=self.speed,
                n_steps=self.n_steps,
                cfg_strength=self.cfg,
                solver=self.solver,
                engine=self.engine,
                diffusion_steps=self.diffusion_steps
            )
            self.finished_signal.emit(str(self.output_path), text)
        except Exception as e:
            self.error_signal.emit(str(e))


class DiarizationWorker(QThread):
    """Descobre os falantes da conversa e salva uma amostra de cada um."""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)

    def __init__(self, source_audio: str, samples_dir: str, num_speakers: int | None,
                 min_speakers: int | None, max_speakers: int | None):
        super().__init__()
        self.source_audio = source_audio
        self.samples_dir = samples_dir
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def run(self):
        try:
            self.progress_signal.emit("Carregando modelo de diarização (pyannote)...")
            from inference.diarizer import SpeakerDiarizer

            diarizer = SpeakerDiarizer()
            self.progress_signal.emit("Analisando a conversa e separando os falantes...")
            segments = diarizer.diarize(
                self.source_audio,
                num_speakers=self.num_speakers,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers
            )
            if not segments:
                self.error_signal.emit("Nenhuma fala foi detectada neste áudio.")
                return

            self.progress_signal.emit("Salvando amostras de cada falante...")
            samples = SpeakerDiarizer.export_speaker_samples(
                self.source_audio, segments, self.samples_dir
            )

            total = sum(seg.duration for seg in segments)
            resumo = []
            for spk in SpeakerDiarizer.speakers(segments):
                spk_segs = [x for x in segments if x.speaker == spk]
                dur = sum(x.duration for x in spk_segs)
                resumo.append({
                    "speaker": spk,
                    "seconds": dur,
                    "share": (dur / total * 100) if total else 0.0,
                    "count": len(spk_segs),
                    "sample": samples.get(spk, "")
                })
            self.finished_signal.emit(resumo)
        except Exception as e:
            self.error_signal.emit(str(e))


class SpeakerSwapWorker(QThread):
    """Troca a voz de um único falante, mantendo os demais intactos."""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str, list)
    error_signal = pyqtSignal(str)

    def __init__(self, cloner: VoiceCloner, source_audio: str, target_audio: str,
                 speaker: str, target_ref_text: str, output_path: str,
                 num_speakers: int | None, min_speakers: int | None, max_speakers: int | None,
                 n_steps: int, fit_mode: str, seed: int | None,
                 engine: str = "seedvc", f0_condition: bool = True, diffusion_steps: int = 25):
        super().__init__()
        self.cloner = cloner
        self.source_audio = source_audio
        self.target_audio = target_audio
        self.speaker = speaker
        self.target_ref_text = target_ref_text
        self.output_path = output_path
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.n_steps = n_steps
        self.fit_mode = fit_mode
        self.seed = seed
        self.engine = engine
        self.f0_condition = f0_condition
        self.diffusion_steps = diffusion_steps

    def run(self):
        try:
            if self.engine == "seedvc":
                self.progress_signal.emit(
                    "Carregando o Seed-VC (na primeira vez baixa os modelos, pode levar alguns minutos)..."
                )
            self.progress_signal.emit(f"Diarizando e localizando os trechos de {self.speaker}...")

            def on_progress(feito, total, texto):
                trecho = f' — "{texto[:40]}"' if texto else ""
                self.progress_signal.emit(f"Convertendo trecho {feito}/{total}{trecho}")

            _, _, report = self.cloner.convert_speaker(
                source_audio_path=self.source_audio,
                target_ref_audio=self.target_audio,
                speaker=self.speaker,
                target_ref_text=self.target_ref_text,
                output_path=self.output_path,
                num_speakers=self.num_speakers,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
                n_steps=self.n_steps,
                fit_mode=self.fit_mode,
                seed=self.seed,
                engine=self.engine,
                f0_condition=self.f0_condition,
                diffusion_steps=self.diffusion_steps
            )
            self.finished_signal.emit(str(self.output_path), report)
        except Exception as e:
            self.error_signal.emit(str(e))


class TrainingWorker(QThread):
    epoch_signal = pyqtSignal(int, int, float)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, data_dir: str, epochs: int, batch_size: int, lr: float, resume_ckpt: str | None = None):
        super().__init__()
        self.data_dir = data_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.resume_ckpt = resume_ckpt
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            config_path = BASE_DIR / "configs" / "default_config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            config["training"]["max_epochs"] = self.epochs
            config["training"]["batch_size"] = self.batch_size
            config["training"]["learning_rate"] = self.lr

            self.log_signal.emit(f"[Treino] Inicializando Trainer com PyTorch em {self.data_dir}...")
            trainer = Trainer(config)

            if self.resume_ckpt and Path(self.resume_ckpt).exists():
                self.log_signal.emit(f"[Treino] Carregando checkpoint: {self.resume_ckpt}")
                trainer.load_checkpoint(self.resume_ckpt)

            dataset = VoiceCloningDataset(
                data_dir=self.data_dir,
                sample_rate=config["audio"]["sample_rate"]
            )
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=voice_cloning_collate_fn,
                num_workers=0
            )

            max_epochs = self.epochs
            save_every = config["training"].get("save_every_epochs", 5)

            for epoch in range(trainer.start_epoch, max_epochs):
                if not self._is_running:
                    self.log_signal.emit("[Treino] Treinamento interrompido pelo usuário.")
                    break

                self.log_signal.emit(f"[Treino] Executando Época {epoch + 1}/{max_epochs}...")
                avg_loss = trainer.train_epoch(dataloader, epoch)
                self.epoch_signal.emit(epoch + 1, max_epochs, avg_loss)
                self.log_signal.emit(f"[Treino] Época {epoch + 1}/{max_epochs} - Loss Média: {avg_loss:.4f}")

                if (epoch + 1) % save_every == 0 or (epoch + 1) == max_epochs:
                    trainer.save_checkpoint(epoch)

            self.finished_signal.emit("Treinamento finalizado com sucesso! Checkpoints salvos.")
        except Exception as e:
            self.error_signal.emit(str(e))


class VoiceClonerMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎙️ Clonador de Voz Moderno (Flow Matching DiT)")
        # 980px e a altura em que a aba de Troca de Falante cabe inteira sem
        # rolagem; fit_to_screen() reduz depois se o monitor for menor, e a area
        # rolavel cobre o resto.
        self.resize(1020, 980)
        self.setMinimumSize(880, 600)
        self.setStyleSheet(DARK_STYLESHEET)

        self.cloner = None
        self.last_generated_audio = None
        self.last_converted_audio = None
        self.last_swapped_audio = None
        self.speaker_samples = {}
        self.library = VoiceLibrary()
        # Combos de personagem espalhados pelas abas, para atualizar todos de
        # uma vez quando a biblioteca mudar.
        self._combos_personagem = []
        self.init_ui()
        self.check_gpu_status()
        self.fit_to_screen()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Header Title
        header_layout = QHBoxLayout()
        title_lbl = QLabel("🎙️ Clonador de Voz Moderno (TTS & Áudio ➔ Áudio)")
        title_lbl.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title_lbl.setStyleSheet("color: #00E676;")
        
        self.gpu_badge = QLabel("🖥️ Dispositivo: Detectando...")
        self.gpu_badge.setStyleSheet("color: #A1A1AA; font-size: 12px; padding: 4px 8px; background-color: #202024; border-radius: 4px;")
        
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()
        header_layout.addWidget(self.gpu_badge)
        main_layout.addLayout(header_layout)

        # Tab Widget
        self.tabs = QTabWidget()
        self.tab_inference = QWidget()
        self.tab_vc = QWidget()
        self.tab_swap = QWidget()
        self.tab_characters = QWidget()
        self.tab_training = QWidget()
        self.tab_about = QWidget()

        self.setup_inference_tab()
        self.setup_vc_tab()
        self.setup_swap_tab()
        self.setup_characters_tab()
        self.setup_training_tab()
        self.setup_about_tab()

        self.tabs.addTab(self.tab_inference, "💬 Texto para Voz (TTS)")
        self.tabs.addTab(self.tab_vc, "🔄 Áudio para Áudio (Voice-to-Voice)")
        self.tabs.addTab(self.tab_swap, "🎭 Troca de Falante (Conversa)")
        self.tabs.addTab(self.tab_characters, "👥 Personagens")
        self.tabs.addTab(self.tab_training, "🏋️ Treinamento / Fine-Tuning")
        self.tabs.addTab(self.tab_about, "ℹ️ Informações & Status")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        main_layout.addWidget(self.tabs)

        # Status Bar Inferior
        self.status_bar = QLabel("Pronto para clonar e converter vozes.")
        self.status_bar.setStyleSheet("color: #71717A; font-size: 11px; padding: 4px;")
        main_layout.addWidget(self.status_bar)

    def on_tab_changed(self, indice: int):
        """
        Ao sair da aba de Personagens, os seletores das outras abas precisam
        refletir o que foi criado ou removido lá.
        """
        if self.tabs.widget(indice) is not self.tab_characters:
            self.refresh_character_combos()

    def fit_to_screen(self):
        """
        Nunca abrir maior que o monitor: a janela pede 880px de altura para a
        aba de Troca de Falante caber sem rolagem, o que não existe em telas
        pequenas ou com escala alta do Windows.
        """
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        disponivel = screen.availableGeometry()
        largura = min(self.width(), disponivel.width() - 40)
        altura = min(self.height(), disponivel.height() - 60)
        self.setMinimumSize(min(880, largura), min(600, altura))
        self.resize(largura, altura)
        moldura = self.frameGeometry()
        moldura.moveCenter(disponivel.center())
        self.move(moldura.topLeft())

    def check_gpu_status(self):
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            self.gpu_badge.setText(f"🚀 GPU: {gpu_name}")
            self.gpu_badge.setStyleSheet("color: #00E676; font-size: 12px; padding: 4px 8px; background-color: #202024; border-radius: 4px;")
        else:
            self.gpu_badge.setText("🖥️ Dispositivo: CPU")

    def _linha_personagem(self, campo_arquivo: QLineEdit, campo_texto: QLineEdit | None = None) -> QHBoxLayout:
        """
        Monta a linha "Personagem: [combo]" que aparece acima do campo de
        arquivo nas abas que pedem uma voz de referência.

        Escolher um personagem resolve a referência dele (juntando os áudios, se
        houver mais de um) e preenche o campo de arquivo — assim o resto do
        pipeline continua recebendo um caminho, sem saber da biblioteca.
        """
        linha = QHBoxLayout()
        linha.addWidget(QLabel("Personagem:"))

        combo = QComboBox()
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(14)
        combo.setToolTip("Vozes salvas na aba 👥 Personagens. Escolher uma preenche o arquivo abaixo.")
        linha.addWidget(combo, 1)

        aviso = QLabel("")
        aviso.setStyleSheet("color: #71717A; font-size: 11px; font-weight: normal;")
        linha.addWidget(aviso, 2)

        def ao_escolher():
            nome = combo.currentData()
            if not nome:
                aviso.setText("")
                return
            try:
                caminho = self.library.referencia(nome)
            except ValueError as e:
                aviso.setText("⚠️ sem áudio válido")
                aviso.setStyleSheet("color: #FFB020; font-size: 11px; font-weight: bold;")
                QMessageBox.warning(self, "Personagem sem áudio", str(e))
                combo.setCurrentIndex(0)
                return
            campo_arquivo.setText(caminho)
            aviso.setText(f"✓ {self.library.resumo(nome)}")
            aviso.setStyleSheet("color: #00E676; font-size: 11px; font-weight: normal;")
            personagem = self.library.obter(nome)
            if campo_texto is not None and personagem is not None and personagem.ref_text:
                campo_texto.setText(personagem.ref_text)

        combo.currentIndexChanged.connect(ao_escolher)
        self._combos_personagem.append((combo, aviso))
        return linha

    def refresh_character_combos(self):
        """Repopula os seletores de personagem sem perder a escolha atual."""
        nomes = self.library.nomes()
        for combo, aviso in self._combos_personagem:
            anterior = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("— escolher arquivo manualmente —", None)
            for nome in nomes:
                combo.addItem(nome, nome)
            indice = combo.findData(anterior) if anterior else 0
            combo.setCurrentIndex(indice if indice >= 0 else 0)
            combo.blockSignals(False)
            if indice <= 0:
                aviso.setText("")

    def setup_inference_tab(self):
        layout = QVBoxLayout(self.tab_inference)
        layout.setSpacing(12)

        # Grupo 1: Áudio de Referência
        ref_group = QGroupBox("1. Áudio de Referência (Voz a ser clonada - 3 a 10s)")
        ref_vlayout = QVBoxLayout(ref_group)
        
        ref_hlayout = QHBoxLayout()
        self.ref_path_edit = QLineEdit()
        self.ref_path_edit.setPlaceholderText("Selecione um arquivo de áudio (.wav, .mp3, .flac)...")
        ref_btn = QPushButton("📁 Escolher Áudio")
        ref_btn.clicked.connect(lambda: self.select_audio_file(self.ref_path_edit))
        self.play_ref_btn = QPushButton("▶️ Ouvir")
        self.play_ref_btn.clicked.connect(lambda: self.play_audio(self.ref_path_edit.text()))
        ref_hlayout.addWidget(self.ref_path_edit)
        ref_hlayout.addWidget(ref_btn)
        ref_hlayout.addWidget(self.play_ref_btn)
        ref_vlayout.addLayout(ref_hlayout)

        self.ref_text_edit = QLineEdit()
        self.ref_text_edit.setPlaceholderText("O que a pessoa fala no áudio de referência? (Opcional, deixe vazio para automático)")
        ref_vlayout.addWidget(self.ref_text_edit)

        ref_vlayout.insertLayout(0, self._linha_personagem(self.ref_path_edit, self.ref_text_edit))
        layout.addWidget(ref_group)

        # Grupo 2: Texto Desejado
        text_group = QGroupBox("2. Texto a ser falado com a voz clonada")
        text_layout = QVBoxLayout(text_group)
        self.target_text_edit = QTextEdit()
        self.target_text_edit.setPlaceholderText("Digite aqui o texto que você deseja que a voz clonada fale...")
        self.target_text_edit.setText("Olá! Agora estou falando em português brasileiro nativo, sem sotaque e com clareza perfeita.")
        self.target_text_edit.setMaximumHeight(85)
        text_layout.addWidget(self.target_text_edit)
        layout.addWidget(text_group)

        # Grupo 3: Parâmetros
        params_group = QGroupBox("3. Parâmetros de Inferência")
        grid = QGridLayout(params_group)

        grid.addWidget(QLabel("Modelo / Idioma:"), 0, 0)
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("🇧🇷 Português Brasileiro (PT-BR) [Nativo / Sem Sotaque]", "pt-br")
        self.lang_combo.addItem("🇺🇸 Inglês / Multilíngue (EN Base)", "en")
        grid.addWidget(self.lang_combo, 0, 1, 1, 2)

        grid.addWidget(QLabel("Checkpoint (.pt opcional):"), 1, 0)
        self.ckpt_path_edit = QLineEdit()
        self.ckpt_path_edit.setPlaceholderText("Opcional: checkpoints/best_model.pt (deixe vazio para modelo nativo)")
        ckpt_btn = QPushButton("📁 Procurar")
        ckpt_btn.clicked.connect(self.select_checkpoint)
        grid.addWidget(self.ckpt_path_edit, 1, 1)
        grid.addWidget(ckpt_btn, 1, 2)

        grid.addWidget(QLabel("Velocidade:"), 2, 0)
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.5, 2.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(1.0)
        grid.addWidget(self.speed_spin, 2, 1)

        grid.addWidget(QLabel("Passos ODE:"), 2, 2)
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(8, 128)
        self.steps_spin.setSingleStep(4)
        self.steps_spin.setValue(32)
        grid.addWidget(self.steps_spin, 2, 3)

        layout.addWidget(params_group)

        # Barra de Progresso
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Botão Principal
        self.synthesize_btn = QPushButton("🚀 Clonar e Sintetizar Voz (TTS)")
        self.synthesize_btn.setObjectName("primaryBtn")
        self.synthesize_btn.clicked.connect(self.start_synthesis)
        layout.addWidget(self.synthesize_btn)

        # Resultado
        res_group = QGroupBox("4. Áudio Gerado")
        res_layout = QHBoxLayout(res_group)
        self.res_label = QLabel("Nenhum áudio gerado ainda.")
        self.res_label.setStyleSheet("color: #A1A1AA;")
        self.play_res_btn = QPushButton("▶️ Reproduzir Áudio")
        self.play_res_btn.setEnabled(False)
        self.play_res_btn.clicked.connect(lambda: self.play_audio(self.last_generated_audio))
        self.save_res_btn = QPushButton("💾 Salvar Como...")
        self.save_res_btn.setEnabled(False)
        self.save_res_btn.clicked.connect(lambda: self.save_audio_file(self.last_generated_audio))

        res_layout.addWidget(self.res_label, 1)
        res_layout.addWidget(self.play_res_btn)
        res_layout.addWidget(self.save_res_btn)
        layout.addWidget(res_group)

    def setup_vc_tab(self):
        layout = QVBoxLayout(self.tab_vc)
        layout.setSpacing(12)

        # Grupo 1: Áudio de Origem (Source Audio)
        src_group = QGroupBox("1. Áudio de Origem (A gravação ou fala que você quer converter)")
        src_layout = QHBoxLayout(src_group)
        self.vc_src_edit = QLineEdit()
        self.vc_src_edit.setPlaceholderText("Selecione o áudio original que será transformado...")
        src_btn = QPushButton("📁 Escolher Áudio de Origem")
        src_btn.clicked.connect(lambda: self.select_audio_file(self.vc_src_edit))
        self.vc_play_src_btn = QPushButton("▶️ Ouvir Origem")
        self.vc_play_src_btn.clicked.connect(lambda: self.play_audio(self.vc_src_edit.text()))
        src_layout.addWidget(self.vc_src_edit)
        src_layout.addWidget(src_btn)
        src_layout.addWidget(self.vc_play_src_btn)
        layout.addWidget(src_group)

        # Grupo 2: Voz Alvo (Target Voice)
        tgt_group = QGroupBox("2. Voz Alvo de Destino (A pessoa cuja voz será o resultado final - 3 a 10s)")
        tgt_vlayout = QVBoxLayout(tgt_group)
        tgt_hlayout = QHBoxLayout()
        self.vc_tgt_edit = QLineEdit()
        self.vc_tgt_edit.setPlaceholderText("Selecione a amostra da voz que você quer clonar...")
        tgt_btn = QPushButton("📁 Escolher Voz Alvo")
        tgt_btn.clicked.connect(lambda: self.select_audio_file(self.vc_tgt_edit))
        self.vc_play_tgt_btn = QPushButton("▶️ Ouvir Alvo")
        self.vc_play_tgt_btn.clicked.connect(lambda: self.play_audio(self.vc_tgt_edit.text()))
        tgt_hlayout.addWidget(self.vc_tgt_edit)
        tgt_hlayout.addWidget(tgt_btn)
        tgt_hlayout.addWidget(self.vc_play_tgt_btn)
        tgt_vlayout.addLayout(tgt_hlayout)

        self.vc_tgt_text_edit = QLineEdit()
        self.vc_tgt_text_edit.setPlaceholderText("O que a pessoa alvo fala no áudio de exemplo? (Opcional)")
        tgt_vlayout.addWidget(self.vc_tgt_text_edit)

        tgt_vlayout.insertLayout(0, self._linha_personagem(self.vc_tgt_edit, self.vc_tgt_text_edit))
        layout.addWidget(tgt_group)

        # Parâmetros VC
        vc_params_group = QGroupBox("3. Parâmetros de Conversão")
        vc_grid = QGridLayout(vc_params_group)

        vc_grid.addWidget(QLabel("Motor:"), 0, 0)
        self.vc_engine_combo = QComboBox()
        self.vc_engine_combo.addItem("🎯 Seed-VC — mesma duração e inflexão", "seedvc")
        self.vc_engine_combo.addItem("💬 F5-TTS — refala o texto reconhecido", "f5")
        self.vc_engine_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.vc_engine_combo.setMinimumContentsLength(16)
        self.vc_engine_combo.setToolTip(
            "Seed-VC: converte quadro a quadro, sem passar por texto. A saída tem o mesmo\n"
            "tamanho da entrada e mantém a entoação — nenhuma palavra se perde.\n\n"
            "F5-TTS: transcreve com o Whisper e refala. Muda a duração e a inflexão, e o que\n"
            "o ASR não ouvir some. Use só quando quiser alterar o que é dito."
        )
        self.vc_engine_combo.currentIndexChanged.connect(self.on_vc_engine_changed)
        vc_grid.addWidget(self.vc_engine_combo, 0, 1)

        self.vc_diffusion_lbl = QLabel("Difusão:")
        vc_grid.addWidget(self.vc_diffusion_lbl, 0, 2)
        self.vc_diffusion_spin = QSpinBox()
        self.vc_diffusion_spin.setRange(4, 100)
        self.vc_diffusion_spin.setValue(25)
        self.vc_diffusion_spin.setToolTip("Passos de difusão do Seed-VC. Mais = melhor e mais lento.")
        vc_grid.addWidget(self.vc_diffusion_spin, 0, 3)

        self.vc_lang_lbl = QLabel("Modelo Base:")
        vc_grid.addWidget(self.vc_lang_lbl, 1, 0)
        self.vc_lang_combo = QComboBox()
        self.vc_lang_combo.addItem("🇧🇷 Português Brasileiro (PT-BR) [Nativo]", "pt-br")
        self.vc_lang_combo.addItem("🇺🇸 Inglês / Multilíngue", "en")
        vc_grid.addWidget(self.vc_lang_combo, 1, 1, 1, 3)

        self.vc_steps_lbl = QLabel("Passos ODE:")
        vc_grid.addWidget(self.vc_steps_lbl, 2, 0)
        self.vc_steps_spin = QSpinBox()
        self.vc_steps_spin.setRange(8, 128)
        self.vc_steps_spin.setValue(32)
        vc_grid.addWidget(self.vc_steps_spin, 2, 1)

        layout.addWidget(vc_params_group)

        # Progresso VC
        self.vc_progress_bar = QProgressBar()
        self.vc_progress_bar.setRange(0, 0)
        self.vc_progress_bar.setVisible(False)
        layout.addWidget(self.vc_progress_bar)

        # Botão Principal VC
        self.vc_btn = QPushButton("🔄 Converter Voz (Áudio ➔ Áudio)")
        self.vc_btn.setObjectName("primaryBtn")
        self.vc_btn.clicked.connect(self.start_voice_conversion)
        layout.addWidget(self.vc_btn)

        # Resultado VC
        vc_res_group = QGroupBox("4. Áudio Convertido & Texto Reconhecido")
        vc_res_vlayout = QVBoxLayout(vc_res_group)

        self.vc_transcribed_lbl = QLabel("Texto Reconhecido do Áudio de Origem: (nenhum)")
        self.vc_transcribed_lbl.setStyleSheet("color: #00E676; font-style: italic;")
        vc_res_vlayout.addWidget(self.vc_transcribed_lbl)

        vc_res_hlayout = QHBoxLayout()
        self.vc_res_label = QLabel("Nenhum áudio convertido ainda.")
        self.vc_res_label.setStyleSheet("color: #A1A1AA;")
        self.vc_play_res_btn = QPushButton("▶️ Reproduzir Áudio Convertido")
        self.vc_play_res_btn.setEnabled(False)
        self.vc_play_res_btn.clicked.connect(lambda: self.play_audio(self.last_converted_audio))
        self.vc_save_res_btn = QPushButton("💾 Salvar Como...")
        self.vc_save_res_btn.setEnabled(False)
        self.vc_save_res_btn.clicked.connect(lambda: self.save_audio_file(self.last_converted_audio))

        vc_res_hlayout.addWidget(self.vc_res_label, 1)
        vc_res_hlayout.addWidget(self.vc_play_res_btn)
        vc_res_hlayout.addWidget(self.vc_save_res_btn)
        vc_res_vlayout.addLayout(vc_res_hlayout)
        layout.addWidget(vc_res_group)

        # So agora: o handler mexe em widgets do grupo de resultado.
        self.on_vc_engine_changed()

    def setup_swap_tab(self):
        # Esta aba tem cinco grupos e não cabe na altura útil da janela em telas
        # menores. Sem a área rolável o Qt espreme todos os widgets em vez de
        # cortar, o que deixa a aba ilegível.
        outer = QVBoxLayout(self.tab_swap)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(2, 2, 8, 2)
        layout.setSpacing(8)

        # Grupo 1: a conversa inteira
        src_group = QGroupBox("1. Conversa (áudio com duas ou mais pessoas)")
        src_layout = QHBoxLayout(src_group)
        self.sw_src_edit = QLineEdit()
        self.sw_src_edit.setMinimumWidth(140)
        self.sw_src_edit.setPlaceholderText("Selecione a conversa inteira — não precisa recortar nada...")
        sw_src_btn = QPushButton("📁 Escolher Conversa")
        sw_src_btn.clicked.connect(lambda: self.select_audio_file(self.sw_src_edit))
        sw_play_src_btn = QPushButton("▶️ Ouvir")
        sw_play_src_btn.clicked.connect(lambda: self.play_audio(self.sw_src_edit.text()))
        src_layout.addWidget(self.sw_src_edit)
        src_layout.addWidget(sw_src_btn)
        src_layout.addWidget(sw_play_src_btn)
        layout.addWidget(src_group)

        # Grupo 2: detectar quem fala
        det_group = QGroupBox("2. Detectar Falantes")
        det_group.setToolTip("Descubra quem é quem ouvindo a amostra de cada falante antes de escolher")
        det_vlayout = QVBoxLayout(det_group)

        det_hlayout = QHBoxLayout()
        det_hlayout.addWidget(QLabel("Falantes:"))
        self.sw_numspk_spin = QSpinBox()
        self.sw_numspk_spin.setRange(0, 20)
        self.sw_numspk_spin.setValue(0)
        self.sw_numspk_spin.setSpecialValueText("auto")
        self.sw_numspk_spin.setToolTip("0 = deixa o modelo descobrir sozinho quantas pessoas falam")
        det_hlayout.addWidget(self.sw_numspk_spin)
        det_hlayout.addWidget(QLabel("Máx:"))
        self.sw_maxspk_spin = QSpinBox()
        self.sw_maxspk_spin.setRange(0, 20)
        self.sw_maxspk_spin.setValue(0)
        self.sw_maxspk_spin.setSpecialValueText("livre")
        det_hlayout.addWidget(self.sw_maxspk_spin)
        self.sw_detect_btn = QPushButton("🔍 Detectar")
        self.sw_detect_btn.setToolTip("Analisa a conversa e descobre quantas pessoas falam")
        # Mudar a contagem sozinho nao refaz nada; sem este aviso o usuario mexe
        # nas setas e fica esperando a lista mudar.
        self.sw_numspk_spin.valueChanged.connect(self.on_speaker_count_changed)
        self.sw_maxspk_spin.valueChanged.connect(self.on_speaker_count_changed)
        self.sw_detect_btn.clicked.connect(self.start_diarization)
        det_hlayout.addWidget(self.sw_detect_btn)
        det_hlayout.addStretch()
        det_vlayout.addLayout(det_hlayout)

        # Lista e botão de amostra lado a lado: economiza uma linha inteira.
        lista_hlayout = QHBoxLayout()
        self.sw_hint_lbl = QLabel(
            "Deixe em \"auto\" para o modelo decidir. Se ele errar a contagem, ajuste e clique em 🔍 Detectar de novo."
        )
        self.sw_hint_lbl.setStyleSheet("color: #71717A; font-size: 11px; font-weight: normal;")
        self.sw_hint_lbl.setWordWrap(True)
        det_vlayout.addWidget(self.sw_hint_lbl)

        self.sw_speaker_list = QListWidget()
        self.sw_speaker_list.setMinimumWidth(200)
        self.sw_speaker_list.setMinimumHeight(84)
        self.sw_speaker_list.setMaximumHeight(112)
        self.sw_speaker_list.itemDoubleClicked.connect(self.play_selected_speaker)
        lista_hlayout.addWidget(self.sw_speaker_list, 1)

        self.sw_play_sample_btn = QPushButton("▶️ Ouvir\nAmostra")
        self.sw_play_sample_btn.setEnabled(False)
        self.sw_play_sample_btn.setToolTip("Toca um trecho do falante selecionado (ou dê duplo clique na lista)")
        self.sw_play_sample_btn.clicked.connect(self.play_selected_speaker)
        lista_hlayout.addWidget(self.sw_play_sample_btn)
        det_vlayout.addLayout(lista_hlayout)
        layout.addWidget(det_group)

        # Grupo 3: a voz nova
        tgt_group = QGroupBox("3. Voz Nova (3 a 10s)")
        tgt_vlayout = QVBoxLayout(tgt_group)
        tgt_hlayout = QHBoxLayout()
        self.sw_tgt_edit = QLineEdit()
        self.sw_tgt_edit.setMinimumWidth(140)
        self.sw_tgt_edit.setPlaceholderText("Selecione a amostra da voz que entrará no lugar...")
        sw_tgt_btn = QPushButton("📁 Escolher Voz Nova")
        sw_tgt_btn.clicked.connect(lambda: self.select_audio_file(self.sw_tgt_edit))
        sw_play_tgt_btn = QPushButton("▶️ Ouvir")
        sw_play_tgt_btn.clicked.connect(lambda: self.play_audio(self.sw_tgt_edit.text()))
        tgt_hlayout.addWidget(self.sw_tgt_edit)
        tgt_hlayout.addWidget(sw_tgt_btn)
        tgt_hlayout.addWidget(sw_play_tgt_btn)
        tgt_vlayout.addLayout(tgt_hlayout)
        self.sw_tgt_text_edit = QLineEdit()
        self.sw_tgt_text_edit.setPlaceholderText("O que a voz nova fala na amostra? (Opcional)")
        tgt_vlayout.addWidget(self.sw_tgt_text_edit)

        tgt_vlayout.insertLayout(0, self._linha_personagem(self.sw_tgt_edit, self.sw_tgt_text_edit))
        layout.addWidget(tgt_group)

        # Grupo 4: parametros
        par_group = QGroupBox("4. Parâmetros")
        par_grid = QGridLayout(par_group)

        par_grid.setVerticalSpacing(6)

        par_grid.addWidget(QLabel("Motor:"), 0, 0)
        self.sw_engine_combo = QComboBox()
        self.sw_engine_combo.addItem("🎯 Seed-VC — mantém duração e inflexão (lip sync)", "seedvc")
        self.sw_engine_combo.addItem("💬 F5-TTS — refala o texto (muda duração)", "f5")
        self.sw_engine_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.sw_engine_combo.setMinimumContentsLength(16)
        self.sw_engine_combo.setToolTip(
            "Seed-VC: voice conversion quadro a quadro. Não passa por texto, então a duração\n"
            "é idêntica à do original e a entoação é preservada — o lip sync se mantém.\n\n"
            "F5-TTS: transcreve com o Whisper e ressintetiza. A duração e a inflexão são\n"
            "reinventadas, e palavras que o ASR não ouviu somem. Use só para mudar o que é dito."
        )
        self.sw_engine_combo.currentIndexChanged.connect(self.on_engine_changed)
        par_grid.addWidget(self.sw_engine_combo, 0, 1)

        par_grid.addWidget(QLabel("Difusão:"), 0, 2)
        self.sw_diffusion_spin = QSpinBox()
        self.sw_diffusion_spin.setRange(4, 100)
        self.sw_diffusion_spin.setValue(25)
        self.sw_diffusion_spin.setToolTip("Passos de difusão do Seed-VC. Mais = melhor e mais lento.")
        par_grid.addWidget(self.sw_diffusion_spin, 0, 3)

        self.sw_lang_lbl = QLabel("Modelo Base:")
        par_grid.addWidget(self.sw_lang_lbl, 1, 0)
        self.sw_lang_combo = QComboBox()
        self.sw_lang_combo.addItem("🇧🇷 Português Brasileiro (PT-BR) [Nativo]", "pt-br")
        self.sw_lang_combo.addItem("🇺🇸 Inglês / Multilíngue", "en")
        # Sem isto o combo exige a largura do item mais longo e estoura a aba;
        # a lista suspensa continua mostrando o texto completo.
        self.sw_lang_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.sw_lang_combo.setMinimumContentsLength(16)
        par_grid.addWidget(self.sw_lang_combo, 1, 1)

        self.sw_fit_lbl = QLabel("Encaixe:")
        par_grid.addWidget(self.sw_fit_lbl, 1, 2)
        self.sw_fit_combo = QComboBox()
        self.sw_fit_combo.addItem("Manter a sincronia (encaixa no tempo original)", "stretch")
        self.sw_fit_combo.addItem("Manter o ritmo natural (empurra a linha do tempo)", "pad")
        self.sw_fit_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.sw_fit_combo.setMinimumContentsLength(16)
        self.sw_fit_combo.setToolTip(
            "Sincronia: a fala gerada é encaixada no tempo exato do trecho original.\n"
            "Ritmo natural: preserva a velocidade da fala e empurra a linha do tempo."
        )
        par_grid.addWidget(self.sw_fit_combo, 1, 3)

        self.sw_steps_lbl = QLabel("Passos ODE:")
        par_grid.addWidget(self.sw_steps_lbl, 2, 0)
        self.sw_steps_spin = QSpinBox()
        self.sw_steps_spin.setRange(8, 128)
        self.sw_steps_spin.setValue(32)
        par_grid.addWidget(self.sw_steps_spin, 2, 1)

        self.sw_seed_lbl = QLabel("Seed:")
        par_grid.addWidget(self.sw_seed_lbl, 2, 2)
        self.sw_seed_spin = QSpinBox()
        self.sw_seed_spin.setRange(0, 2147483647)
        self.sw_seed_spin.setValue(0)
        self.sw_seed_spin.setSpecialValueText("aleatória")
        self.sw_seed_spin.setToolTip(
            "0 = sorteia a cada rodada. Um trecho que saiu ruim costuma melhorar só rodando de novo; "
            "fixe a seed quando quiser repetir um resultado bom."
        )
        par_grid.addWidget(self.sw_seed_spin, 2, 3)
        layout.addWidget(par_group)
        self.on_engine_changed()

        self.sw_progress_bar = QProgressBar()
        self.sw_progress_bar.setRange(0, 0)
        self.sw_progress_bar.setVisible(False)
        layout.addWidget(self.sw_progress_bar)

        self.sw_btn = QPushButton("🎭 Trocar a Voz do Falante Selecionado")
        self.sw_btn.setObjectName("primaryBtn")
        self.sw_btn.setEnabled(False)
        self.sw_btn.clicked.connect(self.start_speaker_swap)
        layout.addWidget(self.sw_btn)

        # Grupo 5: resultado
        res_group = QGroupBox("5. Resultado")
        res_vlayout = QVBoxLayout(res_group)
        self.sw_report = QTextEdit()
        self.sw_report.setReadOnly(True)
        self.sw_report.setMinimumWidth(200)
        self.sw_report.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.sw_report.setMinimumHeight(86)
        self.sw_report.setMaximumHeight(112)
        self.sw_report.setFont(QFont("Consolas", 9))
        self.sw_report.setPlaceholderText("O relatório trecho a trecho aparece aqui depois da troca.")
        res_vlayout.addWidget(self.sw_report)

        res_hlayout = QHBoxLayout()
        self.sw_res_label = QLabel("Nenhuma conversa processada ainda.")
        self.sw_res_label.setStyleSheet("color: #A1A1AA;")
        self.sw_res_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.sw_res_label.setMinimumWidth(120)
        self.sw_play_res_btn = QPushButton("▶️ Ouvir")
        self.sw_play_res_btn.setEnabled(False)
        self.sw_play_res_btn.clicked.connect(lambda: self.play_audio(self.last_swapped_audio))
        self.sw_save_res_btn = QPushButton("💾 Salvar")
        self.sw_save_res_btn.setEnabled(False)
        self.sw_save_res_btn.clicked.connect(lambda: self.save_audio_file(self.last_swapped_audio))
        res_hlayout.addWidget(self.sw_res_label, 1)
        res_hlayout.addWidget(self.sw_play_res_btn)
        res_hlayout.addWidget(self.sw_save_res_btn)
        res_vlayout.addLayout(res_hlayout)
        layout.addWidget(res_group)

    # --- Handlers da aba de troca de falante --------------------------------

    def on_engine_changed(self):
        """
        Os dois motores não compartilham parâmetros: idioma, encaixe, passos ODE
        e seed só existem no caminho F5-TTS; difusão só no Seed-VC. Deixar tudo
        visível sugere controles que não fazem nada.
        """
        usa_f5 = self.sw_engine_combo.currentData() == "f5"
        for w in (self.sw_lang_lbl, self.sw_lang_combo, self.sw_fit_lbl, self.sw_fit_combo,
                  self.sw_steps_lbl, self.sw_steps_spin, self.sw_seed_lbl, self.sw_seed_spin):
            w.setVisible(usa_f5)
        self.sw_diffusion_spin.setVisible(not usa_f5)

    def on_speaker_count_changed(self):
        """A contagem mudou: a lista atual ficou desatualizada até redetectar."""
        if self.sw_speaker_list.count() == 0:
            return
        self.sw_hint_lbl.setText("⚠️ Contagem alterada — clique em 🔍 Detectar para refazer a análise.")
        self.sw_hint_lbl.setStyleSheet("color: #FFB020; font-size: 11px; font-weight: bold;")

    def start_diarization(self):
        src_path = self.sw_src_edit.text().strip()
        if not src_path or not Path(src_path).exists():
            QMessageBox.warning(self, "Aviso", "Selecione primeiro o áudio da conversa.")
            return

        self.sw_detect_btn.setEnabled(False)
        self.sw_btn.setEnabled(False)
        self.sw_progress_bar.setVisible(True)
        self.sw_speaker_list.clear()
        self.status_bar.setText("Detectando falantes...")

        num = self.sw_numspk_spin.value() or None
        mx = self.sw_maxspk_spin.value() or None

        self.diar_worker = DiarizationWorker(
            source_audio=src_path,
            samples_dir=str(BASE_DIR / "speaker_samples"),
            num_speakers=num,
            min_speakers=None,
            max_speakers=mx
        )
        self.diar_worker.progress_signal.connect(lambda msg: self.status_bar.setText(msg))
        self.diar_worker.finished_signal.connect(self.on_diarization_finished)
        self.diar_worker.error_signal.connect(self.on_diarization_error)
        self.diar_worker.start()

    def on_diarization_finished(self, resumo: list):
        self.sw_detect_btn.setEnabled(True)
        self.sw_progress_bar.setVisible(False)
        self.speaker_samples = {r["speaker"]: r["sample"] for r in resumo}

        for r in resumo:
            item = QListWidgetItem(
                f'{r["speaker"]}  —  {r["seconds"]:.1f}s em {r["count"]} trechos  ({r["share"]:.1f}% da fala)'
            )
            item.setData(Qt.ItemDataRole.UserRole, r["speaker"])
            self.sw_speaker_list.addItem(item)

        if resumo:
            self.sw_speaker_list.setCurrentRow(0)
            self.sw_play_sample_btn.setEnabled(True)
            self.sw_btn.setEnabled(True)

        forcado = self.sw_numspk_spin.value()
        if forcado:
            self.sw_hint_lbl.setText(f"Contagem forçada em {forcado}. Ouça as amostras para confirmar quem é quem.")
        else:
            self.sw_hint_lbl.setText(
                f"{len(resumo)} falante(s) detectado(s) automaticamente. "
                f"Se o número estiver errado, ajuste ao lado e clique em 🔍 Detectar de novo."
            )
        self.sw_hint_lbl.setStyleSheet("color: #71717A; font-size: 11px; font-weight: normal;")

        self.status_bar.setText(
            f"{len(resumo)} falante(s) detectado(s). Ouça as amostras e escolha qual substituir."
        )

    def on_diarization_error(self, err_msg: str):
        self.sw_detect_btn.setEnabled(True)
        self.sw_progress_bar.setVisible(False)
        self.status_bar.setText("Erro na detecção de falantes.")
        QMessageBox.critical(self, "Erro na Diarização", f"Ocorreu um erro:\n{err_msg}")

    def play_selected_speaker(self):
        item = self.sw_speaker_list.currentItem()
        if item is None:
            return
        speaker = item.data(Qt.ItemDataRole.UserRole)
        sample = self.speaker_samples.get(speaker)
        if sample and Path(sample).exists():
            self.play_audio(sample)

    def start_speaker_swap(self):
        src_path = self.sw_src_edit.text().strip()
        tgt_path = self.sw_tgt_edit.text().strip()
        item = self.sw_speaker_list.currentItem()

        if not src_path or not Path(src_path).exists():
            QMessageBox.warning(self, "Aviso", "Selecione o áudio da conversa.")
            return
        if item is None:
            QMessageBox.warning(self, "Aviso", "Detecte os falantes e escolha qual deles substituir.")
            return
        if not tgt_path or not Path(tgt_path).exists():
            QMessageBox.warning(self, "Aviso", "Selecione a voz nova que entrará no lugar.")
            return

        speaker = item.data(Qt.ItemDataRole.UserRole)
        self.sw_btn.setEnabled(False)
        self.sw_detect_btn.setEnabled(False)
        self.sw_progress_bar.setVisible(True)
        self.sw_report.clear()
        self.status_bar.setText(f"Trocando a voz de {speaker}...")

        engine = self.sw_engine_combo.currentData() or "seedvc"
        selected_lang = self.sw_lang_combo.currentData() or "pt-br"
        if self.cloner is None or getattr(self.cloner, "language", None) != selected_lang:
            self.cloner = VoiceCloner(language=selected_lang)

        self.swap_worker = SpeakerSwapWorker(
            cloner=self.cloner,
            source_audio=src_path,
            target_audio=tgt_path,
            speaker=speaker,
            target_ref_text=self.sw_tgt_text_edit.text().strip(),
            output_path=str(BASE_DIR / "swap_output.wav"),
            num_speakers=self.sw_numspk_spin.value() or None,
            min_speakers=None,
            max_speakers=self.sw_maxspk_spin.value() or None,
            n_steps=self.sw_steps_spin.value(),
            fit_mode=self.sw_fit_combo.currentData() or "stretch",
            seed=self.sw_seed_spin.value() or None,
            engine=engine,
            f0_condition=True,
            diffusion_steps=self.sw_diffusion_spin.value()
        )
        self.swap_worker.progress_signal.connect(lambda msg: self.status_bar.setText(msg))
        self.swap_worker.finished_signal.connect(self.on_swap_finished)
        self.swap_worker.error_signal.connect(self.on_swap_error)
        self.swap_worker.start()

    def on_swap_finished(self, out_path: str, report: list):
        self.sw_btn.setEnabled(True)
        self.sw_detect_btn.setEnabled(True)
        self.sw_progress_bar.setVisible(False)
        self.last_swapped_audio = out_path

        linhas = []
        apertados = 0
        for r in report:
            if r.get("status") != "substituido":
                linhas.append(f'{r["start"]:7.2f}s  (mantido — nada reconhecido)')
                continue
            if not r.get("text"):
                # Seed-VC: sem texto, a duração é a mesma por construção
                linhas.append(
                    f'{r["start"]:7.2f}s - {r["end"]:7.2f}s  '
                    f'{r.get("slot_seconds", 0):5.2f}s convertidos (duração preservada)'
                )
                continue
            marca = ""
            if r.get("tight"):
                marca = "  <-- APERTADO"
                apertados += 1
            linhas.append(
                f'{r["start"]:7.2f}s  slot {r.get("slot_seconds", 0):5.2f}s / '
                f'natural {r.get("estimated_seconds", 0):5.2f}s  '
                f'"{r.get("text", "")[:42]}"{marca}'
            )
        if apertados:
            linhas.append("")
            linhas.append(
                f"⚠️ {apertados} trecho(s) APERTADO: o tempo original não comporta o texto falado, "
                f"então a fala sai acelerada. Tente o encaixe 'Manter o ritmo natural'."
            )
        self.sw_report.setPlainText("\n".join(linhas))

        trocados = sum(1 for r in report if r.get("status") == "substituido")
        self.sw_res_label.setText(f"{trocados} trecho(s) trocados — {Path(out_path).name}")
        self.sw_res_label.setToolTip(out_path)
        self.sw_res_label.setStyleSheet("color: #00E676; font-weight: bold;")
        self.sw_play_res_btn.setEnabled(True)
        self.sw_save_res_btn.setEnabled(True)
        self.status_bar.setText("Troca de falante concluída!")

    def on_swap_error(self, err_msg: str):
        self.sw_btn.setEnabled(True)
        self.sw_detect_btn.setEnabled(True)
        self.sw_progress_bar.setVisible(False)
        self.status_bar.setText("Erro na troca de falante.")
        QMessageBox.critical(self, "Erro na Troca de Falante", f"Ocorreu um erro:\n{err_msg}")

    def setup_characters_tab(self):
        layout = QVBoxLayout(self.tab_characters)
        layout.setSpacing(10)

        explicacao = QLabel(
            "Junte um ou mais áudios sob o nome de um personagem. "
            "Depois é só escolher o personagem nas outras abas, sem procurar arquivo. "
            "Com vários áudios, eles são emendados numa referência única (até 25s), "
            "o que costuma melhorar a clonagem."
        )
        explicacao.setWordWrap(True)
        explicacao.setStyleSheet("color: #A1A1AA; font-size: 12px;")
        layout.addWidget(explicacao)

        colunas = QHBoxLayout()

        # --- coluna esquerda: personagens
        grupo_pers = QGroupBox("Personagens")
        vbox_pers = QVBoxLayout(grupo_pers)
        self.ch_list = QListWidget()
        self.ch_list.setMinimumWidth(180)
        self.ch_list.currentItemChanged.connect(self.on_character_selected)
        vbox_pers.addWidget(self.ch_list)

        botoes_pers = QHBoxLayout()
        btn_novo = QPushButton("➕ Novo")
        btn_novo.clicked.connect(self.on_character_new)
        self.ch_rename_btn = QPushButton("✏️ Renomear")
        self.ch_rename_btn.clicked.connect(self.on_character_rename)
        self.ch_delete_btn = QPushButton("🗑️ Remover")
        self.ch_delete_btn.clicked.connect(self.on_character_delete)
        for b in (btn_novo, self.ch_rename_btn, self.ch_delete_btn):
            botoes_pers.addWidget(b)
        vbox_pers.addLayout(botoes_pers)
        colunas.addWidget(grupo_pers, 1)

        # --- coluna direita: audios do personagem
        grupo_audios = QGroupBox("Áudios do personagem")
        vbox_audios = QVBoxLayout(grupo_audios)

        self.ch_info_lbl = QLabel("Selecione ou crie um personagem.")
        self.ch_info_lbl.setStyleSheet("color: #A1A1AA; font-size: 12px; font-weight: normal;")
        vbox_audios.addWidget(self.ch_info_lbl)

        self.ch_files_list = QListWidget()
        self.ch_files_list.setMinimumWidth(300)
        self.ch_files_list.itemDoubleClicked.connect(self.on_character_play_file)
        vbox_audios.addWidget(self.ch_files_list)

        botoes_audio = QHBoxLayout()
        self.ch_add_btn = QPushButton("📁 Adicionar áudios...")
        self.ch_add_btn.clicked.connect(self.on_character_add_files)
        self.ch_play_btn = QPushButton("▶️ Ouvir")
        self.ch_play_btn.clicked.connect(self.on_character_play_file)
        self.ch_remove_file_btn = QPushButton("➖ Remover do personagem")
        self.ch_remove_file_btn.clicked.connect(self.on_character_remove_file)
        for b in (self.ch_add_btn, self.ch_play_btn, self.ch_remove_file_btn):
            botoes_audio.addWidget(b)
        vbox_audios.addLayout(botoes_audio)

        self.ch_reftext_edit = QLineEdit()
        self.ch_reftext_edit.setPlaceholderText(
            "Transcrição do áudio (opcional — só o motor F5-TTS usa; o Seed-VC não precisa)"
        )
        self.ch_reftext_edit.editingFinished.connect(self.on_character_reftext_saved)
        vbox_audios.addWidget(self.ch_reftext_edit)

        colunas.addLayout(QVBoxLayout()) if False else None
        colunas.addWidget(grupo_audios, 2)
        layout.addLayout(colunas)

        self.refresh_character_list()

    # --- Handlers da aba de personagens -------------------------------------

    def personagem_selecionado(self) -> str | None:
        item = self.ch_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def refresh_character_list(self, selecionar: str | None = None):
        anterior = selecionar or self.personagem_selecionado()
        self.ch_list.blockSignals(True)
        self.ch_list.clear()
        for nome in self.library.nomes():
            item = QListWidgetItem(f"{nome}  ({self.library.resumo(nome)})")
            item.setData(Qt.ItemDataRole.UserRole, nome)
            self.ch_list.addItem(item)
        self.ch_list.blockSignals(False)

        if anterior:
            for i in range(self.ch_list.count()):
                if self.ch_list.item(i).data(Qt.ItemDataRole.UserRole) == anterior:
                    self.ch_list.setCurrentRow(i)
                    break
        elif self.ch_list.count():
            self.ch_list.setCurrentRow(0)

        self.on_character_selected()
        self.refresh_character_combos()

    def on_character_selected(self, *args):
        nome = self.personagem_selecionado()
        tem = nome is not None
        for w in (self.ch_rename_btn, self.ch_delete_btn, self.ch_add_btn,
                  self.ch_play_btn, self.ch_remove_file_btn, self.ch_reftext_edit):
            w.setEnabled(tem)

        self.ch_files_list.clear()
        if not tem:
            self.ch_info_lbl.setText("Selecione ou crie um personagem.")
            self.ch_reftext_edit.clear()
            return

        personagem = self.library.obter(nome)
        for arquivo in (personagem.arquivos if personagem else []):
            caminho = self.library.caminho_absoluto(arquivo)
            existe = caminho.exists()
            rotulo = caminho.name if existe else f"{caminho.name}  ⚠️ não encontrado"
            item = QListWidgetItem(rotulo)
            item.setData(Qt.ItemDataRole.UserRole, arquivo)
            item.setToolTip(str(caminho))
            if not existe:
                item.setForeground(QColor("#FFB020"))
            self.ch_files_list.addItem(item)

        if self.ch_files_list.count():
            self.ch_files_list.setCurrentRow(0)
        self.ch_info_lbl.setText(f"{nome} — {self.library.resumo(nome)}")
        self.ch_reftext_edit.setText(personagem.ref_text if personagem else "")

    def on_character_new(self):
        from PyQt6.QtWidgets import QInputDialog
        nome, ok = QInputDialog.getText(self, "Novo personagem", "Nome do personagem:")
        if not ok or not nome.strip():
            return
        try:
            self.library.criar(nome)
        except ValueError as e:
            QMessageBox.warning(self, "Não foi possível criar", str(e))
            return
        self.refresh_character_list(selecionar=nome.strip())
        self.status_bar.setText(f"Personagem '{nome.strip()}' criado. Adicione os áudios dele.")

    def on_character_rename(self):
        nome = self.personagem_selecionado()
        if not nome:
            return
        from PyQt6.QtWidgets import QInputDialog
        novo, ok = QInputDialog.getText(self, "Renomear personagem", "Novo nome:", text=nome)
        if not ok or not novo.strip() or novo.strip() == nome:
            return
        try:
            self.library.renomear(nome, novo)
        except ValueError as e:
            QMessageBox.warning(self, "Não foi possível renomear", str(e))
            return
        self.refresh_character_list(selecionar=novo.strip())

    def on_character_delete(self):
        nome = self.personagem_selecionado()
        if not nome:
            return
        resposta = QMessageBox.question(
            self, "Remover personagem",
            f"Remover '{nome}' da biblioteca?\n\n"
            f"Os arquivos de áudio continuam no disco — só a associação é apagada."
        )
        if resposta != QMessageBox.StandardButton.Yes:
            return
        self.library.remover(nome)
        self.refresh_character_list()
        self.status_bar.setText(f"Personagem '{nome}' removido.")

    def on_character_add_files(self):
        nome = self.personagem_selecionado()
        if not nome:
            return
        caminhos, _ = QFileDialog.getOpenFileNames(
            self, f"Áudios de {nome}", "",
            "Áudio (*.wav *.mp3 *.flac *.ogg *.m4a);;Todos os arquivos (*)"
        )
        if not caminhos:
            return
        novos = self.library.adicionar_arquivos(nome, caminhos)
        self.refresh_character_list(selecionar=nome)
        ignorados = len(caminhos) - novos
        recado = f"{novos} áudio(s) adicionados a '{nome}'."
        if ignorados:
            recado += f" {ignorados} já estava(m) na lista."
        self.status_bar.setText(recado)

    def on_character_remove_file(self):
        nome = self.personagem_selecionado()
        item = self.ch_files_list.currentItem()
        if not nome or item is None:
            return
        self.library.remover_arquivo(nome, item.data(Qt.ItemDataRole.UserRole))
        self.refresh_character_list(selecionar=nome)

    def on_character_play_file(self):
        item = self.ch_files_list.currentItem()
        if item is None:
            return
        caminho = self.library.caminho_absoluto(item.data(Qt.ItemDataRole.UserRole))
        if caminho.exists():
            self.play_audio(str(caminho))

    def on_character_reftext_saved(self):
        nome = self.personagem_selecionado()
        if nome:
            self.library.definir_ref_text(nome, self.ch_reftext_edit.text())

    def setup_training_tab(self):
        layout = QVBoxLayout(self.tab_training)
        layout.setSpacing(12)

        data_group = QGroupBox("Diretório de Treinamento")
        data_layout = QHBoxLayout(data_group)
        self.train_dir_edit = QLineEdit()
        self.train_dir_edit.setText("data/demo_speaker")
        self.train_dir_edit.setPlaceholderText("Pasta com arquivos de áudio .wav / .mp3...")
        dir_btn = QPushButton("📁 Escolher Pasta")
        dir_btn.clicked.connect(self.select_train_dir)
        data_layout.addWidget(self.train_dir_edit)
        data_layout.addWidget(dir_btn)
        layout.addWidget(data_group)

        cfg_group = QGroupBox("Hiperparâmetros de Treino")
        grid = QGridLayout(cfg_group)

        grid.addWidget(QLabel("Épocas:"), 0, 0)
        self.train_epochs_spin = QSpinBox()
        self.train_epochs_spin.setRange(1, 1000)
        self.train_epochs_spin.setValue(20)
        grid.addWidget(self.train_epochs_spin, 0, 1)

        grid.addWidget(QLabel("Batch Size:"), 0, 2)
        self.train_bs_spin = QSpinBox()
        self.train_bs_spin.setRange(1, 64)
        self.train_bs_spin.setValue(4)
        grid.addWidget(self.train_bs_spin, 0, 3)

        grid.addWidget(QLabel("Taxa de Aprendizado (LR):"), 1, 0)
        self.train_lr_spin = QDoubleSpinBox()
        self.train_lr_spin.setDecimals(6)
        self.train_lr_spin.setRange(0.000001, 0.01)
        self.train_lr_spin.setValue(0.0002)
        grid.addWidget(self.train_lr_spin, 1, 1)

        grid.addWidget(QLabel("Continuar Checkpoint:"), 1, 2)
        self.train_resume_edit = QLineEdit()
        self.train_resume_edit.setPlaceholderText("Opcional: checkpoints/checkpoint_epoch_10.pt")
        grid.addWidget(self.train_resume_edit, 1, 3)

        layout.addWidget(cfg_group)

        self.train_prog_bar = QProgressBar()
        self.train_prog_bar.setRange(0, 100)
        self.train_prog_bar.setValue(0)
        layout.addWidget(self.train_prog_bar)

        btn_layout = QHBoxLayout()
        self.start_train_btn = QPushButton("⚡ Iniciar Treinamento")
        self.start_train_btn.setObjectName("primaryBtn")
        self.start_train_btn.clicked.connect(self.start_training)
        self.stop_train_btn = QPushButton("⏹️ Parar")
        self.stop_train_btn.setEnabled(False)
        self.stop_train_btn.clicked.connect(self.stop_training)
        btn_layout.addWidget(self.start_train_btn)
        btn_layout.addWidget(self.stop_train_btn)
        layout.addLayout(btn_layout)

        log_group = QGroupBox("Logs de Treinamento em Tempo Real")
        log_layout = QVBoxLayout(log_group)
        self.train_log_text = QTextEdit()
        self.train_log_text.setReadOnly(True)
        self.train_log_text.setStyleSheet("background-color: #0D0D10; color: #00E676; font-family: 'Consolas', monospace;")
        log_layout.addWidget(self.train_log_text)
        layout.addWidget(log_group)

    def setup_about_tab(self):
        layout = QVBoxLayout(self.tab_about)
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setHtml(
            """
            <h2>🎙️ Clonador de Voz Moderno (Flow Matching DiT)</h2>
            <p>O sistema conta com duas modalidades principais:</p>
            <ul>
                <li><b>1. Texto para Voz (TTS)</b>: Digite qualquer frase e ouça com a voz clonada a partir de 3 segundos de referência.</li>
                <li><b>2. Áudio para Áudio (Voice-to-Voice Conversion)</b>: Envie um áudio de uma pessoa falando ou cantando e transforme na voz de outra pessoa (mantendo a mensagem original).</li>
                <li><b>3. Treinamento / Fine-Tuning</b>: Treine modelos especializados em pastas com suas vozes gravadas.</li>
            </ul>
            <p><b>Modelo Base:</b> F5-TTS Brazilian Portuguese (PT-BR) + Neural Vocos 24kHz acelerado por NVIDIA RTX CUDA.</p>
            """
        )
        layout.addWidget(info_text)

    def select_audio_file(self, target_line_edit: QLineEdit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecionar Arquivo de Áudio", "", "Áudios (*.wav *.mp3 *.flac *.ogg)")
        if file_path:
            target_line_edit.setText(file_path)

    def select_checkpoint(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecionar Checkpoint", "checkpoints", "Model Checkpoints (*.pt *.pth *.safetensors)")
        if file_path:
            self.ckpt_path_edit.setText(file_path)

    def select_train_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Selecionar Pasta do Dataset")
        if dir_path:
            self.train_dir_edit.setText(dir_path)

    def play_audio(self, audio_path: str):
        if not audio_path or not Path(audio_path).exists():
            QMessageBox.warning(self, "Aviso", "Arquivo de áudio não encontrado.")
            return
        try:
            winsound.PlaySound(str(audio_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            QMessageBox.warning(self, "Erro de Reprodução", f"Não foi possível reproduzir o áudio: {e}")

    def save_audio_file(self, audio_path: str):
        if not audio_path or not Path(audio_path).exists():
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "Salvar Áudio", "audio_clonado.wav", "Arquivos WAV (*.wav)")
        if save_path:
            import shutil
            shutil.copy2(audio_path, save_path)
            QMessageBox.information(self, "Salvo", f"Áudio salvo em: {save_path}")

    def start_synthesis(self):
        ref_path = self.ref_path_edit.text().strip()
        text = self.target_text_edit.toPlainText().strip()
        ckpt = self.ckpt_path_edit.text().strip()

        if not ref_path or not Path(ref_path).exists():
            QMessageBox.warning(self, "Aviso", "Por favor selecione um arquivo de áudio de referência válido.")
            return
        if not text:
            QMessageBox.warning(self, "Aviso", "Por favor digite o texto a ser sintetizado.")
            return

        self.synthesize_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_bar.setText("Inicializando modelo e sintetizando voz...")

        selected_lang = self.lang_combo.currentData() or "pt-br"
        if self.cloner is None or ckpt or getattr(self.cloner, "language", None) != selected_lang:
            self.cloner = VoiceCloner(
                checkpoint_path=ckpt if ckpt and Path(ckpt).exists() else None,
                language=selected_lang
            )

        ref_text = self.ref_text_edit.text().strip()
        out_file = BASE_DIR / "gui_output.wav"
        self.worker = InferenceWorker(
            cloner=self.cloner,
            ref_audio=ref_path,
            text=text,
            ref_text=ref_text,
            output_path=str(out_file),
            speed=self.speed_spin.value(),
            n_steps=self.steps_spin.value(),
            cfg=self.cfg_spin.value(),
            solver=self.solver_combo.currentText()
        )
        self.worker.progress_signal.connect(lambda msg: self.status_bar.setText(msg))
        self.worker.finished_signal.connect(self.on_synthesis_finished)
        self.worker.error_signal.connect(self.on_synthesis_error)
        self.worker.start()

    def on_synthesis_finished(self, out_path: str):
        self.synthesize_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.last_generated_audio = out_path
        self.res_label.setText(f"Áudio gerado com sucesso: {Path(out_path).name}")
        self.res_label.setStyleSheet("color: #00E676; font-weight: bold;")
        self.play_res_btn.setEnabled(True)
        self.save_res_btn.setEnabled(True)
        self.status_bar.setText("Síntese concluída com sucesso!")
        self.play_audio(out_path)

    def on_synthesis_error(self, err_msg: str):
        self.synthesize_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.setText("Erro na síntese.")
        QMessageBox.critical(self, "Erro de Síntese", f"Ocorreu um erro durante a clonagem:\n{err_msg}")

    def on_vc_engine_changed(self):
        """O Seed-VC não usa modelo de idioma nem passos ODE; o F5 não usa difusão."""
        usa_f5 = self.vc_engine_combo.currentData() == "f5"
        for w in (self.vc_lang_lbl, self.vc_lang_combo, self.vc_steps_lbl, self.vc_steps_spin):
            w.setVisible(usa_f5)
        for w in (self.vc_diffusion_lbl, self.vc_diffusion_spin):
            w.setVisible(not usa_f5)
        # O label continua visivel nos dois motores, mas so o F5 tem texto
        # reconhecido para mostrar.
        self.vc_transcribed_lbl.setText(
            "Texto Reconhecido do Áudio de Origem: (nenhum)" if usa_f5
            else "Sem transcrição: o Seed-VC converte direto o áudio, sem passar por texto."
        )

    def start_voice_conversion(self):
        src_path = self.vc_src_edit.text().strip()
        tgt_path = self.vc_tgt_edit.text().strip()

        if not src_path or not Path(src_path).exists():
            QMessageBox.warning(self, "Aviso", "Por favor selecione o áudio de origem (quem está falando).")
            return
        if not tgt_path or not Path(tgt_path).exists():
            QMessageBox.warning(self, "Aviso", "Por favor selecione a voz alvo de destino.")
            return

        self.vc_btn.setEnabled(False)
        self.vc_progress_bar.setVisible(True)
        self.status_bar.setText("Convertendo para a voz alvo...")

        selected_lang = self.vc_lang_combo.currentData() or "pt-br"
        if self.cloner is None or getattr(self.cloner, "language", None) != selected_lang:
            self.cloner = VoiceCloner(language=selected_lang)

        tgt_ref_text = self.vc_tgt_text_edit.text().strip()
        out_file = BASE_DIR / "vc_output.wav"

        self.vc_worker = VoiceConversionWorker(
            cloner=self.cloner,
            source_audio=src_path,
            target_audio=tgt_path,
            target_ref_text=tgt_ref_text,
            output_path=str(out_file),
            speed=1.0,
            n_steps=self.vc_steps_spin.value(),
            cfg=2.0,
            solver="euler",
            engine=self.vc_engine_combo.currentData() or "seedvc",
            diffusion_steps=self.vc_diffusion_spin.value()
        )
        self.vc_worker.progress_signal.connect(lambda msg: self.status_bar.setText(msg))
        self.vc_worker.finished_signal.connect(self.on_vc_finished)
        self.vc_worker.error_signal.connect(self.on_vc_error)
        self.vc_worker.start()

    def on_vc_finished(self, out_path: str, transcribed_text: str):
        self.vc_btn.setEnabled(True)
        self.vc_progress_bar.setVisible(False)
        self.last_converted_audio = out_path
        if transcribed_text:
            self.vc_transcribed_lbl.setText(f"Texto Reconhecido: \"{transcribed_text}\"")
        else:
            # Seed-VC não transcreve: não há texto a mostrar.
            self.vc_transcribed_lbl.setText("Convertido quadro a quadro (duração e entoação preservadas).")
        self.vc_res_label.setText(f"Áudio convertido: {Path(out_path).name}")
        self.vc_res_label.setStyleSheet("color: #00E676; font-weight: bold;")
        self.vc_play_res_btn.setEnabled(True)
        self.vc_save_res_btn.setEnabled(True)
        self.status_bar.setText("Conversão Áudio ➔ Áudio concluída!")
        self.play_audio(out_path)

    def on_vc_error(self, err_msg: str):
        self.vc_btn.setEnabled(True)
        self.vc_progress_bar.setVisible(False)
        self.status_bar.setText("Erro na conversão.")
        QMessageBox.critical(self, "Erro de Conversão", f"Ocorreu um erro:\n{err_msg}")

    def start_training(self):
        data_dir = self.train_dir_edit.text().strip()
        if not data_dir or not Path(data_dir).exists():
            QMessageBox.warning(self, "Aviso", "Diretório de treino não encontrado.")
            return

        self.start_train_btn.setEnabled(False)
        self.stop_train_btn.setEnabled(True)
        self.train_log_text.clear()

        self.train_worker = TrainingWorker(
            data_dir=data_dir,
            epochs=self.train_epochs_spin.value(),
            batch_size=self.train_bs_spin.value(),
            lr=self.train_lr_spin.value(),
            resume_ckpt=self.train_resume_edit.text().strip() or None
        )
        self.train_worker.log_signal.connect(self.append_train_log)
        self.train_worker.epoch_signal.connect(self.update_train_progress)
        self.train_worker.finished_signal.connect(self.on_train_finished)
        self.train_worker.error_signal.connect(self.on_train_error)
        self.train_worker.start()

    def append_train_log(self, text: str):
        self.train_log_text.append(text)

    def update_train_progress(self, current: int, total: int, loss: float):
        percent = int((current / total) * 100)
        self.train_prog_bar.setValue(percent)

    def stop_training(self):
        if hasattr(self, "train_worker") and self.train_worker.isRunning():
            self.train_worker.stop()
            self.stop_train_btn.setEnabled(False)

    def on_train_finished(self, msg: str):
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        self.append_train_log(f"\n[SUCESSO] {msg}")
        QMessageBox.information(self, "Treinamento Concluído", msg)

    def on_train_error(self, err_msg: str):
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        self.append_train_log(f"\n[ERRO] {err_msg}")
        QMessageBox.critical(self, "Erro de Treinamento", f"Ocorreu um erro:\n{err_msg}")


def main():
    app = QApplication(sys.argv)
    window = VoiceClonerMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
