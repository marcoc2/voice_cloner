"""
Widget de forma de onda com regiões marcáveis por falante.

Desenha o áudio e deixa marcar trechos à mão, cada um atribuído a um falante.
Serve para o caso em que a diarização automática erra a contagem ou as
fronteiras e você prefere marcar você mesmo.

Não depende de biblioteca de plotagem: o desenho é feito no paintEvent, o que
mantém o controle fino sobre arrastar bordas, zoom e seleção.
"""
import os
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

# Resolução do resumo de picos: 200 baldes por segundo (5 ms). Suficiente para
# desenhar com precisão e leve o bastante para um arquivo de 20 minutos.
BALDES_POR_SEGUNDO = 200

# Cores dos falantes, na ordem em que forem criados.
PALETA = [
    "#00E676", "#40C4FF", "#FFB020", "#FF5C8A",
    "#B388FF", "#FFD54F", "#4DD0E1", "#FF8A65",
]

ALTURA_REGUA = 18
MARGEM_BORDA = 6   # tolerância em pixels para pegar a borda de uma região


@dataclass
class Regiao:
    """Um trecho marcado à mão, atribuído a um falante pelo índice."""
    inicio: float
    fim: float
    falante: int

    @property
    def duracao(self) -> float:
        return max(0.0, self.fim - self.inicio)

    def contem(self, t: float) -> bool:
        return self.inicio <= t <= self.fim


def cor_do_falante(indice: int) -> QColor:
    return QColor(PALETA[indice % len(PALETA)])


class WaveformView(QWidget):
    """
    Mostra a onda e as regiões. Interação:

      arrastar no vazio  -> cria uma região para o falante ativo
      clicar numa região -> seleciona
      arrastar a borda   -> redimensiona
      arrastar o meio    -> move
      duplo clique       -> pede para tocar o trecho
      Delete             -> apaga a região selecionada
      roda do mouse      -> rola;  Ctrl+roda -> zoom
    """

    regioesMudaram = pyqtSignal()
    selecaoMudou = pyqtSignal(int)
    pediuTocar = pyqtSignal(float, float)
    cursorMovido = pyqtSignal(float)
    arquivoSolto = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(190)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self._arrasto_ativo = False

        self.picos_min = np.zeros(0, dtype=np.float32)
        self.picos_max = np.zeros(0, dtype=np.float32)
        self.duracao = 0.0

        self.regioes: list[Regiao] = []
        self.nomes_falantes: list[str] = []
        self.falante_ativo = 0
        self.selecionada = -1

        # Janela visível, em segundos
        self.inicio_visivel = 0.0
        self.span_visivel = 1.0

        self._arrastando = None   # 'nova' | 'move' | 'inicio' | 'fim'
        self._ancora = 0.0
        self._t_press = 0.0

    # --- dados ---------------------------------------------------------------

    def carregar_audio(self, caminho: str) -> float:
        """Lê o arquivo e monta o resumo de picos. Devolve a duração."""
        import librosa

        y, sr = librosa.load(caminho, sr=None, mono=True)
        self.duracao = len(y) / sr if sr else 0.0

        n_baldes = max(1, int(self.duracao * BALDES_POR_SEGUNDO))
        por_balde = max(1, len(y) // n_baldes)
        util = por_balde * (len(y) // por_balde)
        if util:
            blocos = y[:util].reshape(-1, por_balde)
            self.picos_min = blocos.min(axis=1).astype(np.float32)
            self.picos_max = blocos.max(axis=1).astype(np.float32)
        else:
            self.picos_min = np.zeros(1, dtype=np.float32)
            self.picos_max = np.zeros(1, dtype=np.float32)

        self.regioes = []
        self.selecionada = -1
        self.inicio_visivel = 0.0
        self.span_visivel = self.duracao or 1.0
        self.update()
        return self.duracao

    def definir_falantes(self, nomes: list[str]):
        self.nomes_falantes = list(nomes)
        if self.falante_ativo >= len(nomes):
            self.falante_ativo = max(0, len(nomes) - 1)
        # Regiões de falantes que deixaram de existir voltam para o primeiro.
        for r in self.regioes:
            if r.falante >= len(nomes):
                r.falante = 0
        self.update()

    def definir_regioes(self, regioes: list[Regiao]):
        self.regioes = sorted(regioes, key=lambda r: r.inicio)
        self.selecionada = -1
        self.update()
        self.regioesMudaram.emit()

    def regioes_do_falante(self, indice: int) -> list[Regiao]:
        return [r for r in sorted(self.regioes, key=lambda x: x.inicio) if r.falante == indice]

    def apagar_selecionada(self):
        if 0 <= self.selecionada < len(self.regioes):
            del self.regioes[self.selecionada]
            self.selecionada = -1
            self.selecaoMudou.emit(-1)
            self.regioesMudaram.emit()
            self.update()

    def limpar_regioes(self):
        self.regioes = []
        self.selecionada = -1
        self.selecaoMudou.emit(-1)
        self.regioesMudaram.emit()
        self.update()

    # --- conversão tempo <-> pixel ------------------------------------------

    def _x_de_t(self, t: float) -> float:
        if self.span_visivel <= 0:
            return 0.0
        return (t - self.inicio_visivel) / self.span_visivel * self.width()

    def _t_de_x(self, x: float) -> float:
        if self.width() <= 0:
            return 0.0
        t = self.inicio_visivel + (x / self.width()) * self.span_visivel
        return max(0.0, min(self.duracao, t))

    # --- zoom e rolagem ------------------------------------------------------

    def zoom(self, fator: float, t_foco: float | None = None):
        if self.duracao <= 0:
            return
        if t_foco is None:
            t_foco = self.inicio_visivel + self.span_visivel / 2
        novo = max(0.05, min(self.duracao, self.span_visivel * fator))
        # Mantém o ponto sob o cursor no lugar.
        prop = (t_foco - self.inicio_visivel) / self.span_visivel if self.span_visivel else 0.5
        self.inicio_visivel = max(0.0, min(self.duracao - novo, t_foco - prop * novo))
        self.span_visivel = novo
        self.update()

    def ver_tudo(self):
        self.inicio_visivel = 0.0
        self.span_visivel = self.duracao or 1.0
        self.update()

    def rolar(self, segundos: float):
        limite = max(0.0, self.duracao - self.span_visivel)
        self.inicio_visivel = max(0.0, min(limite, self.inicio_visivel + segundos))
        self.update()

    def wheelEvent(self, evento):
        if evento.modifiers() & Qt.KeyboardModifier.ControlModifier:
            passos = evento.angleDelta().y() / 120.0
            self.zoom(0.8 ** passos, self._t_de_x(evento.position().x()))
        else:
            self.rolar(-evento.angleDelta().y() / 120.0 * self.span_visivel * 0.15)

    def keyPressEvent(self, evento):
        if evento.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.apagar_selecionada()
        elif evento.key() == Qt.Key.Key_Left:
            self.rolar(-self.span_visivel * 0.1)
        elif evento.key() == Qt.Key.Key_Right:
            self.rolar(self.span_visivel * 0.1)
        else:
            super().keyPressEvent(evento)

    # --- interação -----------------------------------------------------------

    def _regiao_em(self, x: float) -> tuple[int, str | None]:
        """Qual região está sob o pixel x, e se é borda ou meio."""
        t = self._t_de_x(x)
        for i, r in enumerate(self.regioes):
            xi, xf = self._x_de_t(r.inicio), self._x_de_t(r.fim)
            if abs(x - xi) <= MARGEM_BORDA:
                return i, "inicio"
            if abs(x - xf) <= MARGEM_BORDA:
                return i, "fim"
            if r.contem(t):
                return i, "meio"
        return -1, None

    def mousePressEvent(self, evento):
        if evento.button() != Qt.MouseButton.LeftButton or self.duracao <= 0:
            return
        x = evento.position().x()
        t = self._t_de_x(x)
        self._t_press = t

        indice, parte = self._regiao_em(x)
        if indice >= 0:
            self.selecionada = indice
            self.selecaoMudou.emit(indice)
            r = self.regioes[indice]
            if parte == "inicio":
                self._arrastando, self._ancora = "inicio", r.fim
            elif parte == "fim":
                self._arrastando, self._ancora = "fim", r.inicio
            else:
                self._arrastando, self._ancora = "move", t - r.inicio
        else:
            if not self.nomes_falantes:
                return
            nova = Regiao(t, t, self.falante_ativo)
            self.regioes.append(nova)
            self.selecionada = len(self.regioes) - 1
            self.selecaoMudou.emit(self.selecionada)
            self._arrastando, self._ancora = "nova", t

        self.cursorMovido.emit(t)
        self.update()

    def mouseMoveEvent(self, evento):
        x = evento.position().x()
        t = self._t_de_x(x)

        if self._arrastando is None:
            _, parte = self._regiao_em(x)
            if parte in ("inicio", "fim"):
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif parte == "meio":
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)
            return

        if not (0 <= self.selecionada < len(self.regioes)):
            return
        r = self.regioes[self.selecionada]

        if self._arrastando == "nova":
            r.inicio, r.fim = min(self._ancora, t), max(self._ancora, t)
        elif self._arrastando == "inicio":
            r.inicio, r.fim = min(t, self._ancora), max(t, self._ancora)
        elif self._arrastando == "fim":
            r.inicio, r.fim = min(self._ancora, t), max(self._ancora, t)
        elif self._arrastando == "move":
            largura = r.duracao
            novo_inicio = max(0.0, min(self.duracao - largura, t - self._ancora))
            r.inicio, r.fim = novo_inicio, novo_inicio + largura

        self.update()

    def mouseReleaseEvent(self, evento):
        if self._arrastando is None:
            return
        self._arrastando = None
        self.setCursor(Qt.CursorShape.CrossCursor)

        # Um clique sem arrastar nao deve virar uma regiao de duracao zero.
        if 0 <= self.selecionada < len(self.regioes):
            if self.regioes[self.selecionada].duracao < 0.05:
                del self.regioes[self.selecionada]
                self.selecionada = -1
                self.selecaoMudou.emit(-1)

        self.regioes.sort(key=lambda r: r.inicio)
        self.regioesMudaram.emit()
        self.update()

    def mouseDoubleClickEvent(self, evento):
        indice, _ = self._regiao_em(evento.position().x())
        if indice >= 0:
            r = self.regioes[indice]
            self.pediuTocar.emit(r.inicio, r.fim)

    # --- arrastar e soltar ---------------------------------------------------

    def dragEnterEvent(self, evento):
        if evento.mimeData().hasUrls():
            evento.acceptProposedAction()
            self._arrasto_ativo = True
            self.update()

    def dragLeaveEvent(self, evento):
        self._arrasto_ativo = False
        self.update()

    def dropEvent(self, evento):
        self._arrasto_ativo = False
        self.update()
        for url in evento.mimeData().urls():
            # Normaliza as barras: o Qt entrega C:/... e o Windows mostra C:\...
            caminho = os.path.normpath(url.toLocalFile()) if url.toLocalFile() else ""
            if caminho:
                self.arquivoSolto.emit(caminho)
                evento.acceptProposedAction()
                return

    # --- desenho -------------------------------------------------------------

    def paintEvent(self, evento):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        largura, altura = self.width(), self.height()

        p.fillRect(self.rect(), QColor("#141418"))

        if self.duracao <= 0 or not len(self.picos_max):
            p.setPen(QColor("#71717A"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Arraste um vídeo ou áudio aqui, ou use o botão Abrir")
            if self._arrasto_ativo:
                self._moldura_arrasto(p)
            return

        area_topo = ALTURA_REGUA
        area_altura = altura - ALTURA_REGUA
        meio = area_topo + area_altura / 2

        self._desenhar_regua(p, largura)
        self._desenhar_regioes(p, area_topo, area_altura)

        # --- onda
        i0 = int(self.inicio_visivel * BALDES_POR_SEGUNDO)
        i1 = int((self.inicio_visivel + self.span_visivel) * BALDES_POR_SEGUNDO)
        i0 = max(0, min(len(self.picos_max) - 1, i0))
        i1 = max(i0 + 1, min(len(self.picos_max), i1))

        bordas = np.linspace(i0, i1, largura + 1).astype(int)
        bordas = np.clip(bordas, 0, len(self.picos_max) - 1)
        inicios = bordas[:-1]
        validos = inicios < bordas[1:]

        if validos.any():
            baixo = np.minimum.reduceat(self.picos_min, inicios)
            alto = np.maximum.reduceat(self.picos_max, inicios)
            escala = area_altura / 2 * 0.92
            p.setPen(QPen(QColor("#5A5A66"), 1))
            for x in range(largura):
                if not validos[x]:
                    continue
                y1 = meio - alto[x] * escala
                y2 = meio - baixo[x] * escala
                p.drawLine(x, int(y1), x, int(y2))

        p.setPen(QPen(QColor("#3F3F46"), 1))
        p.drawLine(0, int(meio), largura, int(meio))

        self._desenhar_bordas_selecao(p, area_topo, area_altura)

        if self._arrasto_ativo:
            self._moldura_arrasto(p)

    def _moldura_arrasto(self, p: QPainter):
        """Realce enquanto um arquivo está sendo arrastado por cima."""
        p.setPen(QPen(QColor("#00E676"), 3, Qt.PenStyle.DashLine))
        p.drawRect(self.rect().adjusted(2, 2, -3, -3))
        p.setPen(QColor("#00E676"))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Solte para abrir")

    def _desenhar_regua(self, p: QPainter, largura: int):
        p.fillRect(QRect(0, 0, largura, ALTURA_REGUA), QColor("#18181B"))
        p.setFont(QFont("Segoe UI", 7))

        # Escolhe um passo "redondo" que caiba na janela visível.
        for passo in (0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600):
            if self.span_visivel / passo <= 12:
                break

        t = (int(self.inicio_visivel / passo)) * passo
        while t <= self.inicio_visivel + self.span_visivel:
            x = self._x_de_t(t)
            if 0 <= x <= largura:
                p.setPen(QColor("#3F3F46"))
                p.drawLine(int(x), 0, int(x), ALTURA_REGUA)
                p.setPen(QColor("#A1A1AA"))
                minutos, segundos = divmod(t, 60)
                rotulo = f"{int(minutos)}:{segundos:04.1f}" if self.duracao >= 60 else f"{t:.1f}s"
                p.drawText(int(x) + 3, ALTURA_REGUA - 5, rotulo)
            t += passo

    def _desenhar_regioes(self, p: QPainter, topo: int, altura: float):
        p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        for i, r in enumerate(self.regioes):
            x1, x2 = self._x_de_t(r.inicio), self._x_de_t(r.fim)
            if x2 < 0 or x1 > self.width():
                continue
            cor = cor_do_falante(r.falante)
            preenchimento = QColor(cor)
            preenchimento.setAlpha(70 if i != self.selecionada else 110)
            p.fillRect(QRect(int(x1), topo, max(1, int(x2 - x1)), int(altura)), preenchimento)

            p.setPen(QPen(cor, 2 if i == self.selecionada else 1))
            p.drawLine(int(x1), topo, int(x1), topo + int(altura))
            p.drawLine(int(x2), topo, int(x2), topo + int(altura))

            if x2 - x1 > 34 and r.falante < len(self.nomes_falantes):
                p.setPen(cor)
                p.drawText(int(x1) + 4, topo + 13, self.nomes_falantes[r.falante][:16])

    def _desenhar_bordas_selecao(self, p: QPainter, topo: int, altura: float):
        if not (0 <= self.selecionada < len(self.regioes)):
            return
        r = self.regioes[self.selecionada]
        cor = cor_do_falante(r.falante)
        for x in (self._x_de_t(r.inicio), self._x_de_t(r.fim)):
            p.fillRect(QRect(int(x) - 2, topo, 4, int(altura)), cor)
