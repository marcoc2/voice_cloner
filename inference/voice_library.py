"""
Biblioteca de personagens: relaciona um nome a um ou mais arquivos de áudio.

Serve para não precisar caçar o mp3 certo toda vez que quiser usar uma voz.
Cada personagem guarda uma lista de arquivos; na hora de usar, a biblioteca
monta uma referência única a partir deles.

O registro fica em `voices/characters.json`, com caminhos relativos à raiz do
projeto sempre que possível, para a pasta continuar funcionando se você mover o
projeto de lugar.
"""
import json
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np
import soundfile as sf

PASTA_VOZES = RAIZ / "voices"
ARQUIVO_REGISTRO = PASTA_VOZES / "characters.json"
PASTA_CACHE = PASTA_VOZES / "_cache"

# O F5-TTS recorta a referência em ~12s e o Seed-VC usa no máximo 25s. Passar
# disso é só desperdício de processamento.
MAX_SEGUNDOS_REFERENCIA = 25.0
SILENCIO_ENTRE_ARQUIVOS = 0.3
TAXA_REFERENCIA = 24000


@dataclass
class Personagem:
    """Um personagem e os áudios que representam a voz dele."""
    nome: str
    arquivos: list[str] = field(default_factory=list)
    ref_text: str = ""
    notas: str = ""
    criado_em: str = ""

    def para_dict(self) -> dict:
        return {
            "nome": self.nome,
            "arquivos": list(self.arquivos),
            "ref_text": self.ref_text,
            "notas": self.notas,
            "criado_em": self.criado_em,
        }

    @staticmethod
    def de_dict(d: dict) -> "Personagem":
        return Personagem(
            nome=str(d.get("nome", "")).strip(),
            arquivos=[str(a) for a in d.get("arquivos", [])],
            ref_text=str(d.get("ref_text", "")),
            notas=str(d.get("notas", "")),
            criado_em=str(d.get("criado_em", "")),
        )


def _chave(nome: str) -> str:
    """Compara nomes ignorando acento e caixa, para não criar duplicata boba."""
    sem_acento = unicodedata.normalize("NFKD", nome)
    sem_acento = "".join(c for c in sem_acento if not unicodedata.combining(c))
    return sem_acento.strip().lower()


def _nome_seguro(nome: str) -> str:
    """Nome de arquivo derivado do personagem, sem caractere problemático."""
    base = _chave(nome).replace(" ", "_")
    return "".join(c for c in base if c.isalnum() or c in "_-") or "personagem"


class VoiceLibrary:
    """
    Registro de personagens em disco. Todas as operações salvam na hora — a
    interface não precisa lembrar de gravar.
    """

    def __init__(self, caminho_registro: str | Path | None = None):
        self.caminho = Path(caminho_registro) if caminho_registro else ARQUIVO_REGISTRO
        self.personagens: list[Personagem] = []
        self.carregar()

    # --- persistência -------------------------------------------------------

    def carregar(self) -> list[Personagem]:
        self.personagens = []
        if not self.caminho.exists():
            return self.personagens
        try:
            with open(self.caminho, "r", encoding="utf-8") as f:
                dados = json.load(f)
            self.personagens = [Personagem.de_dict(d) for d in dados.get("personagens", [])]
            self.personagens = [p for p in self.personagens if p.nome]
        except (json.JSONDecodeError, OSError) as e:
            print(f"[VoiceLibrary] Nao foi possivel ler '{self.caminho}': {e}")
        return self.personagens

    def salvar(self):
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        dados = {"versao": 1, "personagens": [p.para_dict() for p in self.personagens]}
        with open(self.caminho, "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

    # --- consultas ----------------------------------------------------------

    def nomes(self) -> list[str]:
        return sorted((p.nome for p in self.personagens), key=_chave)

    def obter(self, nome: str) -> Personagem | None:
        alvo = _chave(nome)
        for p in self.personagens:
            if _chave(p.nome) == alvo:
                return p
        return None

    def caminho_absoluto(self, arquivo: str) -> Path:
        """Aceita caminho relativo à raiz do projeto ou absoluto."""
        p = Path(arquivo)
        return p if p.is_absolute() else (RAIZ / p)

    def arquivos_existentes(self, nome: str) -> list[Path]:
        p = self.obter(nome)
        if p is None:
            return []
        return [c for c in (self.caminho_absoluto(a) for a in p.arquivos) if c.exists()]

    def duracao_total(self, nome: str) -> float:
        total = 0.0
        for caminho in self.arquivos_existentes(nome):
            try:
                total += float(sf.info(str(caminho)).duration)
            except Exception:
                pass
        return total

    # --- edição -------------------------------------------------------------

    def criar(self, nome: str) -> Personagem:
        nome = nome.strip()
        if not nome:
            raise ValueError("O nome do personagem nao pode ser vazio.")
        if self.obter(nome) is not None:
            raise ValueError(f"Ja existe um personagem chamado '{nome}'.")
        p = Personagem(nome=nome, criado_em=datetime.now().strftime("%Y-%m-%d %H:%M"))
        self.personagens.append(p)
        self.salvar()
        return p

    def renomear(self, nome: str, novo_nome: str):
        novo_nome = novo_nome.strip()
        if not novo_nome:
            raise ValueError("O nome do personagem nao pode ser vazio.")
        p = self.obter(nome)
        if p is None:
            raise ValueError(f"Personagem '{nome}' nao encontrado.")
        existente = self.obter(novo_nome)
        if existente is not None and existente is not p:
            raise ValueError(f"Ja existe um personagem chamado '{novo_nome}'.")
        self._limpar_cache(p.nome)
        p.nome = novo_nome
        self.salvar()

    def remover(self, nome: str):
        p = self.obter(nome)
        if p is None:
            return
        self._limpar_cache(p.nome)
        self.personagens = [x for x in self.personagens if x is not p]
        self.salvar()

    def adicionar_arquivos(self, nome: str, caminhos: list[str | Path]) -> int:
        """Adiciona áudios ao personagem. Retorna quantos entraram de fato."""
        p = self.obter(nome)
        if p is None:
            raise ValueError(f"Personagem '{nome}' nao encontrado.")

        ja_tem = {str(self.caminho_absoluto(a)).lower() for a in p.arquivos}
        novos = 0
        for caminho in caminhos:
            absoluto = Path(caminho).resolve()
            if not absoluto.exists() or str(absoluto).lower() in ja_tem:
                continue
            # Guarda relativo quando o arquivo esta dentro do projeto.
            try:
                registro = str(absoluto.relative_to(RAIZ)).replace("\\", "/")
            except ValueError:
                registro = str(absoluto)
            p.arquivos.append(registro)
            ja_tem.add(str(absoluto).lower())
            novos += 1

        if novos:
            self._limpar_cache(p.nome)
            self.salvar()
        return novos

    def remover_arquivo(self, nome: str, arquivo: str):
        p = self.obter(nome)
        if p is None:
            return
        alvo = str(self.caminho_absoluto(arquivo)).lower()
        p.arquivos = [a for a in p.arquivos if str(self.caminho_absoluto(a)).lower() != alvo]
        self._limpar_cache(p.nome)
        self.salvar()

    def definir_ref_text(self, nome: str, texto: str):
        p = self.obter(nome)
        if p is None:
            return
        p.ref_text = texto.strip()
        self.salvar()

    # --- referência ---------------------------------------------------------

    def _caminho_cache(self, nome: str) -> Path:
        return PASTA_CACHE / f"{_nome_seguro(nome)}.wav"

    def _limpar_cache(self, nome: str):
        alvo = self._caminho_cache(nome)
        try:
            alvo.unlink()
        except OSError:
            pass

    def referencia(self, nome: str, max_segundos: float = MAX_SEGUNDOS_REFERENCIA) -> str:
        """
        Devolve UM arquivo de áudio para usar como referência deste personagem.

        Com um arquivo só, devolve ele mesmo. Com vários, junta os áudios (com
        um respiro entre eles) até o limite, porque mais material de voz melhora
        a clonagem. O resultado fica em cache e é refeito quando a lista muda.
        """
        arquivos = self.arquivos_existentes(nome)
        if not arquivos:
            raise ValueError(
                f"O personagem '{nome}' nao tem nenhum arquivo de audio valido. "
                f"Adicione pelo menos um na aba de Personagens."
            )

        if len(arquivos) == 1:
            return str(arquivos[0])

        cache = self._caminho_cache(nome)
        if cache.exists():
            # Refaz se algum arquivo de origem mudou depois do cache.
            try:
                if cache.stat().st_mtime >= max(a.stat().st_mtime for a in arquivos):
                    return str(cache)
            except OSError:
                pass

        import librosa

        respiro = np.zeros(int(SILENCIO_ENTRE_ARQUIVOS * TAXA_REFERENCIA), dtype=np.float32)
        pedacos: list[np.ndarray] = []
        acumulado = 0.0
        for caminho in arquivos:
            if acumulado >= max_segundos:
                break
            try:
                y, _ = librosa.load(str(caminho), sr=TAXA_REFERENCIA, mono=True)
            except Exception as e:
                print(f"[VoiceLibrary] Ignorando '{caminho.name}': {e}")
                continue
            restante = max_segundos - acumulado
            if len(y) / TAXA_REFERENCIA > restante:
                y = y[: int(restante * TAXA_REFERENCIA)]
            if not len(y):
                continue
            pedacos.append(np.ascontiguousarray(y, dtype=np.float32))
            pedacos.append(respiro)
            acumulado += len(y) / TAXA_REFERENCIA

        if not pedacos:
            raise ValueError(f"Nenhum audio de '{nome}' pode ser lido.")

        juntado = np.concatenate(pedacos[:-1])
        cache.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(cache), juntado, TAXA_REFERENCIA)
        print(f"[VoiceLibrary] Referencia de '{nome}': {len(arquivos)} arquivo(s), "
              f"{len(juntado)/TAXA_REFERENCIA:.1f}s -> {cache.name}")
        return str(cache)

    def resumo(self, nome: str) -> str:
        """Linha curta para mostrar na interface."""
        arquivos = self.arquivos_existentes(nome)
        p = self.obter(nome)
        faltando = (len(p.arquivos) - len(arquivos)) if p else 0
        texto = f"{len(arquivos)} arquivo(s), {self.duracao_total(nome):.1f}s"
        if faltando:
            texto += f" — {faltando} nao encontrado(s)"
        return texto
